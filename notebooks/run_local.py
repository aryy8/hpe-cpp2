"""Run the notebook 01 + 02 pipeline locally, outside Colab.

Local disk is the binding constraint (a couple of GB free), so this is parameterised to run a
scoped smoke version of the pipeline by default: a shorter window and a smaller healthy
cohort than the Colab configuration. The logic is identical to the notebooks, so a clean run
here means the notebooks are sound.

Usage:
    python3 run_local.py --warmup 2026-02-01 --start 2026-03-01 --end 2026-03-31 --healthy 5000

Every stage is resumable: existing parquet files are skipped.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2_public_creds as bc

# --------------------------------------------------------------------------- SMART selection
SMART_CORE = [5, 187, 188, 197, 198]
SMART_ERR = [1, 7, 10, 183, 184, 199]
SMART_WEAR = [4, 9, 12, 192, 193, 240, 241, 242]
SMART_ENV = [3, 189, 190, 191, 194]
SMART_IDS = sorted(set(SMART_CORE + SMART_ERR + SMART_WEAR + SMART_ENV))
NORMALIZED_IDS = sorted(set(SMART_CORE + [1, 7, 10, 184, 199]))

META_COLS = ["date", "serial_number", "model", "capacity_bytes", "failure",
             "datacenter", "cluster_id", "vault_id", "pod_id", "pod_slot_num"]
PANEL_COLS = (META_COLS
              + [f"smart_{i}_raw" for i in SMART_IDS]
              + [f"smart_{i}_normalized" for i in NORMALIZED_IDS])


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 1e9


def month_starts(a: date, b: date) -> list[tuple[int, int]]:
    out, y, m = [], a.year, a.month
    while (y, m) <= (b.year, b.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def month_bounds(y: int, m: int) -> tuple[date, date]:
    return date(y, m, 1), (date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1))


# --------------------------------------------------------------------------- connection
def connect_pinned(verbose: bool = True):
    """Connect and pin the newest Iceberg snapshot explicitly.

    Relying on `version => '?'` guessing is what silently produced a zero-row table in Colab:
    the guess resolved a stale metadata file, the scan succeeded, and every count came back 0.
    Resolving the newest snapshot id and pinning it makes the choice explicit, and the
    validation below turns any remaining mismatch into a loud failure instead of empty data.
    """
    import duckdb

    key_id, secret = bc.get_credentials(verbose=verbose)
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL iceberg; LOAD iceberg;")
    con.execute(
        "CREATE OR REPLACE SECRET b2 (TYPE s3, KEY_ID ?, SECRET ?, REGION ?, ENDPOINT ?);",
        [key_id, secret, bc.REGION, bc.ENDPOINT],
    )
    con.execute("SET unsafe_enable_version_guessing = true;")

    snap = con.execute(
        f"SELECT snapshot_id, timestamp_ms FROM iceberg_snapshots('{bc.ICEBERG_URI}', "
        f"version => '?') ORDER BY sequence_number DESC LIMIT 1"
    ).fetchone()
    if snap is None:
        raise RuntimeError("no Iceberg snapshots returned - cannot resolve the table")
    snapshot_id, ts = snap
    log(f"pinned snapshot {snapshot_id} (committed {ts})")

    con.execute(
        f"CREATE OR REPLACE VIEW drivestats AS SELECT * FROM iceberg_scan("
        f"'{bc.ICEBERG_URI}', snapshot_from_id => {snapshot_id}, allow_moved_paths => true);"
    )
    return con


def validate(con, need_from: date, need_to: date) -> None:
    """Fail loudly if the resolved snapshot cannot cover the requested window.

    Deliberately probes the two boundary dates rather than asking for a global
    min(date)/max(date). The table spans 2013 to the present, so an unfiltered aggregate has to
    touch every year and takes many minutes, whereas a date-equality predicate is pushed down
    to a single day's files and returns in seconds.
    """
    for label, d in (("window start", need_from), ("window end", need_to)):
        t0 = time.time()
        n = con.execute("SELECT count(*) FROM drivestats WHERE date = ?", [d]).fetchone()[0]
        log(f"probe {label} {d}: {n:,} rows ({time.time() - t0:.0f}s)")
        if n == 0:
            raise RuntimeError(
                f"{d} returned ZERO rows. This is the stale-snapshot failure mode: the scan "
                "succeeds but the resolved metadata predates the data, so everything downstream "
                "sees plausible-looking zeros. Upgrade duckdb (pip install -U duckdb), restart "
                "the kernel, and re-run."
            )
        if n < 100_000:
            raise RuntimeError(
                f"only {n:,} rows on {d}; expected >300k for this fleet. Investigate before "
                "extracting."
            )
    log("validation PASSED")


# --------------------------------------------------------------------------- stages
def build_registry(con, path: str, warmup: date, end: date) -> pd.DataFrame:
    if os.path.exists(path):
        log(f"registry exists, loading {path}")
        return pd.read_parquet(path)

    log("building drive registry (scans the window on a few columns)")
    t0 = time.time()
    reg = con.execute(
        """
        SELECT serial_number,
               any_value(model)        AS model,
               max(capacity_bytes)     AS capacity_bytes,
               min(date)               AS first_seen,
               max(date)               AS last_seen,
               count(*)                AS n_days,
               max(failure)            AS ever_failed,
               max(CASE WHEN failure = 1 THEN date END) AS failure_date,
               any_value(datacenter)   AS datacenter,
               any_value(cluster_id)   AS cluster_id,
               any_value(vault_id)     AS vault_id
        FROM drivestats
        WHERE date BETWEEN ? AND ?
        GROUP BY serial_number
        """,
        [warmup, end],
    ).fetchdf()
    log(f"registry: {len(reg):,} drives, {int(reg.ever_failed.sum()):,} failed "
        f"({time.time() - t0:.0f}s)")
    reg.to_parquet(path, index=False, compression="zstd")
    return reg


def select_cohort(reg: pd.DataFrame, path: str, n_healthy: int, seed: int = 42) -> pd.DataFrame:
    if os.path.exists(path):
        log(f"cohort exists, loading {path}")
        return pd.read_parquet(path)

    rng = np.random.default_rng(seed)
    min_days = max(7, int(reg.n_days.median() * 0.5))

    failed = reg[reg.ever_failed == 1].copy()
    healthy = reg[(reg.ever_failed == 0) & (reg.n_days >= min_days)].copy()
    log(f"eligible healthy drives (>= {min_days} days): {len(healthy):,}")

    frac = min(1.0, n_healthy / max(len(healthy), 1))
    picked = []
    for _, grp in healthy.groupby("model", observed=True):
        take = min(len(grp), max(min(len(grp), 10), int(round(len(grp) * frac))))
        idx = rng.choice(grp.index.values, size=take, replace=False)
        sub = grp.loc[idx].copy()
        sub["sample_weight"] = len(grp) / take
        picked.append(sub)

    healthy_s = pd.concat(picked)
    failed["sample_weight"] = 1.0
    cohort = pd.concat([failed, healthy_s]).reset_index(drop=True)

    log(f"cohort: {len(cohort):,} drives ({len(failed):,} failed + {len(healthy_s):,} healthy), "
        f"expected ~{int(cohort.n_days.sum()):,} panel rows")
    cohort.to_parquet(path, index=False, compression="zstd")
    return cohort


def materialise_panel(con, cohort: pd.DataFrame, outdir: str,
                      warmup: date, end: date, min_free_gb: float) -> None:
    os.makedirs(outdir, exist_ok=True)
    con.register("cohort_df", cohort[["serial_number", "sample_weight"]])
    con.execute("CREATE OR REPLACE TABLE cohort_tbl AS SELECT * FROM cohort_df")

    sel = ", ".join(f"d.{c}" for c in PANEL_COLS)
    for y, m in month_starts(warmup, end):
        out = f"{outdir}/panel_{y}_{m:02d}.parquet"
        if os.path.exists(out):
            log(f"skip panel {y}-{m:02d} ({os.path.getsize(out)/1e6:.0f} MB)")
            continue
        if free_gb(outdir) < min_free_gb:
            log(f"STOP: only {free_gb(outdir):.2f} GB free, below the {min_free_gb} GB floor")
            return
        a, b = month_bounds(y, m)
        t0 = time.time()
        log(f"extracting panel {y}-{m:02d} ...")
        con.execute(
            f"COPY (SELECT {sel}, c.sample_weight FROM drivestats d "
            f"JOIN cohort_tbl c USING (serial_number) "
            f"WHERE d.date >= DATE '{a}' AND d.date < DATE '{b}') "
            f"TO '{out}' (FORMAT parquet, COMPRESSION zstd)"
        )
        log(f"  wrote {os.path.basename(out)} {os.path.getsize(out)/1e6:.0f} MB "
            f"in {time.time()-t0:.0f}s | free {free_gb(outdir):.2f} GB")


def materialise_fleet(con, outdir: str, start: date, end: date, min_free_gb: float) -> None:
    os.makedirs(outdir, exist_ok=True)
    for y, m in month_starts(start, end):
        out = f"{outdir}/fleet_{y}_{m:02d}.parquet"
        if os.path.exists(out):
            log(f"skip fleet {y}-{m:02d} ({os.path.getsize(out)/1e6:.0f} MB)")
            continue
        if free_gb(outdir) < min_free_gb:
            log(f"STOP: only {free_gb(outdir):.2f} GB free, below the {min_free_gb} GB floor")
            return
        a, b = month_bounds(y, m)
        t0 = time.time()
        log(f"extracting fleet {y}-{m:02d} ...")
        con.execute(
            f"COPY (SELECT date, serial_number, model, capacity_bytes, failure, datacenter, "
            f"cluster_id, vault_id, pod_id, pod_slot_num FROM drivestats "
            f"WHERE date >= DATE '{a}' AND date < DATE '{b}') "
            f"TO '{out}' (FORMAT parquet, COMPRESSION zstd)"
        )
        log(f"  wrote {os.path.basename(out)} {os.path.getsize(out)/1e6:.0f} MB "
            f"in {time.time()-t0:.0f}s | free {free_gb(outdir):.2f} GB")


def verify_panel(outdir: str) -> None:
    lf = pl.scan_parquet(f"{outdir}/panel_*.parquet")
    s = lf.select(rows=pl.len(), drives=pl.col("serial_number").n_unique(),
                  days=pl.col("date").n_unique(), lo=pl.col("date").min(),
                  hi=pl.col("date").max(), fails=pl.col("failure").sum()).collect()
    log("panel summary: " + ", ".join(f"{k}={v}" for k, v in s.to_dicts()[0].items()))

    per = lf.group_by("serial_number").agg(n=pl.len()).collect()["n"]
    med, one = per.median(), (per == 1).sum()
    log(f"days/drive: median={med} min={per.min()} max={per.max()} | single-day drives={one:,} "
        f"({100*one/len(per):.1f}%)")

    window = s["days"][0]
    assert med >= 0.5 * window, (
        f"median {med} days/drive vs a {window}-day window - the panel looks row-sampled, "
        "which is exactly the defect that invalidated the earlier processed file")
    log("PASS: per-drive daily time series is intact")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.expanduser("~/hpe_capstone"))
    p.add_argument("--warmup", default="2026-02-01")
    p.add_argument("--start", default="2026-03-01")
    p.add_argument("--end", default="2026-03-31")
    p.add_argument("--healthy", type=int, default=5000)
    p.add_argument("--min-free-gb", type=float, default=0.6)
    p.add_argument("--skip-fleet", action="store_true")
    a = p.parse_args()

    warmup = datetime.strptime(a.warmup, "%Y-%m-%d").date()
    start = datetime.strptime(a.start, "%Y-%m-%d").date()
    end = datetime.strptime(a.end, "%Y-%m-%d").date()

    root = a.root
    os.makedirs(root, exist_ok=True)
    log(f"root={root} window: warmup {warmup} -> labelled {start}..{end}")
    log(f"free disk: {free_gb(root):.2f} GB")

    con = connect_pinned()
    validate(con, warmup, end)

    reg = build_registry(con, f"{root}/drive_registry.parquet", warmup, end)
    cohort = select_cohort(reg, f"{root}/cohort.parquet", a.healthy)
    materialise_panel(con, cohort, f"{root}/panel", warmup, end, a.min_free_gb)
    if not a.skip_fleet:
        materialise_fleet(con, f"{root}/fleet_minimal", start, end, a.min_free_gb)
    verify_panel(f"{root}/panel")

    log("DONE")
    total = 0.0
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".parquet"):
                total += os.path.getsize(os.path.join(r, f)) / 1e6
    log(f"total artifacts: {total:.1f} MB | free {free_gb(root):.2f} GB")


if __name__ == "__main__":
    main()
