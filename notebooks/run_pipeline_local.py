"""Run the notebook 01 + 02 stages locally against the cached month parquet.

This is the scoped local equivalent of the Colab flow: it proves the pipeline logic end to end
on one month of real Drive Stats without needing Drive or a long remote extraction. Everything
here is identical in substance to the notebooks; only the window is smaller.

Produces artifacts and figures under --root (default ~/hpe_capstone).
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import polars as pl

plt.rcParams.update({
    "figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})

TEAL, RUST, SAND = "#1f7a6f", "#c1553b", "#8a6d3b"


def log(m: str) -> None:
    print(m, flush=True)


def header(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def manufacturer_expr() -> pl.Expr:
    m = pl.col("model").str.to_uppercase()
    return (pl.when(m.str.starts_with("ST")).then(pl.lit("Seagate"))
              .when(m.str.contains(r"^(WDC|WUH|WSH|WD)")).then(pl.lit("WDC"))
              .when(m.str.contains(r"^(HGST|HUH|HMS|HDS)")).then(pl.lit("HGST"))
              .when(m.str.contains(r"^(TOSHIBA|MD|MG|MQ)")).then(pl.lit("Toshiba"))
              .otherwise(pl.lit("Other")).alias("manufacturer"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/hpe_capstone"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--healthy", type=int, default=6000)
    ap.add_argument("--figdir", default="/Users/aryan/HPE/notebooks/figures")
    args = ap.parse_args()

    cache = args.cache or f"{args.root}/raw_2026_03.parquet"
    os.makedirs(args.figdir, exist_ok=True)
    lf = pl.scan_parquet(cache)

    raw_cols = sorted(c for c in lf.collect_schema().names() if c.endswith("_raw"))
    norm_cols = sorted(c for c in lf.collect_schema().names() if c.endswith("_normalized"))

    # ---------------------------------------------------------------- 1. overview
    header("1. FLEET OVERVIEW (notebook 01 section 4 / notebook 02 section 2)")
    ov = lf.select(rows=pl.len(), drives=pl.col("serial_number").n_unique(),
                   days=pl.col("date").n_unique(), lo=pl.col("date").min(),
                   hi=pl.col("date").max(), failures=pl.col("failure").sum()).collect()
    for k, v in ov.to_dicts()[0].items():
        log(f"  {k:10s} {v:,}" if isinstance(v, (int, float)) else f"  {k:10s} {v}")

    topo = lf.select(
        rows=pl.len(),
        has_datacenter=pl.col("datacenter").count(),
        has_cluster=pl.col("cluster_id").count(),
        has_vault=pl.col("vault_id").count(),
        has_pod=pl.col("pod_id").count(),
        has_slot=pl.col("pod_slot_num").count(),
    ).collect().to_dicts()[0]
    n = topo["rows"]
    log("\n  topology population:")
    for k in ("has_datacenter", "has_cluster", "has_vault", "has_pod", "has_slot"):
        log(f"    {k:16s} {100 * topo[k] / n:6.2f}%")

    # ---------------------------------------------------------------- 2. registry
    header("2. DRIVE REGISTRY (notebook 01 section 5)")
    registry = (lf.group_by("serial_number")
                  .agg(model=pl.col("model").first(),
                       capacity_bytes=pl.col("capacity_bytes").max(),
                       first_seen=pl.col("date").min(),
                       last_seen=pl.col("date").max(),
                       n_days=pl.len(),
                       ever_failed=pl.col("failure").max(),
                       failure_date=pl.when(pl.col("failure") == 1)
                                      .then(pl.col("date")).otherwise(None).max(),
                       vault_id=pl.col("vault_id").first())
                  .collect())
    registry = registry.with_columns(manufacturer_expr(),
                                     capacity_tb=(pl.col("capacity_bytes") / 1e12).round(1))
    reg_path = f"{args.root}/drive_registry.parquet"
    registry.write_parquet(reg_path, compression="zstd")
    log(f"  drives {len(registry):,} | failed {registry['ever_failed'].sum():,}")
    log(f"  saved {reg_path} ({os.path.getsize(reg_path)/1e6:.1f} MB)")

    fleet_last = registry["last_seen"].max()
    censored = ((registry["last_seen"] < fleet_last) & (registry["ever_failed"] == 0)).sum()
    log(f"  fleet last day {fleet_last} | left early without failure flag: {censored:,} "
        f"({100*censored/len(registry):.1f}%)  <- censoring, NOT negatives")
    log(f"  observed days per drive: median={registry['n_days'].median()} "
        f"min={registry['n_days'].min()} max={registry['n_days'].max()}")

    # ---------------------------------------------------------------- 3. manufacturer / AFR
    header("3. MANUFACTURER MIX AND FAILURE RATES (notebook 02 sections 3, 5)")
    rp = registry.to_pandas()
    mfr = (rp.groupby("manufacturer")
             .agg(drives=("serial_number", "size"), failures=("ever_failed", "sum"),
                  drive_days=("n_days", "sum"))
             .assign(afr_pct=lambda d: (d.failures / d.drive_days * 365 * 100).round(2))
             .sort_values("drives", ascending=False))
    log(mfr.to_string())

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].bar(mfr.index, mfr.drives, color=TEAL)
    ax[0].set_title("Drives by manufacturer"); ax[0].tick_params(axis="x", rotation=15)
    ax[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v/1000:.0f}k"))
    sub = mfr[mfr.drives > 500]
    ax[1].bar(sub.index, sub.afr_pct, color=RUST)
    ax[1].set_title("Annualised failure rate (%)"); ax[1].tick_params(axis="x", rotation=15)
    plt.tight_layout(); plt.savefig(f"{args.figdir}/01_manufacturer.png"); plt.close()

    fleet_afr = rp.ever_failed.sum() / rp.n_days.sum() * 365 * 100
    log(f"\n  fleet AFR over this month: {fleet_afr:.2f}%  "
        f"[Backblaze publish 1.24% for the full Q1 2026]")
    daily_hazard = rp.ever_failed.sum() / rp.n_days.sum()
    log(f"  per-drive-day failure probability: {daily_hazard:.6f} (1 in {1/daily_hazard:,.0f})")
    for h in (7, 14, 30):
        log(f"    ~{h:2d}-day positive prevalence: {daily_hazard*h*100:.3f}%")

    # ---------------------------------------------------------------- 4. SMART availability
    header("4. SMART AVAILABILITY BY MANUFACTURER (notebook 02 section 4)")
    mfr_lookup = registry.select(["serial_number", "manufacturer"])
    matrix = (lf.select(["serial_number"] + raw_cols)
                .join(mfr_lookup.lazy(), on="serial_number", how="left")
                .group_by("manufacturer")
                .agg([(100 * pl.col(c).is_not_null().mean()).round(1).alias(c)
                      for c in raw_cols])
                .collect().to_pandas().set_index("manufacturer"))
    keep = mfr[mfr.drives > 500].index
    matrix = matrix.loc[matrix.index.isin(keep)]
    matrix.columns = [c.replace("smart_", "").replace("_raw", "") for c in matrix.columns]
    matrix = matrix.reindex(sorted(matrix.columns, key=int), axis=1)

    partial = matrix.loc[:, (matrix < 99).any()]
    log("  attributes that are NOT universally available (% of rows non-null):")
    log(partial.to_string())

    fig, ax = plt.subplots(figsize=(13, 2.6))
    im = ax.imshow(matrix.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(matrix.columns))); ax.set_xticklabels(matrix.columns, fontsize=7)
    ax.set_yticks(range(len(matrix.index))); ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title("SMART raw attribute availability by manufacturer (% non-null)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.values[i, j]
            if v < 99:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01); ax.grid(False)
    plt.tight_layout(); plt.savefig(f"{args.figdir}/02_smart_availability.png"); plt.close()

    # ---------------------------------------------------------------- 5. bathtub
    header("5. AGE AND CAPACITY (notebook 02 section 5)")
    age = (lf.select(["serial_number", "smart_9_raw"])
             .filter(pl.col("smart_9_raw").is_not_null())
             .group_by("serial_number").agg(poh=pl.col("smart_9_raw").max())
             .collect().to_pandas())
    age["age_years"] = age.poh / 8766.0
    age = age.merge(rp[["serial_number", "ever_failed", "n_days"]], on="serial_number")
    bins = [0, .5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 100]
    labels = ["<0.5", ".5-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-10", "10+"]
    age["bracket"] = pd.cut(age.age_years, bins=bins, labels=labels)
    bath = (age.groupby("bracket", observed=True)
              .agg(drives=("serial_number", "size"), failures=("ever_failed", "sum"),
                   drive_days=("n_days", "sum"))
              .assign(afr_pct=lambda d: (d.failures / d.drive_days * 365 * 100).round(2)))
    log(bath.to_string())

    cap = (rp[rp.capacity_tb > 0].groupby("capacity_tb")
             .agg(drives=("serial_number", "size"), failures=("ever_failed", "sum"),
                  drive_days=("n_days", "sum"))
             .query("drives >= 500")
             .assign(afr_pct=lambda d: (d.failures / d.drive_days * 365 * 100).round(2)))
    log("\n" + cap.to_string())

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(bath.index.astype(str), bath.afr_pct, "o-", color=RUST, lw=1.8)
    ax[0].set_title("Bathtub curve: AFR vs drive age"); ax[0].set_xlabel("age (years)")
    ax[0].tick_params(axis="x", rotation=40)
    ax[1].bar(cap.index.astype(str), cap.afr_pct, color=RUST)
    ax[1].set_title("AFR by capacity"); ax[1].set_xlabel("TB")
    plt.tight_layout(); plt.savefig(f"{args.figdir}/03_age_capacity.png"); plt.close()

    # ---------------------------------------------------------------- 6. tome geometry
    header("6. TOME GEOMETRY AND THE SAFETY INVARIANT (notebook 02 section 6)")
    ref = lf.select(pl.col("date").max()).collect().item()
    snap = (lf.filter(pl.col("date") == ref)
              .select(["serial_number", "capacity_bytes", "vault_id", "pod_id", "pod_slot_num"])
              .collect())
    log(f"  reference day {ref}: {len(snap):,} drives")

    tome = (snap.filter(pl.col("pod_slot_num").is_not_null())
                .group_by(["vault_id", "pod_slot_num"]).agg(n=pl.len()))
    log("\n  drives per redundancy set (vault_id, pod_slot_num):")
    log(tome["n"].value_counts().sort("count", descending=True).head(6)
        .to_pandas().to_string(index=False))

    pods = snap.group_by("vault_id").agg(pods=pl.col("pod_id").n_unique())
    log("\n  pods per vault:")
    log(pods["pods"].value_counts().sort("count", descending=True).head(5)
        .to_pandas().to_string(index=False))

    triple = (snap.filter(pl.col("pod_slot_num").is_not_null())
                  .group_by(["vault_id", "pod_id", "pod_slot_num"]).agg(n=pl.len()))
    log("\n  drives per (vault, pod, slot) triple -- 1 means the triple is a unique drive slot:")
    log(triple["n"].value_counts().sort("n").head(5).to_pandas().to_string(index=False))

    vc = (snap.group_by("vault_id")
              .agg(cap_tb=(pl.col("capacity_bytes").median() / 1e12).round(1), drives=pl.len())
              .with_columns(parity=pl.when(pl.col("cap_tb") >= 14).then(pl.lit(5))
                                     .when(pl.col("cap_tb") >= 10).then(pl.lit(4))
                                     .otherwise(pl.lit(3)))
              .to_pandas())
    log(f"\n  vaults {vc.vault_id.nunique():,} | parity budget assignment:")
    log(vc.groupby(["cap_tb", "parity"]).agg(vaults=("vault_id", "size"),
                                             drives=("drives", "sum")).to_string())

    # ---------------------------------------------------------------- 7. cohort + panel
    header("7. COHORT AND PANEL (notebook 01 sections 6, 8)")
    rng = np.random.default_rng(42)
    med_days = int(registry["n_days"].median())
    min_days = max(7, med_days // 2)
    failed = rp[rp.ever_failed == 1].copy()
    healthy = rp[(rp.ever_failed == 0) & (rp.n_days >= min_days)].copy()
    frac = min(1.0, args.healthy / max(len(healthy), 1))
    picked = []
    for _, grp in healthy.groupby("model", observed=True):
        take = min(len(grp), max(min(len(grp), 10), int(round(len(grp) * frac))))
        idx = rng.choice(grp.index.values, size=take, replace=False)
        sub = grp.loc[idx].copy(); sub["sample_weight"] = len(grp) / take
        picked.append(sub)
    healthy_s = pd.concat(picked); failed["sample_weight"] = 1.0
    cohort = pd.concat([failed, healthy_s]).reset_index(drop=True)
    log(f"  cohort {len(cohort):,} drives = {len(failed):,} failed + {len(healthy_s):,} healthy "
        f"(sampled from {len(healthy):,}, min {min_days} days)")
    cohort.to_parquet(f"{args.root}/cohort.parquet", index=False, compression="zstd")

    os.makedirs(f"{args.root}/panel", exist_ok=True)
    panel_path = f"{args.root}/panel/panel_2026_03.parquet"
    (lf.join(pl.from_pandas(cohort[["serial_number", "sample_weight"]]).lazy(),
             on="serial_number", how="inner")
       .sort(["serial_number", "date"])
       .collect()
       .write_parquet(panel_path, compression="zstd"))
    log(f"  saved {panel_path} ({os.path.getsize(panel_path)/1e6:.1f} MB)")

    panel = pl.scan_parquet(panel_path)
    per = panel.group_by("serial_number").agg(n=pl.len()).collect()["n"]
    window = panel.select(pl.col("date").n_unique()).collect().item()
    log(f"\n  VALIDATION -- days per drive: median={per.median()} min={per.min()} "
        f"max={per.max()} | window={window} days")
    log(f"  single-day drives: {(per == 1).sum():,} ({100*(per==1).mean():.1f}%)")
    assert per.median() >= 0.5 * window, "panel looks row-sampled, not drive-sampled"
    log("  PASS: per-drive daily time series intact "
        "(the earlier processed file had median 2 days and 47% single-day drives)")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(per.to_numpy(), bins=31, color=TEAL, edgecolor="white", linewidth=.4)
    ax.set_title("Observed days per drive (panel)"); ax.set_xlabel("days")
    plt.tight_layout(); plt.savefig(f"{args.figdir}/04_days_per_drive.png"); plt.close()

    # ---------------------------------------------------------------- 8. trajectories
    header("8. PRE-FAILURE TRAJECTORIES AND LEAD TIME (notebook 02 sections 7, 8)")
    fd = (registry.filter(pl.col("ever_failed") == 1)
                  .select(["serial_number", "failure_date"])
                  .drop_nulls())
    traj = (lf.select(["serial_number", "date"] + raw_cols)
              .join(fd.lazy(), on="serial_number", how="inner")
              .with_columns(dtf=(pl.col("failure_date") - pl.col("date")).dt.total_days())
              .filter((pl.col("dtf") >= 0) & (pl.col("dtf") <= 30))
              .collect())
    log(f"  failed drives {len(fd):,} | pre-failure rows {len(traj):,}")

    healthy_ser = registry.filter(pl.col("ever_failed") == 0)["serial_number"]
    baseline = (lf.select(["serial_number"] + raw_cols)
                  .filter(pl.col("serial_number").is_in(healthy_ser.to_list()))
                  .select([pl.col(c).median().alias(c) for c in raw_cols]).collect())

    focus = [c for c in ["smart_5_raw", "smart_197_raw", "smart_198_raw", "smart_187_raw",
                         "smart_1_raw", "smart_7_raw", "smart_194_raw", "smart_199_raw",
                         "smart_9_raw"] if c in raw_cols]
    curves = (traj.group_by("dtf").agg([pl.col(c).median().alias(c) for c in focus])
                  .sort("dtf", descending=True).to_pandas())
    fig, axes = plt.subplots(3, 3, figsize=(12, 7.5))
    for ax, c in zip(axes.ravel(), focus):
        ax.plot(curves.dtf, curves[c], color=RUST, lw=1.6, label="failing")
        b = baseline[c][0]
        if b is not None:
            ax.axhline(b, color=TEAL, ls="--", lw=1.1, label="healthy median")
        ax.invert_xaxis(); ax.set_title(c.replace("_raw", ""), fontsize=9)
        ax.set_xlabel("days before failure", fontsize=7); ax.legend(fontsize=6)
    for ax in axes.ravel()[len(focus):]:
        ax.axis("off")
    plt.suptitle("Median attribute trajectory approaching failure (failure at right)", y=1.0)
    plt.tight_layout(); plt.savefig(f"{args.figdir}/05_trajectories.png"); plt.close()

    sector = [c for c in ["smart_5_raw", "smart_197_raw", "smart_198_raw"] if c in raw_cols]
    onset = (traj.with_columns(
                 anom=pl.any_horizontal([pl.col(c) > 0 for c in sector]))
                 .filter(pl.col("anom"))
                 .group_by("serial_number").agg(lead=pl.col("dtf").max())
                 .to_pandas())
    nf, nw = len(fd), len(onset)
    log(f"\n  showed a sector anomaly within 30d : {nw:,} / {nf:,} ({100*nw/nf:.1f}%)")
    log(f"  NEVER showed one (silent failures)  : {nf-nw:,} ({100*(nf-nw)/nf:.1f}%)"
        f"  <- ceiling on recall from these attributes alone")
    if nw:
        log("\n  lead time from first anomaly to failure (days):")
        log("  " + onset.lead.describe(percentiles=[.1, .25, .5, .75, .9])
            .round(1).to_string().replace("\n", "\n  "))
        log(f"\n  measured median lead : {onset.lead.median():.0f} days")
        log(f"  SMART-Z reported     : 7 days median (max 56) on an independent fleet")
        for h in (7, 14, 30):
            log(f"    caught with >= {h:2d}d lead : {(onset.lead >= h).mean()*100:5.1f}%")

        fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
        ax[0].hist(onset.lead, bins=30, color=TEAL, edgecolor="white", linewidth=.4)
        ax[0].axvline(7, color=RUST, ls="--", lw=1.4, label="7d horizon")
        ax[0].axvline(onset.lead.median(), color="black", ls=":", lw=1.4,
                      label=f"median {onset.lead.median():.0f}d")
        ax[0].set_title("Warning lead time available"); ax[0].set_xlabel("days before failure")
        ax[0].legend(fontsize=7)
        s = np.sort(onset.lead.values)
        ax[1].plot(s, 1 - np.arange(len(s)) / len(s), color=TEAL, lw=1.8)
        ax[1].axvline(7, color=RUST, ls="--", lw=1.2)
        ax[1].set_title("Fraction of failures with >= X days warning"); ax[1].set_xlabel("days")
        plt.tight_layout(); plt.savefig(f"{args.figdir}/06_lead_time.png"); plt.close()

    header("DONE")
    tot = 0.0
    for r, _, fs in os.walk(args.root):
        for f in fs:
            if f.endswith(".parquet"):
                tot += os.path.getsize(os.path.join(r, f)) / 1e6
    log(f"  artifacts: {tot:.1f} MB under {args.root}")
    log(f"  figures written to {args.figdir}")


if __name__ == "__main__":
    main()
