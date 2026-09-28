"""Cache one month of Drive Stats (all drives, selected columns) into a local parquet file.

Rationale: the remote Iceberg scan is the slow part of the pipeline. Building the registry, the
cohort panel and the fleet table each need the same underlying month, so scanning remotely three
times is wasteful. This does a single remote scan into a local cache, after which every later
stage reads locally.

Usage:
    python3 fetch_month_cache.py --start 2026-03-01 --end 2026-04-01
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_local import PANEL_COLS, connect_pinned, log, validate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/hpe_capstone"))
    ap.add_argument("--start", default="2026-03-01")
    ap.add_argument("--end", default="2026-04-01", help="exclusive upper bound")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    os.makedirs(args.root, exist_ok=True)
    out = f"{args.root}/raw_{start.year}_{start.month:02d}.parquet"
    if os.path.exists(out):
        log(f"cache already present: {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
        return

    con = connect_pinned()
    validate(con, start, end - __import__("datetime").timedelta(days=1))

    cols = ", ".join(PANEL_COLS)
    log(f"extracting {len(PANEL_COLS)} columns for {start} .. {end} (exclusive)")
    t0 = time.time()
    con.execute(
        f"COPY (SELECT {cols} FROM drivestats "
        f"WHERE date >= DATE '{start}' AND date < DATE '{end}') "
        f"TO '{out}' (FORMAT parquet, COMPRESSION zstd)"
    )
    log(f"wrote {out} {os.path.getsize(out) / 1e6:.0f} MB in {time.time() - t0:.0f}s")

    summary = con.execute(
        f"SELECT count(*) AS rows, count(DISTINCT serial_number) AS drives, "
        f"sum(failure) AS failures, count(DISTINCT date) AS days "
        f"FROM read_parquet('{out}')"
    ).fetchdf()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
