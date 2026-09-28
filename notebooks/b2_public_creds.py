"""Resolve the public read-only Backblaze Drive Stats credentials.

Backblaze publishes a read-only application key for the Drive Stats dataset on its public
documentation page so that anyone can query the Iceberg table. Rather than pasting those
values into source control, this module resolves them at runtime in priority order:

    1. B2_KEY_ID / B2_SECRET environment variables (use these to override).
    2. A local untracked file, default ~/.b2_drivestats (KEY=VALUE lines).
    3. Scraped from Backblaze's public documentation page.

The scrape is last because it is the slowest and the page layout can change; the env vars
are first so CI and notebooks can inject values without touching this file.
"""

from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path

DOCS_URL = "https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data"
CRED_FILE = Path(os.environ.get("B2_CRED_FILE", Path.home() / ".b2_drivestats"))

REGION = "us-west-004"
ENDPOINT = "s3.us-west-004.backblazeb2.com"
ICEBERG_URI = "s3://drivestats-iceberg/drivestats"


def _from_env() -> tuple[str, str] | None:
    k, s = os.environ.get("B2_KEY_ID"), os.environ.get("B2_SECRET")
    return (k, s) if k and s else None


def _from_file() -> tuple[str, str] | None:
    if not CRED_FILE.exists():
        return None
    vals: dict[str, str] = {}
    for line in CRED_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        vals[key.strip()] = val.strip().strip("'\"")
    k, s = vals.get("B2_KEY_ID"), vals.get("B2_SECRET")
    return (k, s) if k and s else None


def _fetch(url: str) -> str:
    # python.org macOS builds ship without system CA roots, so fall back to certifi's bundle
    # rather than disabling verification.
    ctx = None
    try:
        import ssl

        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _from_docs() -> tuple[str, str] | None:
    html = _fetch(DOCS_URL)

    # The docs list the key id and secret as adjacent labelled values. Match on shape: the
    # key id is a long numeric-ish token starting 004, the secret starts with K004.
    key_ids = re.findall(r"\b(00[0-9a-f]{20,30})\b", html)
    secrets = re.findall(r"\b(K00[0-9A-Za-z/+]{20,40})\b", html)
    if key_ids and secrets:
        return key_ids[0], secrets[0]
    return None


def get_credentials(verbose: bool = True) -> tuple[str, str]:
    """Return (key_id, secret), raising if no source yields them."""
    for label, fn in (("environment", _from_env),
                      (f"file {CRED_FILE}", _from_file),
                      ("Backblaze public docs", _from_docs)):
        try:
            got = fn()
        except Exception as exc:  # a failing source should fall through, not abort
            if verbose:
                print(f"  {label}: unavailable ({type(exc).__name__}: {exc})")
            continue
        if got:
            if verbose:
                print(f"  resolved credentials from {label}")
            return got
        if verbose:
            print(f"  {label}: not found")

    raise RuntimeError(
        "Could not resolve Drive Stats credentials. Set B2_KEY_ID and B2_SECRET, or write "
        f"them as KEY=VALUE lines into {CRED_FILE}. The current public values are listed at "
        f"{DOCS_URL}"
    )


def connect(verbose: bool = True):
    """Return a DuckDB connection exposing the Drive Stats Iceberg table as `drivestats`."""
    import duckdb

    key_id, secret = get_credentials(verbose=verbose)
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL iceberg; LOAD iceberg;")
    con.execute(
        "CREATE OR REPLACE SECRET b2 (TYPE s3, KEY_ID ?, SECRET ?, REGION ?, ENDPOINT ?);",
        [key_id, secret, REGION, ENDPOINT],
    )
    # Required so DuckDB can resolve the current metadata version from a bare table location.
    con.execute("SET unsafe_enable_version_guessing = true;")
    con.execute(
        f"CREATE OR REPLACE VIEW drivestats AS SELECT * FROM "
        f"iceberg_scan('{ICEBERG_URI}', version => '?', allow_moved_paths => true);"
    )
    return con


if __name__ == "__main__":
    con = connect()
    print(con.execute(
        "SELECT count(*) AS rows, count(DISTINCT serial_number) AS drives, sum(failure) AS fails "
        "FROM drivestats WHERE date = DATE '2026-03-15'"
    ).fetchdf().to_string(index=False))
