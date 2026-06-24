"""Load FileMaker CSV exports (/tmp/fmp-export/<source>/*.csv) into the app DB as
snapshot tables `fmp_snap_<table>`.

Why: the FileMaker mirror in Supabase (us-east) is resource-crashed and unreliable,
and the brain-extract memory says not to depend on it at runtime. Instead we extract
the biofield + product tables straight from the local FileMaker file and snapshot
them here, so the Causal Chain Report tool reads a stable local copy.

All columns are TEXT (FileMaker exports are text). Each table is dropped + recreated
so reloads are idempotent. These rows are PII/PHI (client emails, names, chains) —
keep them in the app's runtime DB; never commit the snapshot to git.
"""
import csv
import sqlite3
from pathlib import Path

PREFIX = "fmp_snap_"


def _safe_ident(s):
    out = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in (s or "").strip())
    if not out:
        out = "col"
    if out[0].isdigit():
        out = "_" + out
    return out


def _dedup(cols):
    seen, out = {}, []
    for c in cols:
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        out.append(c)
    return out


def snapshot_csv_dir(export_dir, db_path, prefix=PREFIX):
    """Load every <table>.csv in export_dir into `<prefix><table>` (TEXT cols,
    full replace). Returns {table: row_count}."""
    export_dir = Path(export_dir)
    counts = {}
    with sqlite3.connect(db_path) as cx:
        for csv_path in sorted(export_dir.glob("*.csv")):
            table = csv_path.stem
            tname = prefix + _safe_ident(table)
            with csv_path.open(encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, [])
                cols = _dedup([_safe_ident(h) for h in header])
                cx.execute(f"DROP TABLE IF EXISTS {tname}")
                if not cols:
                    cx.execute(f"CREATE TABLE {tname} (_empty TEXT)")
                    counts[table] = 0
                    continue
                cx.execute(f"CREATE TABLE {tname} ("
                           + ", ".join(f'"{c}" TEXT' for c in cols) + ")")
                ph = ",".join("?" * len(cols))
                n, batch = 0, []
                for row in reader:
                    row = (row + [""] * len(cols))[:len(cols)]
                    batch.append(row)
                    n += 1
                    if len(batch) >= 1000:
                        cx.executemany(f"INSERT INTO {tname} VALUES ({ph})", batch)
                        batch = []
                if batch:
                    cx.executemany(f"INSERT INTO {tname} VALUES ({ph})", batch)
                counts[table] = n
        cx.commit()
    return counts
