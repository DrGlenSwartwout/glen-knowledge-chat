#!/usr/bin/env python3
"""Seed the inventory ledger from FMP baseline (ingredients.extras) + Phase-3b receipts."""
import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard import inventory as inv  # noqa: E402


def _db_path():
    base = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))
    return str(Path(base) / "chat_log.db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=_db_path())
    ap.add_argument("--write", action="store_true", help="commit (default: dry-run, rolled back)")
    args = ap.parse_args()
    cx = sqlite3.connect(args.db)
    cx.row_factory = sqlite3.Row
    try:
        inv.init_inventory_schema(cx)
        nb = inv.seed_baselines(cx)
        nr = inv.seed_receipts(cx)
        if args.write:
            cx.commit()
            print(f"WROTE baselines={nb} receipts={nr}")
        else:
            cx.rollback()
            print(f"DRY-RUN would insert baselines={nb} receipts={nr} (rolled back)")
    finally:
        cx.close()


if __name__ == "__main__":
    main()
