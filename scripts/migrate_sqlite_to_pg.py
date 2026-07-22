#!/usr/bin/env python3
"""Operator CLI for the P05 SQLite -> Postgres migration tool.

    migrate_sqlite_to_pg.py preflight <sqlite_path>
    migrate_sqlite_to_pg.py copy <sqlite_path> [--truncate]
    migrate_sqlite_to_pg.py verify <sqlite_path>
    migrate_sqlite_to_pg.py full <sqlite_path> [--force] [--truncate]

`preflight` scans the SQLite source for rows that would violate a UNIQUE
constraint on the Postgres side, using TWO sources of unique-key knowledge:
  1. the SQLite source's own UNIQUE indexes (scripts.pgmig.dedup.scan_db).
  2. -- when a Postgres backend is configured (DB_BACKEND=postgres +
     PG_DSN) -- the POSTGRES TARGET SCHEMA's own UNIQUE indexes
     (scripts.pgmig.introspect.pg_unique_indexes +
     scripts.pgmig.dedup.scan_against_targets).
(2) exists because a UNIQUE index can silently fail to build on the SQLite
source (CREATE UNIQUE INDEX raises when duplicate rows already exist, and
app code may swallow that) -- the index is then simply absent from SQLite,
so (1) alone never looks at those columns. The fresh Postgres schema still
enforces that index, so skipping (2) would let `copy` fail or silently lose
rows at cutover. Both scans are always attempted for preflight/full; (2) is
skipped (with a printed note, not a silent no-op) only when PG isn't
configured at all -- `preflight` itself never requires Postgres.

`copy`/`verify`/`full` assume the Postgres target schema and its tables
already exist (schema creation is a separate step -- see the P05 task
brief); they operate on the schema `dashboard.dbschema.schema_for_path`
derives from `sqlite_path`'s basename, via the shipped `dashboard.db`
adapter.

Exit codes:
    0   success (clean preflight / copy+verify OK / verify all-ok)
    1   copy produced row-level errors (copy.any_errors == True)
    2   environment misconfigured (DB_BACKEND/PG_DSN missing for a command
        that needs Postgres)
    3   preflight found collisions (and, for `full`, --force wasn't given)
    4   verify found a row-count mismatch
    5   preflight INCOMPLETE: PG is configured (DB_BACKEND=postgres + PG_DSN)
        but the PG-target unique-index cross-check itself failed to run
        (e.g. introspection query error) -- this is NOT the same as the
        legitimate sqlite-only skip (which stays exit 0); a configured
        cross-check that couldn't run must never look like a clean
        preflight, since it's the exact check the P06 cutover gate relies
        on. `full` aborts on this (like a dirty preflight) unless --force.
"""
import argparse
import sys
from typing import Dict, List, Optional, TextIO, Tuple

from scripts.pgmig import copy as copy_mod
from scripts.pgmig import dedup
from scripts.pgmig import introspect
from scripts.pgmig import verify as verify_mod
from dashboard import db as db_mod

EXIT_OK = 0
EXIT_COPY_ERRORS = 1
EXIT_ENV = 2
EXIT_PREFLIGHT_DIRTY = 3
EXIT_VERIFY_MISMATCH = 4
EXIT_PREFLIGHT_INCOMPLETE = 5


class PgTargetCrossCheckError(Exception):
    """Raised by `_pg_target_map` when PG IS configured (DB_BACKEND=postgres
    + PG_DSN) but the target-schema unique-index introspection itself fails.

    This must propagate rather than degrade to {} like the "PG not
    configured" case does -- the operator asked for the cross-check by
    configuring PG, so a failure here is not a legitimate skip. Swallowing
    it would let preflight/full report a clean/OK result while the exact
    check the P06 cutover gate relies on silently never ran (false
    assurance)."""


def _pg_configured() -> bool:
    import os
    return (os.environ.get("DB_BACKEND", "").strip().lower() == "postgres"
            and bool(os.environ.get("PG_DSN")))


def _require_pg_env(out: TextIO) -> bool:
    """True if DB_BACKEND=postgres and PG_DSN are both set; otherwise prints
    a loud, specific error to `out` and returns False. Callers that need
    Postgres (copy/verify/full) must check this before doing anything else;
    `preflight` alone does NOT require it (it degrades to sqlite-only)."""
    import os
    backend = os.environ.get("DB_BACKEND", "").strip().lower()
    dsn = os.environ.get("PG_DSN")
    ok = True
    if backend != "postgres":
        print(f"ERROR: DB_BACKEND must be 'postgres' (got {backend!r}). "
              "Set DB_BACKEND=postgres before running copy/verify/full.", file=out)
        ok = False
    if not dsn:
        print("ERROR: PG_DSN is not set. Set PG_DSN to the target Postgres "
              "DSN before running copy/verify/full.", file=out)
        ok = False
    return ok


def _merge_findings(sqlite_findings: List[Dict], pg_findings: List[Dict]) -> List[Dict]:
    """Merge scan_db's sqlite-source findings with scan_against_targets'
    pg-target findings, deduping on (table, key_cols) -- a genuine collision
    surfaced by BOTH scans (same table, same key columns) is reported once,
    tagged to show it was seen from both sides, rather than double-counted."""
    merged: List[Dict] = []
    seen: Dict[Tuple, Dict] = {}
    for f in sqlite_findings:
        key = (f["table"], tuple(f["key_cols"]))
        seen[key] = f
        merged.append(f)
    for f in pg_findings:
        key = (f["table"], tuple(f["key_cols"]))
        if key in seen:
            seen[key]["source"] = "sqlite+pg-target"
            continue
        seen[key] = f
        merged.append(f)
    return merged


def _pg_target_map(sqlite_path: str, out: TextIO) -> Dict[str, List[List[str]]]:
    """The Postgres target schema's own unique-index column sets, or {} (with
    a printed note) if PG isn't configured at all -- that is the ONE
    legitimate quiet degradation (preflight itself never requires Postgres).

    Raises PgTargetCrossCheckError if PG IS configured but the introspection
    query fails for any reason -- see that class's docstring for why this
    must NOT be swallowed into a silent {} the way the "not configured" case
    is."""
    if not _pg_configured():
        print("(PG-target cross-check skipped: DB_BACKEND=postgres / PG_DSN not configured)",
              file=out)
        return {}
    try:
        from dashboard.dbschema import schema_for_path
        schema = schema_for_path(sqlite_path)
        pg_cx = db_mod.connect(sqlite_path)
        try:
            return introspect.pg_unique_indexes(pg_cx, schema)
        finally:
            pg_cx.close()
    except Exception as exc:
        raise PgTargetCrossCheckError(str(exc)) from exc


def _run_preflight(sqlite_path: str, out: TextIO) -> Tuple[List[Dict], int]:
    """Runs both the sqlite-source scan and (if configured) the pg-target
    cross-check, prints a combined report, and returns (findings, exit_code).

    If PG is configured but the cross-check itself fails to run, this is
    reported as an INCOMPLETE preflight (exit 5), distinct from both a clean
    preflight (0) and a dirty one (3) -- it must never look like either."""
    try:
        target_map = _pg_target_map(sqlite_path, out)
    except PgTargetCrossCheckError as exc:
        print(f"PREFLIGHT INCOMPLETE (target cross-check could not run: {exc})", file=out)
        return [], EXIT_PREFLIGHT_INCOMPLETE

    sqlite_findings = dedup.scan_db(sqlite_path)
    pg_findings = dedup.scan_against_targets(sqlite_path, target_map) if target_map else []
    findings = _merge_findings(sqlite_findings, pg_findings)

    if not findings:
        print("PREFLIGHT CLEAN: 0 tables with unique-key collisions.", file=out)
        return findings, EXIT_OK

    print(f"PREFLIGHT DIRTY: {len(findings)} table(s) with unique-key collisions:", file=out)
    for f in findings:
        print(f"  - {f['table']}  key={f['key_cols']}  groups={f['n_groups']} "
              f"excess_rows={f['n_excess_rows']}  source={f.get('source', 'sqlite')}", file=out)
        for ex in f.get("examples", [])[:3]:
            print(f"      e.g. key={ex['key']} count={ex['count']}", file=out)
    return findings, EXIT_PREFLIGHT_DIRTY


def _print_copy_summary(results: List[Dict], out: TextIO) -> bool:
    """Prints the per-table copy summary; returns True if any row errored."""
    print("COPY SUMMARY:", file=out)
    for r in sorted(results, key=lambda r: r["table"]):
        line = (f"  - {r['table']}: source_rows={r['source_rows']} "
                f"inserted={r['inserted']} conflicts={r['conflicts']}")
        if r.get("note"):
            line += f"  ({r['note']})"
        print(line, file=out)
        if r.get("errors"):
            print(f"    !! {len(r['errors'])} ROW ERROR(S) in {r['table']}:", file=out)
            for e in r["errors"][:5]:
                print(f"       row={e['row']} error={e['error']}", file=out)
    failed = copy_mod.any_errors(results)
    print("COPY FAILED: one or more rows produced errors (see above)." if failed
          else "COPY OK.", file=out)
    return failed


def _print_verify_summary(results: List[Dict], out: TextIO) -> bool:
    """Prints the parity table; returns True if all-ok."""
    print("PARITY:", file=out)
    for r in results:
        mark = "OK" if r["ok"] else "MISMATCH"
        print(f"  - {r['table']}: sqlite={r['sqlite']} postgres={r['postgres']}  [{mark}]",
              file=out)
    ok = verify_mod.all_ok(results)
    print("ROW-COUNT PARITY OK (counts only -- content not compared)" if ok
          else "PARITY FAILED: mismatches above.", file=out)
    return ok


def cmd_preflight(args: argparse.Namespace, out: Optional[TextIO] = None) -> int:
    out = out or sys.stdout
    _findings, code = _run_preflight(args.sqlite_path, out)
    return code


def cmd_copy(args: argparse.Namespace, out: Optional[TextIO] = None) -> int:
    out = out or sys.stdout
    if not _require_pg_env(out):
        return EXIT_ENV
    results = copy_mod.copy_all(args.sqlite_path, truncate=args.truncate)
    return EXIT_COPY_ERRORS if _print_copy_summary(results, out) else EXIT_OK


def cmd_verify(args: argparse.Namespace, out: Optional[TextIO] = None) -> int:
    out = out or sys.stdout
    if not _require_pg_env(out):
        return EXIT_ENV
    results = verify_mod.parity(args.sqlite_path)
    return EXIT_OK if _print_verify_summary(results, out) else EXIT_VERIFY_MISMATCH


def cmd_full(args: argparse.Namespace, out: Optional[TextIO] = None) -> int:
    out = out or sys.stdout
    if not _require_pg_env(out):
        return EXIT_ENV

    _findings, preflight_code = _run_preflight(args.sqlite_path, out)
    if preflight_code != EXIT_OK:
        if not args.force:
            if preflight_code == EXIT_PREFLIGHT_INCOMPLETE:
                print("ABORTING: preflight is INCOMPLETE (the PG-target cross-check "
                      "could not run) above; re-run with --force to proceed anyway "
                      "(only after understanding why the cross-check failed).", file=out)
            else:
                print("ABORTING: preflight found collisions above; re-run with --force "
                      "to proceed anyway (only after resolving/accepting them).", file=out)
            return preflight_code
        print("--force set: proceeding past a dirty/incomplete preflight.", file=out)

    results = copy_mod.copy_all(args.sqlite_path, truncate=args.truncate)
    if _print_copy_summary(results, out):
        # A failed row must fail the run even if parity would otherwise pass.
        return EXIT_COPY_ERRORS

    ver = verify_mod.parity(args.sqlite_path)
    if not _print_verify_summary(ver, out):
        return EXIT_VERIFY_MISMATCH

    print("FULL MIGRATION OK.", file=out)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="migrate_sqlite_to_pg",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("preflight",
                         help="Dedup-scan the sqlite source (+ PG-target cross-check if configured).")
    pf.add_argument("sqlite_path")
    pf.set_defaults(func=cmd_preflight)

    cp = sub.add_parser("copy", help="Copy the sqlite source into its Postgres target schema.")
    cp.add_argument("sqlite_path")
    cp.add_argument("--truncate", action="store_true",
                     help="TRUNCATE every target table (child-to-parent) before copying.")
    cp.set_defaults(func=cmd_copy)

    vf = sub.add_parser("verify", help="Row-count parity check: sqlite source vs Postgres target.")
    vf.add_argument("sqlite_path")
    vf.set_defaults(func=cmd_verify)

    fl = sub.add_parser("full", help="preflight -> copy -> verify, in one run.")
    fl.add_argument("sqlite_path")
    fl.add_argument("--force", action="store_true",
                     help="Proceed past a dirty preflight instead of aborting.")
    fl.add_argument("--truncate", action="store_true",
                     help="TRUNCATE every target table (child-to-parent) before copying.")
    fl.set_defaults(func=cmd_full)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
