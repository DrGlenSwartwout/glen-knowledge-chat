"""One-time (idempotent) normalization of practitioners.country to ISO 3166-1
alpha-2 codes, plus an audit report + geocode mop-up for newly-consistent rows.

Background: the practitioners table accumulated mixed country values — mostly
ISO-2 codes, but ~50 rows with full names ("Slovenia"), typos ("United Sates"),
and an ambiguous "Georgia". Exact-match country filtering in the finder needs a
clean column, so this rewrites every row through normalize_country().

What it does:
  1. Reads all distinct country values + counts (the "before" audit).
  2. Computes normalize_country() for each and UPDATEs only the rows that change.
  3. Flags any value that did NOT resolve to a 2-letter ISO code (manual review).
  4. Runs the global geocode sweep so newly-consistent international rows that
     lack coordinates get geocoded.

Idempotent: a second run reports zero changes.

Invoke:
  doppler run --project remedy-match --config prd -- \
    python3 -m scrapers.practitioner_finder.migrate_normalize_country [--dry-run]
"""
import argparse
import sys

from db_supabase import supabase_cursor
from scrapers.practitioner_finder.normalize import normalize_country


def _is_iso2(v: str | None) -> bool:
    return bool(v) and len(v) == 2 and v.isalpha() and v.isupper()


def _distinct_country_counts() -> list[tuple[str | None, int]]:
    with supabase_cursor() as cur:
        cur.execute(
            "SELECT country, count(*) AS n FROM practitioners "
            "GROUP BY country ORDER BY n DESC"
        )
        return [(r["country"], r["n"]) for r in cur.fetchall()]


def _apply_normalization(dry_run: bool) -> tuple[int, list[tuple[str, str, int]], list[str]]:
    """Returns (rows_changed, [(old, new, count)], [unresolved_values])."""
    before = _distinct_country_counts()
    changes: list[tuple[str, str, int]] = []
    unresolved: list[str] = []
    rows_changed = 0

    for old, n in before:
        new = normalize_country(old)
        if not _is_iso2(new):
            unresolved.append(f"{old!r} -> {new!r} (n={n})")
        if new != old:
            changes.append((str(old), str(new), n))
            if not dry_run:
                with supabase_cursor() as cur:
                    # NULL old values can't be matched with `=`; handle both.
                    if old is None:
                        cur.execute(
                            "UPDATE practitioners SET country=%s, updated_at=now() "
                            "WHERE country IS NULL",
                            (new,),
                        )
                    else:
                        cur.execute(
                            "UPDATE practitioners SET country=%s, updated_at=now() "
                            "WHERE country=%s",
                            (new, old),
                        )
                    rows_changed += cur.rowcount
        # In dry-run we still want a row estimate
        elif dry_run:
            pass

    if dry_run:
        rows_changed = sum(c for _, _, c in changes)
    return rows_changed, changes, unresolved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing.")
    ap.add_argument("--skip-geocode", action="store_true",
                    help="Skip the post-normalization geocode sweep.")
    args = ap.parse_args()

    print("=== country normalization audit ===")
    rows_changed, changes, unresolved = _apply_normalization(args.dry_run)

    if changes:
        print(f"\n{'DRY-RUN: would change' if args.dry_run else 'changed'} "
              f"{len(changes)} distinct value(s), {rows_changed} row(s):")
        for old, new, n in changes:
            print(f"  {old!r:30} -> {new!r:6} (n={n})")
    else:
        print("\nNo changes — all country values already normalized.")

    if unresolved:
        print(f"\n⚠ {len(unresolved)} value(s) did NOT resolve to an ISO-2 code "
              f"(left unchanged, review manually):")
        for u in unresolved:
            print(f"  {u}")

    if not args.dry_run and not args.skip_geocode:
        from scrapers.practitioner_finder.run_all import _global_geocode_sweep
        attempted, succeeded = _global_geocode_sweep()
        print(f"\ngeocode sweep: {succeeded}/{attempted} newly geocoded")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
