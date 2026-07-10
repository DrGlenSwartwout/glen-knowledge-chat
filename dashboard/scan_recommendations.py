"""A scan's own recommendations, mirrored from the local e4l.db into prod.

Production cannot read e4l.db, so `02 Skills/e4l-scan-recommendations-push.py` POSTs
these rows to a console-gated endpoint. Pure sqlite: no Flask, no network.

`section` is carried verbatim from e4l_scan_results.section_context — the scan PDF's
own "INFOCEUTICALS" / "MIHEALTH FUNCTIONS" headings. It is NOT re-derived from
protocol_days, which only correlates. "The five infoceuticals" is therefore a query.

Keyed on (email, scan_id, priority_rank) — rank is unique per scan in the source
data, but item_code is NOT: a two-column PDF layout can flatten to the same
item_code printed twice at two different ranks in one scan (e.g. scan 542814's
ER1/ER10/ER36 each appear at two ranks), and both rows are real and must both be
stored. A scan's rows are therefore replaced atomically via `replace_scan` — delete
this (email, scan_id)'s rows and insert the full incoming set in one transaction —
rather than upserted per item, which would let the second of a duplicated pair
silently overwrite the first.

An empty or all-invalid item list must never delete a client's stored
recommendations — that hazard (an empty extraction wiping real data) was found and
fixed in the sibling local script; `replace_scan` returns 0 and touches nothing
when there is nothing valid to write.
"""
import sqlite3
from datetime import datetime, timezone

SECTION_INFOCEUTICAL = "Infoceuticals"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS scan_recommendations (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT NOT NULL,
            scan_id       TEXT NOT NULL,
            scan_date     TEXT,
            item_code     TEXT NOT NULL,
            priority_rank INTEGER,
            protocol_days INTEGER,
            section       TEXT,
            category      TEXT,
            label         TEXT,
            synced_at     TEXT,
            UNIQUE(email, scan_id, priority_rank)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_sr_email ON scan_recommendations(email)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_sr_scan ON scan_recommendations(email, scan_id)")
    cx.commit()


def replace_scan(cx, email, scan_id, scan_date, items):
    """Atomically replace one scan's rows. Returns rows actually inserted.

    An item with no item_code is skipped rather than stored blank — a blank code
    would join to no e4l_item and render as an empty remedy on the portal.

    If, after skipping invalid items, there is nothing left to write — including
    the case where `items` was empty to begin with — this deletes NOTHING and
    returns 0. A client's stored recommendations must never be erased by an empty
    or malformed extraction. Otherwise the existing (email, scan_id) rows are
    deleted and the full incoming set is inserted, committed once as a single
    transaction so a reader never observes the scan mid-replace.
    """
    e, sid = _norm(email), str(scan_id or "").strip()
    if not e or not sid:
        return 0

    valid = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        code = (it.get("item_code") or "").strip()
        if not code:
            continue
        valid.append((it, code))

    if not valid:
        return 0

    now = _now()
    date = (scan_date or "").strip()
    cx.execute(
        "DELETE FROM scan_recommendations WHERE email=? AND scan_id=?", (e, sid))
    for it, code in valid:
        cx.execute(
            "INSERT INTO scan_recommendations "
            "(email, scan_id, scan_date, item_code, priority_rank, protocol_days, "
            " section, category, label, synced_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (e, sid, date, code, it.get("priority_rank"),
             it.get("protocol_days"), it.get("section"), it.get("category"),
             it.get("label"), now))
    cx.commit()
    return len(valid)


def for_scan(cx, email, scan_id):
    rows = cx.execute(
        "SELECT * FROM scan_recommendations WHERE email=? AND scan_id=? "
        "ORDER BY priority_rank", (_norm(email), str(scan_id or "").strip())).fetchall()
    return [dict(r) for r in rows]


def infoceuticals_for_scan(cx, email, scan_id):
    rows = cx.execute(
        "SELECT * FROM scan_recommendations WHERE email=? AND scan_id=? AND section=? "
        "ORDER BY priority_rank",
        (_norm(email), str(scan_id or "").strip(), SECTION_INFOCEUTICAL)).fetchall()
    return [dict(r) for r in rows]


def scan_dates_for(cx, email):
    """This client's E4L scan dates, newest first. NOTE: these are SCAN dates, not
    published-report dates — a report can be filed under a date on which the client has
    no scan, so the card must key off these."""
    rows = cx.execute(
        "SELECT DISTINCT scan_date FROM scan_recommendations WHERE email=? AND scan_date<>'' "
        "ORDER BY scan_date DESC", (_norm(email),)).fetchall()
    return [r[0] for r in rows]


def for_scan_date(cx, email, scan_date):
    rows = cx.execute(
        "SELECT * FROM scan_recommendations WHERE email=? AND scan_date=? ORDER BY priority_rank",
        (_norm(email), (scan_date or "").strip())).fetchall()
    return [dict(r) for r in rows]


def split_by_section(rows):
    """(infoceuticals, mihealth), rank order preserved. ER/MR are miHealth device cycles,
    not products — they are shown but never carry an order button."""
    info = [r for r in rows if r.get("section") == SECTION_INFOCEUTICAL]
    mih = [r for r in rows if r.get("section") != SECTION_INFOCEUTICAL]
    return info, mih
