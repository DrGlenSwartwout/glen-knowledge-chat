"""A scan's own recommendations, mirrored from the local e4l.db into prod.

Production cannot read e4l.db, so `02 Skills/e4l-scan-recommendations-push.py` POSTs
these rows to a console-gated endpoint. Pure sqlite: no Flask, no network.

`section` is carried verbatim from e4l_scan_results.section_context — the scan PDF's
own "INFOCEUTICALS" / "MIHEALTH FUNCTIONS" headings. It is NOT re-derived from
protocol_days, which only correlates. "The five infoceuticals" is therefore a query.

Keyed on (email, scan_id, item_code): a re-push UPDATEs. The pusher is idempotent and
runs daily, so duplicate rows would compound silently.
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
            UNIQUE(email, scan_id, item_code)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_sr_email ON scan_recommendations(email)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_sr_scan ON scan_recommendations(email, scan_id)")
    cx.commit()


def upsert_recommendations(cx, email, scan_id, scan_date, items):
    """Write one scan's items. Returns rows written. An item with no item_code is
    skipped rather than stored blank — a blank code would join to no e4l_item and
    render as an empty remedy on the portal."""
    e, sid = _norm(email), str(scan_id or "").strip()
    if not e or not sid:
        return 0
    written = 0
    for it in items or []:
        if not isinstance(it, dict):
            continue
        code = (it.get("item_code") or "").strip()
        if not code:
            continue
        cx.execute(
            "INSERT INTO scan_recommendations "
            "(email, scan_id, scan_date, item_code, priority_rank, protocol_days, "
            " section, category, label, synced_at) VALUES (?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(email, scan_id, item_code) DO UPDATE SET "
            "scan_date=excluded.scan_date, priority_rank=excluded.priority_rank, "
            "protocol_days=excluded.protocol_days, section=excluded.section, "
            "category=excluded.category, label=excluded.label, synced_at=excluded.synced_at",
            (e, sid, (scan_date or "").strip(), code, it.get("priority_rank"),
             it.get("protocol_days"), it.get("section"), it.get("category"),
             it.get("label"), _now()))
        written += 1
    cx.commit()
    return written


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
