"""Begin #4a store: per-scan funnel Biofield reveal (interpretation auto-shown +
ranked remedies, blurred until the top is approved). Distinct from
portal_biofield_reports."""
import json
import sqlite3
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _rows_cursor(cx):
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    return cur


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS biofield_reveals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            scan_date TEXT NOT NULL,
            interpretation_json TEXT NOT NULL DEFAULT '{}',
            remedies_json TEXT NOT NULL DEFAULT '[]',
            first_approved INTEGER NOT NULL DEFAULT 0,
            token_hash TEXT,
            approved_at TEXT, approved_by TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE(email, scan_date)
        )
    """)
    # Additive column for the ingest guardrail (idempotent).
    try:
        cx.execute("ALTER TABLE biofield_reveals ADD COLUMN dropped TEXT NOT NULL DEFAULT '[]'")
    except Exception:
        pass
    try:
        cx.execute("ALTER TABLE biofield_reveals ADD COLUMN layers_json TEXT NOT NULL DEFAULT '[]'")
    except Exception:
        pass
    try:
        cx.execute("ALTER TABLE biofield_reveals ADD COLUMN notified_at TEXT")
    except Exception:
        pass
    # Client's "request review" action (Option B): the reveal-page button stamps
    # this, and the console bucketing promotes a requested reveal into "Review these".
    try:
        cx.execute("ALTER TABLE biofield_reveals ADD COLUMN requested_at TEXT")
    except Exception:
        pass
    cx.commit()


def _row(r):
    if r is None:
        return None
    d = dict(r)
    d["interpretation"] = json.loads(d.pop("interpretation_json") or "{}")
    d["remedies"] = json.loads(d.pop("remedies_json") or "[]")
    d["dropped"] = json.loads(d.pop("dropped", "[]") or "[]")
    d["layers"] = json.loads(d.pop("layers_json", "[]") or "[]")
    d["first_approved"] = bool(d.get("first_approved"))
    return d


# Curated remedy substitutions, applied on the CLIENT read path
# (get_by_token_hash) so a matched remedy whose product isn't purchasable is
# swapped for the equivalent sellable SKU EVERYWHERE it renders with an order
# button — the same row feeds display, entitlement (_biofield_visible_slugs),
# live pricing, and checkout, so substituting once here keeps all four in sync
# (swapping only the display would post a slug checkout then rejects).
# "Relax" (syntropy 364-relax) has no purchasable SKU -> "Stress Release"
# (syntropy 249-stress-release, /begin/buy/stress-release resolves). Glen 2026-07-14.
# Keyed by lowercased slug OR lowercased trimmed name; idempotent (the swapped-in
# slug isn't itself a key).
REMEDY_SUBSTITUTIONS = {
    "relax": {
        "name": "Stress Release",
        "slug": "stress-release",
        "meaning": ("Stress Release is a CBD-synergy botanical formulation that supports "
                    "a calm, balanced nervous system and multi-phase relief from everyday stress."),
    },
}


def _sub_for(rem):
    if not isinstance(rem, dict):
        return None
    for key in ((rem.get("slug") or "").strip().lower(), (rem.get("name") or "").strip().lower()):
        if key and key in REMEDY_SUBSTITUTIONS:
            return REMEDY_SUBSTITUTIONS[key]
    return None


def _apply_sub(rem):
    sub = _sub_for(rem)
    if not sub:
        return
    for field in ("name", "slug", "meaning"):
        if sub.get(field):
            rem[field] = sub[field]


def apply_remedy_substitutions(row):
    """Swap curated non-purchasable remedies for their sellable equivalent, in
    row['remedies'] and each row['layers'][].remedy. Mutates and returns row
    (passes None through). Never raises."""
    if not isinstance(row, dict):
        return row
    try:
        for rem in row.get("remedies") or []:
            _apply_sub(rem)
        for layer in row.get("layers") or []:
            if isinstance(layer, dict):
                _apply_sub(layer.get("remedy"))
    except Exception:
        pass
    return row


def upsert(cx, email, scan_date, interpretation, remedies, source, layers=None):
    """Insert or update a reveal. Content updates only while not yet approved (matcher
    re-push). Returns (id, is_new). layers defaults to [] when not provided."""
    email = (email or "").strip().lower()
    now = _now()
    lj = json.dumps(layers or [])
    existing = cx.execute(
        "SELECT id, first_approved FROM biofield_reveals WHERE email=? AND scan_date=?",
        (email, scan_date)).fetchone()
    if existing is not None:
        rid, approved = existing[0], existing[1]
        if not approved:
            cx.execute(
                "UPDATE biofield_reveals SET interpretation_json=?, remedies_json=?, layers_json=?, updated_at=? WHERE id=?",
                (json.dumps(interpretation or {}), json.dumps(remedies or []), lj, now, rid))
            cx.commit()
        return rid, False
    cur = cx.execute(
        "INSERT INTO biofield_reveals (email, scan_date, interpretation_json, remedies_json, layers_json, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (email, scan_date, json.dumps(interpretation or {}), json.dumps(remedies or []), lj, now, now))
    cx.commit()
    new_id = cur.lastrowid
    # A free member who spent >= $100 since their last reveal earns this one fully un-blurred.
    maybe_unlock_for_spend(cx, email, new_id)
    return new_id, True


def set_token(cx, rid, token_hash):
    cx.execute("UPDATE biofield_reveals SET token_hash=?, updated_at=? WHERE id=?",
               (token_hash, _now(), rid))
    cx.commit()


def set_interpretation(cx, rid, interpretation):
    cx.execute("UPDATE biofield_reveals SET interpretation_json=?, updated_at=? WHERE id=?",
               (json.dumps(interpretation or {}), _now(), rid))
    cx.commit()


def set_remedies(cx, rid, remedies):
    cx.execute("UPDATE biofield_reveals SET remedies_json=?, updated_at=? WHERE id=?",
               (json.dumps(remedies or []), _now(), rid))
    cx.commit()


def set_layers(cx, rid, layers):
    cx.execute("UPDATE biofield_reveals SET layers_json=?, updated_at=? WHERE id=?",
               (json.dumps(layers or []), _now(), rid))
    cx.commit()


def set_dropped(cx, rid, names):
    cx.execute("UPDATE biofield_reveals SET dropped=?, updated_at=? WHERE id=?",
               (json.dumps(names or []), _now(), rid))
    cx.commit()


def delete(cx, rid):
    cx.execute("DELETE FROM biofield_reveals WHERE id=?", (int(rid),))
    cx.commit()


def mark_requested(cx, rid):
    """Stamp the client's 'request review' action on this reveal, once. Idempotent:
    returns True only on the first request; later calls are no-ops that preserve the
    original timestamp."""
    cur = cx.execute(
        "UPDATE biofield_reveals SET requested_at=?, updated_at=? "
        "WHERE id=? AND (requested_at IS NULL OR requested_at='')",
        (_now(), _now(), rid))
    cx.commit()
    return cur.rowcount == 1


def approve_first(cx, rid, by):
    now = _now()
    cur = cx.execute(
        "UPDATE biofield_reveals SET first_approved=1, approved_at=?, approved_by=?, updated_at=? WHERE id=?",
        (now, by, now, rid))
    cx.commit()
    return cur.rowcount == 1


def list_pending(cx):
    rows = _rows_cursor(cx).execute(
        "SELECT * FROM biofield_reveals WHERE first_approved=0 ORDER BY id DESC").fetchall()
    return [_row(r) for r in rows]


def list_approved(cx, limit=50):
    rows = _rows_cursor(cx).execute(
        "SELECT * FROM biofield_reveals WHERE first_approved=1 ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


def list_for_email(cx, email):
    """All reveals for an email, newest scan_date first (row dicts via _row)."""
    email = (email or "").strip().lower()
    rows = _rows_cursor(cx).execute(
        "SELECT * FROM biofield_reveals WHERE email=? ORDER BY scan_date DESC, id DESC",
        (email,)).fetchall()
    return [_row(r) for r in rows]


def set_notified(cx, rid):
    cx.execute("UPDATE biofield_reveals SET notified_at=?, updated_at=? WHERE id=?",
               (_now(), _now(), rid))
    cx.commit()


def list_approved_unnotified(cx, limit=200):
    return [_row(r) for r in _rows_cursor(cx).execute(
        "SELECT * FROM biofield_reveals WHERE first_approved=1 AND (notified_at IS NULL OR notified_at='') "
        "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]


def get(cx, rid):
    return _row(_rows_cursor(cx).execute("SELECT * FROM biofield_reveals WHERE id=?", (rid,)).fetchone())


def get_by_token_hash(cx, th):
    # Client read path: apply curated remedy substitutions so an unpurchasable
    # matched remedy renders + orders as its sellable equivalent everywhere.
    return apply_remedy_substitutions(_row(_rows_cursor(cx).execute(
        "SELECT * FROM biofield_reveals WHERE token_hash=?", (th,)).fetchone()))


# ---------------------------------------------------------------------------
# One-time free top-remedy unblock ledger
# ---------------------------------------------------------------------------

def init_free_unlocks(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS biofield_free_unlocks (email TEXT PRIMARY KEY, reveal_id INTEGER, granted_at TEXT)")
    cx.commit()


def free_unlock_reveal_id(cx, email):
    r = cx.execute("SELECT reveal_id FROM biofield_free_unlocks WHERE email=?",
                   ((email or "").strip().lower(),)).fetchone()
    return r[0] if r else None


def record_free_unlock(cx, email, reveal_id):
    """One-time free top-remedy unblock. True only on the first grant for this email."""
    cur = cx.execute(
        "INSERT OR IGNORE INTO biofield_free_unlocks (email, reveal_id, granted_at) VALUES (?,?,?)",
        ((email or "").strip().lower(), reveal_id, _now()))
    cx.commit()
    return cur.rowcount == 1


# ---------------------------------------------------------------------------
# Spend-earned full reveal unlock — distinct from the one-time top-remedy
# free_unlock above. A free member who places a paid order >= $100 since their
# last reveal earns their NEXT reveal fully un-blurred. Keyed per reveal_id
# (repeatable per period, non-stacking), consumed via the existing `paid` gate.
# ---------------------------------------------------------------------------
SPEND_UNLOCK_CENTS = 10000   # $100


def init_spend_unlocks(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS biofield_reveal_spend_unlocks "
               "(reveal_id INTEGER PRIMARY KEY, email TEXT, granted_at TEXT)")
    cx.commit()


def record_spend_unlock(cx, reveal_id, email):
    """Grant a full-reveal unlock for one reveal. Idempotent."""
    init_spend_unlocks(cx)
    cur = cx.execute(
        "INSERT OR IGNORE INTO biofield_reveal_spend_unlocks (reveal_id, email, granted_at) VALUES (?,?,?)",
        (reveal_id, (email or "").strip().lower(), _now()))
    cx.commit()
    return cur.rowcount == 1


def is_spend_unlocked(cx, reveal_id):
    """True when this reveal was earned by qualifying spend (full un-blur)."""
    if reveal_id is None:
        return False
    try:
        init_spend_unlocks(cx)
        r = cx.execute("SELECT 1 FROM biofield_reveal_spend_unlocks WHERE reveal_id=?",
                       (reveal_id,)).fetchone()
        return r is not None
    except Exception:
        return False


def maybe_unlock_for_spend(cx, email, reveal_id):
    """Grant a full-reveal unlock for `reveal_id` when the email placed a paid
    order >= $100 AFTER their previous reveal. Best-effort — never raises into the
    reveal-creation path. Returns True when a new unlock was granted."""
    try:
        email = (email or "").strip().lower()
        if not email or reveal_id is None:
            return False
        prev = cx.execute(
            "SELECT created_at FROM biofield_reveals WHERE email=? AND id<? ORDER BY id DESC LIMIT 1",
            (email, reveal_id)).fetchone()
        since_ts = (prev[0] if prev else "") or ""
        qualifying = cx.execute(
            "SELECT 1 FROM orders WHERE lower(email)=? AND total_cents>=? "
            "AND (lower(coalesce(pay_status,''))='paid' OR paid_cents>=?) "
            "AND created_at > ? LIMIT 1",
            (email, SPEND_UNLOCK_CENTS, SPEND_UNLOCK_CENTS, since_ts)).fetchone()
        if qualifying is None:
            return False
        return record_spend_unlock(cx, reveal_id, email)
    except Exception:
        return False
