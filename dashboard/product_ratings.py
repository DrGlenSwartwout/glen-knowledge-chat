"""The product-keyed purity-ratings cache and its never-downgrade state machine.
Pure sqlite; caller passes cx. product_key is the natural PRIMARY KEY, so there
is no surrogate id and no cur.lastrowid (which raises on the Postgres adapter).

status order: requested/unrated (0) -> screened (1) -> ai_draft (2) -> confirmed (3).
A row never moves to a lower rank. 'unrated' means a screen ran but no excipient
data was available; it holds no color and cannot advance.

Writes follow the SELECT-then-INSERT/UPDATE pattern of dashboard/supplement_reviews.py
(no `ON CONFLICT ... excluded` upsert) and stamp timestamps in Python (no SQL
`datetime('now')`), because both of those forms vary by backend and this table
must run on the Postgres adapter in prod.
"""
import json
from datetime import datetime, timezone

_RANK = {"requested": 0, "unrated": 0, "screened": 1, "ai_draft": 2, "confirmed": 3}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_tables(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS product_ratings (
        product_key TEXT PRIMARY KEY,
        brand TEXT, product_name TEXT,
        fullscript_slug TEXT, fullscript_external_id TEXT,
        other_ingredients_raw TEXT, other_ingredients_parsed TEXT,
        color TEXT, red_hits TEXT, yellow_hits TEXT, avoidlist_version TEXT,
        tier2_score REAL, tier2_json TEXT, best_ff TEXT,
        status TEXT NOT NULL,
        requested_at TEXT, screened_at TEXT, drafted_at TEXT, confirmed_at TEXT, updated_at TEXT)""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_prat_color ON product_ratings(color)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_prat_status ON product_ratings(status)")
    _cols = {r[1] for r in cx.execute("PRAGMA table_info(product_ratings)")}
    if "requested_by" not in _cols:
        cx.execute("ALTER TABLE product_ratings ADD COLUMN requested_by TEXT")
    cx.commit()


def get(cx, product_key):
    r = cx.execute("SELECT * FROM product_ratings WHERE product_key=?", (product_key,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["red_hits"] = json.loads(d["red_hits"]) if d.get("red_hits") else []
    d["yellow_hits"] = json.loads(d["yellow_hits"]) if d.get("yellow_hits") else []
    d["other_ingredients_parsed"] = (
        json.loads(d["other_ingredients_parsed"]) if d.get("other_ingredients_parsed") else []
    )
    return d


def request(cx, product_key, *, brand, product_name, requested_by):
    """Create a 'requested' row for a product if none exists. If a row exists at
    ANY status it is returned untouched -- request never creates a duplicate and
    never downgrades a further-along row."""
    existing = get(cx, product_key)
    if existing is not None:
        return {"created": False, "status": existing["status"]}
    now = _now()
    cx.execute("""INSERT INTO product_ratings
        (product_key, brand, product_name, status, requested_by, requested_at, updated_at)
        VALUES (?,?,?,?,?,?,?)""",
        (product_key, brand, product_name, "requested", requested_by, now, now))
    cx.commit()
    return {"created": True, "status": "requested"}


def record_screen(cx, product_key, *, brand, product_name,
                  other_ingredients_raw, other_ingredients_parsed, screen):
    """Insert or update a row from a screen result. Never downgrades a row already
    at ai_draft/confirmed. An 'unrated' screen (no data) lands status 'unrated',
    color NULL -- never green."""
    existing = get(cx, product_key)
    if existing is not None and _RANK[existing["status"]] >= _RANK["ai_draft"]:
        return  # confirmed/ai_draft rows are not walked back by a re-screen
    color = screen["color"]
    if color == "unrated":
        new_status, stored_color = "unrated", None
    else:
        new_status, stored_color = "screened", color
    now = _now()
    parsed = json.dumps(other_ingredients_parsed or [])
    reds = json.dumps(screen.get("red_hits") or [])
    yellows = json.dumps(screen.get("yellow_hits") or [])
    version = screen.get("avoidlist_version")
    if existing is None:
        cx.execute("""INSERT INTO product_ratings
            (product_key, brand, product_name, other_ingredients_raw, other_ingredients_parsed,
             color, red_hits, yellow_hits, avoidlist_version, status, screened_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (product_key, brand, product_name, other_ingredients_raw, parsed,
             stored_color, reds, yellows, version, new_status, now, now))
    else:
        cx.execute("""UPDATE product_ratings SET
             brand=?, product_name=?, other_ingredients_raw=?, other_ingredients_parsed=?,
             color=?, red_hits=?, yellow_hits=?, avoidlist_version=?, status=?,
             screened_at=?, updated_at=? WHERE product_key=?""",
            (brand, product_name, other_ingredients_raw, parsed,
             stored_color, reds, yellows, version, new_status, now, now, product_key))
    cx.commit()


def set_tier2(cx, product_key, score, detail_json):
    """Advance a screened, non-red row to ai_draft. Reds never run tier-2, and an
    unrated row has no color to rank -- both raise."""
    row = get(cx, product_key)
    if not row:
        raise ValueError("no such product_rating")
    if row["status"] != "screened" or row["color"] not in ("yellow", "green"):
        raise ValueError(f"cannot run tier-2 on status={row['status']} color={row['color']}")
    now = _now()
    cx.execute("UPDATE product_ratings SET tier2_score=?, tier2_json=?, status='ai_draft', "
               "drafted_at=?, updated_at=? WHERE product_key=?",
               (score, detail_json, now, now, product_key))
    cx.commit()


def confirm(cx, product_key):
    """Confirm a red screened row (reds skip tier-2) or a yellow/green ai_draft row."""
    row = get(cx, product_key)
    if not row:
        raise ValueError("no such product_rating")
    ok = (row["status"] == "screened" and row["color"] == "red") or row["status"] == "ai_draft"
    if not ok:
        raise ValueError(f"cannot confirm status={row['status']} color={row['color']}")
    now = _now()
    cx.execute("UPDATE product_ratings SET status='confirmed', confirmed_at=?, "
               "updated_at=? WHERE product_key=?", (now, now, product_key))
    cx.commit()
