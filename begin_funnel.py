"""Journey-state engine for the /begin progressive-disclosure funnel.

Pure functions over a sqlite3 connection. Routes in app.py manage the
connection + _db_lock; tests pass their own connection. See
docs/superpowers/specs/2026-05-28-progressive-disclosure-funnel-design.md
"""

import json
import sqlite3
import urllib.parse

from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_journey_tables(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS journey_state (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT,
            email           TEXT,
            first_name      TEXT,
            ref_slug        TEXT,
            current_rung    TEXT    DEFAULT 'arrival',
            unlocked_gates  TEXT    DEFAULT '[]',
            awareness_stage TEXT    DEFAULT 'unknown',
            path            TEXT    DEFAULT 'none',
            tos_agreed_at   TEXT,
            tos_version     TEXT,
            last_signal     TEXT,
            created_at      TEXT    NOT NULL,
            updated_at      TEXT    NOT NULL
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_journey_session ON journey_state(session_id)")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_journey_email   ON journey_state(email)")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS journey_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            session_id  TEXT,
            email       TEXT,
            trigger     TEXT NOT NULL,
            detail      TEXT DEFAULT '',
            rung_before TEXT,
            rung_after  TEXT
        )
    """)
    cx.commit()


RUNGS = ["arrival", "listening", "inquire", "personalize", "free_tier",
         "explore_voice", "assess", "choose_path", "ascend", "advocate"]
RUNG_INDEX = {r: i for i, r in enumerate(RUNGS)}

# All accepted unlock triggers. The page (slice 1) only fires the first six;
# the rest are accepted so the engine spine is forward-compatible with later
# slices (rooms built in slices 4-6).
VALID_TRIGGERS = {
    "load", "video", "scroll", "question", "name", "email", "tos",
    "voice", "scan", "quiz", "paid_fork", "purchase", "share_video",
}

# Gate keys stored in unlocked_gates (email/tos drive their own columns, but
# are still recorded as gates for completeness).
GATE_TRIGGERS = VALID_TRIGGERS - {"load"}

# ---------------------------------------------------------------------------
# Awareness-stage inference (Slice 2)
# ---------------------------------------------------------------------------

AWARENESS_RANK = {"unknown": 0, "problem": 1, "solution": 2, "product": 3, "most": 4}

# Deliberately MINIMAL seed lists (refined from data over time, not hand-tuned
# here). Case-insensitive substring match on recent chat text.
_PRODUCT_KEYWORDS = ["e4l", "evox", "neuro magnesium", "retina renew", "zyto",
                     "voice scan", "bioenergetic"]
_SOLUTION_KEYWORDS = ["detox", "cleanse", "frequency", "remedy", "supplement",
                      "protocol", "natural healing", "biofield", "energetic"]
_PROBLEM_KEYWORDS = ["tired", "fatigue", "pain", "can't sleep", "cant sleep",
                     "insomnia", "anxious", "anxiety", "bloated", "headache",
                     "stress", "vision"]
_PRODUCT_GATES = {"paid_fork", "purchase", "scan", "quiz", "voice"}


def _max_awareness(a, b):
    return a if AWARENESS_RANK.get(a, 0) >= AWARENESS_RANK.get(b, 0) else b


def infer_awareness_heuristic(want, gates, query_texts):
    """Cold-start awareness signal from explicit intent, gates opened, and recent
    chat text. Returns one of AWARENESS_RANK's keys."""
    if want:
        return "most"
    gates = set(gates or ())
    if gates & _PRODUCT_GATES:
        return "product"
    text = " ".join(query_texts or []).lower()
    if any(k in text for k in _PRODUCT_KEYWORDS):
        return "product"
    if any(k in text for k in _SOLUTION_KEYWORDS):
        return "solution"
    if any(k in text for k in _PROBLEM_KEYWORDS):
        return "problem"
    return "unknown"


WANT_TARGETS = {
    "e4l":     "https://truly.vip/E4L",
    "quiz":    "https://healing.scoreapp.com",
    "join":    "https://truly.vip/Join",
    "results": "https://truly.vip/Results",
}
# not-yet-built rooms (no Slice 2 redirect): "voice", "path", "ash"


def resolve_want(want, ref=""):
    """Return the threaded external URL for a live ?want= target, else None."""
    key = (want or "").strip().lower()
    base = WANT_TARGETS.get(key)
    if not base:
        return None
    slug = (ref or "remedy-match").strip() or "remedy-match"
    sep = "&" if "?" in base else "?"
    return (f"{base}{sep}utm_source={urllib.parse.quote(slug)}"
            f"&utm_medium=affiliate&utm_campaign=begin-deeplink-{key}")


def compute_rung(gates, email, tos_agreed):
    """Derive the highest rung reached. Monotonic in ladder order. The
    free_tier rung specifically requires BOTH an email and ToS agreement."""
    gates = set(gates or ())
    rung = "arrival"
    if "video" in gates or "scroll" in gates:
        rung = "listening"
    if "question" in gates:
        rung = "inquire"
    if "name" in gates:
        rung = "personalize"
    if email and tos_agreed:
        rung = "free_tier"
    if "voice" in gates:
        rung = "explore_voice"
    if "scan" in gates or "quiz" in gates:
        rung = "assess"
    if "paid_fork" in gates:
        rung = "choose_path"
    if "purchase" in gates:
        rung = "ascend"
    if "share_video" in gates:
        rung = "advocate"
    return rung


_RUNG_LAYERS = {
    "arrival":      ["layer0"],
    "listening":    ["layer0", "layer1"],
    "inquire":      ["layer0", "layer1", "layer2"],
    "personalize":  ["layer0", "layer1", "layer2", "layer3"],
    "free_tier":    ["layer0", "layer1", "layer2", "layer3", "layer4", "layer5"],
}
# Full unfolding surface (layer0-5) stays visible at every rung at/above free_tier.
_ALL_LAYERS = ["layer0", "layer1", "layer2", "layer3", "layer4", "layer5"]


def reveal_for(rung):
    if RUNG_INDEX.get(rung, 0) >= RUNG_INDEX["free_tier"]:
        return list(_ALL_LAYERS)
    return list(_RUNG_LAYERS.get(rung, ["layer0"]))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _row_for_session(cx, session_id):
    cx.row_factory = sqlite3.Row
    return cx.execute(
        "SELECT * FROM journey_state WHERE session_id=? ORDER BY id DESC LIMIT 1",
        (session_id,)).fetchone()


# ---------------------------------------------------------------------------
# record_unlock — mutating entry point
# ---------------------------------------------------------------------------

def record_unlock(cx, *, session_id, trigger, email="", detail="",
                  first_name="", tos=False, ref_slug="", tos_version=""):
    if trigger not in VALID_TRIGGERS:
        raise ValueError(f"invalid trigger: {trigger!r}")
    cx.row_factory = sqlite3.Row
    now = _now()
    row = _row_for_session(cx, session_id)

    if row is None:
        gates = set()
        existing = dict(email="", first_name="", ref_slug="",
                        tos_agreed_at=None, tos_version=None,
                        created_at=now)
    else:
        gates = set(json.loads(row["unlocked_gates"] or "[]"))
        existing = dict(row)

    rung_before = compute_rung(
        gates, existing.get("email") or "",
        bool(existing.get("tos_agreed_at")))

    if trigger in GATE_TRIGGERS:
        gates.add(trigger)
    new_email = (email or existing.get("email") or "").strip().lower()
    new_first = (first_name or existing.get("first_name") or "").strip()
    new_ref = (ref_slug or existing.get("ref_slug") or "").strip()
    tos_at = existing.get("tos_agreed_at")
    tos_ver = existing.get("tos_version")
    if trigger == "tos" or tos:
        tos_at = tos_at or now
        tos_ver = tos_version or tos_ver

    rung_after = compute_rung(gates, new_email, bool(tos_at))
    gates_json = json.dumps(sorted(gates))

    if row is None:
        cx.execute("""
            INSERT INTO journey_state
              (session_id, email, first_name, ref_slug, current_rung,
               unlocked_gates, awareness_stage, path, tos_agreed_at,
               tos_version, last_signal, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (session_id, new_email, new_first, new_ref, rung_after,
              gates_json, "unknown", "none", tos_at, tos_ver, trigger,
              now, now))
    else:
        cx.execute("""
            UPDATE journey_state SET
              email=?, first_name=?, ref_slug=?, current_rung=?,
              unlocked_gates=?, tos_agreed_at=?, tos_version=?,
              last_signal=?, updated_at=?
            WHERE id=?
        """, (new_email, new_first, new_ref, rung_after, gates_json,
              tos_at, tos_ver, trigger, now, row["id"]))

    cx.execute("""
        INSERT INTO journey_events
          (ts, session_id, email, trigger, detail, rung_before, rung_after)
        VALUES (?,?,?,?,?,?,?)
    """, (now, session_id, new_email, trigger, detail[:500],
          rung_before, rung_after))
    cx.commit()
    state = get_state(cx, session_id=session_id)
    state["rung_before"] = rung_before
    return state


# ---------------------------------------------------------------------------
# get_state — non-destructive read + email aggregation
# ---------------------------------------------------------------------------

def _default_state(session_id, email):
    return {
        "session_id": session_id, "email": email or "", "first_name": "",
        "ref_slug": "", "current_rung": "arrival", "unlocked_gates": [],
        "awareness_stage": "unknown", "path": "none",
        "tos_agreed_at": None, "tos_version": None,
        "reveal": reveal_for("arrival"), "surfaced_cards": [],
    }


def get_state(cx, session_id="", email=""):
    """Return the visitor's aggregated journey state. When an email is known,
    union the gates across ALL rows sharing that email (cross-device
    continuity) plus the current session row. Non-destructive."""
    cx.row_factory = sqlite3.Row
    email = (email or "").strip().lower()
    rows = []
    seen = set()
    if email:
        for r in cx.execute(
                "SELECT * FROM journey_state WHERE LOWER(email)=?", (email,)):
            rows.append(r); seen.add(r["id"])
    if session_id:
        r = _row_for_session(cx, session_id)
        if r is not None and r["id"] not in seen:
            rows.append(r)
    if not rows:
        return _default_state(session_id, email)

    gates = set()
    first_name = ""
    ref_slug = ""
    email_final = email
    tos_at = None
    tos_ver = None
    path = "none"
    awareness = "unknown"
    created_at = None
    for r in rows:
        gates |= set(json.loads(r["unlocked_gates"] or "[]"))
        first_name = first_name or (r["first_name"] or "")
        ref_slug = ref_slug or (r["ref_slug"] or "")
        email_final = email_final or (r["email"] or "")
        tos_at = tos_at or r["tos_agreed_at"]
        tos_ver = tos_ver or r["tos_version"]
        if (r["path"] or "none") != "none":
            path = r["path"]
        if (r["awareness_stage"] or "unknown") != "unknown":
            awareness = r["awareness_stage"]
        if created_at is None or (r["created_at"] and r["created_at"] < created_at):
            created_at = r["created_at"]

    rung = compute_rung(gates, email_final, bool(tos_at))
    return {
        "session_id": session_id, "email": email_final, "first_name": first_name,
        "ref_slug": ref_slug, "current_rung": rung,
        "unlocked_gates": sorted(gates), "awareness_stage": awareness,
        "path": path, "tos_agreed_at": tos_at, "tos_version": tos_ver,
        "reveal": reveal_for(rung), "surfaced_cards": [],
    }
