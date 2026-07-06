"""Peer matching store (community arc c2a). Pure sqlite. Paid, opted-in members are
matched on shared liked-topics (read from community_signals); a mutual 'connect'
opens a source='peer' coach_threads thread. Privacy: proposals are anonymous
(member_ref only) until mutual; no 'you were passed on' signal; blocks/skips/matches
are excluded. Eligibility (paid) + reveal + thread live in the routes."""
import hashlib


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _lc(email):
    return (email or "").strip().lower()


def _ref_key():
    import os
    return (os.environ.get("PEER_REF_SALT") or os.environ.get("CONSOLE_SECRET") or "").encode("utf-8")


def member_ref(email):
    """Anonymized handle for a member. HMAC-salted with a server secret so a party
    who already knows a target's email cannot precompute their ref and confirm, from a
    proposal, that they are opted in and share topics (pre-mutual de-anonymization).
    Falls back to a plain hash only when no secret is configured (local/test) so the
    handle stays deterministic there."""
    import hmac
    msg = _lc(email).encode("utf-8")
    key = _ref_key()
    if key:
        return hmac.new(key, msg, hashlib.sha256).hexdigest()[:16]
    return hashlib.sha256(msg).hexdigest()[:16]


_DDL = """
CREATE TABLE IF NOT EXISTS peer_optin (
    member_email TEXT PRIMARY KEY,
    active INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS peer_interest (
    from_email TEXT NOT NULL,
    to_email TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT,
    UNIQUE(from_email, to_email)
);
CREATE TABLE IF NOT EXISTS peer_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    a_email TEXT NOT NULL,
    b_email TEXT NOT NULL,
    thread_id INTEGER,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT,
    UNIQUE(a_email, b_email)
);
"""


def init_peer_tables(cx):
    cx.executescript(_DDL)
    cx.commit()


def set_optin(cx, email, active):
    cx.execute("INSERT INTO peer_optin (member_email, active, updated_at) VALUES (?,?,?) "
               "ON CONFLICT(member_email) DO UPDATE SET active=excluded.active, "
               "updated_at=excluded.updated_at", (_lc(email), 1 if active else 0, _now()))
    cx.commit()


def is_opted_in(cx, email):
    row = cx.execute("SELECT active FROM peer_optin WHERE member_email=?", (_lc(email),)).fetchone()
    return bool(row and row["active"])


def opted_in_members(cx):
    return [r["member_email"] for r in
            cx.execute("SELECT member_email FROM peer_optin WHERE active=1").fetchall()]


def _topics(cx, email, signal):
    return {r["target_key"] for r in cx.execute(
        "SELECT target_key FROM community_signals WHERE email=? AND target_type='topic' "
        "AND signal=?", (_lc(email), signal)).fetchall()}


def liked_topics(cx, email):
    return _topics(cx, email, "like")


def blocked_topics(cx, email):
    return _topics(cx, email, "block")


def record_interest(cx, from_email, to_email, kind):
    cx.execute("INSERT INTO peer_interest (from_email, to_email, kind, created_at) "
               "VALUES (?,?,?,?) ON CONFLICT(from_email, to_email) DO UPDATE SET "
               "kind=excluded.kind, created_at=excluded.created_at",
               (_lc(from_email), _lc(to_email), kind, _now()))
    cx.commit()


def interest_kind(cx, from_email, to_email):
    row = cx.execute("SELECT kind FROM peer_interest WHERE from_email=? AND to_email=?",
                     (_lc(from_email), _lc(to_email))).fetchone()
    return row["kind"] if row else None


def _person_blocked(cx, blocker, blocked):
    # A future person-block UI must key on member_ref computed AT READ TIME (as here),
    # never persist the ref as a stored value: member_ref is HMAC-salted, so a stored
    # ref would stop matching if the salt (CONSOLE_SECRET/PEER_REF_SALT) ever rotates.
    row = cx.execute("SELECT 1 FROM community_signals WHERE email=? AND target_type='person' "
                     "AND target_key=? AND signal='block'",
                     (_lc(blocker), member_ref(blocked))).fetchone()
    return row is not None


def _pair_has_match(cx, e1, e2):
    a, b = sorted([_lc(e1), _lc(e2)])
    return cx.execute("SELECT 1 FROM peer_matches WHERE a_email=? AND b_email=?",
                      (a, b)).fetchone() is not None


def next_candidate(cx, me, is_paid=None):
    me = _lc(me)
    mine = liked_topics(cx, me) - blocked_topics(cx, me)
    if not mine:
        return None
    best = None
    for n in opted_in_members(cx):
        if n == me:
            continue
        if is_paid is not None and not is_paid(n):              # no longer a current paid member
            continue
        if _pair_has_match(cx, me, n):
            continue
        if interest_kind(cx, me, n) is not None:            # I already acted on them
            continue
        if interest_kind(cx, n, me) == "skip":              # they passed on me
            continue
        if _person_blocked(cx, me, n) or _person_blocked(cx, n, me):
            continue
        shared = mine & (liked_topics(cx, n) - blocked_topics(cx, n))
        if not shared:
            continue
        score = len(shared)
        if best is None or score > best[0] or (score == best[0] and member_ref(n) < best[1]):
            best = (score, member_ref(n), sorted(shared))
    if best is None:
        return None
    return {"member_ref": best[1], "shared_topics": best[2]}


def resolve_ref(cx, me, ref):
    me = _lc(me)
    for n in opted_in_members(cx):
        if n != me and member_ref(n) == ref:
            return n
    return None


def create_match(cx, a_email, b_email, thread_id):
    a, b = sorted([_lc(a_email), _lc(b_email)])
    cx.execute("INSERT OR IGNORE INTO peer_matches (a_email, b_email, thread_id, status, "
               "created_at) VALUES (?,?,?, 'active', ?)", (a, b, thread_id, _now()))
    cx.commit()


def match_for_pair(cx, e1, e2):
    a, b = sorted([_lc(e1), _lc(e2)])
    row = cx.execute("SELECT * FROM peer_matches WHERE a_email=? AND b_email=?", (a, b)).fetchone()
    return dict(row) if row else None


def matches_for(cx, me):
    me = _lc(me)
    rows = cx.execute("SELECT a_email, b_email, thread_id, status FROM peer_matches "
                      "WHERE a_email=? OR b_email=? ORDER BY id", (me, me)).fetchall()
    out = []
    for r in rows:
        other = r["b_email"] if r["a_email"] == me else r["a_email"]
        out.append({"other_email": other, "thread_id": r["thread_id"], "status": r["status"]})
    return out


def end_match(cx, thread_id):
    cx.execute("UPDATE peer_matches SET status='ended' WHERE thread_id=?", (thread_id,))
    cx.commit()
