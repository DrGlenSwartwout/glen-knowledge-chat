"""Per-member data-sharing consent: normalized, revocable, timestamped grants."""
import datetime

DATA_SCOPES = ("chat", "scans", "results", "video")
PURPOSES = ("improve_ai", "research", "marketing")

# Friendly member-facing toggle -> underlying grants + attribution
TOGGLE_MAP = {
    "improve_ai_chat":   {"grants": [("chat", "improve_ai")], "attribution": "anonymized"},
    "research_results":  {"grants": [("scans", "research"), ("results", "research"),
                                     ("results", "improve_ai")], "attribution": "anonymized"},
    "share_story":       {"grants": [("scans", "marketing"), ("results", "marketing")],
                          "attribution": "attributed"},
    "video_testimonial": {"grants": [("video", "marketing")], "attribution": "attributed"},
}

def _now():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def init_data_sharing_tables(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS member_data_sharing (
        email TEXT PRIMARY KEY,
        attribution TEXT NOT NULL DEFAULT 'anonymized',
        tier INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS member_data_sharing_grants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL,
        data_scope TEXT NOT NULL,
        purpose TEXT NOT NULL,
        granted_at TEXT NOT NULL,
        revoked_at TEXT)""")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_mdsg_email ON member_data_sharing_grants(email)")
    cx.commit()

def expand_toggles(toggles):
    grants, attribution = set(), "anonymized"
    for key, on in (toggles or {}).items():
        if on and key in TOGGLE_MAP:
            spec = TOGGLE_MAP[key]
            grants.update(tuple(g) for g in spec["grants"])
            if spec["attribution"] == "attributed":
                attribution = "attributed"
    return grants, attribution

def derive_tier(grants, attribution):
    scopes = {s for s, _ in grants}
    purposes_by_scope = {(s, p) for s, p in grants}
    if "video" in scopes and attribution == "attributed":
        return 4
    if any(p == "marketing" for _, p in grants) and attribution == "attributed":
        return 3
    if scopes & {"scans", "results"}:
        return 2
    if "chat" in scopes:
        return 1
    return 0

def _active_grants(cx, email):
    rows = cx.execute(
        "SELECT data_scope, purpose FROM member_data_sharing_grants "
        "WHERE email=? AND revoked_at IS NULL", (email.lower(),)).fetchall()
    return {(r[0], r[1]) for r in rows}

def set_consent(cx, email, toggles):
    email = email.lower()
    desired, attribution = expand_toggles(toggles)
    current = _active_grants(cx, email)
    now = _now()
    # Revoke grants no longer desired
    for scope, purpose in current - desired:
        cx.execute("UPDATE member_data_sharing_grants SET revoked_at=? "
                   "WHERE email=? AND data_scope=? AND purpose=? AND revoked_at IS NULL",
                   (now, email, scope, purpose))
    # Add newly-desired grants
    for scope, purpose in desired - current:
        cx.execute("INSERT INTO member_data_sharing_grants "
                   "(email, data_scope, purpose, granted_at) VALUES (?,?,?,?)",
                   (email, scope, purpose, now))
    tier = derive_tier(desired, attribution)
    cx.execute("INSERT INTO member_data_sharing (email, attribution, tier, updated_at) "
               "VALUES (?,?,?,?) ON CONFLICT(email) DO UPDATE SET "
               "attribution=excluded.attribution, tier=excluded.tier, updated_at=excluded.updated_at",
               (email, attribution, tier, now))
    cx.commit()
    return {"tier": tier, "attribution": attribution,
            "grants": sorted([list(g) for g in desired])}

def get_consent(cx, email):
    email = email.lower()
    row = cx.execute("SELECT attribution, tier FROM member_data_sharing WHERE email=?",
                     (email,)).fetchone()
    grants = sorted([list(g) for g in _active_grants(cx, email)])
    if not row:
        return {"tier": 0, "attribution": "anonymized", "grants": grants}
    return {"tier": row[1], "attribution": row[0], "grants": grants}
