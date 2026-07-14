"""Tier -> reward mapping with an idempotent, never-downgrade ledger."""
import datetime

# Cumulative: tier N includes all rewards from tiers <= N of the same mode.
REWARD_MAP = {
    1: [{"reward_type": "founding_badge", "mode": "auto"}],
    2: [{"reward_type": "free_reveal_unlock", "mode": "auto"},
        {"reward_type": "early_access", "mode": "auto"}],
    3: [{"reward_type": "store_credit", "mode": "pending"}],
    4: [{"reward_type": "video_perk", "mode": "pending"}],
}
_STATUS_RANK = {"pending": 0, "granted": 1, "fulfilled": 2}

def _now():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def init_reward_tables(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS member_reward_grants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL,
        reward_type TEXT NOT NULL,
        tier INTEGER NOT NULL,
        status TEXT NOT NULL,
        granted_by TEXT,
        granted_at TEXT NOT NULL,
        notes TEXT)""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_mrg_once "
               "ON member_reward_grants(email, reward_type)")
    cx.commit()

def _rewards_for(tier):
    out = []
    for t in range(1, tier + 1):
        out.extend(REWARD_MAP.get(t, []))
    return out

def grant_rewards_for_tier(cx, email, tier, free_unlock_fn=None):
    email = email.lower()
    newly = []
    for spec in _rewards_for(tier):
        rt = spec["reward_type"]
        exists = cx.execute("SELECT 1 FROM member_reward_grants "
                            "WHERE email=? AND reward_type=?", (email, rt)).fetchone()
        if exists:
            continue  # never re-grant, never downgrade
        status = "granted" if spec["mode"] == "auto" else "pending"
        cx.execute("INSERT INTO member_reward_grants "
                   "(email, reward_type, tier, status, granted_by, granted_at) "
                   "VALUES (?,?,?,?,?,?)",
                   (email, rt, tier, status,
                    "auto" if status == "granted" else None, _now()))
        if rt == "free_reveal_unlock" and free_unlock_fn is not None:
            free_unlock_fn(cx, email)  # existing biofield free-unlock mechanism
        newly.append(rt)
    cx.commit()
    return newly

def rewards_for_email(cx, email):
    rows = cx.execute("SELECT reward_type, status FROM member_reward_grants WHERE email=?",
                      (email.lower(),)).fetchall()
    return {r[0]: r[1] for r in rows}
