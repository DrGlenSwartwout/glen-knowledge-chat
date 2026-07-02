"""Cohort-based pricing — Phase 1 foundation.

A client belongs to cohort(s); each cohort carries a declarative pricing policy.
This module is the pure data + policy layer: tables, membership, and a
side-effect-free evaluator. Wiring into the order pricers lives in app.py behind
COHORT_PRICING_ENABLED (default off) — so this ships dark, changing nothing until
cohorts exist and the flag is flipped.

Resolution (in the pricer): explicit line edit > per-client negotiated rate
(floor) > LOWEST of the held cohorts' policy prices > standard volume/list.
This module owns only the "lowest of held cohort policies" step.

Policy types (policy_json):
  {"type": "flat_ff",     "cents": 5000}                 # flat price for every FF
  {"type": "per_sku",     "prices": {"slug": cents, ...}}
  {"type": "percent_off", "pct": 15, "scope": "ff"|"all"}
`volume` and `reorder_loyalty` are NOT evaluated here (they need the pricing
engine / order history) — they resolve in the caller; absent a matching policy,
the line falls through to the standard volume/list price.
"""
import json
import sqlite3
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def init_tables(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS cohorts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            description TEXT,
            policy_json TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            is_default INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    cx.execute("""
        CREATE TABLE IF NOT EXISTS cohort_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            cohort_key TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'admin',
            joined_at TEXT NOT NULL,
            expires_at TEXT,
            UNIQUE(email, cohort_key)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_cohort_members_email ON cohort_members(email)")
    cx.commit()


# ── cohort catalog ───────────────────────────────────────────────────────────

def upsert_cohort(cx, *, key, name, policy, description="", active=True, is_default=False):
    key = (key or "").strip()
    if not key or not name or not isinstance(policy, dict):
        raise ValueError("key, name, and a policy dict are required")
    validate_policy(policy)  # raises on a malformed policy
    cx.execute(
        "INSERT INTO cohorts (key, name, description, policy_json, active, is_default, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
        "name=excluded.name, description=excluded.description, policy_json=excluded.policy_json, "
        "active=excluded.active, is_default=excluded.is_default, updated_at=excluded.updated_at",
        (key, name, description, json.dumps(policy), 1 if active else 0,
         1 if is_default else 0, _now(), _now()))
    cx.commit()


def get_cohort(cx, key):
    row = cx.execute("SELECT key, name, description, policy_json, active, is_default "
                     "FROM cohorts WHERE key=?", ((key or "").strip(),)).fetchone()
    if not row:
        return None
    return {"key": row[0], "name": row[1], "description": row[2],
            "policy": json.loads(row[3] or "{}"), "active": bool(row[4]),
            "is_default": bool(row[5])}


def list_cohorts(cx):
    return [{"key": r[0], "name": r[1], "description": r[2], "policy": json.loads(r[3] or "{}"),
             "active": bool(r[4]), "is_default": bool(r[5])}
            for r in cx.execute(
                "SELECT key, name, description, policy_json, active, is_default "
                "FROM cohorts ORDER BY key").fetchall()]


# ── membership ───────────────────────────────────────────────────────────────

def add_member(cx, email, cohort_key, *, source="admin", expires_at=None):
    email, cohort_key = _norm(email), (cohort_key or "").strip()
    if not email or not cohort_key:
        raise ValueError("email and cohort_key required")
    cx.execute(
        "INSERT INTO cohort_members (email, cohort_key, source, joined_at, expires_at) "
        "VALUES (?,?,?,?,?) ON CONFLICT(email, cohort_key) DO UPDATE SET "
        "source=excluded.source, expires_at=excluded.expires_at",
        (email, cohort_key, source, _now(), expires_at))
    cx.commit()


def remove_member(cx, email, cohort_key):
    cur = cx.execute("DELETE FROM cohort_members WHERE email=? AND cohort_key=?",
                     (_norm(email), (cohort_key or "").strip()))
    cx.commit()
    return cur.rowcount > 0


def member_cohorts(cx, email, *, now=None):
    """Active (unexpired) cohorts this client holds, with their policies. Expired
    memberships and inactive cohorts are excluded."""
    email = _norm(email)
    if not email:
        return []
    now = now or _now()
    rows = cx.execute(
        "SELECT c.key, c.name, c.policy_json FROM cohort_members m "
        "JOIN cohorts c ON c.key=m.cohort_key "
        "WHERE m.email=? AND c.active=1 AND (m.expires_at IS NULL OR m.expires_at > ?)",
        (email, now)).fetchall()
    return [{"key": r[0], "name": r[1], "policy": json.loads(r[2] or "{}")} for r in rows]


# ── policy evaluation (pure) ─────────────────────────────────────────────────

_KNOWN_TYPES = {"flat_ff", "per_sku", "percent_off", "volume", "reorder_loyalty"}


def validate_policy(policy):
    t = (policy or {}).get("type")
    if t not in _KNOWN_TYPES:
        raise ValueError(f"unknown policy type: {t!r}")
    if t == "flat_ff" and not isinstance(policy.get("cents"), int):
        raise ValueError("flat_ff needs int cents")
    if t == "per_sku" and not isinstance(policy.get("prices"), dict):
        raise ValueError("per_sku needs a prices dict")
    if t == "percent_off":
        if not isinstance(policy.get("pct"), (int, float)):
            raise ValueError("percent_off needs numeric pct")
        if policy.get("scope") not in ("ff", "all", None):
            raise ValueError("percent_off scope must be ff|all")
    return True


def policy_unit_cents(policy, *, slug, list_cents, is_ff):
    """The unit price this policy sets for the given product, or None if it doesn't
    apply. Pure. list_cents = the product's list price; is_ff = FF-eligible."""
    t = (policy or {}).get("type")
    if t == "flat_ff":
        return int(policy["cents"]) if is_ff else None
    if t == "per_sku":
        v = (policy.get("prices") or {}).get(slug)
        return int(v) if v is not None else None
    if t == "percent_off":
        scope = policy.get("scope") or "all"
        if scope == "ff" and not is_ff:
            return None
        return int(round(int(list_cents) * (1 - float(policy["pct"]) / 100.0)))
    # volume / reorder_loyalty resolve in the caller, not here.
    return None


def best_cohort_price(cohorts, *, slug, list_cents, is_ff):
    """Lowest unit price across the client's held cohort policies, or None if none
    applies to this product. `cohorts` = member_cohorts() output."""
    prices = []
    for c in (cohorts or []):
        try:
            pc = policy_unit_cents(c.get("policy") or {}, slug=slug,
                                   list_cents=list_cents, is_ff=is_ff)
        except Exception:
            pc = None
        if pc is not None and pc >= 0:
            prices.append(pc)
    return min(prices) if prices else None
