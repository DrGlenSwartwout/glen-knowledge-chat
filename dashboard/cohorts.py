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
    if t == "reorder_loyalty" and not isinstance(policy.get("ff_cents"), int):
        raise ValueError("reorder_loyalty needs int ff_cents")
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


def active_reorder_loyalty(cx):
    """Active reorder_loyalty cohorts (applied automatically to earners — no explicit
    membership needed). Usually one, holding the FF reorder rate."""
    return [c for c in list_cohorts(cx)
            if c.get("active") and (c.get("policy") or {}).get("type") == "reorder_loyalty"]


def reorder_loyalty_price(cohorts, *, slug, is_ff, earned_slugs):
    """Lowest FF reorder-loyalty price for this line — but ONLY when the SKU is FF
    and EARNED (the caller decides earning: paid Biofield + previously purchased).
    `cohorts` = reorder_loyalty cohorts (e.g. active_reorder_loyalty()). Pure."""
    if not is_ff or slug not in (earned_slugs or set()):
        return None
    prices = [int(c["policy"]["ff_cents"]) for c in (cohorts or [])
              if (c.get("policy") or {}).get("type") == "reorder_loyalty"
              and isinstance((c.get("policy") or {}).get("ff_cents"), int)]
    return min(prices) if prices else None


def candidate_cohorts(cx, email):
    """Active cohorts the client is NOT already in — the plans switch-to-save can
    offer them. Excludes reorder_loyalty (automatic for earners, not a 'plan' to
    switch to)."""
    have = {c["key"] for c in member_cohorts(cx, email)}
    return [c for c in list_cohorts(cx)
            if c.get("active") and c["key"] not in have
            and (c.get("policy") or {}).get("type") != "reorder_loyalty"]


def savings_offer(lines, current_units, candidate_list, *, earned_slugs=None):
    """Would a plan the client ISN'T on make this cart cheaper? For each candidate
    cohort, price the cart as lowest-wins vs the current per-line prices; return the
    single best cheaper plan or None.
      lines        = [{slug, qty, list_cents, is_ff}]
      current_units= [cents] current effective UNIT price per line (same order)
    Returns {cohort_key, cohort_name, current_total_cents, new_total_cents,
    savings_cents} or None. Pure."""
    if not lines or not candidate_list:
        return None
    current_total = sum(int(u) * int(ln["qty"]) for u, ln in zip(current_units, lines))
    best = None
    for c in candidate_list:
        pol = c.get("policy") or {}
        new_total = 0
        for u, ln in zip(current_units, lines):
            if pol.get("type") == "reorder_loyalty":
                cand = reorder_loyalty_price([c], slug=ln["slug"], is_ff=ln["is_ff"],
                                             earned_slugs=earned_slugs)
            else:
                cand = policy_unit_cents(pol, slug=ln["slug"],
                                         list_cents=ln["list_cents"], is_ff=ln["is_ff"])
            eff = min(int(u), cand) if cand is not None else int(u)
            new_total += eff * int(ln["qty"])
        if new_total < current_total and (best is None or new_total < best["new_total_cents"]):
            best = {"cohort_key": c["key"], "cohort_name": c["name"],
                    "current_total_cents": current_total, "new_total_cents": new_total,
                    "savings_cents": current_total - new_total}
    return best


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
