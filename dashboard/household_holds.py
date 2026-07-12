"""Household hold-and-batch: hold a Family-Plan household's new shippable orders
up to N days so same-household orders combine into ONE parcel via
dashboard/combined_shipments. Pure sqlite; the hold layer sits UPSTREAM of the
combined-shipment layer and never touches labels/tracking/delivery.
"""
import hashlib
import os
import secrets
from datetime import datetime, timezone, timedelta

from dashboard import orders as _orders
from dashboard import family_plan as _fp
from dashboard import household as _hh

_TERMINAL = _orders._TERMINAL_STATUSES  # ("shipped","delivered","done","cancelled")
_DEPENDENT_NO_EMAIL = {"pet", "child"}  # accounts we never email an invite to


def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat()


def _lc(e):
    return (e or "").strip().lower()


def _enabled():
    return str(os.environ.get("HOUSEHOLD_AUTO_BATCH_ENABLED", "")).strip().lower() \
        in ("1", "true", "yes", "on")


def init_hold_tables(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS household_holds (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            caregiver_email TEXT NOT NULL,
            household_key  TEXT NOT NULL,
            status         TEXT NOT NULL DEFAULT 'open',
            opened_at      TEXT NOT NULL,
            hold_until     TEXT NOT NULL,
            invite_sent_at TEXT,
            release_token_hash TEXT,
            released_at    TEXT,
            released_by    TEXT,
            updated_at     TEXT
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_hold_status ON household_holds(status)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_hold_cg ON household_holds(caregiver_email, status)")
    cx.commit()


def caregiver_of(cx, buyer_email):
    """The active-plan caregiver covering this buyer, or None. A buyer who holds
    their own plan is their own caregiver.

    `family_plan.covers()` is the single source of truth for WHETHER a buyer is
    entitled (which statuses count as active, the share-consent gate). We gate on
    it first so this module can never diverge from that policy; the walk below only
    RESOLVES the caregiver's identity, and is reached solely when covers() is True."""
    e = _lc(buyer_email)
    if not e or not _fp.covers(cx, e):
        return None
    if _fp.is_active(cx, e):
        return e
    for cg in _hh.caregivers_for(cx, e):
        if cg["share_consent"] and _fp.is_active(cx, cg["primary_email"]):
            return cg["primary_email"]
    return None


def eligible_for_hold(cx, order):
    if order is None:
        return False
    if (order.get("status") or "") in _TERMINAL:
        return False
    if (order.get("channel") or "") == "pickup":
        return False
    if order.get("hold_group_id") is not None:
        return False
    if order.get("group_shipment_id") is not None:
        return False
    return caregiver_of(cx, order.get("email")) is not None


def _open_group_for(cx, caregiver_email, household_key):
    row = cx.execute(
        "SELECT * FROM household_holds WHERE caregiver_email=? AND household_key=? "
        "AND status='open' ORDER BY id DESC LIMIT 1",
        (_lc(caregiver_email), _lc(household_key))).fetchone()
    return dict(row) if row else None


def get_hold(cx, group_id):
    row = cx.execute("SELECT * FROM household_holds WHERE id=?", (group_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["members"] = orders_in_hold(cx, group_id)
    return d


def orders_in_hold(cx, group_id):
    return _orders.orders_in_hold_group(cx, group_id)


def open_or_join_hold(cx, order_id, *, caregiver_email, household_key, hold_days=4, now=None):
    """Open a new hold group for this order, or join it to the household's
    already-open group. The hold window belongs to the GROUP and is anchored
    to the FIRST order's arrival: a sibling order joining an open group does
    NOT push out hold_until — only a later extend_hold (Task 4) moves it."""
    now = now or _now()
    existing = _open_group_for(cx, caregiver_email, household_key)
    if existing:
        _orders.set_order_hold_group(cx, order_id, existing["id"])
        cx.execute("UPDATE household_holds SET updated_at=? WHERE id=?",
                   (_iso(now), existing["id"]))
        cx.commit()
        return {"group_id": existing["id"], "opened": False, "joined": True}
    hold_until = _iso(now + timedelta(days=int(hold_days)))
    cur = cx.execute(
        "INSERT INTO household_holds (caregiver_email, household_key, status, "
        "opened_at, hold_until, updated_at) VALUES (?,?,'open',?,?,?)",
        (_lc(caregiver_email), _lc(household_key), _iso(now), hold_until, _iso(now)))
    gid = int(cur.lastrowid)
    _orders.set_order_hold_group(cx, order_id, gid)
    cx.commit()
    return {"group_id": gid, "opened": True, "joined": False}


def release_hold(cx, group_id, *, by):
    """Close an open hold group. Returns its non-cancelled member order ids so the
    caller can hand them to combined_shipments.create_shipment. Idempotent: a
    non-open group returns its ids with the existing status."""
    hold = get_hold(cx, group_id)
    if hold is None:
        return {"group_id": group_id, "order_ids": [], "status": "missing"}
    order_ids = [m["id"] for m in hold["members"]]
    if hold["status"] != "open":
        return {"group_id": group_id, "order_ids": order_ids, "status": hold["status"]}
    cx.execute("UPDATE household_holds SET status='released', released_at=?, released_by=?, "
               "updated_at=? WHERE id=?",
               (_iso(_now()), str(by or ""), _iso(_now()), group_id))
    cx.commit()
    return {"group_id": group_id, "order_ids": order_ids, "status": "released"}


def extend_hold(cx, group_id, days, *, now=None):
    """Push a hold group's deadline out by `days` from its CURRENT hold_until
    (not from now). Raises ValueError if the group isn't open."""
    hold = get_hold(cx, group_id)
    if hold is None or hold["status"] != "open":
        raise ValueError(f"hold #{group_id} is not open")
    base = datetime.fromisoformat(hold["hold_until"])
    new_until = _iso(base + timedelta(days=int(days)))
    cx.execute("UPDATE household_holds SET hold_until=?, updated_at=? WHERE id=?",
               (new_until, _iso(now or _now()), group_id))
    cx.commit()
    return get_hold(cx, group_id)


def due_holds(cx, now=None):
    now = now or _now()
    rows = cx.execute(
        "SELECT * FROM household_holds WHERE status='open' AND hold_until <= ? "
        "ORDER BY id", (_iso(now),)).fetchall()
    return [dict(r) for r in rows]


def _tok_hash(raw):
    return hashlib.sha256(("household-hold:" + (raw or "")).encode("utf-8")).hexdigest()


def set_release_token(cx, group_id):
    """Mint a one-time release token for this hold group, embeddable in an email
    link. Only its SHA-256 hash is stored; the raw token is returned once and
    never persisted."""
    raw = secrets.token_urlsafe(32)
    cx.execute("UPDATE household_holds SET release_token_hash=?, updated_at=? WHERE id=?",
               (_tok_hash(raw), _iso(_now()), group_id))
    cx.commit()
    return raw


def hold_by_release_token(cx, raw_token):
    """Look up a hold group by its raw release token. None for an empty or
    unknown token."""
    raw_token = (raw_token or "").strip()
    if not raw_token:
        return None
    row = cx.execute("SELECT * FROM household_holds WHERE release_token_hash=?",
                     (_tok_hash(raw_token),)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["members"] = orders_in_hold(cx, d["id"])
    return d
