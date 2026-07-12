"""Household hold-and-batch: hold a Family-Plan household's new shippable orders
up to N days so same-household orders combine into ONE parcel via
dashboard/combined_shipments. Pure sqlite; the hold layer sits UPSTREAM of the
combined-shipment layer and never touches labels/tracking/delivery.
"""
import os
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
    their own plan is their own caregiver."""
    e = _lc(buyer_email)
    if not e:
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
