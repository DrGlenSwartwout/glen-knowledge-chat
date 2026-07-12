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
_NO_EMAIL_EXACT = {"pet", "child"}


def _never_email(relationship):
    """Household members we never send an invite to: children, and any animal —
    whether tagged with the legacy bare 'pet' or a species-namespaced
    'animal:<species>' (e.g. 'animal:cat', 'animal:dog')."""
    r = (relationship or "").strip().lower()
    return r in _NO_EMAIL_EXACT or r.startswith("animal")


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


def maybe_hold_new_order(cx, order_id, *, now=None):
    """Called right after a shippable order is created. If the auto-batch flag is
    on and this order belongs to an active Family-Plan household, open or join a
    hold group and return the hold result; otherwise return None (ship as normal)."""
    if not _enabled():
        return None
    order = _orders.get_order(cx, order_id)
    if not eligible_for_hold(cx, order):
        return None
    cg = caregiver_of(cx, order.get("email"))
    if not cg:
        return None
    return open_or_join_hold(cx, order_id, caregiver_email=cg,
                             household_key=cg, now=now)


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


def remove_from_hold(cx, order_id):
    """Called when a held order is cancelled: drop it from its hold group. If
    that leaves the group with no remaining (non-cancelled) members, close the
    group as 'cancelled' — an empty hold has nothing left to combine."""
    order = _orders.get_order(cx, order_id)
    gid = order.get("hold_group_id") if order else None
    if gid is None:
        return {"ok": False, "reason": "not in a hold"}
    _orders.set_order_hold_group(cx, order_id, None)
    remaining = orders_in_hold(cx, gid)
    if not remaining:
        cx.execute("UPDATE household_holds SET status='cancelled', updated_at=? "
                   "WHERE id=? AND status='open'", (_iso(_now()), gid))
        cx.commit()
    return {"ok": True, "group_id": gid, "remaining": len(remaining)}


def invite_recipients(cx, group_id):
    hold = get_hold(cx, group_id)
    cg = hold["caregiver_email"]
    cc = []
    for m in _hh.viewable_members_for(cx, cg):
        if _never_email(m.get("relationship")):
            continue
        if _lc(m["email"]) and _lc(m["email"]) != _lc(cg):
            cc.append(_lc(m["email"]))
    return {"to": _lc(cg), "cc": cc}


def compose_invite(hold, ship_date, release_url):
    members = hold.get("members") or []
    lines = []
    for m in members:
        who = m.get("name") or m.get("email") or f"order #{m.get('id')}"
        lines.append(f"  • {who}")
    items = "\n".join(lines) if lines else "  • your order"
    subject = "Your household order is being prepared to ship"
    body = (
        f"A shipment for your household is being prepared to go out on {ship_date}.\n\n"
        f"It currently includes:\n{items}\n\n"
        "If anyone else in your household wants to add something, just place their "
        "order in the next few days and it will ship together in the same box.\n\n"
        f"Or if nothing else is coming, ship it now: {release_url}\n\n"
        "In wellness,\nDr. Glen & Rae"
    )
    html = (
        f"<p>A shipment for your household is being prepared to go out on "
        f"<strong>{ship_date}</strong>.</p>"
        f"<p>It currently includes:</p><ul>"
        + "".join(f"<li>{(m.get('name') or m.get('email') or ('order #' + str(m.get('id'))))}</li>"
                  for m in members)
        + "</ul>"
        "<p>If anyone else in your household wants to add something, just place "
        "their order in the next few days and it will ship together in the same box.</p>"
        f"<p>Or if nothing else is coming, "
        f"<a href='{release_url}'>ship it now</a>.</p>"
        "<p>In wellness,<br>Dr. Glen &amp; Rae</p>"
    )
    return {"subject": subject, "body": body, "html": html}


def send_invite(cx, group_id, *, base_url, now=None):
    """One invite per group. Mints the release token, composes, sends via Gmail,
    stamps invite_sent_at. No-op if already sent."""
    hold = get_hold(cx, group_id)
    if hold is None or hold.get("invite_sent_at"):
        return {"skipped": "already_sent_or_missing"}
    raw = set_release_token(cx, group_id)
    ship_date = datetime.fromisoformat(hold["hold_until"]).strftime("%B %-d")
    release_url = f"{base_url.rstrip('/')}/hold/{raw}/ship"
    rec = invite_recipients(cx, group_id)
    msg = compose_invite(get_hold(cx, group_id), ship_date, release_url)
    from dashboard import inbox as _inbox
    res = _inbox.send_email(rec["to"], msg["subject"], msg["body"], html=msg["html"])
    cx.execute("UPDATE household_holds SET invite_sent_at=?, updated_at=? WHERE id=?",
               (_iso(now or _now()), _iso(now or _now()), group_id))
    cx.commit()
    return {"sent_to": rec["to"], "cc": rec["cc"], "send_result": res}


# ── Board actions (self-register on import) ──────────────────────────────────
from dashboard.actions import action, LOW_WRITE
from dashboard.rbac import OWNER, OPS


def _cx_of(params, ctx):
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection")
    return cx


def _extend_exec(params, ctx):
    cx = _cx_of(params, ctx)
    gid = int(params["group_id"])
    days = int(params.get("days", 2))
    hold = extend_hold(cx, gid, days)
    return {"group_id": gid, "hold_until": hold["hold_until"],
            "message": f"Hold #{gid} extended to {hold['hold_until'][:10]}."}


def _release_exec(params, ctx):
    from dashboard import combined_shipments as _cs
    cx = _cx_of(params, ctx)
    gid = int(params["group_id"])
    res = release_hold(cx, gid, by="operator")
    ids = res["order_ids"]
    made = None
    if len(ids) >= 2:
        made = _cs.create_shipment(cx, ids, created_by="operator-release")
    return {"group_id": gid, "order_ids": ids,
            "shipment_id": (made["id"] if made else None),
            "message": (f"Hold #{gid} released; combined shipment "
                        f"#{made['id']} created." if made
                        else f"Hold #{gid} released ({len(ids)} order).")}


action(key="holds.extend", module="orders", title="Extend household hold",
       description="Push a household hold group's ship-by deadline out N days (default 2).",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS), reversible=True)(_extend_exec)

action(key="holds.release", module="orders", title="Release household hold now",
       description="Close a household hold and send its orders to fulfillment "
                   "(combining 2+ into one shipment).",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS), reversible=False)(_release_exec)
