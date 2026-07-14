"""Single source of truth for the operator's 'next action' per record type.

Pure resolvers (normalized record dict -> descriptor) shared by the per-page
buttons and the unified /console Next Action queue. Listers + aggregate live
below. Adding a record type = write a resolver + a lister + register both.
"""
import json
import os
import urllib.parse

TYPE_PRIORITY = ["order", "invoice", "household", "biofield_reveal", "handoff", "ff_match_draft", "reward_grant"]

_DONE = {"actionable": False}


def resolve_order(rec):
    status = rec.get("status")
    if status not in ("new", "packed"):
        return dict(_DONE)
    oid = rec.get("id")
    who = rec.get("name") or rec.get("email") or "unknown"
    total = (rec.get("total_cents") or 0) / 100
    n = rec.get("item_count") or 0
    summary = f"#{oid} · {who} · ${total:.2f} · {n} item{'' if n == 1 else 's'}"
    age = rec.get("age_ts", "")
    if status == "new":
        return {
            "type": "order", "id": oid, "actionable": True, "state": "new",
            "label": "Pack",
            "action": {"kind": "dispatch", "keys": ["orders.mark_packed"],
                       "body": {"order_id": oid}},
            "confirm": False,
            "secondary": {"label": "Open order",
                          "action": {"kind": "link", "url": "/console/orders"},
                          "confirm": False},
            "summary": summary, "age_ts": age,
        }
    return {   # packed
        "type": "order", "id": oid, "actionable": True, "state": "packed",
        "label": "Open to ship",
        "action": {"kind": "link", "url": "/console/orders"},
        "confirm": False, "secondary": None,
        "summary": summary, "age_ts": age,
    }


def resolve_invoice(rec):
    if rec.get("status") not in ("proposed", "confirmed"):
        return dict(_DONE)
    oid = rec.get("id")
    who = rec.get("name") or rec.get("email") or "unknown"
    total = (rec.get("total_cents") or 0) / 100
    n = rec.get("item_count") or 0
    summary = f"#{oid} · {who} · ${total:.2f} · {n} item{'' if n == 1 else 's'}"
    age = rec.get("age_ts", "")
    if not rec.get("invoice_sent_at"):
        return {
            "type": "invoice", "id": oid, "actionable": True, "state": "unsent",
            "label": "Send invoice",
            "action": {"kind": "dispatch", "keys": ["orders.send_invoice"],
                       "body": {"order_id": oid}},
            "confirm": True,
            "secondary": {"label": "Open order",
                          "action": {"kind": "link", "url": "/console/orders"},
                          "confirm": False},
            "summary": summary, "age_ts": age,
        }
    if rec.get("pay_status") != "paid":
        return {
            "type": "invoice", "id": oid, "actionable": True, "state": "sent_unpaid",
            "label": "Record payment",
            "action": {"kind": "link", "url": "/console/orders"},
            "confirm": False, "secondary": None,
            "summary": summary, "age_ts": age,
        }
    return dict(_DONE)


def resolve_household_hold(rec):
    gid = rec.get("group_id")
    care = rec.get("caregiver", "")
    n = rec.get("n_orders") or 0
    due = rec.get("hold_until", "")
    tail = "overdue" if rec.get("overdue") else (f"due {due[:10]}" if due else "due ?")
    summary = f"{care} · {n} order{'' if n == 1 else 's'} · {tail}"
    return {
        "type": "household", "id": gid, "actionable": True,
        "state": "overdue" if rec.get("overdue") else "holding",
        "label": "Release now",
        "action": {"kind": "dispatch", "keys": ["holds.release"], "body": {"group_id": gid}},
        "confirm": True,
        "secondary": {"label": "Extend 2 days",
                      "action": {"kind": "dispatch", "keys": ["holds.extend"],
                                 "body": {"group_id": gid, "days": 2}},
                      "confirm": False},
        "summary": summary, "age_ts": rec.get("age_ts", ""),
    }


def resolve_biofield_reveal(rec):
    rid = rec.get("id")
    summary = f"{rec.get('email','')} · scan {rec.get('scan_date','')}"
    age = rec.get("age_ts", "")
    if not rec.get("first_approved"):
        return {
            "type": "biofield_reveal", "id": rid, "actionable": True, "state": "draft",
            "label": "Approve & send",
            "action": {"kind": "dispatch",
                       "keys": ["biofield_reveal.approve", "biofield_reveal.send"],
                       "body": {"id": rid}},
            "confirm": True,
            "secondary": {"label": "Approve only, don't email",
                          "action": {"kind": "dispatch",
                                     "keys": ["biofield_reveal.approve"],
                                     "body": {"id": rid}},
                          "confirm": False},
            "summary": summary, "age_ts": age,
        }
    if not rec.get("notified_at"):
        return {
            "type": "biofield_reveal", "id": rid, "actionable": True,
            "state": "approved_unsent", "label": "Send reveal link",
            "action": {"kind": "dispatch", "keys": ["biofield_reveal.send"],
                       "body": {"id": rid}},
            "confirm": True, "secondary": None,
            "summary": summary, "age_ts": age,
        }
    return dict(_DONE)


def resolve_ff_match_draft(rec):
    if rec.get("status") != "draft":
        return dict(_DONE)
    email = rec.get("email", ""); when = rec.get("scan_date", "")
    return {
        "type": "ff_match_draft", "id": None, "actionable": True, "state": "draft",
        "label": "Publish",
        "action": {"kind": "post", "url": "/api/console/ff-match-drafts/publish",
                   "body": {"email": email, "scan_date": when}},
        "confirm": True,
        "secondary": {"label": "Open to edit",
                      "action": {"kind": "link", "url": "/console/ff-drafts"},
                      "confirm": False},
        "summary": f"{email} · scan {when}", "age_ts": rec.get("age_ts", ""),
    }


def resolve_handoff(rec):
    if rec.get("biofield_status") != "ai_draft":
        return dict(_DONE)
    email = rec.get("email", "")
    return {
        "type": "handoff", "id": None, "actionable": True, "state": "needs_publish",
        "label": "Review & publish",
        "action": {"kind": "link",
                   "url": "/console/biofield-portal?email=" + urllib.parse.quote(email)},
        "confirm": False,
        "secondary": None,
        "summary": f"{email}", "age_ts": rec.get("age_ts", ""),
    }


def resolve_reward_grant(rec):
    if _reward_gifts_flag_on():
        action = {"kind": "link", "url": "/console/rewards"}
        label = "Choose reward gift"
    else:
        action = {"kind": "dispatch", "keys": ["reward.fulfill"], "body": {"grant_id": rec["id"]}}
        label = "Approve reward"

    return {
        "type": "reward_grant", "id": rec["id"], "actionable": True, "state": "pending",
        "label": label,
        "action": action,
        "confirm": False,
        "secondary": {"label": "Dismiss",
                      "action": {"kind": "dispatch", "keys": ["reward.dismiss"],
                                 "body": {"grant_id": rec["id"]}},
                      "confirm": True},
        "summary": f"{rec['reward_type']} for {rec['email']} — Tier {rec['tier']} (shared attributed data)",
        "age_ts": rec["age_ts"],
    }


def _order_records(cx):
    # Exclude orders currently held by an OPEN household hold-and-batch group —
    # the batch owns them and already surfaces via resolve_household_hold's
    # "Release now" row. An order whose hold has released stays packable here.
    rows = cx.execute(
        "SELECT id, email, name, items_json, total_cents, status, created_at "
        "FROM orders WHERE status IN ('new','packed') "
        "AND (hold_group_id IS NULL "
        "OR hold_group_id NOT IN (SELECT id FROM household_holds WHERE status='open'))")
    out = []
    for r in rows:
        try:
            n = len(json.loads(r["items_json"] or "[]"))
        except Exception:
            n = 0
        out.append({"id": r["id"], "email": r["email"], "name": r["name"],
                    "total_cents": r["total_cents"], "item_count": n,
                    "status": r["status"], "age_ts": r["created_at"]})
    return out


def _invoice_records(cx):
    rows = cx.execute(
        "SELECT id, email, name, items_json, total_cents, status, pay_status, "
        "invoice_sent_at, created_at FROM orders WHERE status IN ('proposed','confirmed')")
    out = []
    for r in rows:
        try:
            n = len(json.loads(r["items_json"] or "[]"))
        except Exception:
            n = 0
        out.append({"id": r["id"], "email": r["email"], "name": r["name"],
                    "total_cents": r["total_cents"], "item_count": n,
                    "status": r["status"], "pay_status": r["pay_status"],
                    "invoice_sent_at": r["invoice_sent_at"], "age_ts": r["created_at"]})
    return out


def _household_hold_records(cx):
    from dashboard import household_holds as _hh
    now_iso = _hh._iso(_hh._now())
    out = []
    for h in _hh.list_open_holds(cx):
        hu = h.get("hold_until") or ""
        out.append({"group_id": h["id"], "caregiver": h.get("caregiver_email", ""),
                    "n_orders": len(h.get("members") or []),
                    "hold_until": hu, "overdue": bool(hu and hu <= now_iso),
                    "age_ts": hu})
    return out


def _reveal_records(cx):
    rows = cx.execute(
        "SELECT id, email, scan_date, first_approved, notified_at, created_at "
        "FROM biofield_reveals "
        "WHERE first_approved=0 OR (first_approved=1 AND (notified_at IS NULL OR notified_at=''))")
    return [{"id": r["id"], "email": r["email"], "scan_date": r["scan_date"],
             "first_approved": r["first_approved"], "notified_at": r["notified_at"],
             "age_ts": r["created_at"]} for r in rows]


def _ff_records(cx):
    rows = cx.execute(
        "SELECT email, scan_date, status, created_at FROM ff_match_drafts WHERE status='draft'")
    return [{"email": r["email"], "scan_date": r["scan_date"], "status": r["status"],
             "age_ts": r["created_at"]} for r in rows]


def _handoff_records(cx):
    # Mirror api_console_handoffs (app.py:13534-13551): a handoff = a client_portals
    # row whose content_json.biofield_status == 'ai_draft'. Confirmed columns on
    # client_portals (dashboard/client_portal.py:25-40): id, token_hash, email, name,
    # content_json, created_at, updated_at. age_ts uses the real updated_at column.
    out = []
    for r in cx.execute("SELECT email, content_json, updated_at FROM client_portals"):
        try:
            st = (json.loads(r["content_json"] or "{}") or {}).get("biofield_status")
        except Exception:
            st = None
        if st == "ai_draft":
            out.append({"email": r["email"], "biofield_status": st,
                        "age_ts": r["updated_at"] or ""})
    return out


def _data_sharing_flag_on():
    return os.environ.get("DATA_SHARING_REWARD_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def _reward_gifts_flag_on():
    return os.environ.get("REWARD_GIFTS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def _reward_records(cx):
    if not _data_sharing_flag_on():   # read env directly — no app import (avoids cycle)
        return []
    rows = cx.execute(
        "SELECT id, email, reward_type, tier, granted_at FROM member_reward_grants "
        "WHERE status='pending' ORDER BY granted_at")
    return [{"id": r["id"], "email": r["email"], "reward_type": r["reward_type"],
             "tier": r["tier"], "age_ts": r["granted_at"]} for r in rows]


def list_actionable(cx):
    items = ([resolve_order(r) for r in _order_records(cx)]
             + [resolve_invoice(r) for r in _invoice_records(cx)]
             + [resolve_household_hold(r) for r in _household_hold_records(cx)]
             + [resolve_biofield_reveal(r) for r in _reveal_records(cx)]
             + [resolve_handoff(r) for r in _handoff_records(cx)]
             + [resolve_ff_match_draft(r) for r in _ff_records(cx)]
             + [resolve_reward_grant(r) for r in _reward_records(cx)])
    items = [d for d in items if d.get("actionable")]
    prio = {t: i for i, t in enumerate(TYPE_PRIORITY)}
    items.sort(key=lambda d: (prio.get(d["type"], 99), d.get("age_ts") or ""))
    return items
