"""Per-client "client-360" hub: read + assemble everything the console knows
about one client (by email). Pure functions take open sqlite connections so
they are testable offline. No writes."""
import sqlite3

_SOURCE_ACTION = {
    "biofield": {"kind": "link", "target": "/console/biofield-portal"},
    "scan":     {"kind": "link", "target": "/console/ff-drafts"},
    "intake":   {"kind": "link", "target": "/console/crm"},
    "chat":     {"kind": "link", "target": "/console/crm"},
}
_SOURCE_LABEL = {"biofield": "Biofield", "scan": "Scan",
                 "intake": "Intake", "chat": "Chat"}
_FULFILLED = ("shipped", "delivered", "done", "fulfilled")


def _exists(cx, sql, params):
    """True if the query returns a row; False if the table is absent."""
    try:
        return cx.execute(sql, params).fetchone() is not None
    except sqlite3.OperationalError:
        return False


def _detect_source(cx, email):
    """Recommendation source for a client, in priority order. None if no
    concrete recommendation record exists."""
    if _exists(cx, "SELECT 1 FROM biofield_reveals WHERE lower(email)=? LIMIT 1", (email,)):
        return "biofield"
    if _exists(cx, "SELECT 1 FROM ff_match_drafts WHERE lower(email)=? LIMIT 1", (email,)):
        return "scan"
    if _exists(cx, "SELECT 1 FROM intake_responses WHERE lower(email)=? AND status='submitted' LIMIT 1", (email,)):
        return "intake"
    if _exists(cx, "SELECT 1 FROM inquiries WHERE lower(client_email)=? LIMIT 1", (email,)):
        return "chat"
    return None


def process_strip(cx, email):
    """The client's CURRENT in-flight cycle as sequence-status stages.
    cx: LOG_DB connection (row_factory=sqlite3.Row). Read-only."""
    e = (email or "").strip().lower()
    source = _detect_source(cx, e)
    order = cx.execute(
        "SELECT id, COALESCE(status,'') status, COALESCE(pay_status,'') pay, "
        "COALESCE(invoice_sent_at,'') sent FROM orders "
        "WHERE lower(COALESCE(email,''))=? AND COALESCE(status,'')<>'cancelled' "
        "ORDER BY id DESC LIMIT 1", (e,)).fetchone()
    oid = order["id"] if order else None
    status = order["status"] if order else ""
    pay = order["pay"] if order else ""
    sent = order["sent"] if order else ""

    rec_action = _SOURCE_ACTION.get(source, {"kind": "none"}) if source else {"kind": "none"}
    stages = [
        {"key": "recommendation",
         "label": _SOURCE_LABEL.get(source, "Recommendation"),
         "done": source is not None, "source": source, "action": rec_action},
        {"key": "invoice", "label": "Invoice", "done": order is not None,
         "action": {"kind": "link", "target": "/console/orders"} if order else {"kind": "none"}},
        {"key": "sent", "label": "Sent", "done": bool(sent),
         "action": ({"kind": "dispatch", "target": "orders.send_invoice", "order_id": oid}
                    if order and not sent else {"kind": "link", "target": "/console/orders"})},
        {"key": "paid", "label": "Paid", "done": pay == "paid",
         "action": {"kind": "link", "target": "/console/orders"} if order else {"kind": "none"}},
        {"key": "fulfilled", "label": "Fulfilled", "done": status in _FULFILLED,
         "action": {"kind": "link", "target": "/console/orders"} if order else {"kind": "none"}},
    ]
    return {"source": source, "order_id": oid, "stages": stages}
