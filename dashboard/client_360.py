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


def _detect_sources(cx, email):
    """All applicable sources by presence, in priority order (multi-badge)."""
    out = []
    if _exists(cx, "SELECT 1 FROM biofield_reveals WHERE lower(email)=? LIMIT 1", (email,)):
        out.append("biofield")
    if _exists(cx, "SELECT 1 FROM ff_match_drafts WHERE lower(email)=? LIMIT 1", (email,)):
        out.append("scan")
    if _exists(cx, "SELECT 1 FROM intake_responses WHERE lower(email)=? AND status='submitted' LIMIT 1", (email,)):
        out.append("intake")
    if _exists(cx, "SELECT 1 FROM inquiries WHERE lower(client_email)=? LIMIT 1", (email,)):
        out.append("chat")
    return out


def process_strip(cx, email):
    """The client's CURRENT in-flight cycle as sequence-status stages.
    cx: LOG_DB connection (row_factory=sqlite3.Row). Read-only."""
    e = (email or "").strip().lower()
    sources = _detect_sources(cx, e)
    source = sources[0] if sources else None
    try:
        order = cx.execute(
            "SELECT id, COALESCE(status,'') status, COALESCE(pay_status,'') pay, "
            "COALESCE(invoice_sent_at,'') sent FROM orders "
            "WHERE lower(COALESCE(email,''))=? AND COALESCE(status,'')<>'cancelled' "
            "ORDER BY id DESC LIMIT 1", (e,)).fetchone()
    except sqlite3.OperationalError:
        order = None
    oid = order["id"] if order else None
    status = order["status"] if order else ""
    pay = order["pay"] if order else ""
    sent = order["sent"] if order else ""

    rec_action = _SOURCE_ACTION.get(source, {"kind": "none"}) if source else {"kind": "none"}
    stages = [
        {"key": "recommendation",
         "label": _SOURCE_LABEL.get(source, "Recommendation"),
         "done": bool(sources), "source": source, "sources": sources, "action": rec_action},
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
    return {"source": source, "sources": sources, "order_id": oid, "stages": stages}


def client_tags_for_email(email, *, e4l_path=None):
    """Clinical tags for a client from the synced e4l.db. Degrades to empty
    lists when the db/table/row is unavailable — never raises."""
    from dashboard import biofield_e4l, clinical_tags_console
    empty = {"active": [], "suggested": []}
    path = biofield_e4l._db_path(e4l_path)
    cx = biofield_e4l._connect_ro(path)
    if cx is None:
        return empty
    try:
        row = cx.execute("SELECT client_id FROM e4l_clients WHERE lower(email)=lower(?)",
                         ((email or "").strip(),)).fetchone()
        if not row:
            return empty
        data = clinical_tags_console.client_tags(cx, row["client_id"])
        return {"active": data.get("active", []), "suggested": data.get("suggested", [])}
    except sqlite3.Error:
        return empty
    finally:
        cx.close()


def _person(cx, email):
    empty = {"name": "", "email": email, "phone": "", "location": "",
             "profession": "", "order_count": 0, "last_order_date": ""}
    try:
        r = cx.execute(
            "SELECT name, email, COALESCE(phone,'') phone, COALESCE(city,'') city, "
            "COALESCE(state,'') state, COALESCE(island,'') island, "
            "COALESCE(profession,'') profession, COALESCE(order_count,0) oc, "
            "COALESCE(last_order_date,'') lod FROM people WHERE lower(email)=? LIMIT 1",
            (email,)).fetchone()
    except sqlite3.OperationalError:
        return empty
    if not r:
        return empty
    loc = ", ".join(p for p in (r["city"], r["state"]) if p) or (r["island"] or "")
    return {"name": r["name"] or "", "email": r["email"] or email, "phone": r["phone"],
            "location": loc, "profession": r["profession"],
            "order_count": r["oc"], "last_order_date": r["lod"]}


def _tests(cx, email):
    from dashboard import client_scans, biofield_reveals
    by_date = {}
    try:
        for s in client_scans.scans_for(cx, email):
            by_date[s["scan_date"]] = "scan"
    except sqlite3.OperationalError:
        pass
    try:
        for rv in biofield_reveals.list_for_email(cx, email):
            by_date[rv["scan_date"]] = "biofield"   # biofield wins on a shared date
    except Exception:
        pass
    return [{"date": d, "type": t}
            for d, t in sorted(by_date.items(), key=lambda kv: kv[0], reverse=True)]


def _invoices(cx, email):
    from dashboard import order_payments, fmp_orders
    out = {"total_paid_cents": 0, "open_balance_cents": 0, "orders": [], "fmp": []}
    try:
        rows = cx.execute(
            "SELECT id, COALESCE(status,'') status, COALESCE(created_at,'') created_at "
            "FROM orders "
            "WHERE lower(COALESCE(email,''))=? AND COALESCE(status,'')<>'cancelled' "
            "ORDER BY id DESC", (email,)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    for r in rows:
        try:
            bal = order_payments.balance(cx, r["id"])
        except sqlite3.OperationalError:
            continue
        out["orders"].append({
            "id": r["id"], "date": r["created_at"], "status": r["status"],
            "total_cents": bal["invoice_cents"], "paid_cents": bal["paid_cents"],
            "balance_cents": bal["balance_cents"],
            "edit_url": f"/orders/new?edit_order={r['id']}"})
        out["total_paid_cents"] += bal["paid_cents"]
        if bal["balance_cents"] > 0:
            out["open_balance_cents"] += bal["balance_cents"]
    try:
        out["fmp"] = fmp_orders.client_order_history(cx, email=email)
    except Exception:
        out["fmp"] = []
    finally:
        cx.row_factory = sqlite3.Row
    return out


def _comms(cx, email):
    from dashboard import recent_comms
    try:
        rc = recent_comms.recent_comms(cx, email, days_window=3650)
    except Exception:
        return []
    out = []
    for q in rc.get("recent_inquiries", []):
        topic = q.get("main_challenge") or q.get("main_goal") or "inquiry"
        out.append({"date": q.get("created_at") or "", "topic": topic, "source": "inquiry"})
    for q in rc.get("recent_queries", []):
        out.append({"date": q.get("ts") or "", "topic": q.get("question") or "", "source": "query"})
    for f in rc.get("recent_feedback", []):
        topic = ", ".join(f.get("topics") or []) or f.get("summary") or "feedback"
        out.append({"date": f.get("received_at") or "", "topic": topic, "source": "feedback"})
    if rc.get("intake_summary"):
        out.append({"date": "", "topic": "Intake on file", "source": "intake"})
    out.sort(key=lambda c: c["date"], reverse=True)
    return out


def _recommendations(cx, email):
    from dashboard import recommendation_events
    try:
        return recommendation_events.product_sources(cx, email)
    except Exception:
        return []


def bundle(cx, email, *, e4l_path=None):
    """Assemble the full client-360 payload. cx: LOG_DB connection
    (row_factory=sqlite3.Row). Read-only; never raises on missing data."""
    e = (email or "").strip().lower()
    return {
        "person": _person(cx, e),
        "clinical": client_tags_for_email(e, e4l_path=e4l_path),
        "tests": _tests(cx, e),
        "invoices": _invoices(cx, e),
        "comms": _comms(cx, e),
        "process": process_strip(cx, e),
        "recommendations": _recommendations(cx, e),
    }
