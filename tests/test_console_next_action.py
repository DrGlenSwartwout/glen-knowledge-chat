import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dashboard.console_next_action as na


def test_reveal_draft_offers_approve_send_plus_approve_only():
    d = na.resolve_biofield_reveal(
        {"id": 7, "email": "a@b.co", "scan_date": "2026-07-01",
         "first_approved": 0, "notified_at": None, "age_ts": "2026-07-01T00:00:00"})
    assert d["actionable"] and d["state"] == "draft"
    assert d["label"] == "Approve & send" and d["confirm"] is True
    assert d["action"] == {"kind": "dispatch",
                           "keys": ["biofield_reveal.approve", "biofield_reveal.send"],
                           "body": {"id": 7}}
    assert d["secondary"]["label"] == "Approve only, don't email"
    assert d["secondary"]["action"]["keys"] == ["biofield_reveal.approve"]


def test_reveal_approved_unsent_offers_send():
    d = na.resolve_biofield_reveal(
        {"id": 9, "email": "a@b.co", "scan_date": "x", "first_approved": 1,
         "notified_at": None, "age_ts": "t"})
    assert d["actionable"] and d["state"] == "approved_unsent"
    assert d["label"] == "Send reveal link"
    assert d["action"]["keys"] == ["biofield_reveal.send"] and d["secondary"] is None


def test_reveal_sent_is_done():
    d = na.resolve_biofield_reveal(
        {"id": 9, "first_approved": 1, "notified_at": "2026-07-02T00:00:00"})
    assert d == {"actionable": False}


def test_ff_draft_offers_publish_and_edit_link():
    d = na.resolve_ff_match_draft(
        {"email": "a@b.co", "scan_date": "2026-07-01", "status": "draft", "age_ts": "t"})
    assert d["actionable"] and d["label"] == "Publish" and d["confirm"] is True
    assert d["action"] == {"kind": "post", "url": "/api/console/ff-match-drafts/publish",
                           "body": {"email": "a@b.co", "scan_date": "2026-07-01"}}
    assert d["secondary"]["action"] == {"kind": "link", "url": "/console/ff-drafts"}


def test_ff_published_is_done():
    assert na.resolve_ff_match_draft({"status": "published"}) == {"actionable": False}


def test_handoff_ai_draft_offers_composer_deep_link():
    d = na.resolve_handoff({"email": "a@b.co", "biofield_status": "ai_draft", "age_ts": "t"})
    assert d["actionable"] and d["label"] == "Review & publish"
    assert d["action"] == {"kind": "link",
                           "url": "/console/biofield-portal?email=a%40b.co"}
    assert d["confirm"] is False
    assert d["secondary"] is None


def test_handoff_confirmed_is_done():
    assert na.resolve_handoff({"biofield_status": "confirmed"}) == {"actionable": False}


def test_order_new_offers_pack_and_open_secondary():
    d = na.resolve_order({"id": 12, "name": "Jane Doe", "email": "j@d.co",
                          "total_cents": 14000, "item_count": 2,
                          "status": "new", "age_ts": "2026-07-01T00:00:00"})
    assert d["actionable"] and d["state"] == "new"
    assert d["label"] == "Pack" and d["confirm"] is False
    assert d["action"] == {"kind": "dispatch", "keys": ["orders.mark_packed"],
                           "body": {"order_id": 12}}
    assert d["secondary"]["label"] == "Open order"
    assert d["secondary"]["action"] == {"kind": "link", "url": "/console/orders"}
    assert d["summary"] == "#12 · Jane Doe · $140.00 · 2 items"


def test_order_packed_deeplinks_to_ship_no_secondary():
    d = na.resolve_order({"id": 5, "name": "", "email": "a@b.co",
                          "total_cents": 7000, "item_count": 1,
                          "status": "packed", "age_ts": "t"})
    assert d["actionable"] and d["state"] == "packed"
    assert d["label"] == "Open to ship"
    assert d["action"] == {"kind": "link", "url": "/console/orders"}
    assert d["secondary"] is None
    assert d["summary"] == "#5 · a@b.co · $70.00 · 1 item"   # name falls back to email; singular


def test_order_terminal_states_are_done():
    for s in ("shipped", "delivered", "done", "cancelled", "proposed", "confirmed", "paid"):
        assert na.resolve_order({"id": 1, "status": s}) == {"actionable": False}, s


def test_order_first_in_priority():
    assert na.TYPE_PRIORITY[0] == "order"


def test_invoice_unsent_offers_send():
    d = na.resolve_invoice({"id": 3, "name": "Acme", "email": "a@b.co",
                            "total_cents": 30000, "item_count": 1, "status": "proposed",
                            "pay_status": "unpaid", "invoice_sent_at": None, "age_ts": "t"})
    assert d["actionable"] and d["state"] == "unsent"
    assert d["label"] == "Send invoice" and d["confirm"] is True
    assert d["action"] == {"kind": "dispatch", "keys": ["orders.send_invoice"],
                           "body": {"order_id": 3}}
    assert d["secondary"]["action"] == {"kind": "link", "url": "/console/orders"}
    assert d["summary"] == "#3 · Acme · $300.00 · 1 item"


def test_invoice_sent_unpaid_offers_record_payment_deeplink():
    d = na.resolve_invoice({"id": 4, "name": "", "email": "a@b.co", "total_cents": 5000,
                            "item_count": 2, "status": "confirmed", "pay_status": "unpaid",
                            "invoice_sent_at": "2026-07-01T00:00:00", "age_ts": "t"})
    assert d["actionable"] and d["state"] == "sent_unpaid"
    assert d["label"] == "Record payment"
    assert d["action"] == {"kind": "link", "url": "/console/orders"}
    assert d["secondary"] is None
    assert d["summary"] == "#4 · a@b.co · $50.00 · 2 items"


def test_invoice_non_billing_status_is_done():
    for s in ("new", "packed", "shipped", "done", "cancelled", "delivered", "paid"):
        assert na.resolve_invoice({"id": 1, "status": s}) == {"actionable": False}, s


def test_invoice_sent_and_paid_is_done():
    assert na.resolve_invoice({"id": 1, "status": "proposed", "pay_status": "paid",
                               "invoice_sent_at": "t"}) == {"actionable": False}


def test_invoice_priority_after_order():
    assert na.TYPE_PRIORITY[:2] == ["order", "invoice"]


def test_aggregate_orders_before_invoices(monkeypatch):
    import sqlite3
    from dashboard import biofield_reveals, ff_match_drafts, orders
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    biofield_reveals.init_table(cx); ff_match_drafts.init_table(cx); orders.init_orders_table(cx)
    # an order (status new) and an unsent invoice (status proposed) — both on the orders table
    cx.execute("INSERT INTO orders (created_at,source,external_ref,email,name,items_json,"
               "total_cents,status) VALUES "
               "('2026-07-01T00:00:00','t','i1','o@b.co','O','[]',5000,'new')")
    cx.execute("INSERT INTO orders (created_at,source,external_ref,email,name,items_json,"
               "total_cents,status,invoice_sent_at) VALUES "
               "('2026-07-01T00:00:00','t','i2','v@b.co','V','[]',9000,'proposed',NULL)")
    cx.commit()
    monkeypatch.setattr(na, "_handoff_records", lambda cx: [])
    monkeypatch.setattr(na, "_household_hold_records", lambda cx: [])
    items = na.list_actionable(cx)
    types = [d["type"] for d in items]
    assert types == ["order", "invoice"]   # order sorts before invoice


def test_aggregate_lists_open_orders_first(monkeypatch):
    import sqlite3
    from dashboard import biofield_reveals, ff_match_drafts
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    biofield_reveals.init_table(cx); ff_match_drafts.init_table(cx)
    # orders table: prefer the module's own initializer; fall back to a minimal
    # table if importing dashboard.orders can't run bare in this env.
    try:
        from dashboard import orders as _ord
        _ord.init_orders_table(cx)
    except Exception:
        cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                   "email TEXT, name TEXT, items_json TEXT, total_cents INTEGER, "
                   "status TEXT, created_at TEXT)")
    cx.execute("INSERT INTO orders (source,external_ref,email,name,items_json,total_cents,status,created_at) "
               "VALUES ('test','o1','o@b.co','Ord One','[{\"name\":\"x\"}]',5000,'new','2026-07-01T00:00:00')")
    cx.execute("INSERT INTO orders (source,external_ref,email,name,items_json,total_cents,status,created_at) "
               "VALUES ('test','o2','done@b.co','Done','[]',9000,'shipped','2026-07-02T00:00:00')")
    cx.execute("INSERT INTO ff_match_drafts (email,scan_date,items_json,status,created_at,updated_at) "
               "VALUES ('f@b.co','s','[]','draft','2026-07-01T00:00:00','2026-07-01T00:00:00')")
    cx.commit()
    monkeypatch.setattr(na, "_handoff_records", lambda cx: [])
    monkeypatch.setattr(na, "_household_hold_records", lambda cx: [])
    items = na.list_actionable(cx)
    types = [d["type"] for d in items]
    assert types[0] == "order"                      # order sorts first
    assert "order" in types and "ff_match_draft" in types
    assert all("done@b.co" not in d["summary"] for d in items)  # shipped order skipped


def _seed_cx():
    import sqlite3
    from dashboard import biofield_reveals, ff_match_drafts
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    biofield_reveals.init_table(cx)
    ff_match_drafts.init_table(cx)
    try:
        from dashboard import orders as _ord
        _ord.init_orders_table(cx)
    except Exception:
        cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                   "email TEXT, name TEXT, items_json TEXT, total_cents INTEGER, "
                   "status TEXT, created_at TEXT)")
    return cx


def test_household_overdue_offers_release_now():
    d = na.resolve_household_hold({"group_id": 8, "caregiver": "care@x.co",
                                   "n_orders": 3, "hold_until": "2026-07-10T00:00:00+00:00",
                                   "overdue": True, "age_ts": "2026-07-10T00:00:00+00:00"})
    assert d["type"] == "household" and d["actionable"] and d["state"] == "overdue"
    assert d["label"] == "Release now" and d["confirm"] is True
    assert d["action"] == {"kind": "dispatch", "keys": ["holds.release"],
                           "body": {"group_id": 8}}
    assert d["secondary"]["label"] == "Extend 2 days"
    assert d["secondary"]["action"] == {"kind": "dispatch", "keys": ["holds.extend"],
                                        "body": {"group_id": 8, "days": 2}}
    assert d["secondary"]["confirm"] is False
    assert d["summary"] == "care@x.co · 3 orders · overdue"


def test_household_holding_shows_due_date_and_singular():
    d = na.resolve_household_hold({"group_id": 2, "caregiver": "c@x.co", "n_orders": 1,
                                   "hold_until": "2026-08-01T12:00:00+00:00",
                                   "overdue": False, "age_ts": "2026-08-01T12:00:00+00:00"})
    assert d["state"] == "holding"
    assert d["summary"] == "c@x.co · 1 order · due 2026-08-01"


def test_household_priority_position():
    assert na.TYPE_PRIORITY[:3] == ["order", "invoice", "household"]


def test_household_lister_marks_overdue(monkeypatch):
    import sqlite3
    from dashboard import household_holds as hh
    from dashboard import orders as _ord
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    hh.init_hold_tables(cx)
    _ord.init_orders_table(cx)  # list_open_holds() joins member orders via orders_in_hold_group
    # one overdue open hold (hold_until far in the past)
    cx.execute("INSERT INTO household_holds (caregiver_email, household_key, status, "
               "opened_at, hold_until) VALUES ('care@x.co','hh1','open',"
               "'2026-07-01T00:00:00+00:00','2026-07-02T00:00:00+00:00')")
    cx.commit()
    recs = na._household_hold_records(cx)
    assert len(recs) == 1
    r = recs[0]
    assert r["caregiver"] == "care@x.co" and r["overdue"] is True
    assert r["group_id"] and r["hold_until"] == "2026-07-02T00:00:00+00:00"
    assert r["age_ts"] == r["hold_until"]


def test_aggregate_household_after_invoice(monkeypatch):
    # synthetic household record so we don't need to seed the holds table here
    monkeypatch.setattr(na, "_household_hold_records", lambda cx: [
        {"group_id": 1, "caregiver": "c@x.co", "n_orders": 2,
         "hold_until": "2026-07-02T00:00:00+00:00", "overdue": True,
         "age_ts": "2026-07-02T00:00:00+00:00"}])
    monkeypatch.setattr(na, "_order_records", lambda cx: [])
    monkeypatch.setattr(na, "_invoice_records", lambda cx: [])
    monkeypatch.setattr(na, "_reveal_records", lambda cx: [])
    monkeypatch.setattr(na, "_handoff_records", lambda cx: [])
    monkeypatch.setattr(na, "_ff_records", lambda cx: [])
    items = na.list_actionable(None)
    assert [d["type"] for d in items] == ["household"]
    assert items[0]["label"] == "Release now"


def test_aggregate_orders_by_type_then_age_and_skips_done(monkeypatch):
    cx = _seed_cx()
    now = "2026-07-01T00:00:00"; later = "2026-07-02T00:00:00"
    # two reveals: one draft (older) + one sent (done); one ff draft; one handoff
    cx.execute("INSERT INTO biofield_reveals (email,scan_date,interpretation_json,"
               "remedies_json,first_approved,notified_at,created_at,updated_at) "
               "VALUES ('r@b.co','s1','{}','[]',0,NULL,?,?)", (now, now))
    cx.execute("INSERT INTO biofield_reveals (email,scan_date,interpretation_json,"
               "remedies_json,first_approved,notified_at,created_at,updated_at) "
               "VALUES ('done@b.co','s2','{}','[]',1,?,?,?)", (later, later, later))
    cx.execute("INSERT INTO ff_match_drafts (email,scan_date,items_json,status,"
               "created_at,updated_at) VALUES ('f@b.co','s3','[]','draft',?,?)", (now, now))
    cx.commit()
    # stub the handoff lister to avoid needing the client_portals schema in this unit
    monkeypatch.setattr(na, "_handoff_records",
                        lambda cx: [{"email": "h@b.co", "biofield_status": "ai_draft",
                                     "age_ts": later}])
    # stub the household lister to avoid needing the household_holds schema in this unit
    monkeypatch.setattr(na, "_household_hold_records", lambda cx: [])
    items = na.list_actionable(cx)
    types = [d["type"] for d in items]
    assert types == ["biofield_reveal", "handoff", "ff_match_draft"]  # TYPE_PRIORITY order
    assert "done@b.co" not in [d["summary"].split(" ")[0] for d in items]  # sent reveal skipped
    assert all(d["actionable"] for d in items)
