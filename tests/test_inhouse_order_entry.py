"""Phase 1 in-house order entry: proposed-invoice lifecycle (confirm /
record_payment), person_id persistence, and dashboard.customers helpers."""
import sqlite3
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _orders_db():
    from dashboard import orders as O
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    return O, cx


def _people_db():
    """Minimal people + orders tables for the customers helpers."""
    from dashboard import customers as C, orders as O
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    cx.execute("""CREATE TABLE people (
        id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,
        first_name TEXT DEFAULT '', last_name TEXT DEFAULT '', name TEXT DEFAULT '',
        phone TEXT DEFAULT '', city TEXT DEFAULT '', state TEXT DEFAULT '',
        country TEXT DEFAULT '', source TEXT DEFAULT '', order_count INTEGER DEFAULT 0,
        last_order_date TEXT DEFAULT '', created_at TEXT DEFAULT '', updated_at TEXT DEFAULT '')""")
    cx.commit()
    C.add_people_address_columns(cx)
    O.init_orders_table(cx)
    return C, O, cx


# ── proposed-invoice lifecycle ───────────────────────────────────────────────

def test_proposed_invoice_confirm_then_record_payment():
    O, cx = _orders_db()
    oid = O.upsert_order(cx, source="in-house", external_ref="INH-1", status="proposed",
                         email="c@x.com", name="Cara", person_id=42,
                         items=[{"slug": "bone-builder", "name": "Bone Builder", "qty": 1,
                                 "unit_cents": 7000, "line_cents": 7000}],
                         total_cents=7000)
    row = O.get_order(cx, oid)
    assert row["status"] == "proposed"
    assert row["person_id"] == 42  # person_id persisted

    ctx = {"cx": cx}
    O._confirm_exec({"order_id": oid}, ctx)
    assert O.get_order(cx, oid)["status"] == "confirmed"

    res = O._record_payment_exec({"order_id": oid, "method": "Zelle", "amount_cents": 7000}, ctx)
    paid = O.get_order(cx, oid)
    assert res["status"] == "new"
    assert paid["status"] == "new"          # enters the fulfillment board
    assert paid["pay_status"] == "paid"
    assert paid["pay_method"] == "Zelle"
    assert paid["paid_cents"] == 7000
    assert paid["paid_at"]


def test_confirm_rejects_non_proposed():
    O, cx = _orders_db()
    oid = O.upsert_order(cx, source="funnel", external_ref="INV-9", status="new")
    try:
        O._confirm_exec({"order_id": oid}, {"cx": cx})
        assert False, "expected ValueError on confirming a non-proposed order"
    except ValueError:
        pass


def test_record_payment_rejects_already_fulfilling():
    O, cx = _orders_db()
    oid = O.upsert_order(cx, source="funnel", external_ref="INV-10", status="packed")
    try:
        O._record_payment_exec({"order_id": oid, "method": "card"}, {"cx": cx})
        assert False, "expected ValueError recording payment on a packed order"
    except ValueError:
        pass


def test_record_payment_defaults_amount_to_total():
    O, cx = _orders_db()
    oid = O.upsert_order(cx, source="in-house", external_ref="INH-2", status="confirmed",
                         total_cents=12345)
    O._record_payment_exec({"order_id": oid, "method": "card"}, {"cx": cx})
    assert O.get_order(cx, oid)["paid_cents"] == 12345


def test_confirm_and_record_payment_registered_owner_only():
    from dashboard.actions import ACTION_REGISTRY
    from dashboard.rbac import OWNER
    for key in ("orders.confirm", "orders.record_payment"):
        act = ACTION_REGISTRY[key]
        assert act.permission == (OWNER,), f"{key} must be OWNER-only"


# ── customers helpers ────────────────────────────────────────────────────────

def test_find_people_matches_name_email_phone():
    C, O, cx = _people_db()
    cx.execute("INSERT INTO people (email, name, phone) VALUES (?,?,?)",
               ("jane@x.com", "Jane Smith", "808-555-1212"))
    cx.commit()
    assert C.find_people(cx, "jane")[0]["email"] == "jane@x.com"
    assert C.find_people(cx, "smith")[0]["name"] == "Jane Smith"
    assert C.find_people(cx, "jane@x")[0]["email"] == "jane@x.com"
    assert C.find_people(cx, "") == []


def test_find_people_synthesizes_name_from_first_last_when_blank():
    """A contact with an empty `name` column but populated first/last (common for
    imported records) still matches a name search — and must come back with a
    usable `name` so the order-entry picker fills the Name field, not just email."""
    C, O, cx = _people_db()
    cx.execute("INSERT INTO people (email, first_name, last_name, name) VALUES (?,?,?,?)",
               ("miriam@x.com", "Miriam Lynn", "Nelson", ""))
    cx.commit()
    hit = C.find_people(cx, "miriam")[0]
    assert hit["email"] == "miriam@x.com"
    assert hit["name"] == "Miriam Lynn Nelson"


def test_upsert_person_address_saves_and_normalizes_street():
    C, O, cx = _people_db()
    pid = C.find_or_create_by_email(cx, email="bob@x.com", name="Bob")
    C.upsert_person_address(cx, pid, {"street": "123 Main", "city": "Hilo",
                                      "state": "HI", "zip": "96720", "country": "us"})
    p = C.get_person(cx, pid)
    assert p["address1"] == "123 Main"
    assert p["city"] == "Hilo" and p["state"] == "HI" and p["zip"] == "96720"
    assert p["country"] == "US"


def test_last_address_for_reads_newest_order():
    C, O, cx = _people_db()
    O.upsert_order(cx, source="funnel", external_ref="INV-1", email="kim@x.com",
                   address={"street": "1 Old St", "city": "A", "state": "OR", "zip": "97000"})
    O.upsert_order(cx, source="funnel", external_ref="INV-2", email="kim@x.com",
                   address={"street": "9 New Ave", "city": "B", "state": "WA", "zip": "98000"})
    last = C.last_address_for(cx, "kim@x.com")
    assert last["address1"] == "9 New Ave"
    assert last["state"] == "WA"
    assert C.last_address_for(cx, "nobody@x.com") == {}


def test_find_or_create_by_email_is_idempotent():
    C, O, cx = _people_db()
    a = C.find_or_create_by_email(cx, email="dup@x.com", name="Dup")
    b = C.find_or_create_by_email(cx, email="DUP@x.com", name="Dup2")
    assert a == b  # case-insensitive, no duplicate person
