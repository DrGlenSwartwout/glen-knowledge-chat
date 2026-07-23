import sqlite3
from dashboard import portal_view as pv
from dashboard import household as hh

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    hh.init_household_tables(cx)
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, email TEXT, total_cents INTEGER, "
               "invoice_token TEXT, items_json TEXT, pay_status TEXT, status TEXT)")
    cx.execute("INSERT INTO orders VALUES (1,'michael@x.com',5000,'tok1','[{\"slug\":\"a\"}]','','open')")
    return cx

def test_block_hides_items_when_amount_only():
    cx = _cx()
    hh.add_member(cx, "steve@x.com", "michael@x.com", relationship="partner")
    hh.set_pay_consent(cx, "steve@x.com", "michael@x.com", 1, share_scope="amount_only")
    block = pv._caregiver_pay_block(cx, "steve@x.com", True)
    assert len(block["orders"]) == 1
    o = block["orders"][0]
    assert o["order_id"] == 1 and o["amount_dollars"] == "50.00"
    assert o["items"] is None  # amount_only hides line items

def test_block_empty_without_consent_and_when_disabled():
    cx = _cx()
    hh.add_member(cx, "steve@x.com", "michael@x.com", relationship="partner")
    assert pv._caregiver_pay_block(cx, "steve@x.com", True)["orders"] == []   # no consent
    hh.set_pay_consent(cx, "steve@x.com", "michael@x.com", 1)
    assert pv._caregiver_pay_block(cx, "steve@x.com", False) == {"members": [], "orders": []}  # flag off

def test_block_firewall_keys_only():
    """Structural firewall: block dict keys are exactly {members, orders}, and each
    order dict's keys are a subset of the whitelisted set — no clinical field can
    ride along, no matter what the block's internals do."""
    cx = _cx()
    hh.add_member(cx, "steve@x.com", "michael@x.com", relationship="partner")
    hh.set_pay_consent(cx, "steve@x.com", "michael@x.com", 1, share_scope="line_items")
    block = pv._caregiver_pay_block(cx, "steve@x.com", True)
    assert set(block.keys()) == {"members", "orders"}
    allowed = {"order_id", "beneficiary_email", "beneficiary_name",
               "amount_dollars", "token", "items"}
    assert block["orders"], "expected at least one order to check keys against"
    for o in block["orders"]:
        assert set(o.keys()) <= allowed
