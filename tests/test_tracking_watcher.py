import sqlite3
from cns_tracking_watcher import handle_confirmation
from dashboard.tracking import init_tracking_schema

CONF = """<html><body>
  <p>Order #: <a href="https://cnsb.usps.com/confirmation-page?orderUUID=019e0000-0000-7000-8000-000000000001">019e0000-0000-7000-8000-000000000001</a></p>
  <table class="item-contents-table"><tbody><tr>
    <td class="item-contents-column">
      <table><tbody><tr><td>
        <p class="bold"> Priority Mail&#174;</p>
        <a href="x">4200000000009405530109355300000001</a>
        <p class="bold">Shipped To:</p>
        <p class="pt-5">New Buyer</p>
        <p class="pt-5">1 A St</p>
        <p class="pt-5">RENO NV 89501-0001 US</p>
      </td></tr></tbody></table>
    </td>
    <td class="item-total-column"><p class="price-col-p">$11.99</p></td>
  </tr></tbody></table>
</body></html>"""

def _cx():
    cx = sqlite3.connect(":memory:"); init_tracking_schema(cx); return cx

def test_no_ghl_match_precise_harvest_fills_to_and_persists():
    calls = {}
    def find_contact(name): return None
    def harvest_fn(name):
        return {"email": "new@example.com", "first": "New", "last": "Buyer",
                "phone": None, "source": "eprocessing", "products": []}
    def persist(identity, name):
        calls["persist"] = (identity, name); return {"contact_id": "C1", "onboarded": False}
    drafts = []
    def draft_fn(to, subject, html, text): drafts.append(to); return "D1"
    res = handle_confirmation(CONF, "M1", _cx(), find_contact, draft_fn,
                              harvest_fn=harvest_fn, persist_contact=persist, dry_run=False)[0]
    assert res["to"] == "new@example.com"
    assert res["confidence"] == "harvested"
    assert res["status"] == "drafted"
    assert drafts == ["new@example.com"]
    assert calls["persist"][0]["email"] == "new@example.com"

def test_no_ghl_match_no_harvest_stays_needs_review():
    def find_contact(name): return None
    def harvest_fn(name): return None
    def draft_fn(to, subject, html, text): return "D2"
    res = handle_confirmation(CONF, "M2", _cx(), find_contact, draft_fn,
                              harvest_fn=harvest_fn, persist_contact=lambda i, n: {}, dry_run=False)[0]
    assert res["to"] == "(blank — needs review)"
    assert res["status"] == "needs_review"

def test_harvest_fn_none_is_legacy_behavior():
    def find_contact(name): return None
    def draft_fn(to, subject, html, text): return "D3"
    res = handle_confirmation(CONF, "M3", _cx(), find_contact, draft_fn, dry_run=False)[0]
    assert res["status"] == "needs_review"

def test_dry_run_previews_harvest_without_persisting():
    def find_contact(name): return None
    def harvest_fn(name):
        return {"email": "d@example.com", "first": "New", "last": "Buyer",
                "phone": None, "source": "neworder", "products": []}
    def persist(identity, name): raise AssertionError("must not persist in dry-run")
    res = handle_confirmation(CONF, "M4", _cx(), find_contact, lambda **k: None,
                              harvest_fn=harvest_fn, persist_contact=persist, dry_run=True)[0]
    assert res["to"] == "d@example.com"
    assert res["confidence"] == "harvested"
    assert res["action"] == "would draft (harvested)"
