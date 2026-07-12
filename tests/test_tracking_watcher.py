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


# ── Confidence-gated auto-send ────────────────────────────────────────────────
# When --auto-send is on, a HIGH GHL match or a precise HARVEST auto-sends the
# tracking email; MEDIUM (fuzzy single match) and LOW/none stay drafts for review.

def _high_match(name):
    return {"email": "known@example.com", "contact_id": "CG1",
            "name": name, "confidence": "high"}

def _medium_match(name):
    return {"email": "maybe@example.com", "contact_id": "CG2",
            "name": name, "confidence": "medium"}


def test_auto_send_high_confidence_sends_not_drafts():
    sent, drafted = [], []
    def send_fn(to, subject, html, text): sent.append(to); return "SENT1"
    def draft_fn(to, subject, html, text): drafted.append(to); return "D"
    res = handle_confirmation(CONF, "AS1", _cx(), _high_match, draft_fn,
                              send_fn=send_fn, auto_send=True, dry_run=False)[0]
    assert res["status"] == "sent"
    assert res["action"] == "sent"
    assert res["draft_id"] == "SENT1"
    assert sent == ["known@example.com"]
    assert drafted == []          # never also drafts


def test_auto_send_harvested_sends_and_persists():
    def find_contact(name): return None
    def harvest_fn(name):
        return {"email": "harv@example.com", "first": "New", "last": "Buyer",
                "phone": None, "source": "eprocessing", "products": []}
    persisted = {}
    def persist(identity, name):
        persisted["email"] = identity["email"]; return {"contact_id": "H1", "onboarded": False}
    sent, drafted = [], []
    def send_fn(to, subject, html, text): sent.append(to); return "SENT2"
    def draft_fn(to, subject, html, text): drafted.append(to); return "D"
    res = handle_confirmation(CONF, "AS2", _cx(), find_contact, draft_fn,
                              harvest_fn=harvest_fn, persist_contact=persist,
                              send_fn=send_fn, auto_send=True, dry_run=False)[0]
    assert res["status"] == "sent"
    assert res["confidence"] == "harvested"
    assert sent == ["harv@example.com"]
    assert drafted == []
    assert persisted["email"] == "harv@example.com"   # still ingests the buyer


def test_auto_send_medium_confidence_still_drafts():
    sent, drafted = [], []
    def send_fn(to, subject, html, text): sent.append(to); return "S"
    def draft_fn(to, subject, html, text): drafted.append(to); return "DRAFT2"
    res = handle_confirmation(CONF, "AS3", _cx(), _medium_match, draft_fn,
                              send_fn=send_fn, auto_send=True, dry_run=False)[0]
    assert res["status"] == "drafted"
    assert res["draft_id"] == "DRAFT2"
    assert drafted == ["maybe@example.com"]
    assert sent == []             # fuzzy match is never auto-sent


def test_auto_send_low_confidence_stays_needs_review_no_send():
    def low_match(name):
        return {"email": "amb@example.com", "contact_id": "L1",
                "name": name, "confidence": "low"}
    sent = []
    def send_fn(to, subject, html, text): sent.append(to); return "S"
    def draft_fn(to, subject, html, text): return "D"
    res = handle_confirmation(CONF, "AS4", _cx(), low_match, draft_fn,
                              send_fn=send_fn, auto_send=True, dry_run=False)[0]
    assert res["status"] == "needs_review"
    assert sent == []


def test_auto_send_off_high_confidence_drafts_backward_compatible():
    sent, drafted = [], []
    def send_fn(to, subject, html, text): sent.append(to); return "S"
    def draft_fn(to, subject, html, text): drafted.append(to); return "D5"
    res = handle_confirmation(CONF, "AS5", _cx(), _high_match, draft_fn,
                              send_fn=send_fn, auto_send=False, dry_run=False)[0]
    assert res["status"] == "drafted"
    assert drafted == ["known@example.com"]
    assert sent == []             # auto_send off => legacy draft behavior


def test_dry_run_auto_send_previews_would_send_and_sends_nothing():
    def send_fn(to, subject, html, text): raise AssertionError("must not send in dry-run")
    res = handle_confirmation(CONF, "AS6", _cx(), _high_match, lambda **k: None,
                              send_fn=send_fn, auto_send=True, dry_run=True)[0]
    assert res["action"] == "would send"
    assert res["status"] == "sent"
