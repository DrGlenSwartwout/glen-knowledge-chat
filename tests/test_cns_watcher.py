"""Tests for cns_tracking_watcher.handle_confirmation — the decision core."""

import sqlite3

import pytest

from dashboard.tracking import init_tracking_schema, shipment_exists
from cns_tracking_watcher import handle_confirmation, build_raw


ONE_SHIPMENT = """
<a href="x?orderUUID=019e3cbe-bd74-7a99-8876-2269d7f83860">order</a>
<td class="item-contents-column"><table><tr><td>
  <p class="bold">Priority Mail&#174;</p>
  <a href="x?tLabelsB08522452499405530109355381515251"> 4208522452499405530109355381515251</a>
  <p class="bold">Shipped To:</p>
  <p class="pt-5">Cyndi O'Brien</p>
  <p class="pt-5">1016 W CHICAGO CT</p>
  <p class="pt-5">CHANDLER AZ 85224-5249 US</p>
</td></tr></table></td><td class="item-total-column"><p>$11.99</p></td>
"""

HIGH = lambda name: {"email": "cyndi@example.com", "contact_id": "c1",
                     "name": name, "confidence": "high"}
NONE = lambda name: None


@pytest.fixture
def cx(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "chat_log.db"))
    init_tracking_schema(conn)
    yield conn
    conn.close()


def _recording_draft_fn():
    calls = []
    def fn(to, subject, html, text):
        calls.append({"to": to, "subject": subject})
        return f"draft_{len(calls)}"
    fn.calls = calls
    return fn


def test_dry_run_mutates_nothing(cx):
    draft_fn = _recording_draft_fn()
    res = handle_confirmation(ONE_SHIPMENT, "m1", cx, HIGH, draft_fn, dry_run=True)
    assert res[0]["action"] == "would draft"
    assert res[0]["to"] == "cyndi@example.com"
    assert draft_fn.calls == []                               # no draft created
    assert not shipment_exists(cx, "9405530109355381515251")  # nothing recorded


def test_live_high_confidence_prefills_to_and_records(cx):
    draft_fn = _recording_draft_fn()
    res = handle_confirmation(ONE_SHIPMENT, "m1", cx, HIGH, draft_fn, dry_run=False)
    assert res[0]["action"] == "drafted"
    assert res[0]["status"] == "drafted"
    assert draft_fn.calls[0]["to"] == "cyndi@example.com"
    assert draft_fn.calls[0]["subject"] == "tracking number"
    assert shipment_exists(cx, "9405530109355381515251")


def test_live_no_match_is_needs_review_blank_to(cx):
    draft_fn = _recording_draft_fn()
    res = handle_confirmation(ONE_SHIPMENT, "m1", cx, NONE, draft_fn, dry_run=False)
    assert res[0]["status"] == "needs_review"
    assert res[0]["confidence"] == "none"
    # Draft is still created (blank To) so Glen always has something to send.
    assert draft_fn.calls[0]["to"] is None
    assert shipment_exists(cx, "9405530109355381515251")


def test_idempotent_second_run_skips(cx):
    draft_fn = _recording_draft_fn()
    handle_confirmation(ONE_SHIPMENT, "m1", cx, HIGH, draft_fn, dry_run=False)
    res2 = handle_confirmation(ONE_SHIPMENT, "m1", cx, HIGH, draft_fn, dry_run=False)
    assert res2[0]["action"] == "skipped (already processed)"
    assert len(draft_fn.calls) == 1   # drafted once, never twice


def test_build_raw_is_valid_base64_mime():
    raw = build_raw("tracking number", "<b>hi</b>", "hi", to="x@example.com")
    import base64
    decoded = base64.urlsafe_b64decode(raw).decode()
    assert "Subject: tracking number" in decoded
    assert "To: x@example.com" in decoded
    assert "text/html" in decoded and "text/plain" in decoded
