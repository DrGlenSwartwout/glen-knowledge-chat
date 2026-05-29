"""Tests for dashboard.tracking — parse USPS Click-N-Ship confirmation emails."""

import sqlite3

from dashboard.tracking import (
    normalize_tracking,
    tracking_url,
    parse_cns_confirmation,
    build_tracking_email,
    init_tracking_schema,
    record_shipment,
    shipment_exists,
)


# A faithful slice of a real noreply-ecns@usps.com "Payment Confirmation" email.
# Two shipments: one 3-line address, one with an apartment (4-line) to prove the
# city/state/zip line is found from the END, not by position. The anchor text is
# the clean 34-digit IMpb (the href is charset-mangled in real emails, so we
# parse the anchor TEXT, never the href).
SAMPLE = """
<html><body>
  <td class="header-confirmation">
    <p>Order #: <a href="https://cnsb.usps.com/confirmation-page?orderUUID=019e3cbe-bd74-7a99-8876-2269d7f83860">019e3cbe-bd74-7a99-8876-2269d7f83860</a></p>
  </td>
  <a class="button" href="https://cnsb.usps.com/history/orders/019e3cbe-bd74-7a99-8876-2269d7f83860">View Order Details</a>

  <table class="item-contents-table"><tbody><tr>
    <td class="item-contents-column">
      <table><tbody><tr><td>
        <p class="bold"> Priority Mail&#174;</p>
        <a href="https://tools.usps.com/go/TrackConfirmAction!input.action?tLabelsB08522452499405530109355381515251"> 4208522452499405530109355381515251</a>
        <p class="bold">Scheduled delivery date: 05/21/2026</p>
        <p class="bold">Shipped To:</p>
        <p class="pt-5">Cyndi O'Brien</p>
        <p class="pt-5">1016 W CHICAGO CT</p>
        <p class="pt-5">CHANDLER AZ 85224-5249 US</p>
      </td></tr></tbody></table>
    </td>
    <td class="item-total-column"><p class="price-col-p">$11.99</p></td>
  </tr></tbody></table>

  <table class="item-contents-table"><tbody><tr>
    <td class="item-contents-column">
      <table><tbody><tr><td>
        <p class="bold"> Priority Mail&#174;</p>
        <a href="https://tools.usps.com/go/TrackConfirmAction!input.action?tLabelsB01201853009405530109355381515275"> 4201201853009405530109355381515275</a>
        <p class="bold">Scheduled delivery date: 05/21/2026</p>
        <p class="bold">Shipped To:</p>
        <p class="pt-5">Frank Swartwout</p>
        <p class="pt-5">5 BANNER HILL LN</p>
        <p class="pt-5">APT 2</p>
        <p class="pt-5">AVERILL PARK NY 12018-5300 US</p>
      </td></tr></tbody></table>
    </td>
    <td class="item-total-column"><p class="price-col-p">$21.17</p></td>
  </tr></tbody></table>
</body></html>
"""


# ── normalize_tracking ──────────────────────────────────────────────────────

def test_normalize_already_22_digits():
    assert normalize_tracking("9405530109355379725082") == "9405530109355379725082"


def test_normalize_strips_impb_routing_prefix():
    # 420 + 9-digit ZIP routing + 22-digit tracking
    assert normalize_tracking("4208522452499405530109355381515251") == \
        "9405530109355381515251"


def test_normalize_ignores_spaces():
    assert normalize_tracking("4205 8522 4524 9 9405 5301 0935 5381 5152 51") == \
        "9405530109355381515251"


def test_normalize_returns_none_for_junk():
    assert normalize_tracking("") is None
    assert normalize_tracking("not a number") is None


# ── tracking_url ────────────────────────────────────────────────────────────

def test_tracking_url_is_customer_facing_track_link():
    url = tracking_url("9405530109355381515251")
    assert url == (
        "https://tools.usps.com/go/TrackConfirmAction?"
        "tLabels=9405530109355381515251"
    )


# ── parse_cns_confirmation ──────────────────────────────────────────────────

def test_parse_extracts_order_uuid():
    out = parse_cns_confirmation(SAMPLE)
    assert out["order_uuid"] == "019e3cbe-bd74-7a99-8876-2269d7f83860"


def test_parse_finds_every_shipment():
    out = parse_cns_confirmation(SAMPLE)
    assert len(out["shipments"]) == 2


def test_parse_first_shipment_fields():
    s = parse_cns_confirmation(SAMPLE)["shipments"][0]
    assert s["tracking"] == "9405530109355381515251"
    assert s["recipient_name"] == "Cyndi O'Brien"
    assert s["street"] == "1016 W CHICAGO CT"
    assert s["city"] == "CHANDLER"
    assert s["state"] == "AZ"
    assert s["zip"] == "85224-5249"
    assert "Priority Mail" in s["service"]
    assert s["delivery_date"] == "05/21/2026"


def test_parse_handles_multiline_address():
    # The apartment line must not break city/state/zip detection.
    s = parse_cns_confirmation(SAMPLE)["shipments"][1]
    assert s["recipient_name"] == "Frank Swartwout"
    assert s["city"] == "AVERILL PARK"
    assert s["state"] == "NY"
    assert s["zip"] == "12018-5300"
    assert "APT 2" in s["street"]


def test_parse_empty_html_is_safe():
    out = parse_cns_confirmation("")
    assert out == {"order_uuid": None, "shipments": []}


# ── build_tracking_email ─────────────────────────────────────────────────────

def test_email_has_tracking_link_and_signature():
    msg = build_tracking_email("9405530109355381515251", "Cyndi O'Brien")
    assert msg["subject"] == "tracking number"
    # live USPS tracking link
    assert "tLabels=9405530109355381515251" in msg["html"]
    assert "9405530109355381515251" in msg["text"]
    # personalized greeting (first name only)
    assert "Hi Cyndi," in msg["html"]
    assert "Hi Cyndi," in msg["text"]
    # Glen's sign-off present in both parts
    assert "Dr. Glen Swartwout" in msg["html"]
    assert "Dr. Glen Swartwout" in msg["text"]
    assert "Truly.VIP/ASH" in msg["html"]


def test_email_without_name_omits_greeting():
    msg = build_tracking_email("9405530109355381515251")
    assert "Hi " not in msg["html"]
    assert msg["text"].startswith("Your order is on its way")


# ── shipments table ──────────────────────────────────────────────────────────

def test_record_shipment_is_idempotent(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_tracking_schema(cx)
        init_tracking_schema(cx)  # idempotent
        first = record_shipment(
            cx,
            tracking_number="9405530109355381515251",
            recipient_name="Cyndi O'Brien",
            status="drafted",
        )
        # second insert of the same tracking number is a no-op
        dup = record_shipment(
            cx,
            tracking_number="9405530109355381515251",
            recipient_name="Cyndi O'Brien",
            status="drafted",
        )
        count = cx.execute("SELECT COUNT(*) FROM shipments").fetchone()[0]
        assert first is not None
        assert dup is None
        assert count == 1
        assert shipment_exists(cx, "9405530109355381515251")
        assert not shipment_exists(cx, "0000000000000000000000")
