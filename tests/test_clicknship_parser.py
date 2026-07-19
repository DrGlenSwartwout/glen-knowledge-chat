"""Unit tests for dashboard.clicknship.parse_confirmation.

Pure — no network. Uses a real, saved Click-N-Ship Payment Confirmation
(tests/fixtures/clicknship_sample.eml), fetched once via IMAP from Rae's
Gmail. That sample has TWO packages in one order (two item rows), which
this test exercises directly.
"""

import os
import re

import pytest

from dashboard.clicknship import parse_confirmation

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "clicknship_sample.eml")


@pytest.fixture
def raw_sample():
    with open(FIXTURE_PATH, "rb") as f:
        return f.read()


@pytest.fixture
def shipments(raw_sample):
    return parse_confirmation(raw_sample)


def test_parses_at_least_one_package(shipments):
    assert isinstance(shipments, list)
    assert len(shipments) >= 1


def test_parses_both_packages_in_the_sample(shipments):
    # This fixture is a real two-item order.
    assert len(shipments) == 2


def test_tracking_present_and_digit_length_plausible(shipments):
    for s in shipments:
        assert s["tracking"], s
        assert s["tracking"].isdigit()
        # Normalized USPS tracking numbers run roughly 20-22 digits.
        assert 18 <= len(s["tracking"]) <= 26, s["tracking"]
        # tracking_raw is the untouched visible link text (routing-prefixed).
        assert s["tracking_raw"]
        assert s["tracking_raw"].isdigit()
        assert len(s["tracking_raw"]) >= len(s["tracking"])


def test_tracking_normalization_strips_routing_prefix(shipments):
    for s in shipments:
        # The raw visible text starts with the 420 IMb routing prefix; the
        # normalized tracking number should not carry that prefix and
        # should start with a standard USPS tracking prefix instead.
        assert s["tracking_raw"].startswith("420")
        assert s["tracking"].startswith(("9400", "9405", "9407", "9408", "92", "93", "94"))


def test_scheduled_delivery_present_for_at_least_one_package(shipments):
    dates = [s["scheduled_delivery"] for s in shipments if s["scheduled_delivery"]]
    assert dates, "expected at least one package with a scheduled delivery date"
    for d in dates:
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", d)


def test_recipient_name_city_state_zip_present(shipments):
    for s in shipments:
        assert s["recipient_name"]
        assert s["recipient_city"]
        assert s["recipient_state"]
        assert re.match(r"^[A-Z]{2}$", s["recipient_state"])
        assert s["recipient_zip"]
        assert re.match(r"^\d{5}(-\d{4})?$", s["recipient_zip"])
        assert s["recipient_country"] == "US"


def test_service_extracted(shipments):
    for s in shipments:
        assert s["service"]
        assert "priority mail" in s["service"].lower()


def test_cost_cents_extracted(shipments):
    for s in shipments:
        assert isinstance(s["cost_cents"], int)
        assert s["cost_cents"] > 0


def test_order_uuid_and_placed_on_shared_across_packages(shipments):
    uuids = {s["order_uuid"] for s in shipments}
    placed = {s["placed_on"] for s in shipments}
    assert len(uuids) == 1
    assert re.match(r"^[0-9a-f-]{36}$", list(uuids)[0])
    assert len(placed) == 1
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", list(placed)[0])


def test_second_package_has_company_line(shipments):
    # The sample's second package has a company line between name and street.
    with_company = [s for s in shipments if s["recipient_company"]]
    assert with_company
    assert with_company[0]["recipient_company"]
    assert with_company[0]["recipient_street"]


def test_returns_empty_list_for_non_confirmation_input():
    assert parse_confirmation(b"") == []
    assert parse_confirmation(b"not an email at all") == []
    assert parse_confirmation("<html><body>hello</body></html>") == []


def test_never_raises_on_garbage_bytes():
    # Defensive: malformed/binary input must not raise.
    garbage_inputs = [
        b"\x00\x01\x02\xff\xfe",
        None,
        12345,
        object(),
    ]
    for g in garbage_inputs:
        assert parse_confirmation(g) == []


def test_parse_confirmation_accepts_bare_html_string(raw_sample):
    import email as email_mod

    msg = email_mod.message_from_bytes(raw_sample)
    payload = msg.get_payload(decode=True)
    html_body = payload.decode(msg.get_content_charset() or "utf-8", "replace")

    result = parse_confirmation(html_body)
    assert len(result) == 2
