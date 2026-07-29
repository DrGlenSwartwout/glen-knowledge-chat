from pathlib import Path


SRC = (
    Path(__file__).resolve().parent.parent / "static" / "order-new.html"
).read_text()


def test_hawaii_shipments_request_manual_shipping():
    assert 'state==="HI"' in SRC
    assert "function requestHawaiiShipping(force=false)" in SRC
    assert "Enter the actual shipping cost ($)" in SRC


def test_hawaii_shipping_prompt_runs_before_create_and_edit():
    assert SRC.count("if (!requestHawaiiShipping()) return;") == 2


def test_shipping_quote_uses_manual_hawaii_amount():
    hawaii_branch = SRC.split("async function priceShipping(){", 1)[1].split(
        'fetch("/api/orders/shipping-preview"', 1
    )[0]
    assert "if (isHawaiiShipment())" in hawaii_branch
    assert "requestHawaiiShipping(true)" in hawaii_branch
