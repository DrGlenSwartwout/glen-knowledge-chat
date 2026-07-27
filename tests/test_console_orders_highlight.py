from pathlib import Path


def _html():
    return (Path(__file__).resolve().parent.parent / "static" / "console-orders.html").read_text()


def test_order_card_has_data_oid():
    assert 'data-oid="' in _html()


def test_orders_reads_order_param_and_flashes():
    html = _html()
    assert "URLSearchParams(location.search).get('order')" in html
    assert "scrollIntoView" in html
    assert "ord-flash" in html


def test_order_items_fall_back_to_slug_for_display():
    html = _html()
    assert "i.name||i.slug||''" in html
    assert "it.name||it.slug||('Line '+" in html


def test_dropship_cards_show_recipient_and_shipping_address():
    html = _html()
    assert "function dropShipDetailsHtml(o)" in html
    assert "Drop-ship details" in html
    assert "Shipping address:" in html
    assert "+ dropShipDetailsHtml(o)" in html
