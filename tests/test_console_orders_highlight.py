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
