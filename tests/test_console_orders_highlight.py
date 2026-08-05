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
    assert "Client / ship to:" in html
    assert "Shipping address:" in html
    assert "+ dropShipDetailsHtml(o)" in html
    assert "var displayName = (o.source==='dropship' && o.address && o.address.name)" in html
    assert "esc(displayName)" in html


def test_packed_and_shipped_orders_remain_editable():
    html = _html()
    assert "o.status==='packed') acts = editBtn(o.id)" in html
    assert "o.status==='shipped') acts = editBtn(o.id)" in html
    assert "acts = keep + editBtn(o.id)" in html


def test_order_editor_keeps_ship_to_name_separate():
    html = (Path(__file__).resolve().parent.parent / "static" / "order-new.html").read_text()
    assert 'id="a-name"' in html
    assert 'name:$("a-name").value.trim()' in html
    assert '$("a-name").value=a.name||o.name||""' in html


def test_packed_shipped_and_combined_orders_can_print_existing_label():
    html = _html()
    assert "function labelBtn(o)" in html
    assert "function printShippingLabel(id)" in html
    assert "window.open(o.label_url, '_blank')" in html
    assert "var labelOrder = ms.find" in html
    assert "labelOrder ? labelBtn(labelOrder)" in html


def test_unpurchased_labels_are_bought_then_opened_for_printing():
    html = _html()
    assert "Buy + print shipping label" in html
    assert "function buyAndPrintLabel(id)" in html
    assert "'/api/action/orders.create_label'" in html
    assert "function buyAndPrintGroupLabel(sid)" in html
    assert "'/api/action/shipments.create_label'" in html
    assert "res.status==='done' && out.label_url" in html


def test_easypost_tracking_can_be_emailed_without_retyping():
    html = _html()
    assert "o.tracking_number ? btn(o.id,'orders.send_tracking','Send tracking')" in html
    assert "HH.byId[Number(id)].tracking_number" in html
    assert "prompt('Tracking number (optional):', saved)" in html


def test_expanded_cards_put_actions_above_products():
    html = _html()
    actions = "+ '<div class=\"acts\">'+acts+'</div>'"
    products = "+ (items?'<div class=\"meta prods\">'+items+'</div>':'')"

    order_body = html.split("var body = '<div class=\"card-body\">'", 1)[1].split(
        "return '<div class=\"card\" data-oid=", 1
    )[0]
    shipment_body = html.split("var body = '<div class=\"card-body\">'", 2)[2].split(
        "return '<div class=\"card hh-ship\"", 1
    )[0]

    assert order_body.index(actions) < order_body.index(products)
    assert shipment_body.index(actions) < shipment_body.index(products)
