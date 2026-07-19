from pathlib import Path


def _html():
    return (Path(__file__).resolve().parent.parent / "static" / "console-money.html").read_text()


def test_ar_row_has_data_inv():
    assert 'data-inv="' in _html()


def test_receivables_reads_invoice_param_and_flashes():
    html = _html()
    assert "URLSearchParams(location.search).get('invoice')" in html
    assert "scrollIntoView" in html
    assert "inv-flash" in html  # the transient highlight class


def test_escattr_escapes_double_quotes():
    # Regression: the Void/Send-reminder/Cancel buttons pass a JSON payload through
    # escAttr into a double-quoted onclick="" attribute. If escAttr does not encode
    # the JSON's double quotes, the attribute terminates early and the click fires
    # nothing ("Void gives no response"). escAttr must encode " (and &).
    html = _html()
    assert r"""replace(/"/g,'&quot;')""" in html   # escAttr encodes double quotes
    assert r"""replace(/&/g,'&amp;')""" in html     # ...and ampersands, before the rest


def test_void_button_onclick_wellformed_after_render():
    # Parse a rendered Void button the way a browser would; the onclick must survive
    # intact (no stray attributes from an early-terminated attribute value) and decode
    # back to a valid JSON payload the act() handler can JSON.parse.
    import json
    from html.parser import HTMLParser

    def esc_attr(s):  # mirrors the JS escAttr in console-money.html
        return "'" + str(s).replace("&", "&amp;").replace('"', "&quot;").replace("'", "&#39;") + "'"

    iid = 123
    params_void = json.dumps({"invoice_id": iid}, separators=(",", ":"))
    btn = ('<button class="void" onclick="MoneyReceivables.act(' + str(iid)
           + ",'finance.void_invoice'," + esc_attr(params_void) + ')">Void</button>')

    captured = {}

    class P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "button":
                captured["attrs"] = dict(attrs)

    P().feed(btn)
    attrs = captured["attrs"]
    # No garbage attributes leaked from a truncated onclick value.
    assert set(attrs) == {"class", "onclick"}
    onclick = attrs["onclick"]
    # The decoded handler carries the full, valid JSON payload.
    assert onclick == "MoneyReceivables.act(123,'finance.void_invoice','{\"invoice_id\":123}')"
    json.loads(onclick.split("'")[-2])  # the payload arg round-trips through JSON.parse
