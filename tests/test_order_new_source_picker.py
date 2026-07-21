import app as app_module


def test_order_new_page_has_source_picker():
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    body = c.get("/orders/new").get_data(as_text=True)
    # a per-line Source select + it is threaded into the POST payload
    assert "setSource" in body
    assert "b.source" in body            # linesPayload includes source
    assert ">self<" in body or "'self'" in body or '"self"' in body   # self default option
