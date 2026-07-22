import app as app_module


def test_embed_cta_click_sends_slug():
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    body = c.get("/embed").get_data(as_text=True)
    assert "cta-click" in body
    # the click payload now carries a slug parsed from the product target
    assert "slug: slug" in body
    assert "begin\\/(?:buy|product)\\/" in body
