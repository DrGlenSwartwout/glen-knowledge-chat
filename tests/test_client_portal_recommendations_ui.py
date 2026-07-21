import app as app_module


def test_portal_page_has_recommendations_section_and_fetch():
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    body = c.get("/portal/anytoken").get_data(as_text=True)  # page is static; token resolved client-side
    assert "/recommendations" in body                 # the page fetches the endpoint
    assert "My Recommendations" in body                # the section heading
    assert "renderRecommendations" in body             # the render function exists


def test_portal_page_wires_recommendation_writes():
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    body = c.get("/portal/anytoken").get_data(as_text=True)
    assert "/recommendation/hide" in body
    assert "/recommendation/client-note" in body
    assert "/recommendation/section" in body
