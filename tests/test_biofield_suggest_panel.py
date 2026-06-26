from dashboard.biofield_report_html import render_suggest_panel, render_author_html


def test_panel_renders_picks_and_count():
    data = {"picks": [{"remedy": "Neuro Magnesium", "covers": ["Membrane", "Mitochondria"]}],
            "uncovered": ["Lymph"]}
    h = render_suggest_panel(data)
    assert "Neuro Magnesium" in h and "Membrane, Mitochondria" in h
    assert ">2<" in h or "(2)" in h           # coverage count shown
    assert "Lymph" in h                        # uncovered listed


def test_panel_empty_state():
    h = render_suggest_panel({"picks": [], "uncovered": []})
    assert "No active required stresses" in h


def test_author_page_has_button_and_handler():
    rep = {"test_id": "a7", "client": {"name": "J", "email": "j@x.com"}, "date": "",
           "layers": [], "schedule": []}
    h = render_author_html(rep, [], "")
    assert "Suggest minimal remedies" in h
    assert "function suggestRemedies" in h
    assert "/author/a7/suggest-remedies" in h
    assert "id=suggestpanel" in h
