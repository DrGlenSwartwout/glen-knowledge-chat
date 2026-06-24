"""The authoring editor page: header inputs + editable chain rows + add-row."""
from dashboard.biofield_report_html import render_author_html, render_list_html


def _report():
    return {
        "test_id": "a1", "client": {"name": "Jane Doe", "email": "jane@x.com"},
        "date": "2026-06-23",
        "layers": [
            {"layer": 1, "head": "Night", "most_affected": "Night", "remedy": "TMG",
             "dosage": "1 scoop", "frequency": "daily", "timing": "at night", "rid": 5},
        ],
        "schedule": {"slots": [], "entries": []},
    }


def test_author_page_has_header_rows_and_endpoints():
    html = render_author_html(_report())
    assert "Jane Doe" in html and "2026-06-23" in html       # header prefilled
    assert "TMG" in html and "Night" in html                 # existing row prefilled
    assert "/author/a1/header" in html                       # save header
    assert "/author/a1/row" in html                          # add/save rows
    assert "Add row" in html
    assert "/test/a1" in html                                # link to the read-only report


def test_author_page_escapes_free_text():
    rep = _report()
    rep["client"]["name"] = "<script>x</script>"
    html = render_author_html(rep)
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


def test_list_html_shows_authored_and_new_button():
    html = render_list_html(
        tests=[{"test_id": "10", "name": "Lewis", "email": "l@x.com",
                "date": "2026-06-11", "layer_count": 19}],
        authored=[{"test_id": "a1", "name": "Jane Doe", "email": "j@x.com",
                   "date": "2026-06-23", "layer_count": 1, "authored": True}])
    assert "Jane Doe" in html and "/author/a1" in html       # authored test -> editor
    assert "/author/new" in html                             # New test action
    assert "Lewis" in html                                   # FMP tests still listed
