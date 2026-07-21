import app as app_module


def test_invoice_line_view_carries_source():
    out = app_module._invoice_line_view(
        {"slug": "neuro-magnesium", "name": "Neuro Magnesium", "qty": 1,
         "unit_cents": 7000, "line_cents": 7000, "source": "biofield"})
    assert out["source"] == "biofield"


def test_invoice_line_view_omits_source_when_absent():
    out = app_module._invoice_line_view(
        {"slug": "neuro-magnesium", "name": "Neuro Magnesium", "qty": 1,
         "unit_cents": 7000, "line_cents": 7000})
    assert "source" not in out
