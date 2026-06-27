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
