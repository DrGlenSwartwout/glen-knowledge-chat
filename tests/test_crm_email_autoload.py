from pathlib import Path


def test_crm_reads_email_param():
    html = (Path(__file__).resolve().parent.parent / "static" / "console-crm.html").read_text()
    assert "URLSearchParams(location.search).get('email')" in html
    assert "f.value = em" in html  # input pre-filled from the param
