from dashboard.biofield_report_html import render_report_html

REPORT = {"test_id": "a2", "client": {"name": "K", "email": ""}, "date": "2026-06-24",
          "layers": [], "schedule": {"slots": [], "entries": []}}

def test_has_print_and_pdf_links():
    html = render_report_html(REPORT, "", "", "")
    assert "/test/a2/report" in html        # clean view
    assert "/test/a2/report.pdf" in html    # printable PDF
