import sqlite3
from dashboard import portal_biofield_reports as pbr

def _cx():
    cx = sqlite3.connect(":memory:")
    pbr.init_table(cx)
    return cx

def test_returns_url_for_confirmed_report_with_pdf():
    cx = _cx()
    pbr.upsert_report(cx, "k@example.com", "2026-06-25", "",
                      {"report_pdf": {"url": "https://h/portal-asset/r.pdf"}}, "confirmed")
    assert pbr.report_pdf_urls(cx, ["k@example.com"]) == {"k@example.com": "https://h/portal-asset/r.pdf"}

def test_picks_latest_confirmed_when_multiple():
    cx = _cx()
    pbr.upsert_report(cx, "k@example.com", "2026-06-20", "",
                      {"report_pdf": {"url": "https://h/old.pdf"}}, "confirmed")
    pbr.upsert_report(cx, "k@example.com", "2026-06-25", "",
                      {"report_pdf": {"url": "https://h/new.pdf"}}, "confirmed")
    assert pbr.report_pdf_urls(cx, ["k@example.com"]) == {"k@example.com": "https://h/new.pdf"}

def test_omits_non_confirmed_and_missing_pdf():
    cx = _cx()
    pbr.upsert_report(cx, "draft@example.com", "2026-06-25", "",
                      {"report_pdf": {"url": "https://h/d.pdf"}}, "ai_draft")
    pbr.upsert_report(cx, "nopdf@example.com", "2026-06-25", "",
                      {"greeting": "hi"}, "confirmed")
    assert pbr.report_pdf_urls(cx, ["draft@example.com", "nopdf@example.com"]) == {}

def test_lowercases_and_empty_input():
    cx = _cx()
    pbr.upsert_report(cx, "k@example.com", "2026-06-25", "",
                      {"report_pdf": {"url": "https://h/r.pdf"}}, "confirmed")
    assert pbr.report_pdf_urls(cx, ["K@Example.COM"]) == {"k@example.com": "https://h/r.pdf"}
    assert pbr.report_pdf_urls(cx, []) == {}
