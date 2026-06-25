import importlib, sqlite3, pytest
from datetime import datetime, timezone

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    monkeypatch.setenv("BIOFIELD_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setenv("BIOFIELD_REPORTS_DIR", str(tmp_path / "reports"))
    import biofield_local_app as bla
    importlib.reload(bla)
    app = bla.create_app(db_path=str(tmp_path / "chat_log.db"))
    # seed an authored test with one chain row + narrative
    from dashboard.biofield_authoring import create_test, add_chain_row
    from dashboard.biofield_narrative import save_narrative
    with sqlite3.connect(tmp_path / "chat_log.db") as cx:
        tid = create_test(cx, "Kauilani", "k@x.com", "2026-06-24")
        add_chain_row(cx, tid, 1, "ET4", "ET4", "MSM Lotion", "1 app", "daily", "am")
        save_narrative(cx, tid, "You are healing.")
    return app.test_client(), tid

def test_report_view_is_clean_html(client):
    c, tid = client
    r = c.get(f"/test/{tid}/report")
    assert r.status_code == 200 and r.mimetype == "text/html"
    body = r.get_data(as_text=True)
    assert "Accelerated Self Healing™" in body and "MSM Lotion" in body
    assert "<button" not in body.lower()

def test_report_pdf_downloads_and_saves(client, tmp_path):
    pytest.importorskip("playwright")
    c, tid = client
    r = c.get(f"/test/{tid}/report.pdf")
    assert r.status_code == 200 and r.mimetype == "application/pdf"
    assert r.get_data()[:5] == b"%PDF-"
    saved = list((tmp_path / "reports").glob("report_*.pdf"))
    assert saved, "PDF should also be saved locally for printing/shipping"
