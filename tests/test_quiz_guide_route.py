import importlib, sys
from pathlib import Path
import pytest


def _load_app():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def test_guide_no_pdf_key_degrades_gracefully(monkeypatch):
    app_module = _load_app()
    monkeypatch.delenv("LEAD_MAGNET_PDF_KEY", raising=False)
    monkeypatch.setattr(app_module, "_validate_lead_magnet_guide_link", lambda t: "a@b.com")
    c = app_module.app.test_client()
    r = c.get("/begin/quiz/guide?token=valid")
    assert r.status_code == 200
    assert b"almost ready" in r.data.lower()


def test_pending_page_does_not_promise_an_email_nobody_sends(monkeypatch):
    """LEAD_MAGNET_PDF_KEY is read in exactly one place (the guide route) and no
    sender or cron exists, so "we'll email it to you" was a promise the system
    could never keep. Don't reintroduce it without also building the delivery."""
    app_module = _load_app()
    monkeypatch.delenv("LEAD_MAGNET_PDF_KEY", raising=False)
    monkeypatch.setattr(app_module, "_validate_lead_magnet_guide_link", lambda t: "a@b.com")
    body = app_module.app.test_client().get("/begin/quiz/guide?token=valid").data.lower()
    assert b"email it to you" not in body
    assert b"we'll email" not in body and b"we will email" not in body


def test_pending_page_quotes_the_real_token_lifetime(monkeypatch):
    """Copy derived from the constant, so it cannot drift the way the "expires in
    15 minutes" pages did while their tokens lived 24 hours."""
    app_module = _load_app()
    monkeypatch.delenv("LEAD_MAGNET_PDF_KEY", raising=False)
    monkeypatch.setattr(app_module, "_validate_lead_magnet_guide_link", lambda t: "a@b.com")
    body = app_module.app.test_client().get("/begin/quiz/guide?token=valid").data.decode()
    assert f"{app_module.LEAD_MAGNET_GUIDE_TTL_DAYS} days" in body


def test_guide_invalid_token_friendly(monkeypatch):
    app_module = _load_app()
    monkeypatch.setenv("LEAD_MAGNET_PDF_KEY", "guides/eye-brain.pdf")
    monkeypatch.setattr(app_module, "_validate_lead_magnet_guide_link", lambda t: None)
    c = app_module.app.test_client()
    r = c.get("/begin/quiz/guide?token=bogus")
    assert r.status_code == 200
    assert b"expired" in r.data.lower() or b"fresh" in r.data.lower()


def test_guide_valid_streams_pdf(monkeypatch):
    app_module = _load_app()
    monkeypatch.setenv("LEAD_MAGNET_PDF_KEY", "guides/eye-brain.pdf")
    monkeypatch.setenv("R2_BUCKET", "rm-clips")
    monkeypatch.setattr(app_module, "_validate_lead_magnet_guide_link", lambda t: "a@b.com")

    class _Body:
        def iter_chunks(self, chunk_size=65536):
            yield b"%PDF-1.4 fake"

    class _R2:
        def get_object(self, **kw):
            assert kw["Key"] == "guides/eye-brain.pdf"
            return {"Body": _Body(), "ContentType": "application/pdf", "ContentLength": 13}

    monkeypatch.setattr(app_module, "_r2", lambda: _R2())
    c = app_module.app.test_client()
    r = c.get("/begin/quiz/guide?token=valid")
    assert r.status_code == 200
    assert r.headers["Content-Type"] == "application/pdf"
    assert b"%PDF" in r.data
