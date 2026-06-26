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
    assert b"coming" in r.data.lower() or b"on its way" in r.data.lower()


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
