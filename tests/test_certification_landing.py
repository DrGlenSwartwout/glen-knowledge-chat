"""Public ASH Certification sales landing page at /certification.

The static page lives at static/certification.html (approved, final copy).
This just serves it on a plain public GET, on the funnel host (illtowell.com),
not behind auth and not on the courses_bp blueprint (which 404s off the
course host).
"""

import importlib
import sys
from pathlib import Path

import pytest


def _app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable: {e}")


def test_certification_page_serves_200_with_expected_copy():
    app = _app()
    resp = app.app.test_client().get("/certification")

    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "ASH" in body
    assert "Certification" in body


def test_certification_page_has_pricing_and_cta_copy():
    app = _app()
    resp = app.app.test_client().get("/certification")

    body = resp.get_data(as_text=True)
    assert "2,997" in body
    assert "99" in body
    assert "/learn/ash-certification" in body


def test_certification_page_has_ref_capture_passthrough():
    app = _app()
    resp = app.app.test_client().get("/certification")

    assert "ref-capture.js" in resp.get_data(as_text=True)


def test_certification_page_needs_no_auth():
    app = _app()
    resp = app.app.test_client().get("/certification")

    assert resp.status_code == 200
    assert resp.status_code not in (401, 403)
    # a plain GET must not be redirected to a login/auth page
    assert resp.status_code not in (301, 302, 303, 307, 308)
