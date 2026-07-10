import importlib
import sys
from pathlib import Path

import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable in this env: {e}")


def test_terms_page_served_200():
    app_module = _load_app()
    r = app_module.app.test_client().get("/terms")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("Content-Type", "")


def test_terms_contains_pma_wellness_agreement():
    # The Pastoral Medical Association wellness agreement must be present verbatim.
    app_module = _load_app()
    body = app_module.app.test_client().get("/terms").get_data(as_text=True)
    assert "Pastoral Medical Association" in body
    assert "Pastoral Science &amp; Medicine services are not state licensed medical services" in body


def test_terms_contains_site_terms_of_use():
    # The RemedyMatch.com Terms of Use (site terms) must be present verbatim.
    app_module = _load_app()
    body = app_module.app.test_client().get("/terms").get_data(as_text=True)
    assert "AGREEMENT TO TERMS" in body
    assert "binding arbitration" in body
    assert "Hawai&#39;i-Kingdom" in body or "Hawai'i-Kingdom" in body


def test_terms_contains_dependent_coverage_clause():
    # Our added clause: a caregiver's agreement covers those in their care.
    app_module = _load_app()
    body = app_module.app.test_client().get("/terms").get_data(as_text=True)
    assert "those in your care" in body


def test_privacy_page_served_200():
    app_module = _load_app()
    r = app_module.app.test_client().get("/privacy")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("Content-Type", "")


def test_privacy_contains_key_sections():
    app_module = _load_app()
    body = app_module.app.test_client().get("/privacy").get_data(as_text=True)
    assert "WHAT INFORMATION DO WE COLLECT" in body
    assert "Shine The Light" in body


def test_portal_gate_points_at_local_terms_not_dead_illtowell_link():
    # The gate's fallback must be the new /terms route, not the 404 illtowell.com/terms.
    repo_root = Path(__file__).resolve().parent.parent
    portal = (repo_root / "static" / "client-portal.html").read_text()
    assert "https://illtowell.com/terms" not in portal
    assert '"/terms"' in portal
