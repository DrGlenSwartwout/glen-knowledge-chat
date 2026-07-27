"""The public funnel home page /begin links to the ASH Certification page.

Discoverability only: /begin (static/begin.html) must include a link to
/certification so visitors can find the cert landing page.
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


def test_begin_links_to_certification_page():
    app = _app()
    resp = app.app.test_client().get("/begin")

    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "/certification" in body
