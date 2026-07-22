"""Route-level tests for the free product review. Skips if app import needs
secrets (CI). Dummy keys let the OpenAI/Pinecone/Anthropic clients construct.
Covers the in-portal request, the analyzer draft hand-off, the portal-view
visibility gate (text hidden until confirmed), and the dark-flag 404."""
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("PINECONE_API_KEY", "pc-dummy")
os.environ.setdefault("SUPPLEMENT_REVIEW_ENABLED", "on")

import pytest

try:
    import app
    import dashboard
    from dashboard import client_portal as _cp
    from dashboard import supplement_reviews as _sr
except Exception as e:  # pragma: no cover
    pytest.skip(f"app import needs secrets: {e}", allow_module_level=True)


def _setup(mp, tmp):
    dbp = str(tmp / "chat_log.db")
    mp.setenv("DATA_DIR", str(tmp))
    mp.setenv("SUPPLEMENT_REVIEW_ENABLED", "on")
    mp.setattr(app, "LOG_DB", dbp, raising=False)
    for obj in (app, dashboard):
        mp.setattr(obj, "CONSOLE_SECRET", "sek", raising=False)
    app._init_people_table()  # production inits this at startup; seed it for the test
    cx = sqlite3.connect(dbp)
    _cp.init_client_portal_table(cx)
    _sr.init_table(cx)
    tok = _cp.ensure_token(cx, "client@x.com", "Client")
    cx.commit()
    cx.close()
    return dbp, tok


def test_request_draft_and_visibility_gate(monkeypatch, tmp_path):
    dbp, tok = _setup(monkeypatch, tmp_path)
    c = app.app.test_client()

    # 1. in-portal request (identity from the token)
    r = c.post("/api/product-review/request",
               json={"token": tok, "product_name": "Fish Oil", "product_brand": "OmegaCo"})
    assert r.status_code == 200 and r.get_json()["status"] == "requested"

    cx = sqlite3.connect(dbp)
    rid = _sr.pending_queue(cx)[0]["id"]
    cx.close()

    # 2. analyzer hand-off -> ai_draft (console gated)
    r = c.post("/api/console/product-review/draft",
               headers={"X-Console-Key": "sek"},
               json={"id": rid, "review_text": "Contains soy lecithin."})
    assert r.status_code == 200 and r.get_json()["status"] == "ai_draft"

    # 3. portal view hides an unconfirmed review's text
    v = c.get(f"/api/portal/{tok}/view").get_json()["supplement_review"]
    assert v["status"] == "has_reviews"
    assert v["reviews"][0]["status"] == "ai_draft" and "review" not in v["reviews"][0]

    # 4. confirm (store transition) -> portal now reveals the text
    cx = sqlite3.connect(dbp)
    _sr.set_status(cx, rid, "confirmed")
    cx.close()
    v = c.get(f"/api/portal/{tok}/view").get_json()["supplement_review"]
    assert v["reviews"][0]["status"] == "confirmed"
    assert v["reviews"][0]["review"] == "Contains soy lecithin."


def test_request_404_when_flag_off(monkeypatch, tmp_path):
    dbp, tok = _setup(monkeypatch, tmp_path)
    monkeypatch.setenv("SUPPLEMENT_REVIEW_ENABLED", "")
    r = app.app.test_client().post("/api/product-review/request",
                                   json={"token": tok, "product_name": "X"})
    assert r.status_code == 404


def test_console_draft_requires_secret(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    r = app.app.test_client().post("/api/console/product-review/draft",
                                   json={"id": 1, "review_text": "x"})
    assert r.status_code == 401
