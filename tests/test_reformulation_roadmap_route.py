"""Route tests for the reformulation-roadmap console endpoints. Skips if app import
needs secrets. The generate route on an empty corpus must not call the LLM."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("PINECONE_API_KEY", "pc-dummy")

import pytest

try:
    import app
    import dashboard
except Exception as e:  # pragma: no cover
    pytest.skip(f"app import needs secrets: {e}", allow_module_level=True)


def _setup(mp, tmp):
    dbp = str(tmp / "chat_log.db")
    mp.setenv("DATA_DIR", str(tmp))
    mp.setattr(app, "LOG_DB", dbp, raising=False)
    for o in (app, dashboard):
        mp.setattr(o, "CONSOLE_SECRET", "sek", raising=False)
    return dbp


def test_roadmap_get_is_gated_and_shaped(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    c = app.app.test_client()
    assert c.get("/api/console/reformulation-roadmap").status_code == 401
    j = c.get("/api/console/reformulation-roadmap", headers={"X-Console-Key": "sek"}).get_json()
    assert j["ok"] and j["latest"]["roadmap"] == [] and j["frequency"] == []


def test_generate_empty_corpus_route_is_safe(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    # no reviewed submissions -> generate returns empty without touching the LLM
    j = app.app.test_client().post("/api/console/reformulation-roadmap/generate",
                                   headers={"X-Console-Key": "sek"}).get_json()
    assert j["ok"] and j["n_reviews"] == 0 and j["roadmap"] == []
