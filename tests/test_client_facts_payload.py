import importlib, sqlite3, sys
from pathlib import Path
import pytest
from dashboard import condition_programs

def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try: return importlib.import_module("app")
    except Exception as e: pytest.skip(f"app not importable: {e}")

@pytest.fixture
def wired(monkeypatch, tmp_path):
    app = _app()
    db = tmp_path / "chat_log.db"
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        condition_programs.init_table(cx)
        condition_programs.upsert(cx, "dry-amd", "Dry AMD", False,
            items=[{"slug": "macular-wellness-lutein", "name": "Macular Wellness Lutein"},
                   {"slug": "wholomega", "name": "WholOmega"}],
            modifiers=[{"when": "on_areds2", "action": "remove", "source": "client-reported",
                        "client_default": False,
                        "items": [{"slug": "macular-wellness-lutein", "name": "Macular Wellness Lutein"}]}])
        condition_programs.upsert(cx, "dry-eye", "Dry Eye", False,
            items=[{"slug": "moisturize", "name": "Moisturize"}], modifiers=[])
    monkeypatch.setattr(app, "LOG_DB", db)
    monkeypatch.setattr(app, "_support_programs_enabled", lambda: True)
    return app

def test_amd_program_offers_on_areds2_fact(wired, monkeypatch):
    monkeypatch.setattr(wired, "_client_condition_for", lambda e: "dry-amd")
    sp = wired._support_program_for("pat@x.com")
    cf = sp.get("client_facts")
    assert cf and cf[0]["key"] == "on_areds2"
    assert cf[0]["value"] is False and cf[0]["label"] and cf[0]["hint"]

def test_non_amd_program_omits_client_facts(wired, monkeypatch):
    monkeypatch.setattr(wired, "_client_condition_for", lambda e: "dry-eye")
    sp = wired._support_program_for("pat@x.com")
    assert "client_facts" not in sp
