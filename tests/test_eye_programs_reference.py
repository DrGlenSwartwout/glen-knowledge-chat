import importlib, sys, sqlite3
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


def test_view_builder_shows_all_modifiers_and_urls():
    app = _load_app()
    prog = {"condition_key": "dry-amd", "label": "Dry AMD", "consult_recommended": False,
            "items": [{"slug": "wholomega", "name": "WholOmega", "dose": "4/day",
                       "alts": [{"slug": "brain-cleanse", "name": "Brain Cleanse"}]}],
            "modifiers": [{"when": "drusen", "action": "add", "source": "diagnosis-implied",
                           "client_default": True,
                           "items": [{"slug": "lipid-cleanse", "name": "Lipid Cleanse"}]}]}
    v = app._eye_program_public_view(prog)
    assert v["condition_key"] == "dry-amd" and v["label"] == "Dry AMD"
    assert v["consult_recommended"] is False
    assert v["items"][0]["name"] == "WholOmega" and v["items"][0]["dose"] == "4/day"
    assert v["items"][0]["url"]                              # store URL present
    assert v["items"][0]["alts"][0]["name"] == "Brain Cleanse"
    assert len(v["modifiers"]) == 1                          # modifier shown (not resolved away)
    m = v["modifiers"][0]
    assert m["when"] == "drusen" and m["source"] == "diagnosis-implied" and m["action"] == "add"
    assert m["items"][0]["name"] == "Lipid Cleanse" and m["items"][0]["url"]


def _client(app, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "LOG_DB", str(tmp_path / "chat_log.db"))
    from dashboard import condition_programs as CP
    with sqlite3.connect(str(tmp_path / "chat_log.db")) as cx:
        CP.init_table(cx)
        CP.upsert(cx, "wet-amd", "Wet AMD", True,
                  [{"slug": "wholomega", "name": "WholOmega"}], [])
        CP.upsert(cx, "dry-amd", "Dry AMD", False,
                  [{"slug": "wholomega", "name": "WholOmega"}],
                  [{"when": "drusen", "action": "add", "source": "diagnosis-implied",
                    "client_default": True, "items": [{"slug": "lipid-cleanse", "name": "Lipid Cleanse"}]}])
        cx.commit()
    app.app.config["TESTING"] = True
    return app.app.test_client()


def test_api_eye_programs_public_clinical_order_no_auth(monkeypatch, tmp_path):
    app = _load_app()
    c = _client(app, monkeypatch, tmp_path)
    r = c.get("/api/eye-programs")                           # no key/token -> still 200 (public)
    assert r.status_code == 200
    keys = [p["condition_key"] for p in r.get_json()["programs"]]
    # dry-amd precedes wet-amd in _CLINICAL_ORDER regardless of insert order
    assert keys.index("dry-amd") < keys.index("wet-amd")
    wet = next(p for p in r.get_json()["programs"] if p["condition_key"] == "wet-amd")
    assert wet["consult_recommended"] is True
