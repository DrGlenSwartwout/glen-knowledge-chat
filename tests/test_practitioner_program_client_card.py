"""Task 5: `_practitioner_program_card(email)` -- the client-facing card for a
practitioner-composed program (distinct from the reco-card and the
support-program card). Best-effort: None on any error, flag off, or no saved
program. When a program is saved, items are run through
`_support_program_item_view` (so each carries `url`), and the label comes
from the saved program's condition_key -> condition_programs.get(...).label,
falling back to "Your Practitioner's Program"."""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import condition_programs
from dashboard import practitioner_programs as pp

PID = "doc1"
EMAIL = "pat@x.com"
CONDITION_KEY = "dry-amd"


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture
def wired(monkeypatch, tmp_path):
    app_module = _app()
    db_path = tmp_path / "chat_log.db"
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        condition_programs.init_table(cx)
        pp.init_table(cx)
        condition_programs.upsert(
            cx, CONDITION_KEY, "Dry AMD Support", False,
            items=[{"slug": "wholomega", "name": "WholOmega"}])

    monkeypatch.setattr(app_module, "LOG_DB", db_path)
    monkeypatch.setenv("PROGRAM_COMPOSER_ENABLED", "1")
    return app_module, db_path


def test_flag_off_returns_none(monkeypatch, wired):
    app_module, db_path = wired
    monkeypatch.delenv("PROGRAM_COMPOSER_ENABLED", raising=False)
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        pp.upsert(cx, patient_email=EMAIL, practitioner_id=PID, condition_key=CONDITION_KEY,
                  items=[{"slug": "wholomega", "name": "WholOmega"}], note="start low")
    assert app_module._practitioner_program_card(EMAIL) is None


def test_no_saved_program_returns_none(wired):
    app_module, _db = wired
    assert app_module._practitioner_program_card(EMAIL) is None


def test_saved_program_returns_card_with_items_url_label_note(wired):
    app_module, db_path = wired
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        pp.upsert(cx, patient_email=EMAIL, practitioner_id=PID, condition_key=CONDITION_KEY,
                  items=[{"slug": "wholomega", "name": "WholOmega", "dose": "1/day",
                          "alts": [{"slug": "omega-alt", "name": "Omega Alt"}]}],
                  note="start low")

    card = app_module._practitioner_program_card(EMAIL)
    assert card is not None
    assert card["label"] == "Dry AMD Support"
    assert card["note"] == "start low"
    assert card["items"] == [app_module._support_program_item_view(
        {"slug": "wholomega", "name": "WholOmega", "dose": "1/day",
         "alts": [{"slug": "omega-alt", "name": "Omega Alt"}]})]
    assert card["items"][0]["url"]


def test_saved_program_unknown_condition_key_falls_back_to_default_label(wired):
    app_module, db_path = wired
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        pp.upsert(cx, patient_email=EMAIL, practitioner_id=PID, condition_key="no-such-key",
                  items=[{"slug": "wholomega", "name": "WholOmega"}], note="")

    card = app_module._practitioner_program_card(EMAIL)
    assert card is not None
    assert card["label"] == "Your Practitioner's Program"


def test_practitioner_program_card_swallows_errors(monkeypatch, wired):
    """Verify the best-effort guard: if practitioner_programs.get raises,
    _practitioner_program_card returns None instead of propagating."""
    app_module, _db = wired
    # Monkeypatch the store's get to raise an error
    def boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(pp, "get", boom)
    # The function should catch the error and return None
    assert app_module._practitioner_program_card("someone@example.com") is None
