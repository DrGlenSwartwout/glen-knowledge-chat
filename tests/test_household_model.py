"""Unit tests for household slug generation + candidate dedup helpers.

Pure-function tests — no DB, no network. The slug function generates
URL-safe identifiers; the dedup key sorts person_ids for stable matching
across detection runs.
"""

import importlib
import sys
from pathlib import Path


def _app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return importlib.import_module("app")


def test_household_slug_basic():
    app = _app()
    assert app._household_slug("Savant") == "savant"
    assert app._household_slug("O'Connor") == "o-connor"
    assert app._household_slug("Smith Jones") == "smith-jones"


def test_household_slug_appends_head_firstname_on_collision():
    app = _app()
    existing = {"savant"}
    assert app._household_slug("Savant", "Lotika", existing=existing) == "savant-lotika"
    assert app._household_slug("Savant", "Omika", existing=existing) == "savant-omika"


def test_household_slug_collision_without_firstname_falls_back_to_numeric():
    app = _app()
    existing = {"savant", "savant-2"}
    assert app._household_slug("Savant", existing=existing) == "savant-3"


def test_candidate_dedup_key_sorts_person_ids():
    app = _app()
    assert app._candidate_dedup_key([3, 1, 2]) == "1,2,3"
    assert app._candidate_dedup_key([5]) == "5"
    assert app._candidate_dedup_key([]) == ""
