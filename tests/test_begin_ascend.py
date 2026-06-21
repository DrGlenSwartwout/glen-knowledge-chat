# tests/test_begin_ascend.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load_bf():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("begin_funnel")
    except Exception as e:
        pytest.skip(f"begin_funnel not importable: {e}")


def test_recommend_heal_entry():
    bf = _load_bf()
    assert bf.recommend_ascend("heal") == "biofield-analysis"


def test_recommend_learn_entry():
    bf = _load_bf()
    assert bf.recommend_ascend("learn") == "certification"


def test_recommend_learn_certified_bumps():
    bf = _load_bf()
    assert bf.recommend_ascend("learn", reached={"certification"}) == "one-to-one"


def test_recommend_build_entry():
    bf = _load_bf()
    assert bf.recommend_ascend("build") == "one-to-one"


def test_recommend_build_practitioner_bumps():
    bf = _load_bf()
    assert bf.recommend_ascend("build", reached={"one-to-one"}) == "healing-oasis-tools"


def test_recommend_unknown_goal_falls_back_to_heal():
    bf = _load_bf()
    assert bf.recommend_ascend("nonsense") == "biofield-analysis"
    assert bf.recommend_ascend("") == "biofield-analysis"
    assert bf.recommend_ascend(None) == "biofield-analysis"


def test_recommend_all_reached_returns_track_top():
    bf = _load_bf()
    allslugs = set(bf.TIER_CATALOG.keys())
    assert bf.recommend_ascend("build", reached=allslugs) == "consultant-package"


def test_recommend_returns_valid_catalog_slug():
    bf = _load_bf()
    for goal in ("heal", "learn", "build", "x"):
        assert bf.recommend_ascend(goal) in bf.TIER_CATALOG


def test_ascend_is_valid_trigger():
    bf = _load_bf()
    assert "ascend" in bf.VALID_TRIGGERS
