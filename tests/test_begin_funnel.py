import sqlite3
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _mem():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    return cx


def test_init_creates_journey_tables():
    import begin_funnel as bf
    cx = _mem()
    bf.init_journey_tables(cx)
    names = {r[0] for r in cx.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "journey_state" in names
    assert "journey_events" in names
    state_cols = {r[1] for r in cx.execute("PRAGMA table_info(journey_state)")}
    for c in ("session_id", "email", "first_name", "ref_slug",
              "current_rung", "unlocked_gates", "awareness_stage", "path",
              "tos_agreed_at", "tos_version", "last_signal",
              "created_at", "updated_at"):
        assert c in state_cols, c
    ev_cols = {r[1] for r in cx.execute("PRAGMA table_info(journey_events)")}
    for c in ("ts", "session_id", "email", "trigger", "detail",
              "rung_before", "rung_after"):
        assert c in ev_cols, c


def test_compute_rung_ladder():
    import begin_funnel as bf
    assert bf.compute_rung(set(), "", False) == "arrival"
    assert bf.compute_rung({"video"}, "", False) == "listening"
    assert bf.compute_rung({"scroll"}, "", False) == "listening"
    assert bf.compute_rung({"video", "question"}, "", False) == "inquire"
    assert bf.compute_rung({"question", "name"}, "", False) == "personalize"
    # email WITHOUT tos does not grant free_tier
    assert bf.compute_rung({"name"}, "a@b.com", False) == "personalize"
    # email AND tos grants free_tier
    assert bf.compute_rung({"name"}, "a@b.com", True) == "free_tier"
    # later rungs (forward-compatible spine; rooms built in later slices)
    assert bf.compute_rung({"voice"}, "a@b.com", True) == "explore_voice"
    assert bf.compute_rung({"scan"}, "a@b.com", True) == "assess"
    assert bf.compute_rung({"paid_fork"}, "a@b.com", True) == "choose_path"
    assert bf.compute_rung({"purchase"}, "a@b.com", True) == "ascend"
    assert bf.compute_rung({"share_video"}, "a@b.com", True) == "advocate"


def test_valid_triggers_set():
    import begin_funnel as bf
    for t in ("load", "video", "scroll", "question", "name", "email", "tos",
              "voice", "scan", "quiz", "paid_fork", "purchase", "share_video"):
        assert t in bf.VALID_TRIGGERS
