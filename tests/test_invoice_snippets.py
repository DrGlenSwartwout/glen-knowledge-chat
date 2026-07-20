import sqlite3
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _cx():
    from dashboard import invoice_snippets as S
    cx = sqlite3.connect(":memory:")
    S.init_table(cx)
    return S, cx


def test_add_lists_and_dedups_on_exact_text():
    S, cx = _cx()
    id1 = S.add(cx, "Take with food.")
    id2 = S.add(cx, "Follow-up scan in 30 days.")
    assert id1 and id2 and id1 != id2
    # Same text again is NOT a second row — it upserts (auto-save must not pile up).
    id1b = S.add(cx, "Take with food.")
    assert id1b == id1
    texts = [r["text"] for r in S.list_all(cx)]
    assert sorted(texts) == ["Follow-up scan in 30 days.", "Take with food."]
    assert len(S.list_all(cx)) == 2


def test_add_strips_and_ignores_blank():
    S, cx = _cx()
    assert S.add(cx, "   ") is None
    assert S.add(cx, "") is None
    assert S.add(cx, None) is None
    assert S.list_all(cx) == []
    # Leading/trailing whitespace is trimmed, so " x " and "x" are the same snippet.
    a = S.add(cx, "  Shake well.  ")
    b = S.add(cx, "Shake well.")
    assert a == b
    assert len(S.list_all(cx)) == 1


def test_reuse_floats_to_top():
    S, cx = _cx()
    S.add(cx, "first")
    S.add(cx, "second")
    # Re-using "first" touches last_used_at, so it sorts ahead of "second".
    S.add(cx, "first")
    assert [r["text"] for r in S.list_all(cx)][0] == "first"


def test_remove():
    S, cx = _cx()
    sid = S.add(cx, "prune me")
    S.add(cx, "keep me")
    assert S.remove(cx, sid) is True
    assert [r["text"] for r in S.list_all(cx)] == ["keep me"]
    # Removing a gone/garbage id is a no-op, never raises.
    assert S.remove(cx, sid) is False
    assert S.remove(cx, "not-an-int") is False
    assert S.remove(cx, None) is False
