import sqlite3
from dashboard import ff_match_drafts as d

def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    d.init_table(cx)
    return cx

def test_get_or_create_is_generate_once():
    cx = _cx()
    calls = []
    def make():
        calls.append(1)
        return [{"slug": "x", "name": "X"}]
    r1 = d.get_or_create(cx, "a@b.com", "2026-07-01", make)
    r2 = d.get_or_create(cx, "a@b.com", "2026-07-01", make)  # must NOT regenerate
    assert calls == [1]
    assert r1["items"] == r2["items"] == [{"slug": "x", "name": "X"}]
    assert r1["status"] == "draft"

def test_publish_and_status_filter():
    cx = _cx()
    d.get_or_create(cx, "a@b.com", "2026-07-01", lambda: [{"slug": "x"}])
    assert d.publish(cx, "a@b.com", "2026-07-01") is True
    assert d.get(cx, "a@b.com", "2026-07-01")["status"] == "published"
    assert d.get(cx, "a@b.com", "2026-07-01")["published_at"]
    assert [r["email"] for r in d.list_by_status(cx, "published")] == ["a@b.com"]
    assert d.list_by_status(cx, "draft") == []

def test_publish_missing_row_returns_false():
    assert d.publish(_cx(), "no@b.com", "2026-07-01") is False
