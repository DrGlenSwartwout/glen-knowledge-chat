import sqlite3, time
from dashboard import recommendation_events as re


def _cx():
    cx = sqlite3.connect(":memory:")
    re.init_recommendation_events(cx)
    return cx


def test_each_click_counts():
    cx = _cx()
    assert re.record_click(cx, "A@B.com", "neuro-magnesium", "biofield") is True
    time.sleep(0.001)   # ensure a distinct microsecond timestamp
    assert re.record_click(cx, "a@b.com", "neuro-magnesium", "biofield") is True
    ev = [e for e in re.list_events(cx, "a@b.com") if e["source_key"] == "biofield"]
    assert len(ev) == 2          # two clicks -> two events (unlike sticky self)


def test_click_blank_guard_and_source():
    cx = _cx()
    assert re.record_click(cx, "a@b.com", "", "scan") is False   # blank slug -> no-op
    re.record_click(cx, "a@b.com", "immune-modulation", "scan")
    ev = re.list_events(cx, "a@b.com")
    assert ev[-1]["source_key"] == "scan"
