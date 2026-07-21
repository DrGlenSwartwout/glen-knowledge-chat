import sqlite3
from dashboard import recommendation_events as re


def _cx():
    cx = sqlite3.connect(":memory:")
    re.init_recommendation_events(cx)
    return cx


def test_record_self_idempotent_per_product():
    cx = _cx()
    assert re.record_self(cx, "A@B.com", "neuro-magnesium") is True
    # re-add of the same product -> no new event (stable origin_ref)
    assert re.record_self(cx, "a@b.com", "neuro-magnesium") is False
    ev = re.list_events(cx, "a@b.com")
    assert len(ev) == 1
    assert ev[0]["source_key"] == "self" and ev[0]["origin_ref"] == "self"


def test_record_self_distinct_products_and_blank_guard():
    cx = _cx()
    re.record_self(cx, "a@b.com", "neuro-magnesium")
    re.record_self(cx, "a@b.com", "immune-modulation")
    assert len(re.list_events(cx, "a@b.com")) == 2
    assert re.record_self(cx, "a@b.com", "") is False        # blank slug -> no-op (record_event guard)
