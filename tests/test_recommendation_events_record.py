import sqlite3
from dashboard import recommendation_events as re


def _cx():
    cx = sqlite3.connect(":memory:")
    re.init_recommendation_events(cx)
    return cx


def test_record_inserts_then_dedups():
    cx = _cx()
    assert re.record_event(cx, "A@B.com", "neuro-magnesium", "biofield",
                           occurred_at="2026-07-01", origin_ref="2026-07-01") is True
    # same (email, product, source, origin_ref) -> ignored
    assert re.record_event(cx, "a@b.com", "neuro-magnesium", "biofield",
                           occurred_at="2026-07-01", origin_ref="2026-07-01") is False
    # different origin_ref -> new event
    assert re.record_event(cx, "a@b.com", "neuro-magnesium", "biofield",
                           occurred_at="2026-07-08", origin_ref="2026-07-08") is True
    rows = re.list_events(cx, "a@b.com")
    assert len(rows) == 2
    assert rows[0]["product_key"] == "neuro-magnesium"


def test_record_rejects_empty_email_or_slug():
    cx = _cx()
    assert re.record_event(cx, "", "x", "biofield", occurred_at="d", origin_ref="r") is False
    assert re.record_event(cx, "a@b.com", "", "biofield", occurred_at="d", origin_ref="r") is False
    assert re.list_events(cx, "a@b.com") == []


def test_email_lowercased():
    cx = _cx()
    re.record_event(cx, "MixedCase@X.com", "slug", "purchased", occurred_at="d", origin_ref="7")
    assert len(re.list_events(cx, "mixedcase@x.com")) == 1
