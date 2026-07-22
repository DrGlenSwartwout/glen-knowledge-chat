import sqlite3

from dashboard import health_suggestions as hs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    hs.init_table(cx)
    return cx


def test_add_pending_dedupes_identical_pending_rows():
    cx = _cx()
    hs.add_pending(cx, "a@b.com", "terrain", "3", "mentioned fatigue", source="chat")
    hs.add_pending(cx, "a@b.com", "terrain", "3", "mentioned fatigue again", source="chat")
    assert hs.count_pending(cx, "a@b.com") == 1


def test_resolve_dismissed_clears_pending_count():
    cx = _cx()
    sug_id = hs.add_pending(cx, "a@b.com", "terrain", "3", "mentioned fatigue", source="chat")
    assert hs.resolve(cx, sug_id, "a@b.com", "dismissed") is True
    assert hs.count_pending(cx, "a@b.com") == 0


def test_resolve_scoped_to_email_cannot_resolve_others_row():
    cx = _cx()
    sug_id = hs.add_pending(cx, "a@b.com", "terrain", "3", "mentioned fatigue", source="chat")
    assert hs.resolve(cx, sug_id, "other@b.com", "dismissed") is False
    assert hs.count_pending(cx, "a@b.com") == 1


def test_add_pending_allows_resuggestion_after_prior_resolved():
    cx = _cx()
    sug_id = hs.add_pending(cx, "a@b.com", "terrain", "3", "first mention", source="chat")
    hs.resolve(cx, sug_id, "a@b.com", "confirmed")
    new_id = hs.add_pending(cx, "a@b.com", "terrain", "3", "mentioned again later", source="chat")
    assert new_id is not None
    assert hs.count_pending(cx, "a@b.com") == 1


def test_extract_from_turn_filters_to_editable_fields():
    result = hs.extract_from_turn(
        "client message", "assistant message",
        extractor=lambda c, a: [
            {"field_id": "terrain", "value": 3, "rationale": "x"},
            {"field_id": "first_name", "value": "z", "rationale": "y"},
        ])
    assert result == [{"field_id": "terrain", "value": 3, "rationale": "x"}]


def test_extract_from_turn_defaults_to_empty_seam():
    assert hs.extract_from_turn("client message", "assistant message") == []
