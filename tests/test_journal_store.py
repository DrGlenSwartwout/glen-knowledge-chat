"""Journal entries re-homed from a dead Supabase project to the app's local
sqlite (LOG_DB). These pin the store contract the routes depend on: insert
returns the new id, range-selects honor since/order/limit, and JSON columns
round-trip back to dicts/lists so metadata-based filtering still works."""
import sqlite3
from dashboard import journal_store as js


def _cx():
    cx = sqlite3.connect(":memory:")
    js.init_table(cx)
    return cx


def _rec(recorded_at, *, test=False, entry_type="journal", element="Fire"):
    return {
        "user_id": "glen",
        "recorded_at": recorded_at,
        "duration_seconds": 9.0,
        "transcript": "hello",
        "emotion_scores": {"Calmness": 0.7},
        "tcm_scores": {"elements": {"Fire": 60}},
        "dominant_element": element,
        "dominant_treasure": "Shen",
        "top_emotions": [{"name": "Calmness", "score": 0.7}],
        "polyvagal_state": {"ventral_vagal": 60},
        "congruence": {"score": 0.6, "self_contradictions": ['"a" vs. "b"']},
        "lexical_metrics": {"wpm": 120},
        "top_themes": ["hope"],
        "transcript_embedding": [0.1, 0.2, 0.3],
        "mapper_check": None,
        "metadata": {"test": test, "entry_type": entry_type},
    }


def test_insert_returns_id_and_roundtrips_json():
    cx = _cx()
    out = js.insert(cx, _rec("2026-06-25T10:00:00+00:00"))
    assert isinstance(out, list) and out and out[0].get("id")
    rows = js.select(cx, since_iso="2026-06-01T00:00:00+00:00", order="desc")
    assert len(rows) == 1
    r = rows[0]
    # JSON columns come back as parsed structures (not strings)
    assert r["metadata"]["entry_type"] == "journal"
    assert r["top_emotions"][0]["name"] == "Calmness"
    assert r["congruence"]["self_contradictions"] == ['"a" vs. "b"']  # quote-safe


def test_select_since_order_and_limit():
    cx = _cx()
    js.insert(cx, _rec("2026-06-20T08:00:00+00:00", element="Wood"))
    js.insert(cx, _rec("2026-06-24T08:00:00+00:00", element="Metal"))
    js.insert(cx, _rec("2026-05-01T08:00:00+00:00", element="Water"))  # before cutoff
    # since 2026-06-10, desc, limit 1 -> the most recent in-window
    desc = js.select(cx, since_iso="2026-06-10T00:00:00+00:00", order="desc", limit=1)
    assert len(desc) == 1 and desc[0]["dominant_element"] == "Metal"
    # asc, no limit -> two in-window in ascending order
    asc = js.select(cx, since_iso="2026-06-10T00:00:00+00:00", order="asc")
    assert [r["dominant_element"] for r in asc] == ["Wood", "Metal"]


def test_metadata_filtering_inputs_present():
    # The routes filter on metadata.test / metadata.entry_type in Python; make sure
    # those fields survive the round-trip for both flavors.
    cx = _cx()
    js.insert(cx, _rec("2026-06-25T10:00:00+00:00", test=True))
    js.insert(cx, _rec("2026-06-25T11:00:00+00:00", entry_type="affirmation_reading"))
    rows = js.select(cx, since_iso="2026-06-01T00:00:00+00:00", order="asc")
    assert (rows[0]["metadata"] or {}).get("test") is True
    assert (rows[1]["metadata"] or {}).get("entry_type") == "affirmation_reading"
