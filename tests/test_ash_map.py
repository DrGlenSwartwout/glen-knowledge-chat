import copy
import sqlite3

import dashboard.ash_map as am


def test_twelve_canonical_dimensions_in_order():
    assert am.DIM_KEYS == [
        "body", "mind", "spirit", "inheritance", "personal_history",
        "epigenetics", "symptoms", "terrain", "diagnosis", "treatment",
        "regulation", "prognosis",
    ]
    # ASH_DIMENSIONS carries key/name/meaning for each, in the same order
    assert [d["key"] for d in am.ASH_DIMENSIONS] == am.DIM_KEYS
    for d in am.ASH_DIMENSIONS:
        assert d["name"] and d["meaning"]


def test_state_order_ladder():
    assert am.STATE_ORDER == {"untouched": 0, "opened": 1, "explored": 2, "deep": 3}


def test_norm_email():
    assert am._norm_email("  Foo@Bar.COM ") == "foo@bar.com"


def test_blank_map_has_all_twelve_untouched_and_is_fresh():
    m = am._blank_map()
    assert set(m.keys()) == set(am.DIM_KEYS)
    for k in am.DIM_KEYS:
        assert m[k] == {
            "state": "untouched", "opened_excerpt": "",
            "notes": "", "last_touched_at": None,
        }
    # fresh dict each call — mutating one does not leak into the next
    m["body"]["notes"] = "x"
    assert am._blank_map()["body"]["notes"] == ""


def test_now_iso_is_valid_iso8601_with_seconds():
    from datetime import datetime
    ts = am._now_iso()
    # parseable as ISO 8601 (strip trailing Z for fromisoformat)
    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed.year >= 2026
    # seconds field present: THH:MM:SS.ffffff -> the time part has 3 colon-separated groups
    time_part = ts.split("T")[1].rstrip("Z")
    assert len(time_part.split(":")) == 3, ts


def _mem(summary="", dims=None):
    m = {"summary": summary, "dimensions": am._blank_map()}
    for k, v in (dims or {}).items():
        m["dimensions"][k].update(v)
    return m


def test_merge_bumps_state_forward_and_sets_excerpt_once():
    mem = _mem()
    out = {"dimensions": {"symptoms": {"state": "opened",
            "excerpt": "my knee aches every morning", "notes": "AM knee pain"}},
           "summary": "Cautious, in pain."}
    merged = am.merge_turn(mem, out)
    s = merged["dimensions"]["symptoms"]
    assert s["state"] == "opened"
    assert s["opened_excerpt"] == "my knee aches every morning"
    assert s["notes"] == "AM knee pain"
    assert s["last_touched_at"] is not None
    assert merged["summary"] == "Cautious, in pain."
    # untouched dims stay untouched
    assert merged["dimensions"]["body"]["state"] == "untouched"


def test_merge_never_downgrades_and_preserves_first_excerpt():
    mem = _mem(dims={"symptoms": {"state": "deep",
        "opened_excerpt": "first words", "notes": "old"}})
    out = {"dimensions": {"symptoms": {"state": "opened",
            "excerpt": "second words", "notes": "new detail"}}, "summary": ""}
    merged = am.merge_turn(mem, out)
    s = merged["dimensions"]["symptoms"]
    assert s["state"] == "deep"               # max(deep, opened) = deep, no downgrade
    assert s["opened_excerpt"] == "first words"  # excerpt set once, preserved
    assert s["notes"] == "old\nnew detail"    # appended


def test_merge_dedupes_identical_note_line():
    mem = _mem(dims={"terrain": {"state": "explored", "notes": "low vitality"}})
    out = {"dimensions": {"terrain": {"state": "explored",
            "excerpt": "", "notes": "low vitality"}}, "summary": ""}
    merged = am.merge_turn(mem, out)
    assert merged["dimensions"]["terrain"]["notes"] == "low vitality"  # not duplicated


def test_merge_empty_summary_preserves_prior_and_input_not_mutated():
    mem = _mem(summary="Prior who-they-are.")
    snapshot = copy.deepcopy(mem)
    out = {"dimensions": {"mind": {"state": "opened", "excerpt": "", "notes": "n"}},
           "summary": ""}
    merged = am.merge_turn(mem, out)
    assert merged["summary"] == "Prior who-they-are."  # empty summary keeps prior
    assert mem == snapshot                             # input untouched


def test_merge_ignores_unknown_dimension_keys():
    mem = _mem()
    out = {"dimensions": {"not_a_dim": {"state": "deep", "excerpt": "x", "notes": "y"}},
           "summary": ""}
    merged = am.merge_turn(mem, out)
    assert "not_a_dim" not in merged["dimensions"]
    assert all(merged["dimensions"][k]["state"] == "untouched" for k in am.DIM_KEYS)


def _cx():
    return sqlite3.connect(":memory:")


def test_get_unseen_email_returns_all_untouched_skeleton():
    cx = _cx()
    m = am.get(cx, "  New@User.com ")
    assert m["email"] == "new@user.com"
    assert m["summary"] == ""
    assert set(m["dimensions"].keys()) == set(am.DIM_KEYS)
    assert all(m["dimensions"][k]["state"] == "untouched" for k in am.DIM_KEYS)
    assert m["created_at"] is None and m["updated_at"] is None


def test_upsert_then_get_round_trips_and_backfills_missing_keys():
    cx = _cx()
    am.init_table(cx)
    dims = am._blank_map()
    dims["symptoms"].update({"state": "opened", "opened_excerpt": "knee", "notes": "AM"})
    # store a PARTIAL dimensions map (only one key) to prove get() backfills the rest
    am._upsert(cx, "a@b.com", "A summary.", {"symptoms": dims["symptoms"]})
    got = am.get(cx, "A@B.com")
    assert got["summary"] == "A summary."
    assert got["dimensions"]["symptoms"]["state"] == "opened"
    assert got["dimensions"]["symptoms"]["opened_excerpt"] == "knee"
    # the other 11 keys are backfilled as untouched
    assert got["dimensions"]["body"]["state"] == "untouched"
    assert set(got["dimensions"].keys()) == set(am.DIM_KEYS)
    assert got["created_at"] and got["updated_at"]


def test_upsert_preserves_created_at_on_update():
    cx = _cx()
    am.init_table(cx)
    am._upsert(cx, "a@b.com", "first", am._blank_map())
    first = am.get(cx, "a@b.com")["created_at"]
    am._upsert(cx, "a@b.com", "second", am._blank_map())
    again = am.get(cx, "a@b.com")
    assert again["summary"] == "second"
    assert again["created_at"] == first  # created_at preserved across updates


def test_context_block_first_conversation_line():
    mem = {"summary": "", "dimensions": am._blank_map()}
    assert am.context_block(mem) == (
        "This is your first conversation with them — nothing covered yet."
    )


def test_context_block_populated_sections():
    mem = {"summary": "A tired caregiver in pain.", "dimensions": am._blank_map()}
    mem["dimensions"]["symptoms"].update(
        {"state": "explored", "notes": "AM knee pain\nworse in cold"})
    mem["dimensions"]["terrain"].update(
        {"state": "opened", "opened_excerpt": "I just have no energy left"})
    block = am.context_block(mem)
    assert "Who they are: A tired caregiver in pain." in block
    assert "Already explored (do not re-ask):" in block
    assert "AM knee pain worse in cold" in block          # notes flattened
    assert "Opened, go deeper when they return to it:" in block
    assert '"I just have no energy left"' in block
    assert "Not yet touched:" in block
    # an untouched dim's display name appears in the not-yet-touched list
    assert "Body / States of Matter" in block
    # touched dims are NOT in the not-yet-touched list
    not_touched_line = [l for l in block.splitlines() if l.startswith("Not yet touched:")][0]
    assert "Symptoms / 5 Cardinal Signs" not in not_touched_line
