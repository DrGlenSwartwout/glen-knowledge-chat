import sqlite3
from dashboard import community_signals as _s


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _s.init_signal_tables(cx)
    return cx


def test_toggle_reaction_on_then_off():
    cx = _cx()
    assert _s.toggle_reaction(cx, "A@B.com", 5, "helpful") is True   # added
    assert _s.my_reactions(cx, "a@b.com", 5) == ["helpful"]
    assert _s.toggle_reaction(cx, "a@b.com", 5, "helpful") is False  # removed
    assert _s.my_reactions(cx, "a@b.com", 5) == []


def test_reaction_counts_aggregate_no_emails():
    cx = _cx()
    _s.toggle_reaction(cx, "a@b.com", 5, "helpful")
    _s.toggle_reaction(cx, "c@d.com", 5, "helpful")
    _s.toggle_reaction(cx, "c@d.com", 5, "inspiring")
    counts = _s.reaction_counts(cx, 5)
    assert counts["helpful"] == 2 and counts["inspiring"] == 1
    # aggregate structure only: values are ints, keys are reaction names
    assert all(isinstance(v, int) for v in counts.values())
    assert "a@b.com" not in counts and "c@d.com" not in counts


def test_my_reactions_only_caller():
    cx = _cx()
    _s.toggle_reaction(cx, "a@b.com", 5, "helpful")
    _s.toggle_reaction(cx, "c@d.com", 5, "inspiring")
    assert _s.my_reactions(cx, "a@b.com", 5) == ["helpful"]  # not c@d's


def test_set_signal_upsert_replaces():
    cx = _cx()
    _s.set_signal(cx, "a@b.com", "topic", "sleep", "like")
    _s.set_signal(cx, "a@b.com", "topic", "sleep", "block")  # same target
    sig = _s.my_signals(cx, "a@b.com")
    assert sig["blocks"] == [{"target_type": "topic", "target_key": "sleep"}]
    assert sig["likes"] == []          # like replaced, not doubled
    row_count = cx.execute("SELECT COUNT(*) FROM community_signals").fetchone()[0]
    assert row_count == 1              # one row per (email, target)


def test_clear_signal_deletes():
    cx = _cx()
    _s.set_signal(cx, "a@b.com", "topic", "sleep", "like")
    _s.clear_signal(cx, "a@b.com", "topic", "sleep")
    assert _s.my_signals(cx, "a@b.com") == {"likes": [], "blocks": []}


def test_my_signals_splits_and_scopes():
    cx = _cx()
    _s.set_signal(cx, "a@b.com", "topic", "sleep", "like")
    _s.set_signal(cx, "a@b.com", "person", "x@y.com", "block")
    _s.set_signal(cx, "c@d.com", "topic", "adrenals", "like")  # other member
    sig = _s.my_signals(cx, "a@b.com")
    assert {"target_type": "topic", "target_key": "sleep"} in sig["likes"]
    assert {"target_type": "person", "target_key": "x@y.com"} in sig["blocks"]
    assert all(x["target_key"] != "adrenals" for x in sig["likes"])  # not c@d's
