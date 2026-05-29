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


def test_reveal_for_layers():
    import begin_funnel as bf
    assert bf.reveal_for("arrival") == ["layer0"]
    assert bf.reveal_for("listening") == ["layer0", "layer1"]
    assert bf.reveal_for("inquire") == ["layer0", "layer1", "layer2"]
    assert bf.reveal_for("personalize") == ["layer0", "layer1", "layer2", "layer3"]
    assert bf.reveal_for("free_tier") == [
        "layer0", "layer1", "layer2", "layer3", "layer4", "layer5"]
    # rungs beyond free_tier still expose the full unfolding surface
    assert "layer5" in bf.reveal_for("assess")


def _seeded():
    import begin_funnel as bf
    cx = _mem()
    bf.init_journey_tables(cx)
    return bf, cx


def test_record_unlock_invalid_trigger_raises():
    bf, cx = _seeded()
    with pytest.raises(ValueError):
        bf.record_unlock(cx, session_id="s1", trigger="bogus")


def test_record_unlock_progresses_rung_and_logs_event():
    bf, cx = _seeded()
    st = bf.record_unlock(cx, session_id="s1", trigger="question")
    assert st["current_rung"] == "inquire"
    assert "question" in st["unlocked_gates"]
    st = bf.record_unlock(cx, session_id="s1", trigger="name", first_name="Ada")
    assert st["current_rung"] == "personalize"
    assert st["first_name"] == "Ada"
    assert cx.execute("SELECT COUNT(*) FROM journey_state").fetchone()[0] == 1
    assert cx.execute("SELECT COUNT(*) FROM journey_events").fetchone()[0] == 2
    last = cx.execute(
        "SELECT trigger, rung_before, rung_after FROM journey_events "
        "ORDER BY id DESC LIMIT 1").fetchone()
    assert last["trigger"] == "name"
    assert last["rung_before"] == "inquire"
    assert last["rung_after"] == "personalize"


def test_record_unlock_email_then_tos_reaches_free_tier_and_stamps():
    bf, cx = _seeded()
    bf.record_unlock(cx, session_id="s1", trigger="question")
    bf.record_unlock(cx, session_id="s1", trigger="name", first_name="Ada")
    st = bf.record_unlock(cx, session_id="s1", trigger="email",
                          email="ada@example.com")
    assert st["current_rung"] == "personalize"
    assert st["tos_agreed_at"] is None
    st = bf.record_unlock(cx, session_id="s1", trigger="tos", tos=True,
                          tos_version="rm-tc-2026-05-28")
    assert st["current_rung"] == "free_tier"
    assert st["email"] == "ada@example.com"
    assert st["tos_agreed_at"] is not None
    assert st["tos_version"] == "rm-tc-2026-05-28"


def test_get_state_default_for_unknown_session():
    bf, cx = _seeded()
    st = bf.get_state(cx, session_id="nope")
    assert st["current_rung"] == "arrival"
    assert st["unlocked_gates"] == []
    assert st["reveal"] == ["layer0"]
    assert st["surfaced_cards"] == []


def test_get_state_aggregates_across_sessions_by_email():
    bf, cx = _seeded()
    bf.record_unlock(cx, session_id="A", trigger="question")
    bf.record_unlock(cx, session_id="A", trigger="name", first_name="Ada")
    bf.record_unlock(cx, session_id="A", trigger="email", email="ada@x.com")
    bf.record_unlock(cx, session_id="A", trigger="tos", tos=True)
    bf.record_unlock(cx, session_id="B", trigger="video")
    bf.record_unlock(cx, session_id="B", trigger="email", email="ada@x.com")
    st = bf.get_state(cx, session_id="B", email="ada@x.com")
    assert st["current_rung"] == "free_tier"
    assert set(st["unlocked_gates"]) >= {"question", "name", "video"}
    assert st["first_name"] == "Ada"
    assert "layer5" in st["reveal"]


def test_awareness_rank_order():
    import begin_funnel as bf
    assert bf.AWARENESS_RANK["unknown"] < bf.AWARENESS_RANK["problem"] < \
           bf.AWARENESS_RANK["solution"] < bf.AWARENESS_RANK["product"] < \
           bf.AWARENESS_RANK["most"]


def test_infer_awareness_heuristic():
    import begin_funnel as bf
    assert bf.infer_awareness_heuristic("e4l", set(), []) == "most"
    assert bf.infer_awareness_heuristic("", {"scan"}, []) == "product"
    assert bf.infer_awareness_heuristic("", set(), ["tell me about EVOX"]) == "product"
    assert bf.infer_awareness_heuristic("", set(), ["what about a detox protocol"]) == "solution"
    assert bf.infer_awareness_heuristic("", set(), ["I am so tired and can't sleep"]) == "problem"
    assert bf.infer_awareness_heuristic("", set(), ["hello"]) == "unknown"
    assert bf.infer_awareness_heuristic("", set(), ["detox with E4L"]) == "product"


def test_max_awareness_monotonic():
    import begin_funnel as bf
    assert bf._max_awareness("problem", "product") == "product"
    assert bf._max_awareness("most", "solution") == "most"
    assert bf._max_awareness("unknown", "unknown") == "unknown"


def test_resolve_want_live_targets():
    import begin_funnel as bf
    url = bf.resolve_want("e4l", "Jane")
    assert url.startswith("https://truly.vip/E4L")
    assert "utm_source=Jane" in url
    assert "utm_campaign=begin-deeplink-e4l" in url
    assert bf.resolve_want("quiz", "").startswith("https://healing.scoreapp.com")
    assert "utm_source=remedy-match" in bf.resolve_want("quiz", "")
    assert bf.resolve_want("join", "x").startswith("https://truly.vip/Join")
    assert bf.resolve_want("results", "x").startswith("https://truly.vip/Results")


def test_resolve_want_unknown_or_unbuilt_returns_none():
    import begin_funnel as bf
    # "voice" is now a live internal target (slice 4) — only truly unknown keys return None
    assert bf.resolve_want("ash", "x") is None
    assert bf.resolve_want("", "x") is None
    assert bf.resolve_want("bogus", "x") is None


def test_reveal_for_gate_skip_for_aware():
    import begin_funnel as bf
    assert bf.reveal_for("arrival", "product") == bf._ALL_LAYERS
    assert bf.reveal_for("arrival", "most") == bf._ALL_LAYERS
    assert bf.reveal_for("arrival", "solution") == ["layer0"]
    assert bf.reveal_for("listening", "problem") == ["layer0", "layer1"]
    assert bf.reveal_for("arrival") == ["layer0"]   # default awareness preserves old behavior


def test_awareness_classified_at_column_exists():
    import begin_funnel as bf
    cx = _mem()
    bf.init_journey_tables(cx)
    cols = {r[1] for r in cx.execute("PRAGMA table_info(journey_state)")}
    assert "awareness_classified_at" in cols


def test_set_awareness_persists_upward_and_stamps():
    import begin_funnel as bf
    cx = _mem()
    bf.init_journey_tables(cx)
    bf.record_unlock(cx, session_id="s1", trigger="question")  # creates row
    bf.set_awareness(cx, "s1", "product")
    st = bf.get_state(cx, session_id="s1")
    assert st["awareness_stage"] == "product"
    row = cx.execute("SELECT awareness_classified_at FROM journey_state WHERE session_id='s1'").fetchone()
    assert row[0] is not None
    # never regresses: a lower stage does not overwrite
    bf.set_awareness(cx, "s1", "problem")
    st = bf.get_state(cx, session_id="s1")
    assert st["awareness_stage"] == "product"


def test_deep_link_trigger_valid_but_not_a_gate():
    import begin_funnel as bf
    assert "deep_link" in bf.VALID_TRIGGERS
    assert "deep_link" not in bf.GATE_TRIGGERS


def test_record_unlock_persists_awareness_from_query_texts():
    import begin_funnel as bf
    cx = _mem()
    bf.init_journey_tables(cx)
    st = bf.record_unlock(cx, session_id="s1", trigger="question",
                          query_texts=["tell me about EVOX dosing"])
    assert st["awareness_stage"] == "product"
    assert "layer5" in st["reveal"]   # product-aware gate-skip even at 'inquire' rung


def test_record_unlock_deep_link_sets_most_aware():
    import begin_funnel as bf
    cx = _mem()
    bf.init_journey_tables(cx)
    st = bf.record_unlock(cx, session_id="s1", trigger="deep_link", want="e4l")
    assert st["awareness_stage"] == "most"
    assert st["current_rung"] == "arrival"   # deep_link is NOT a commitment gate
    assert "layer5" in st["reveal"]


def test_get_state_reveal_uses_awareness():
    import begin_funnel as bf
    cx = _mem()
    bf.init_journey_tables(cx)
    bf.record_unlock(cx, session_id="s1", trigger="deep_link", want="voice")
    st = bf.get_state(cx, session_id="s1")
    assert st["awareness_stage"] == "most"
    assert "layer5" in st["reveal"]


def test_record_unlock_awareness_never_regresses_on_update():
    import begin_funnel as bf
    cx = _mem()
    bf.init_journey_tables(cx)
    # first call infers product-aware from the chat text
    st = bf.record_unlock(cx, session_id="s1", trigger="question",
                          query_texts=["how do I use EVOX"])
    assert st["awareness_stage"] == "product"
    # a later call with a plain, signal-free question must NOT drop awareness
    st = bf.record_unlock(cx, session_id="s1", trigger="scroll",
                          query_texts=["hi"])
    assert st["awareness_stage"] == "product"


def test_card_catalog_has_core_keys():
    import begin_funnel as bf
    for k in ("quiz", "e4l_scan", "intake", "voice_distinctions", "product",
              "ash_course", "ash_masterclass", "pay_forward", "practitioner"):
        assert k in bf.CARD_CATALOG
        assert bf.CARD_CATALOG[k]["title"] and bf.CARD_CATALOG[k]["sub"]


def test_card_href_external_threads_utm():
    import begin_funnel as bf
    h = bf.card_href("e4l_scan", "Jane")
    assert h.startswith("https://truly.vip/E4L")
    assert "utm_source=Jane" in h
    assert "utm_campaign=begin-card-e4l_scan" in h
    assert "utm_source=remedy-match" in bf.card_href("quiz", "")


def test_card_href_internal_as_is():
    import begin_funnel as bf
    assert bf.card_href("practitioner", "Jane") == "/practitioner"


def test_card_builds_full_dict():
    import begin_funnel as bf
    c = bf._card("product", "Jane")
    assert c["key"] == "product"
    assert c["title"] and c["sub"]
    assert c["href"].startswith("https://remedymatch.com")


# ---------------------------------------------------------------------------
# Slice 3 Task 2 — surface()
# ---------------------------------------------------------------------------

def _keys(cards):
    return [c["key"] for c in cards]


def test_surface_default_trio_when_no_signal():
    import begin_funnel as bf
    st = {"awareness_stage": "unknown", "current_rung": "arrival", "unlocked_gates": []}
    assert _keys(bf.surface(st, ["hello there"], "")) == ["quiz", "e4l_scan", "intake"]


def test_surface_specific_product_outranks_voice():
    import begin_funnel as bf
    st = {"awareness_stage": "unknown", "current_rung": "inquire", "unlocked_gates": []}
    cards = bf.surface(st, ["does EVOX help — also tell me about voice frequency"], "")
    assert _keys(cards)[0] == "product"


def test_surface_remedy_match_is_product():
    import begin_funnel as bf
    st = {"awareness_stage": "unknown", "current_rung": "inquire", "unlocked_gates": []}
    assert "product" in _keys(bf.surface(st, ["what helps with my insomnia"], ""))


def test_surface_generic_product_routes_to_voice():
    import begin_funnel as bf
    st = {"awareness_stage": "unknown", "current_rung": "inquire", "unlocked_gates": []}
    cards = bf.surface(st, ["what products do you have"], "")
    assert "voice_distinctions" in _keys(cards)
    assert "product" not in _keys(cards)


def test_surface_practitioner_signal_top():
    import begin_funnel as bf
    st = {"awareness_stage": "unknown", "current_rung": "inquire", "unlocked_gates": []}
    cards = bf.surface(st, ["I'm not happy with my dentist, anyone near me?"], "")
    assert _keys(cards)[0] == "practitioner"


def test_surface_caps_at_three():
    import begin_funnel as bf
    st = {"awareness_stage": "most", "current_rung": "assess", "unlocked_gates": []}
    cards = bf.surface(st, ["dentist near me, EVOX voice, learn a course, refer a friend"], "")
    assert len(cards) <= 3


def test_surface_most_aware_masterclass_when_no_specific():
    import begin_funnel as bf
    st = {"awareness_stage": "most", "current_rung": "arrival", "unlocked_gates": []}
    assert _keys(bf.surface(st, ["hello"], "")) == ["ash_masterclass"]


def test_resolve_want_voice_is_internal_room_no_utm():
    import begin_funnel as bf
    assert bf.resolve_want("voice", "Jane") == "/begin/voice"   # internal — no utm threading


def test_resolve_want_external_still_threads():
    import begin_funnel as bf
    h = bf.resolve_want("e4l", "Jane")
    assert h.startswith("https://truly.vip/E4L") and "utm_source=Jane" in h


def test_voice_distinctions_card_points_to_room():
    import begin_funnel as bf
    assert bf.CARD_CATALOG["voice_distinctions"]["internal"] is True
    assert bf.card_href("voice_distinctions", "Jane") == "/begin/voice"


def test_pay_forward_card_points_to_path_room():
    import begin_funnel as bf
    assert bf.CARD_CATALOG["pay_forward"]["internal"] is True
    assert bf.card_href("pay_forward", "Jane") == "/begin/path"


def test_resolve_want_path_is_internal_room():
    import begin_funnel as bf
    assert bf.resolve_want("path", "Jane") == "/begin/path"


def test_record_unlock_stamps_path_and_does_not_reset():
    import begin_funnel as bf
    cx = _mem()
    bf.init_journey_tables(cx)
    st = bf.record_unlock(cx, session_id="s1", trigger="paid_fork", path="pay_forward")
    assert st["path"] == "pay_forward"
    # a later unlock WITHOUT a path must not reset it to 'none'
    st = bf.record_unlock(cx, session_id="s1", trigger="scroll")
    assert st["path"] == "pay_forward"


def test_affiliate_social_links_table_exists():
    import begin_funnel as bf
    cx = _mem()
    bf.init_journey_tables(cx)
    names = {r[0] for r in cx.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "affiliate_social_links" in names
    cols = {r[1] for r in cx.execute("PRAGMA table_info(affiliate_social_links)")}
    for c in ("slug", "url", "points", "views", "likes", "shares", "ts"):
        assert c in cols, c


# ---------------------------------------------------------------------------
# Slice 6 Task 1 — ash_masterclass → /begin/ascend + want=ascend
# ---------------------------------------------------------------------------

def test_ash_masterclass_card_points_to_ascension_room():
    import begin_funnel as bf
    assert bf.CARD_CATALOG["ash_masterclass"]["internal"] is True
    assert bf.card_href("ash_masterclass", "Jane") == "/begin/ascend"


def test_resolve_want_ascend_is_internal_room():
    import begin_funnel as bf
    assert bf.resolve_want("ascend", "Jane") == "/begin/ascend"
