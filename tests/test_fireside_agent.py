from dashboard import fireside_agent as fa


def test_parse_hook_present_strips_marker():
    raw = "I think I know what your body is asking for. Shall we go and find it?\n⟦HOOK⟧"
    clean, hooked = fa.parse_hook(raw)
    assert hooked is True
    assert "⟦HOOK⟧" not in clean
    assert clean.endswith("find it?")


def test_parse_hook_absent():
    clean, hooked = fa.parse_hook("Tell me more about that.")
    assert hooked is False
    assert clean == "Tell me more about that."


def test_hook_eligible_hard_cap():
    assert fa.hook_eligible(8, {}) is True
    assert fa.hook_eligible(9, {"dimensions": {}}) is True


def test_hook_eligible_min_turns_and_dims():
    cov = {"dimensions": {
        "symptoms": {"state": "opened"},
        "terrain": {"state": "explored"},
        "spirit": {"state": "opened"},
        "mind": {"state": "untouched"},
    }}
    assert fa.hook_eligible(4, cov) is True   # 4 turns, 3 touched
    assert fa.hook_eligible(3, cov) is False  # too few turns
    thin = {"dimensions": {"symptoms": {"state": "opened"}}}
    assert fa.hook_eligible(4, thin) is False  # only 1 touched


def test_hook_eligible_early_is_false():
    assert fa.hook_eligible(1, {}) is False
    assert fa.hook_eligible(0, {"dimensions": {}}) is False


def test_build_system_includes_persona_and_context():
    cov = {"summary": "weary traveler", "dimensions": {"symptoms": {"state": "opened", "opened_excerpt": "always tired"}}}
    sys_low = fa.build_system(cov, turn_count=2)
    assert "Glendalf" in sys_low
    assert "weary traveler" in sys_low          # context_block folded in
    assert "do not close" in sys_low.lower()    # hook forbidden early
    sys_hi = fa.build_system(cov, turn_count=8)
    assert "⟦HOOK⟧" in sys_hi                    # hook permitted at cap
    assert "shall we go and find it" in sys_hi.lower()


def test_build_messages_maps_roles_and_appends():
    transcript = [
        {"speaker": "traveler", "text": "hi"},
        {"speaker": "glendalf", "text": "welcome, friend"},
        {"speaker": "traveler", "text": ""},          # empty -> dropped
    ]
    msgs = fa.build_messages(transcript, "I feel stuck")
    assert msgs[0] == {"role": "user", "content": "hi"}
    assert msgs[1] == {"role": "assistant", "content": "welcome, friend"}
    assert msgs[-1] == {"role": "user", "content": "I feel stuck"}
    assert all(m["content"] for m in msgs)         # no empty turns


def test_build_messages_caps_history():
    transcript = [{"speaker": "traveler", "text": f"m{i}"} for i in range(40)]
    msgs = fa.build_messages(transcript, "now")
    # last MAX_HISTORY_TURNS of history + the new user message
    assert len(msgs) == fa.MAX_HISTORY_TURNS + 1
    assert msgs[-1]["content"] == "now"
