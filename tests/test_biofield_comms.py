from dashboard.biofield_comms import comms_to_text


def test_flattens_all_sections():
    ctx = {"intake_summary": "first name: Jane\n  Energy?: Low",
           "recent_inquiries": [{"main_challenge": "fatigue", "main_goal": "more energy"}],
           "recent_queries": [{"question": "why tired"}, {"question": "what helps sleep"}]}
    t = comms_to_text(ctx)
    assert "Jane" in t and "fatigue" in t and "more energy" in t
    assert "why tired" in t and "what helps sleep" in t


def test_empty_context():
    assert comms_to_text({}) == ""
    assert comms_to_text(None) == ""


def test_includes_recent_feedback():
    ctx = {"intake_summary": "", "recent_inquiries": [], "recent_queries": [],
           "recent_feedback": [{"summary": "feels exhausted",
                                "topics": ["sleep"], "conditions": ["adrenal fatigue"]}]}
    t = comms_to_text(ctx)
    assert "feels exhausted" in t and "sleep" in t and "adrenal fatigue" in t


def test_no_feedback_key_unchanged():
    # context without the key behaves exactly as before (back-compat)
    ctx = {"intake_summary": "Jane", "recent_inquiries": [], "recent_queries": []}
    assert comms_to_text(ctx) == "Jane"
