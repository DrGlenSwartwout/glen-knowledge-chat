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
