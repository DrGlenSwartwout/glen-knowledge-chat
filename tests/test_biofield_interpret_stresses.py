from dashboard.biofield_interpret import interpret_stresses


def _c(payload):
    return lambda system, user: payload


def test_extracts_distinct_stress_labels():
    out = interpret_stresses(
        "the stress is liver congestion, also adrenal fatigue, liver congestion again",
        _c('{"stresses": ["Liver congestion", "Adrenal fatigue", "liver congestion"]}'))
    assert out == ["Liver congestion", "Adrenal fatigue"]   # deduped case-insensitively, order kept


def test_empty_transcript_returns_empty():
    assert interpret_stresses("   ", _c('{"stresses": ["x"]}')) == []


def test_handles_garbage_completion():
    assert interpret_stresses("something", _c("not json at all")) == []
