import sqlite3
from dashboard import notify_state as N


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db")); N.init_table(cx); return cx


def test_defaults_and_decide_taper(tmp_path):
    cx = _cx(tmp_path)
    d = N.decide(N.get_state(cx, "a@x.com"))
    assert d["eligible"] is True and d["variant"] == 0
    N.incr_notify(cx, "a@x.com"); N.incr_notify(cx, "a@x.com")     # count=2
    d = N.decide(N.get_state(cx, "a@x.com"))
    assert d["eligible"] is True and d["variant"] == 2             # 3rd (last-call)
    N.incr_notify(cx, "a@x.com")                                    # count=3
    assert N.decide(N.get_state(cx, "a@x.com"))["eligible"] is False  # quiet after 3


def test_opt_out_suppresses_and_in_overrides(tmp_path):
    cx = _cx(tmp_path)
    for _ in range(5): N.incr_notify(cx, "b@x.com")
    N.set_opt(cx, "b@x.com", "in")
    assert N.decide(N.get_state(cx, "b@x.com"))["eligible"] is True
    N.set_opt(cx, "b@x.com", "out")
    assert N.decide(N.get_state(cx, "b@x.com"))["eligible"] is False


def test_engaged_keeps_eligible_past_cap(tmp_path):
    cx = _cx(tmp_path)
    for _ in range(4): N.incr_notify(cx, "c@x.com")
    N.mark_engaged(cx, "c@x.com")
    assert N.decide(N.get_state(cx, "c@x.com"))["eligible"] is True


def test_email_by_phone_reverse_lookup(tmp_path):
    cx = _cx(tmp_path)
    N.set_phone(cx, "ph@x.com", "+1 (555) 123-9999")
    assert N.email_by_phone(cx, "5551239999") == "ph@x.com"
    assert N.email_by_phone(cx, "+15551239999") == "ph@x.com"   # last-10 digits match
    assert N.email_by_phone(cx, "0000000000") is None
