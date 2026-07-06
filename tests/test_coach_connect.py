import sqlite3
from dashboard import coach_directory as _cd
from dashboard import coach_connect as _cc


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _cd.init_coach_tables(cx)
    _cc.init_connect_tables(cx)
    return cx


def test_coach_ref_stable_and_resolves():
    cx = _cx()
    _cd.upsert_volunteer(cx, email="Coach@X.com", name="Cora", focus="sleep",
                         intro_video_url="u", capacity=2, cert_ok=1)
    ref = _cc.coach_ref("coach@x.com")
    assert ref == _cc.coach_ref("COACH@X.com") and len(ref) == 16
    assert _cc.email_for_ref(cx, ref) == "coach@x.com"
    assert _cc.email_for_ref(cx, "deadbeefdeadbeef") is None


def test_email_for_ref_only_active():
    cx = _cx()
    _cd.upsert_volunteer(cx, email="off@x.com", name="Off", focus="f", intro_video_url="u",
                         capacity=2, cert_ok=1)
    _cd.set_active(cx, "off@x.com", 0)
    assert _cc.email_for_ref(cx, _cc.coach_ref("off@x.com")) is None  # inactive not resolvable


def test_multi_apply_allowed_until_matched():
    cx = _cx()
    r1 = _cc.create_request(cx, "c1@x.com", "m@x.com", "Mel", "trouble sleeping")
    r2 = _cc.create_request(cx, "c2@x.com", "m@x.com", "Mel", "and adrenals")
    assert r1 is not None and r2 is not None                    # two pendings OK
    dup = _cc.create_request(cx, "c1@x.com", "m@x.com", "Mel", "dup")
    assert dup is None                                          # already applied to c1
    assert [a["coach_email"] for a in _cc.member_applications(cx, "m@x.com")] == ["c1@x.com", "c2@x.com"]
    # once matched, no new applications
    _cc.set_request_status(cx, r1, "accepted")
    assert _cc.member_has_accepted(cx, "m@x.com") is True
    assert _cc.create_request(cx, "c3@x.com", "m@x.com", "Mel", "more") is None


def test_first_accept_withdraws_other_pendings():
    cx = _cx()
    r1 = _cc.create_request(cx, "c1@x.com", "m@x.com", "Mel", "n1")
    r2 = _cc.create_request(cx, "c2@x.com", "m@x.com", "Mel", "n2")
    _cc.set_request_status(cx, r1, "accepted")
    _cc.withdraw_other_pendings(cx, "m@x.com", r1)
    apps = {a["coach_email"]: a["status"] for a in _cc.member_applications(cx, "m@x.com")}
    assert apps == {"c1@x.com": "accepted"}                     # c2 pending withdrawn (not active)
    assert _cc.request_member(cx, r2) == "m@x.com"


def test_accept_flow_and_count():
    cx = _cx()
    rid = _cc.create_request(cx, "c@x.com", "m@x.com", "Mel", "note")
    assert _cc.accepted_count(cx, "c@x.com") == 0
    _cc.set_request_status(cx, rid, "accepted")
    assert _cc.accepted_count(cx, "c@x.com") == 1
    pend = _cc.requests_for_coach(cx, "c@x.com", status="pending")
    assert pend == []                                     # no longer pending
    _cc.set_request_status(cx, rid, "declined")
    assert _cc.accepted_count(cx, "c@x.com") == 0


def test_requests_for_coach_has_name_note_video_no_email():
    cx = _cx()
    _cc.create_request(cx, "c@x.com", "m@x.com", "Mel", "working on adrenals",
                       member_video_url="/portal-asset/member-ab.mp4")
    pend = _cc.requests_for_coach(cx, "c@x.com")
    assert pend[0]["member_name"] == "Mel" and pend[0]["note"] == "working on adrenals"
    assert pend[0]["member_video_url"] == "/portal-asset/member-ab.mp4"
    assert "member_email" not in pend[0] and "email" not in pend[0]


def test_waitlist_and_interest():
    cx = _cx()
    assert _cc.on_waitlist(cx, "m@x.com") is False
    _cc.join_waitlist(cx, "M@x.com")
    assert _cc.on_waitlist(cx, "m@x.com") is True
    _cc.record_interest(cx, "m@x.com", "glen")
    _cc.record_interest(cx, "m@x.com", "glen")             # idempotent per (member,tier)
    assert cx.execute("SELECT COUNT(*) FROM coaching_interest").fetchone()[0] == 1


def test_list_active_full_has_email_and_capacity():
    cx = _cx()
    _cd.upsert_volunteer(cx, email="c@x.com", name="Cora", focus="sleep",
                         intro_video_url="u", capacity=3, cert_ok=1)
    full = _cd.list_active_full(cx)
    assert full[0]["email"] == "c@x.com" and full[0]["capacity"] == 3


def test_reapply_after_decline_clears_decided_at():
    cx = _cx()
    rid = _cc.create_request(cx, "c@x.com", "m@x.com", "Mel", "first")
    _cc.set_request_status(cx, rid, "declined")     # stamps decided_at
    assert cx.execute("SELECT decided_at FROM coach_requests WHERE id=?", (rid,)).fetchone()[0] is not None
    rid2 = _cc.create_request(cx, "c@x.com", "m@x.com", "Mel", "second chance")   # re-apply
    row = cx.execute("SELECT status, decided_at FROM coach_requests WHERE id=?", (rid2,)).fetchone()
    assert rid2 == rid and row["status"] == "pending" and row["decided_at"] is None  # fresh pending
