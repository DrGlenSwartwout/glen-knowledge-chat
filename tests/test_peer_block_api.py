import json, sqlite3
from unittest import mock
import app as appmod
from dashboard import peer_connect as _pc, community_signals as _cs, coach_threads as _ct


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _member(email, *topics):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx)
        _pc.init_peer_tables(cx); _cs.init_signal_tables(cx); _ct.init_thread_tables(cx)
        _cp.ensure_token(cx, email, email.split("@")[0].title())
        for t in topics:
            _cs.set_signal(cx, email, "topic", t, "like")
        tok = _ev.ensure_portal_token(cx, email, email.split("@")[0]); cx.commit()
    return tok


def _optin(*emails):
    for e in emails:
        with sqlite3.connect(appmod.LOG_DB) as cx:
            cx.row_factory = sqlite3.Row; _pc.init_peer_tables(cx)
            _pc.set_optin(cx, e, True); cx.commit()


def test_block_writes_person_signal():
    c = _client(); me = _member("pb1_me@x.com", "liver"); _member("pb1_n@x.com", "liver")
    _optin("pb1_me@x.com", "pb1_n@x.com")
    ref = _pc.member_ref("pb1_n@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True):
        r = c.post(f"/api/peer/interest?token={me}", json={"member_ref": ref, "kind": "block"})
    d = r.get_json()
    assert d["ok"] is True and d["matched"] is False
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cs.init_signal_tables(cx)
        row = cx.execute("SELECT * FROM community_signals WHERE email='pb1_me@x.com' "
                         "AND target_type='person'").fetchone()
        assert row["target_key"] == ref and row["signal"] == "block"
        assert "pb1_n@x.com" not in json.dumps(dict(row))     # anonymous: no email stored


def test_block_excludes_both_directions():
    # Isolated in-memory db (not appmod.LOG_DB): this exercises peer_connect
    # logic only, and eligible_candidates has no shared-topic filter, so
    # sharing the module-wide LOG_DB with other tests' opted-in members would
    # leave unrelated members in the pool and false-fail this assertion.
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _pc.init_peer_tables(cx); _cs.init_signal_tables(cx)
    for e in ("pb2_me@x.com", "pb2_n@x.com"):
        _pc.set_optin(cx, e, True)
    _cs.set_signal(cx, "pb2_me@x.com", "person", _pc.member_ref("pb2_n@x.com"), "block")
    paid = lambda e: True
    assert _pc.eligible_candidates(cx, "pb2_me@x.com", is_paid=paid) == []     # N gone from mine
    assert _pc.eligible_candidates(cx, "pb2_n@x.com", is_paid=paid) == []       # I'm gone from N's (mutual)


def test_block_supersedes_stale_skip():
    # Isolated in-memory db for the same reason as above.
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _pc.init_peer_tables(cx); _cs.init_signal_tables(cx)
    for e in ("pb3_me@x.com", "pb3_n@x.com"):
        _pc.set_optin(cx, e, True)
    _pc.record_interest(cx, "pb3_me@x.com", "pb3_n@x.com", "skip")
    cx.execute("UPDATE peer_interest SET created_at='2020-01-01T00:00:00+00:00'"); cx.commit()
    _cs.set_signal(cx, "pb3_me@x.com", "person", _pc.member_ref("pb3_n@x.com"), "block")
    # even the stale-skip fallback pass excludes a blocked person
    fb = _pc.eligible_candidates(cx, "pb3_me@x.com", is_paid=lambda e: True,
                                 include_stale_skips=True, cutoff_iso="2026-01-01T00:00:00+00:00")
    assert fb == []


def test_block_stale_ref_404():
    c = _client(); me = _member("pb4_me@x.com", "liver"); _optin("pb4_me@x.com")
    with mock.patch.object(appmod, "_is_paid_member", return_value=True):
        r = c.post(f"/api/peer/interest?token={me}", json={"member_ref": "deadbeefdeadbeef",
                                                           "kind": "block"})
    assert r.status_code == 404
