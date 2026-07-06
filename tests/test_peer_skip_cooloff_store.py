# tests/test_peer_skip_cooloff_store.py
import sqlite3
from dashboard import peer_connect as _pc, community_signals as _cs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _pc.init_peer_tables(cx); _cs.init_signal_tables(cx)
    return cx


def _skip(cx, frm, to, created_at):
    _pc.record_interest(cx, frm, to, "skip")
    cx.execute("UPDATE peer_interest SET created_at=? WHERE from_email=? AND to_email=?",
               (created_at, frm, to)); cx.commit()


def test_default_excludes_all_skips_and_connects():
    cx = _cx()
    for e in ("me@x.com", "sk@x.com", "cn@x.com"):
        _pc.set_optin(cx, e, True)
    _skip(cx, "me@x.com", "sk@x.com", "2020-01-01T00:00:00+00:00")   # even an OLD skip
    _pc.record_interest(cx, "me@x.com", "cn@x.com", "connect")
    assert set(_pc.eligible_candidates(cx, "me@x.com")) == set()      # fresh pass: both excluded


def test_stale_skip_readmitted_only_in_fallback():
    cx = _cx()
    for e in ("me@x.com", "old@x.com", "new@x.com"):
        _pc.set_optin(cx, e, True)
    _skip(cx, "me@x.com", "old@x.com", "2020-01-01T00:00:00+00:00")   # stale
    _skip(cx, "me@x.com", "new@x.com", "2099-01-01T00:00:00+00:00")   # fresh (future = not stale)
    cutoff = "2026-01-01T00:00:00+00:00"
    fresh = set(_pc.eligible_candidates(cx, "me@x.com"))
    fb = set(_pc.eligible_candidates(cx, "me@x.com", include_stale_skips=True, cutoff_iso=cutoff))
    assert fresh == set()                       # both skips excluded in the fresh pass
    assert fb == {"old@x.com"}                  # only the stale skip re-admitted


def test_connect_never_readmitted_even_in_fallback():
    cx = _cx()
    for e in ("me@x.com", "cn@x.com"):
        _pc.set_optin(cx, e, True)
    _pc.record_interest(cx, "me@x.com", "cn@x.com", "connect")
    cx.execute("UPDATE peer_interest SET created_at='2020-01-01T00:00:00+00:00'"); cx.commit()
    assert set(_pc.eligible_candidates(cx, "me@x.com", include_stale_skips=True,
                                       cutoff_iso="2026-01-01T00:00:00+00:00")) == set()


def test_other_exclusions_hold_in_fallback():
    cx = _cx()
    for e in ("me@x.com", "sk@x.com", "theyskip@x.com"):
        _pc.set_optin(cx, e, True)
    _skip(cx, "me@x.com", "sk@x.com", "2020-01-01T00:00:00+00:00")    # stale skip by me
    _skip(cx, "theyskip@x.com", "me@x.com", "2020-01-01T00:00:00+00:00")  # they skipped me (stale)
    fb = set(_pc.eligible_candidates(cx, "me@x.com", include_stale_skips=True,
                                     cutoff_iso="2026-01-01T00:00:00+00:00", is_paid=lambda e: True))
    assert fb == {"sk@x.com"}      # my stale skip re-admitted; a stale skip-of-me stays excluded


def test_my_interest_shape():
    cx = _cx()
    assert _pc._my_interest(cx, "a@x.com", "b@x.com") == (None, None)
    _pc.record_interest(cx, "a@x.com", "b@x.com", "skip")
    kind, at = _pc._my_interest(cx, "a@x.com", "b@x.com")
    assert kind == "skip" and at
