import sqlite3
from dashboard import purity_ratings_access as acc


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    acc.init_table(cx); return cx


def test_paid_membership_can_request():
    cx = _cx()
    assert acc.can_request(cx, "a@b.com", "full") is True


def test_non_paid_without_grant_cannot():
    cx = _cx()
    assert acc.can_request(cx, "a@b.com", "trial") is False
    assert acc.can_request(cx, "a@b.com", "none") is False


def test_explicit_grant_overrides_non_paid():
    cx = _cx()
    acc.set_access(cx, "a@b.com", True, "glen")
    assert acc.can_request(cx, "a@b.com", "none") is True


def test_explicit_revoke_blocks_even_if_row_exists():
    cx = _cx()
    acc.set_access(cx, "a@b.com", True, "glen")
    acc.set_access(cx, "a@b.com", False, "glen")
    assert acc.can_request(cx, "a@b.com", "none") is False
    # but paid membership still passes regardless of the revoke row
    assert acc.can_request(cx, "a@b.com", "full") is True
