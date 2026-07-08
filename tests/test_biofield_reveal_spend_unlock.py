"""Spend-earned full reveal unlock: a free member who places a paid order >= $100
since their last reveal earns their NEXT reveal fully un-blurred (per reveal_id,
non-stacking). See docs/superpowers/specs/2026-07-08-reveal-spend-eligibility-design.md."""
import sqlite3

from dashboard import biofield_reveals as br

EMAIL = "pat@example.com"


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE biofield_reveals (id INTEGER PRIMARY KEY AUTOINCREMENT, "
               "email TEXT, scan_date TEXT, interpretation_json TEXT, remedies_json TEXT, "
               "layers_json TEXT, created_at TEXT, updated_at TEXT, first_approved INTEGER DEFAULT 0)")
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, "
               "total_cents INTEGER, pay_status TEXT, paid_cents INTEGER DEFAULT 0, created_at TEXT)")
    br.init_spend_unlocks(cx)
    return cx


def _order(cx, total, created_at, pay_status="paid", paid_cents=None, email=EMAIL):
    cx.execute("INSERT INTO orders (email,total_cents,pay_status,paid_cents,created_at) VALUES (?,?,?,?,?)",
               (email, total, pay_status, total if paid_cents is None else paid_cents, created_at))
    cx.commit()


def _reveal(cx, scan_date, created_at, email=EMAIL):
    cur = cx.execute("INSERT INTO biofield_reveals (email,scan_date,created_at,updated_at) VALUES (?,?,?,?)",
                     (email, scan_date, created_at, created_at))
    cx.commit()
    return cur.lastrowid


def test_grants_on_qualifying_order_since_last_reveal():
    cx = _cx()
    _reveal(cx, "2026-06-01", "2026-06-01T00:00:00")       # prior reveal
    _order(cx, 12000, "2026-06-15T00:00:00")               # paid $120 after it
    new_id = _reveal(cx, "2026-07-01", "2026-07-01T00:00:00")
    assert br.maybe_unlock_for_spend(cx, EMAIL, new_id) is True
    assert br.is_spend_unlocked(cx, new_id) is True


def test_no_unlock_below_threshold():
    cx = _cx()
    _reveal(cx, "2026-06-01", "2026-06-01T00:00:00")
    _order(cx, 9999, "2026-06-15T00:00:00")                # $99.99
    new_id = _reveal(cx, "2026-07-01", "2026-07-01T00:00:00")
    assert br.maybe_unlock_for_spend(cx, EMAIL, new_id) is False
    assert br.is_spend_unlocked(cx, new_id) is False


def test_no_unlock_when_unpaid():
    cx = _cx()
    _reveal(cx, "2026-06-01", "2026-06-01T00:00:00")
    _order(cx, 12000, "2026-06-15T00:00:00", pay_status="unpaid", paid_cents=0)
    new_id = _reveal(cx, "2026-07-01", "2026-07-01T00:00:00")
    assert br.maybe_unlock_for_spend(cx, EMAIL, new_id) is False


def test_no_unlock_for_order_before_last_reveal():
    cx = _cx()
    _reveal(cx, "2026-06-01", "2026-06-01T00:00:00")
    _order(cx, 12000, "2026-05-01T00:00:00")               # BEFORE the prior reveal
    new_id = _reveal(cx, "2026-07-01", "2026-07-01T00:00:00")
    assert br.maybe_unlock_for_spend(cx, EMAIL, new_id) is False


def test_first_reveal_unlocks_with_no_prior_reveal():
    cx = _cx()
    _order(cx, 15000, "2026-06-15T00:00:00")               # spend before any reveal
    new_id = _reveal(cx, "2026-07-01", "2026-07-01T00:00:00")
    assert br.maybe_unlock_for_spend(cx, EMAIL, new_id) is True


def test_idempotent():
    cx = _cx()
    _order(cx, 15000, "2026-06-15T00:00:00")
    new_id = _reveal(cx, "2026-07-01", "2026-07-01T00:00:00")
    assert br.maybe_unlock_for_spend(cx, EMAIL, new_id) is True
    assert br.maybe_unlock_for_spend(cx, EMAIL, new_id) is False     # already granted
    n = cx.execute("SELECT COUNT(*) FROM biofield_reveal_spend_unlocks WHERE reveal_id=?", (new_id,)).fetchone()[0]
    assert n == 1


def test_two_orders_still_single_unlock():
    cx = _cx()
    _reveal(cx, "2026-06-01", "2026-06-01T00:00:00")
    _order(cx, 12000, "2026-06-10T00:00:00")
    _order(cx, 20000, "2026-06-20T00:00:00")               # second qualifying order, same period
    new_id = _reveal(cx, "2026-07-01", "2026-07-01T00:00:00")
    assert br.maybe_unlock_for_spend(cx, EMAIL, new_id) is True
    n = cx.execute("SELECT COUNT(*) FROM biofield_reveal_spend_unlocks WHERE reveal_id=?", (new_id,)).fetchone()[0]
    assert n == 1                                          # no banking — one unlock on the next reveal


def test_upsert_new_reveal_grants_via_hook():
    cx = _cx()
    _reveal(cx, "2026-06-01", "2026-06-01T00:00:00")       # prior reveal
    _order(cx, 12000, "2026-06-15T00:00:00")               # qualifying order after it
    new_id, is_new = br.upsert(cx, EMAIL, "2026-07-01", {}, [], "test")
    assert is_new is True
    assert br.is_spend_unlocked(cx, new_id) is True        # hook fired on creation


def test_upsert_update_path_does_not_grant():
    cx = _cx()
    # First reveal created with NO qualifying order -> no unlock.
    new_id, is_new = br.upsert(cx, EMAIL, "2026-07-01", {}, [], "test")
    assert is_new is True and br.is_spend_unlocked(cx, new_id) is False
    # Now a qualifying order arrives, then the SAME reveal is re-upserted (update path).
    _order(cx, 12000, "2026-07-05T00:00:00")
    same_id, is_new2 = br.upsert(cx, EMAIL, "2026-07-01", {"x": 1}, [], "test")
    assert same_id == new_id and is_new2 is False
    assert br.is_spend_unlocked(cx, new_id) is False       # update path must not grant
