# tests/test_trial_credit.py
"""Trial-upgrade credit engine — accrual math (PR2, pure module).

A $1 Biofield-trial buyer pays REGULAR price on remedy orders (volume discount is
a paid-member perk). dashboard.trial_credit accrues the member discount they left
on the table over a 30-day window from the trial purchase. Pricing is delegated to
a callback so the app's pricing fns stay in app.py and this math stays testable.

accrued_credit_cents(cx, email, *, price_line, window_days=30) -> int
  price_line(item, order) -> (regular_unit_cents, member_unit_cents, qty)
    return (0, 0, 0) for a non-eligible / unresolvable line.
"""
import json
import sqlite3

from dashboard import orders as _orders
from dashboard import trial_credit as tc

EMAIL = "trial-buyer@example.com"


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _orders.init_orders_table(cx)
    return cx


def _add_order(cx, *, source, ref, created_at, items, email=EMAIL):
    """Insert an order row with an explicit created_at (upsert_order stamps _now())."""
    cx.execute(
        "INSERT INTO orders (created_at, source, external_ref, email, items_json, "
        "total_cents, status) VALUES (?,?,?,?,?,?, 'new')",
        (created_at, source, ref, email, json.dumps(items), 0),
    )
    cx.commit()


# A price_line that treats any item named like "ff-*" as volume-eligible with a
# fixed $69.97 regular and $50.00 member unit; everything else non-eligible.
def _fake_price_line(item, order):
    name = (item.get("name") or "").lower()
    qty = int(item.get("qty") or 1)
    if name.startswith("ff-"):
        return (6997, 5000, qty)
    return (0, 0, 0)


def test_no_trial_order_returns_zero():
    cx = _cx()
    # A reorder exists, but no biofield_trial order -> no window -> 0.
    _add_order(cx, source="reorder", ref="r1", created_at="2026-06-10T00:00:00+00:00",
               items=[{"name": "FF-A", "qty": 2}])
    assert tc.accrued_credit_cents(cx, EMAIL, price_line=_fake_price_line) == 0


def test_sums_in_window_eligible_lines():
    cx = _cx()
    _add_order(cx, source="biofield_trial", ref="bt1",
               created_at="2026-06-01T00:00:00+00:00",
               items=[{"name": "Biofield Analysis", "qty": 1}])
    _add_order(cx, source="reorder", ref="r1", created_at="2026-06-06T00:00:00+00:00",
               items=[{"name": "FF-A", "qty": 2}, {"name": "FF-B", "qty": 1}])
    # (6997-5000)*2 + (6997-5000)*1 = 1997*3 = 5991
    assert tc.accrued_credit_cents(cx, EMAIL, price_line=_fake_price_line) == 5991


def test_ignores_out_of_window_orders():
    cx = _cx()
    _add_order(cx, source="biofield_trial", ref="bt1",
               created_at="2026-06-01T00:00:00+00:00",
               items=[{"name": "Biofield Analysis", "qty": 1}])
    # 40 days after trial start -> outside the 31-day window.
    _add_order(cx, source="reorder", ref="r1", created_at="2026-07-11T00:00:00+00:00",
               items=[{"name": "FF-A", "qty": 5}])
    assert tc.accrued_credit_cents(cx, EMAIL, price_line=_fake_price_line) == 0


def test_day_32_is_just_outside_window():
    cx = _cx()
    _add_order(cx, source="biofield_trial", ref="bt1",
               created_at="2026-06-01T00:00:00+00:00",
               items=[{"name": "Biofield Analysis", "qty": 1}])
    # 2026-07-03 = trial start + 32 days -> just past the 31-day window.
    _add_order(cx, source="reorder", ref="r1", created_at="2026-07-03T00:00:01+00:00",
               items=[{"name": "FF-A", "qty": 1}])
    assert tc.accrued_credit_cents(cx, EMAIL, price_line=_fake_price_line) == 0


def test_ignores_non_eligible_lines():
    cx = _cx()
    _add_order(cx, source="biofield_trial", ref="bt1",
               created_at="2026-06-01T00:00:00+00:00",
               items=[{"name": "Biofield Analysis", "qty": 1}])
    _add_order(cx, source="reorder", ref="r1", created_at="2026-06-05T00:00:00+00:00",
               items=[{"name": "FF-A", "qty": 2}, {"name": "Some Booklet", "qty": 3}])
    # only the FF-A line counts: 1997*2 = 3994
    assert tc.accrued_credit_cents(cx, EMAIL, price_line=_fake_price_line) == 3994


def test_clamps_negative_per_line():
    cx = _cx()
    _add_order(cx, source="biofield_trial", ref="bt1",
               created_at="2026-06-01T00:00:00+00:00",
               items=[{"name": "Biofield Analysis", "qty": 1}])
    _add_order(cx, source="reorder", ref="r1", created_at="2026-06-05T00:00:00+00:00",
               items=[{"name": "FF-A", "qty": 1}])
    # member price > regular -> clamp the line to 0, not negative.
    def inverted(item, order):
        return (5000, 6997, int(item.get("qty") or 1)) if (item.get("name") or "").lower().startswith("ff-") else (0, 0, 0)
    assert tc.accrued_credit_cents(cx, EMAIL, price_line=inverted) == 0


def test_window_boundary_is_inclusive():
    cx = _cx()
    _add_order(cx, source="biofield_trial", ref="bt1",
               created_at="2026-06-01T00:00:00+00:00",
               items=[{"name": "Biofield Analysis", "qty": 1}])
    # exactly 31 days later (2026-07-02) -> still inside [start, start+31d]; this is
    # the day-30-to-31 conversion gap the window was widened to cover.
    _add_order(cx, source="reorder", ref="r1", created_at="2026-07-02T00:00:00+00:00",
               items=[{"name": "FF-A", "qty": 1}])
    assert tc.accrued_credit_cents(cx, EMAIL, price_line=_fake_price_line) == 1997


def test_window_uses_earliest_trial_order():
    cx = _cx()
    # Two trial orders; the earliest defines the window start.
    _add_order(cx, source="biofield_trial", ref="bt-late",
               created_at="2026-06-20T00:00:00+00:00",
               items=[{"name": "Biofield Analysis", "qty": 1}])
    _add_order(cx, source="biofield_trial", ref="bt-early",
               created_at="2026-06-01T00:00:00+00:00",
               items=[{"name": "Biofield Analysis", "qty": 1}])
    # reorder at day 5 from the EARLY trial -> in window.
    _add_order(cx, source="reorder", ref="r1", created_at="2026-06-06T00:00:00+00:00",
               items=[{"name": "FF-A", "qty": 1}])
    assert tc.accrued_credit_cents(cx, EMAIL, price_line=_fake_price_line) == 1997


def test_handles_naive_and_z_suffixed_timestamps():
    cx = _cx()
    _add_order(cx, source="biofield_trial", ref="bt1",
               created_at="2026-06-01T00:00:00Z",
               items=[{"name": "Biofield Analysis", "qty": 1}])
    _add_order(cx, source="reorder", ref="r1", created_at="2026-06-05 12:00:00",
               items=[{"name": "FF-A", "qty": 1}])
    assert tc.accrued_credit_cents(cx, EMAIL, price_line=_fake_price_line) == 1997


def test_email_matching_is_case_insensitive():
    cx = _cx()
    _add_order(cx, source="biofield_trial", ref="bt1",
               created_at="2026-06-01T00:00:00+00:00",
               items=[{"name": "Biofield Analysis", "qty": 1}], email="MixedCase@Example.com")
    _add_order(cx, source="reorder", ref="r1", created_at="2026-06-05T00:00:00+00:00",
               items=[{"name": "FF-A", "qty": 1}], email="MixedCase@Example.com")
    assert tc.accrued_credit_cents(cx, "mixedcase@example.com",
                                   price_line=_fake_price_line) == 1997


def test_trial_window_none_without_trial_order():
    cx = _cx()
    assert tc.trial_window(cx, EMAIL) is None


def test_credit_order_ref_normalises_email():
    assert tc.credit_order_ref("Foo@Bar.com") == "trial-credit:foo@bar.com"
    assert tc.CREDIT_REASON == "trial_upgrade_credit"
