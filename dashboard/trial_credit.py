# dashboard/trial_credit.py
"""Trial-upgrade credit engine (PR2).

A $1 Biofield-trial buyer pays REGULAR price on remedy orders — the quantity/
volume discount is a paid-member perk (see reference_member_gated_qty_pricing /
_is_paid_member). To make conversion attractive we ACCRUE the member discount the
trial buyer left on the table over the trial window (31 days from the trial
purchase — matches the 31-day biofield-trial access grant, so the whole
pre-conversion period is covered, charge fires ~day 31) and, when they convert to
a full paid membership (their first $99 charge clears), hand that amount back as
loyalty POINTS (dashboard.points.credit) that auto-apply to their next remedy
order.

This is a PURE module: a sqlite connection is passed in; no Flask, no Stripe. The
member-vs-regular pricing lives in app.py (the catalog + _qty_unit_cents), so the
caller passes a `price_line` callback and the windowing + summation math stays
here, fully unit-testable.
"""
import json
from datetime import datetime, timedelta, timezone

# 31 days = the biofield-trial access grant length (the first $99 charge / conversion
# fires at ~day 31), so accrual covers the entire pre-conversion trial period.
WINDOW_DAYS = 31
# Ledger reason + deterministic order_ref make the conversion grant idempotent
# (dashboard.points.credit -> has_entry de-dupes on (order_ref, reason)).
CREDIT_REASON = "trial_upgrade_credit"


def credit_order_ref(email: str) -> str:
    """Deterministic points-ledger ref for a buyer's trial-upgrade credit. One per
    buyer, so a cron re-run can never double-credit."""
    return f"trial-credit:{(email or '').strip().lower()}"


def _parse_dt(s):
    """Parse a stored ISO timestamp into a naive UTC datetime (so values written
    tz-aware, with a 'Z' suffix, or naive all compare cleanly). Returns None on
    anything unparseable."""
    if not s:
        return None
    raw = str(s).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        try:
            dt = datetime.fromisoformat(raw[:10])  # date-only fallback
        except ValueError:
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def trial_window(cx, email, *, window_days=WINDOW_DAYS):
    """Return (start, end) naive-UTC datetimes for the buyer's trial window, or
    None if they have no biofield_trial order. The EARLIEST trial order defines
    the window start (the trial actually began then)."""
    em = (email or "").strip().lower()
    if not em:
        return None
    row = cx.execute(
        "SELECT created_at FROM orders WHERE lower(email)=? AND source='biofield_trial' "
        "ORDER BY created_at ASC LIMIT 1",
        (em,),
    ).fetchone()
    if not row:
        return None
    start = _parse_dt(row[0])
    if start is None:
        return None
    return (start, start + timedelta(days=window_days))


def accrued_credit_cents(cx, email, *, price_line, window_days=WINDOW_DAYS) -> int:
    """Sum over the buyer's in-window orders of max(0, regular - member) * qty for
    each volume-eligible line.

    Returns 0 when the buyer has no biofield_trial order (no window to accrue in).

    price_line(item, order) -> (regular_unit_cents, member_unit_cents, qty)
      where `item` is one parsed line dict and `order` is the parsed order dict
      ({"created_at", "source", "items": [...]}); return (0, 0, 0) for a
      non-eligible / unresolvable line.
    """
    win = trial_window(cx, email, window_days=window_days)
    if win is None:
        return 0
    start, end = win
    em = (email or "").strip().lower()

    rows = cx.execute(
        "SELECT created_at, source, items_json FROM orders WHERE lower(email)=? "
        "ORDER BY created_at ASC",
        (em,),
    ).fetchall()

    total = 0
    for created_at, source, items_json in rows:
        cdt = _parse_dt(created_at)
        if cdt is None or cdt < start or cdt > end:
            continue
        try:
            items = json.loads(items_json) if items_json else []
        except (ValueError, TypeError):
            continue
        if not isinstance(items, list):
            continue
        order = {"created_at": created_at, "source": source, "items": items}
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                regular, member, qty = price_line(item, order)
            except Exception:
                continue
            total += max(0, int(regular) - int(member)) * int(qty)
    return max(0, total)
