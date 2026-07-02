"""Pure logic for the program-bundled live-group offer (Mechanic 1).
1 free live-group month per program month purchased, capped at 3 (the max
recommended program length); 0 for a Biofield-only / no-program purchase."""

from dashboard import prepay as _prepay

MEMBERSHIP_AMOUNT_CENTS = _prepay.MONTHLY_ANCHOR_CENTS   # $99/mo founders rate (live group)
MEMBERSHIP_CADENCE_MONTHS = 1
MAX_INCLUDED_MONTHS = 3


def included_group_months(program_months) -> int:
    try:
        m = int(program_months or 0)
    except (TypeError, ValueError):
        return 0
    if m <= 0:
        return 0
    return min(m, MAX_INCLUDED_MONTHS)
