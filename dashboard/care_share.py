"""Cert-scaled recurring fee-share for turnkey continuity (Continuous Care).

Pure math (rate/share) + the per-charge credit orchestration. A doctor who
enrolled a patient earns rate(modules_completed) of every successful membership
charge, credited to their wallet, for as long as the patient stays.
"""

_BASE_RATE = 0.30
_MAX_RATE = 0.50
_MAX_MODULES = 12


def rate(modules_completed):
    """Fee-share fraction, linear in completed cert modules: 0.30 at 0 -> 0.50 at 12."""
    m = max(0, min(_MAX_MODULES, int(modules_completed or 0)))
    return _BASE_RATE + m * ((_MAX_RATE - _BASE_RATE) / _MAX_MODULES)


def share_cents(charge_cents, modules_completed):
    """Doctor's share of one charge, in integer cents (banker-free round-half-up-ish)."""
    return int(round(max(0, int(charge_cents or 0)) * rate(modules_completed)))
