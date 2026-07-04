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


def modules_for_practitioner(pid):
    """Live modules_completed for a practitioner id, or None if not a practitioner."""
    if not pid:
        return None
    from db_supabase import supabase_cursor
    with supabase_cursor() as cur:
        cur.execute("SELECT modules_completed FROM practitioners WHERE id=%s", (str(pid),))
        row = cur.fetchone()
    if not row:
        return None
    return int(row["modules_completed"] or 0)


def credit_for_charge(sub, *, charge_cents, earn=None, resolve_modules=None):
    """Credit the attributed doctor's cert-scaled share of one successful charge.

    Idempotent per event_ref = care_share:<sub_id>:<order_count>. Returns cents
    credited (0 when unattributed, the owner is not a practitioner, or the
    computed share is non-positive).
    """
    pid = (sub or {}).get("attributed_practitioner_id")
    if not pid:
        return 0
    resolve_modules = resolve_modules or modules_for_practitioner
    m = resolve_modules(pid)
    if m is None:
        return 0
    cents = share_cents(charge_cents, m)
    if cents <= 0:
        return 0
    if earn is None:
        from dashboard import wallet as _wallet
        earn = _wallet.earn_care_share
    event_ref = f"care_share:{sub['id']}:{sub['order_count']}"
    earn(str(pid), cents, event_ref=event_ref)
    return cents
