"""Readiness-gate state for the Biofield checkout flow (pure: cx + injected lookups)."""
from dashboard import biofield_store as _store


def gate_state(cx, email, *, has_intake, scan_window_days=7):
    """Compute the readiness gate state for `email`.
    has_intake(email) -> bool: an injected auto-check (e.g. an inbound_leads lookup).
    Returns {paid, booked, items:{photo,intake,scan -> {status}}, booking_unlocked}."""
    row = _store.get(cx, email) or {}
    paid = bool(row.get("paid_at"))
    booked = bool(row.get("booked_at"))

    photo = bool(row.get("photo_on_file"))
    intake = bool(row.get("intake_confirmed")) or bool(has_intake(email))
    scan = bool(row.get("scan_confirmed"))

    def item(ok):
        return {"status": "green" if ok else "needed"}

    items = {"photo": item(photo), "intake": item(intake), "scan": item(scan)}
    booking_unlocked = bool(paid and photo and intake and scan)
    return {"paid": paid, "booked": booked, "items": items,
            "booking_unlocked": booking_unlocked}
