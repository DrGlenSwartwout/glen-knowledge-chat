"""Doctor continuity tooling (C): per-patient continuity view + recommend loop.

Every per-patient read/write goes through authorized_patient() first — a doctor
may only ever touch a patient who has a CONSENTED CONTINUITY link to them.
"""
import sqlite3


def authorized_patient(cx, practitioner_id, patient_email) -> bool:
    """True iff patient_email has an active-consented Continuous Care membership
    attributed to practitioner_id. The single access boundary for all of C."""
    if not practitioner_id or not patient_email:
        return False
    row = cx.execute(
        "SELECT 1 FROM subscriptions WHERE lower(email)=lower(?) "
        "AND attributed_practitioner_id=? AND practitioner_share_consent=1 "
        "AND kind='membership' AND status != 'cancelled' LIMIT 1",
        ((patient_email or "").strip(), str(practitioner_id)),
    ).fetchone()
    return row is not None


def roster(cx, practitioner_id) -> list:
    """This doctor's consented continuity patients — one row per DISTINCT patient
    email. Uses the EXACT SAME predicate as authorized_patient() (attribution +
    consent + membership kind + not-cancelled) so the roster and the gate can
    never disagree: every patient listed here passes the gate, and vice-versa."""
    if not practitioner_id:
        return []
    rows = cx.execute(
        "SELECT DISTINCT lower(email) FROM subscriptions "
        "WHERE attributed_practitioner_id=? AND practitioner_share_consent=1 "
        "AND kind='membership' AND status != 'cancelled'",
        (str(practitioner_id),),
    ).fetchall()
    return [{"email": r[0], "name": _display_name(cx, r[0])} for r in rows]


def _display_name(cx, email) -> str:
    """Best-effort display name for a patient email: the `people` record's name
    if one is on file, else the email's local-part. Simple lookup, no joins —
    tolerates a database with no `people` table (e.g. a bare subscriptions-only
    test connection)."""
    try:
        row = cx.execute(
            "SELECT name FROM people WHERE lower(email)=lower(?) LIMIT 1", (email,)
        ).fetchone()
        if row and row[0]:
            return row[0]
    except sqlite3.OperationalError:
        pass
    return (email or "").split("@")[0]
