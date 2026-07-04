"""Doctor continuity tooling (C): per-patient continuity view + recommend loop.

Every per-patient read/write goes through authorized_patient() first — a doctor
may only ever touch a patient who has a CONSENTED CONTINUITY link to them.
"""


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
