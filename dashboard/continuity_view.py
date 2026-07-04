"""Doctor continuity tooling (C): per-patient continuity view + recommend loop.

Every per-patient read/write goes through authorized_patient() first — a doctor
may only ever touch a patient who has a CONSENTED CONTINUITY link to them.
"""
import sqlite3
from typing import Optional

from dashboard import scan_analysis as _scan
from dashboard import biofield_narrative as _narrative
from dashboard import biofield_portal_publish as _portal
from dashboard import practitioner_recommendations as _pr


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


def latest_biofield_test_id(cx, patient_email) -> str:
    """The patient's most recent authored biofield test id (e.g. "a7"), or None.

    A biofield test maps to a patient by the `email` column on `biofield_auth_tests`
    (see dashboard/biofield_authoring.py; ids are the "a"-prefixed autoincrement
    key). "Most recent" == highest id — the same creation-order ranking
    `biofield_authoring.list_authored` uses (ORDER BY t.id DESC). Tolerant of a
    connection with no biofield tables (returns None)."""
    e = (patient_email or "").strip().lower()
    if not e:
        return None
    try:
        row = cx.execute(
            "SELECT id FROM biofield_auth_tests WHERE lower(email)=? "
            "ORDER BY id DESC LIMIT 1",
            (e,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return ("a" + str(row[0])) if row else None


def _member_price_cents(cx, test_id) -> int:
    """The special (member) price the doctor already published this patient's
    report at, in cents; 0 when never published / the table is absent. Reused as
    the reorder price rather than recomputed — it is exactly what this patient's
    portal charges them."""
    try:
        row = cx.execute(
            "SELECT special_price_cents FROM biofield_portal_published WHERE test_id=?",
            (str(test_id),),
        ).fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0]) if row and row[0] else 0


def patient_view(cx, practitioner_id, patient_email):
    """Gate-first per-patient continuity view for a doctor.

    SECURITY KEYSTONE: calls authorized_patient() FIRST and returns None for a
    patient who is not the doctor's — BEFORE reading any patient data (no scan
    trajectory, no biofield narrative, no reorder build). On the authorized path
    it assembles, by REUSING the existing engines:
      - trajectory      : scan_analysis.get(cx, email)  (the longitudinal artifact)
      - narrative       : biofield_narrative.get_narrative for the latest test
                          (the latest-vs-prior "what changed" read)
      - suggested_step  : biofield_portal_publish.build_portal_content(...)
                          ["content"]["reorder_items"] for the latest test

    Degrades gracefully: a patient with no scans / no biofield test yet yields an
    empty trajectory / empty suggested_step rather than crashing."""
    if not authorized_patient(cx, practitioner_id, patient_email):
        return None
    # --- authorized past this line; ONLY now may we read patient data ---
    trajectory = _scan.get(cx, patient_email) or {}
    test_id = latest_biofield_test_id(cx, patient_email)
    if test_id:
        narrative = _narrative.get_narrative(cx, test_id) or ""
        portal = _portal.build_portal_content(
            cx, test_id, special_price_cents=_member_price_cents(cx, test_id))
        suggested_step = (portal.get("content") or {}).get("reorder_items") or []
    else:
        narrative = ""
        suggested_step = []
    return {
        "trajectory": trajectory,
        "narrative": narrative,
        "suggested_step": suggested_step,
    }


def _item_label(item) -> str:
    if isinstance(item, dict):
        return str(item.get("name") or item.get("slug") or item)
    return str(item)


def _notify_patient(cx, practitioner_id, patient_email, items, note, *, send=None) -> None:
    """Best-effort 'your practitioner has a recommendation for you' email.

    Not wired to biofield_comms/recent_comms — those are read-only comms
    AGGREGATORS (context builders for the intake/balancing loop), not senders.
    The actual reusable send transport in this codebase is
    dashboard.inbox.send_email, exactly as dashboard/cert_notify.py already
    reuses it for other patient/member notifications (same `send=` injection
    pattern for testability). Caller wraps this in try/except; we don't here so
    tests can also call it directly and see failures surface if desired."""
    from dashboard import inbox as _inbox
    send = send or _inbox.send_email
    email = (patient_email or "").strip()
    if not email:
        return
    lines = "\n".join(f"- {_item_label(i)}" for i in (items or []))
    body = (
        "Hi,\n\nYour practitioner has a new recommendation for you"
        + (":\n\n" + lines if lines else ".")
        + ("\n\n" + note if note else "")
        + "\n\nOpen your portal to see the full details.\n\nIn wellness,\nDr. Glen\n"
    )
    send(email, "A new recommendation from your practitioner", body,
         from_name="Dr. Glen Swartwout")


def send_recommendation(cx, practitioner_id, patient_email, items, note) -> Optional[int]:
    """Gate-first recommend action: writes a practitioner_recommendations row for
    a consented-continuity patient, then best-effort notifies the patient.

    SECURITY: calls authorized_patient() FIRST and returns None for a patient who
    is NOT the doctor's — BEFORE any write (no recommendation row is created).
    This is defense in depth: the route ALSO gate-checks before ever calling
    this, so an unauthorized write can't happen even if a future caller forgets
    the route-level check.

    Notification is best-effort: any comms failure is caught and logged, never
    raised — a broken email transport must never fail the recommend."""
    if not authorized_patient(cx, practitioner_id, patient_email):
        return None
    rec_id = _pr.create(
        cx, practitioner_id=practitioner_id, patient_email=patient_email,
        items=items, note=note,
    )
    try:
        _notify_patient(cx, practitioner_id, patient_email, items, note)
    except Exception as e:  # noqa: BLE001 - notification never blocks the recommend
        print(f"[continuity_view] recommend notify failed for {patient_email!r}: {e!r}",
              flush=True)
    return rec_id
