# Doctor Continuity Tooling (C) — Per-Patient Continuity View + Recommend Loop (v1) — Design

**Date:** 2026-07-03
**Status:** Approved (brainstormed with Glen 2026-07-03)
**Repo:** deploy-chat
**Depends on (build after):** #565 turnkey continuity fee-share (`attributed_practitioner_id`, the Continuous Care enrollment card + agreement, `care_share`) — C queries the attribution, extends the enrollment agreement with the consent flag, and reuses the enrollment surface. Also builds on the practitioner portal (Clients tab, `dispensary_stats`), the biofield/scan engines, and the patient portal + reorder rails.

## Summary

Give a doctor the tooling to actually *co-manage* their continuity patients — the thing that justifies the fee-share's up-to-50%. For each of a doctor's **consented continuity patients**, a per-patient continuity view in the practitioner portal shows the patient's scan trajectory over time, a plain-language read of what changed, and a system-suggested next step. The doctor reviews/edits it, optionally adds a note, and pushes it to the patient's portal as a recommendation with one-tap reorder + a notification. This closes the **see → recommend → reorder** loop — the retention-and-outcomes engine the fee-share rewards.

C is mostly *assembly* of existing engines behind one hard authorization boundary — not new analysis.

## Scope

**v1 = the per-patient continuity view + the recommend loop**, one coherent flow across two surfaces (doctor portal + patient portal). Deferred to fast-follows: the **ranked triage roster** (option b — "who needs attention now"), the **portal opt-in back-fill** for patients who enrolled before the consent flag, free-form-message-only comms beyond the attached note, and any program beyond Continuous Care.

## The authorization gate (privacy keystone — trace end-to-end)

C surfaces one person's health scans to their doctor. **Every per-patient read and every recommend-write MUST verify the requesting practitioner owns a consented continuity link to that exact patient** — i.e. there exists a subscription with `attributed_practitioner_id == this practitioner` AND the consent flag set AND `kind == "membership"`, matching the patient's email. A doctor can never see another doctor's patient, a non-consented patient, or a non-continuity patient. This gate is C's load-bearing invariant (the equivalent of #561's "no global-open-total leak"); it is enforced in one place (`continuity_view.authorized_patient(practitioner_id, patient_email)`) that every route calls first, and is the most heavily tested surface.

## Consent (model A)

Add explicit authorization language + a stored consent flag to the "Start Continuous Care" enrollment agreement (the card shipped in #565, `static/practitioner-client.html`): e.g. "…and I authorize sharing my wellness results with my enrolling practitioner." The flag is stored on the membership (a new `subscriptions.practitioner_share_consent INTEGER DEFAULT 0`, set to 1 when the box is checked at dispensary enrollment). C shows only patients with the flag. Patients who enrolled before this language aren't covered — a portal opt-in back-fill is a deferred fast-follow.

## Components

1. **Continuity roster (Clients tab).** List the doctor's consented continuity patients — subscriptions where `attributed_practitioner_id == pid AND practitioner_share_consent == 1 AND kind == "membership"`, resolved to patient email/name. Each row opens the per-patient view. (No ranking in v1 — that's the deferred triage b.)

2. **Per-patient continuity view (the core — all reuse).**
   - **Trajectory:** the patient's scan/biofield reads over time. Reuse `dashboard/scan_analysis.py` (`get(cx, email)` — per-member scan artifact) + `dashboard/biofield_profile.py` to render the trend of key dimensions across the patient's scan history.
   - **What changed:** a plain-language latest-vs-prior read. Reuse `dashboard/biofield_narrative.py` / `biofield_interpret.py`.
   - **Suggested next step:** a recommended remedy/protocol derived from the latest scan. Reuse the biofield analysis's recommended-remedy path (`dashboard/biofield_portal_publish.py` — `resolve_remedy_slug`/`_dosing`/`segment_narrative` already turn a scan into recommended remedies with dosing/pricing).

3. **Doctor action (i + ii).** The doctor reviews/edits the suggested step (add/remove remedies, adjust), optionally adds a free-form note, and clicks "Recommend to patient." Nothing auto-sends; the doctor is the clinical authority.

4. **Landing (X).** The push writes a `practitioner_recommendation` record that surfaces in the patient's portal as "Your practitioner recommends X" with **one-tap add-to-cart at member pricing** (reuse the patient portal + `dashboard/reorder.py`/`reorder_actions.py`), plus a notification/email (reuse `dashboard/biofield_comms.py`/`recent_comms.py`). The $99/mo is the *service*; the recommended remedy is a separate product buy at member pricing — "add to cart," not a silent charge. The patient chooses to buy. A `portal_triage.add_item(...)`-style record may back the patient-facing surface.

## Data model

- `subscriptions.practitioner_share_consent INTEGER NOT NULL DEFAULT 0` (guarded ALTER, per the `attributed_practitioner_id` precedent). Set at dispensary Continuous Care enrollment when the consent box is checked.
- New `practitioner_recommendations` table: `id, practitioner_id, patient_email, items_json, note, status ('sent'|'accepted'|'dismissed'), created_at`. Idempotent/append; the patient portal reads the latest active one.
- New `dashboard/continuity_view.py`: `authorized_patient(practitioner_id, patient_email) -> bool` (the gate); `roster(practitioner_id) -> [patient]`; `patient_view(practitioner_id, patient_email) -> {trajectory, narrative, suggested_step}` (calls the gate first); `send_recommendation(practitioner_id, patient_email, items, note) -> id` (gate first).

## Integration points

- **Practitioner portal:** Clients tab gains the roster + a per-patient view route (`GET /api/practitioner/continuity/<patient_email>` gate-checked) and a recommend route (`POST /api/practitioner/continuity/recommend`). Practitioner identity via the existing `_practitioner_session_pid()`.
- **Patient portal:** render the active `practitioner_recommendation` with one-tap add-to-cart; the add-to-cart uses the patient's existing reorder/member-pricing path.
- **Enrollment (from #565):** the consent checkbox writes `practitioner_share_consent` on `create_membership`.

## Testing

- **Authorization gate (highest priority):** `authorized_patient` returns True only for a consented continuity link; a doctor requesting another doctor's patient, a non-consented patient, or a non-continuity patient → False, and every route returns 403 (no data leak). Fuzz the cross-practitioner case.
- Roster returns only consented continuity patients for the requesting doctor.
- Per-patient view assembles trajectory + narrative + suggested step from the real engines for an authorized patient; 403 for an unauthorized one.
- Recommend writes a `practitioner_recommendation`, surfaces it in the patient portal, notifies, and the add-to-cart lands at member pricing.
- Consent flag set at enrollment when checked; absent otherwise; C excludes non-consented.

## Out of scope / Future (fast-follows)

- **Triage roster (b):** rank continuity patients by who needs attention (trending worse, scan overdue, adherence slipping).
- **Portal opt-in back-fill** for patients enrolled before the consent flag.
- **Doctor-authored free-form messages** beyond the recommendation note; two-way patient↔doctor messaging.
- **Programs beyond Continuous Care.**
- **Branded patient experience (D).**
