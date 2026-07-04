# Branded Patient Experience (D) — Co-Brand the Patient Portal (v1) — Design

**Date:** 2026-07-04
**Status:** Approved (brainstormed with Glen 2026-07-04)
**Repo:** deploy-chat
**Last piece of the practitioner-partnership arc (A–D).** Builds on #565/#575/#576 (attribution) + #572 (continuity C). Reuses `dashboard/practitioner_settings.py` (the existing white-label branding store) and `_last_attributed_practitioner` (#576).

## Summary

The patient portal (`client-portal.html`) is generic Remedy Match today — no sign of the local doctor who is caring for the patient. This surfaces the patient's **attributed doctor's identity** (photo, name, practice name, accent) as a "Your practitioner" band at the top of their portal, so the patient plainly experiences *their local doctor, powered by Remedy Match*. **Co-brand, not white-label:** Remedy Match stays the platform chrome (it's the merchant/fulfiller the patient bought from); the doctor's identity is added prominently, not substituted.

Everything needed already exists — the branding store (`practitioner_settings.branding_json`, prod-readable in `LOG_DB`, with a doctor-facing settings page) and the "patient → their doctor" lookup (`_last_attributed_practitioner`). This wires them into the patient portal.

## Scope

**v1 = the patient portal co-brand band only.** Deferred: the recommendation card + comms/email co-branding, full white-label (doctor's colors/logo replacing RM, practice as the title), and a per-practice depth toggle.

## The mechanic

- **Payload** (`api_client_portal(token)`, app.py ~13773): resolve the patient's email from the portal record → `_last_attributed_practitioner(email)` → pid; if a pid, read that doctor's branding from `practitioner_settings.get_settings(cx, pid)["branding"]` (best-effort, mirroring the existing doctor-portal read at app.py:11459-11470) **and** the doctor's display name from their practitioner record. Include a `practitioner_brand` object in the payload:
  ```
  {"name": <doctor name>, "practice_name": ..., "photo_url": ..., "logo_url": ..., "accent": <brand color>}
  ```
  Only present when the patient has an attributed doctor AND that doctor has non-empty branding. Best-effort — a lookup/read failure must never crash the portal (it just omits the band).
- **Render** (`client-portal.html`): when `practitioner_brand` is present, render a "Your practitioner" band at the top — the doctor's **photo + name + practice name** (e.g. "Your continuity care is guided by Dr. X — [Practice]"), with the doctor's **accent color applied only to that band** (a border/heading tint), NOT a full re-theme. Absent → the portal renders exactly as today (generic Remedy Match). Vanilla JS, matching the file's style.

## Attribution-only — no consent gate

The band is the doctor's **public identity** (photo/name/practice), not patient health data — so it is NOT gated on C's `practitioner_share_consent`. A patient attributed to a doctor sees "your practitioner" whether or not they consented to share their scans. C's tooling gate (which governs the *doctor's* access to *patient* data) is separate and unchanged.

## Data flow

`client-portal.html` GET `/api/portal/<token>` → `api_client_portal` resolves patient email → `_last_attributed_practitioner(email)` (most-recent attributed doctor, #576) → `practitioner_settings.get_settings(LOG_DB, pid)["branding"]` + the doctor's name (practitioner record by pid) → `payload["practitioner_brand"]` → the render band.

## Components / files

- **`app.py`** — `api_client_portal`: add the `practitioner_brand` resolution + payload key (best-effort). A small helper `_patient_practitioner_brand(cx, email) -> dict | None` keeps it isolated and testable (resolves pid → branding + name; returns None when no attributed doctor / no branding).
- **`static/client-portal.html`** — the "Your practitioner" band, rendered from `d.practitioner_brand`.

No schema changes; reuses `practitioner_settings` + `_last_attributed_practitioner`.

## The doctor's name

The doctor's display name comes from their practitioner record (by pid), resolved the same way other code resolves a practitioner (best-effort). If the name is unavailable, the band falls back to `practice_name` alone. `practice_name`, `photo_url`, `logo_url`, `accent` all come from `branding_json`.

## Testing

- **Payload:** an attributed patient whose doctor has branding → `payload["practitioner_brand"]` carries name/practice_name/photo_url/accent. A patient with NO attributed doctor → no `practitioner_brand`. An attributed patient whose doctor has EMPTY branding → no band (or None). A branding-read/lookup failure → payload still returns (band omitted), portal not crashed.
- **No consent coupling:** an attributed patient with `practitioner_share_consent=0` STILL gets `practitioner_brand` (branding is public identity, independent of results-sharing consent).
- **Render:** the band shows photo/name/practice when `practitioner_brand` is present; the portal is byte-unchanged (no band) when it's absent; the accent is applied only to the band, not the global theme.
- **No patient-data leak:** the band exposes only the doctor's public identity (photo/name/practice), never patient scans or another patient's data.

## Out of scope / Future

- Recommendation-card + comms/email co-branding (next surfaces).
- **Full white-label** (practice as the page brand, RM to a "powered by" footer) as a per-practice depth toggle.
- A patient-facing "contact your practitioner" action.
