# Cross-app client navigation — design (teed up 2026-07-08)

**Status:** spec ready to build. Origin: the per-client workflow spans TWO apps — the local Intake (`127.0.0.1:8011`) and the prod portal composer (`illtowell.com/console/biofield-portal`) — and navigation only flows one way, so the operator keeps landing on the wrong app and losing the client.

## Goal

A **client-carrying round-trip** across the two apps so the operator moves through Edit → Report → Invoice (local) ⇄ Publish portal (composer) ⇄ back, without retyping URLs or re-finding the client.

## Current state

- **Local Intake `:8011`** (`biofield_local_app.py`, `dashboard/biofield_report_html._client_tabs`): every client page (`/author/<id>`, `/test/<id>`, `/author/<id>/invoice-view`) carries the **Edit · Report · Invoice · Portal** strip. The **Portal** tab already links OUT to `illtowell.com/console/biofield-portal?email=<email>` (local → composer works).
- **Prod composer** (`static/console-biofield-portal.html`): has the console-section nav (Biofield/Reveals/Intake/Tags) but **no link back** to the local Intake editor for the loaded client. Dead end.

## The build

### 1. Local "by-email" entry route (`biofield_local_app.py`)

`GET /by-email/<email>` → look up the client's authored test (`biofield_auth_tests` by email; if several, newest by id) → `redirect(302)` to `/author/a<id>`. If none found, render the home list filtered/scrolled to that email (or a "no authored test for <email> — create one?" prompt). This is the deep-link target the composer button uses; keeps the composer ignorant of local `a<id>` ids (it only knows email).

### 2. Composer "← Edit in Intake" button (`static/console-biofield-portal.html`)

When an email is loaded, render a button/link → `http://127.0.0.1:8011/by-email/<email>` (new tab). Optimistic (prod can't know whether a local test exists; the local route handles "not found" gracefully).

### 3. Relabel the local Portal tab (`_client_tabs`)

`Portal` → **`Publish portal →`** so it reads as the handoff (it already points at the composer), not just a view.

## The localhost wrinkle (important)

A prod page linking to `127.0.0.1:8011` **only works on the operator's Mac** — it's a dead link for Rae or any other console user. So the "← Edit in Intake" button must be **operator-scoped**:
- Gate it behind a flag (e.g. `LOCAL_INTAKE_LINK_ENABLED`) OR behind the owner role (`_bos_actor().role == OWNER`), so it never renders for non-operators.
- Label it clearly as a local-only tool.

## Open decisions (resolve at build)

1. **Gating** — flag vs owner-role for the composer button. Recommend **owner-role** (auto-correct, no flag to manage). Confirm.
2. **Multiple authored tests for one email** — newest wins (recommend) vs a small chooser. Recommend newest.
3. **No local test found** — home filtered to the email vs a "create authored test" prompt. Recommend home-filtered for v1.
4. **Port/host** — hardcode `127.0.0.1:8011` vs a configurable base. Recommend hardcode for v1 (it's the fixed local server).

## Testing

- `/by-email/<known email>` → 302 to `/author/a<id>`; unknown email → home (no crash).
- Composer renders the "← Edit in Intake" button for the owner only (not for a non-owner actor).
- `_client_tabs` Portal tab now reads "Publish portal →" and still targets the composer with `?email=`.

## Out of scope (v1)

- Single-sign-through / auto-auth across the two apps (each already carries its own key).
- Exposing the localhost link to non-operator console users.
- Deep-linking to a specific tab/scan within the composer beyond `?email=`.
