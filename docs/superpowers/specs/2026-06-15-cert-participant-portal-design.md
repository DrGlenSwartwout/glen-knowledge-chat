# Certification Participant Personal Portal — Build Spec

**Status:** Design (Glen approved Option B, 2026-06-15). Needs Glen's spec sign-off before build.
**Repo:** glen-knowledge-chat (deploy-chat). **Branch:** `sess/b76661d9-cert-portal`.
**Reuses:** the existing practitioner portal ([[project_practitioner_dropship_portal]]).

## Goal

Give each certification participant a personal portal — reusing the practitioner portal — where they can **order at their cert-level discount for personal use immediately**, and **activate reselling (drop-ship + wholesale) only after supplying a resale license you approve**. Licensed participants unlock reselling directly. Plus a self-serve contact-visibility toggle (deferred from the finder work).

## What already exists (reuse as-is)

- Magic-link login → 30-day session token → `/practitioner/portal` (`practitioner_portal.py`: create_magic_link_token / consume_magic_link / create_session_token / practitioner_id_from_session; `_practitioner_session_pid()` app.py:6014).
- Three pages: client/dispensary (`/dispensary/<code>`), wholesale portal (`/practitioner/portal`), drop-ship (`/practitioner/dropship`).
- Cert-level pricing floor `F = 4000 − clamp(modules,0,12)*125` and the blended volume curve (`wholesale_pricing.py`); `order_quote(items, {"modules_completed": N})`.
- Wallet (earn on drop-ship, redeem on orders), practitioner settings (branding + pricing) (`practitioner_settings.py`, `/practitioner/settings`).
- Resale-license **apply → approve** flow: `/wholesale/apply` → `application_status='pending'` → `/admin/wholesale/approve` → `decide_application()` sets `wholesale_unlocked_at` + emails the applicant.
- The 11 cohort participants are already `coach` rows, **locked** (`wholesale_unlocked_at` NULL); they did not self-register, so the current coach auto-unlock did not run on them.

## The two access levels (the model)

| | Personal ordering | Reselling (drop-ship + wholesale) |
|---|---|---|
| **Who** | every cert participant (coach), immediately | requires a **resale license** supplied + **approved** (or a professional license = licensed role) |
| **What** | order products at **cert-level price**, shipped to themselves | the drop-ship/dispensary page (sell to patients) + tax-exempt wholesale ordering |
| **Tax** | charged normally (personal purchase) | resale-exempt (existing `resale_ok` GET logic) |
| **Gate** | none beyond being a cert participant | `wholesale_unlocked_at IS NOT NULL` (set by resale approval / license) |

## Changes to build

### 1. Personal-use ordering (open to cert participants)
- New `POST /api/practitioner/personal/quote` and `POST /api/practitioner/personal/checkout`, session-gated, available to any `coach`/`panel_in_cert`/`panel_certified` (or licensed) **without** the `wholesale_unlocked` block.
- Pricing: reuse `order_quote(items, {"modules_completed": N})` (cert-level). Ships to the participant. Tax applied (not resale-exempt — these participants have no resale cert at this stage).
- Reuse the existing `wholesale_cart` table for the cart, or a parallel personal cart. (Implementer picks the lower-friction reuse; cart semantics are identical.)
- v1 sanity guard: a per-order quantity ceiling to discourage using personal orders as a resale workaround (e.g. ≤ a large-box quantity per SKU). Log if hit.

### 2. Resale activation (gates drop-ship + wholesale)
- The drop-ship routes (`/api/practitioner/dropship/*`, dispensary-code generation) and the wholesale checkout stay gated on `wholesale_unlocked`.
- **Coaches are no longer auto-unlocked.** A logged-in coach who is not unlocked sees a "Activate reselling" panel with a resale-license form (number + state). Submit → reuse the application path to set `application_status='pending'` + `resale_license_number` on their existing record (not a new row) → admin approves at the existing `/admin/wholesale` → `wholesale_unlocked_at` set → drop-ship unlocks.
- **Licensed participants:** if `portal_role='licensed'` (a professional license on file), unlock reselling directly, same as practitioners (no resale cert needed).
- Change the registration coach auto-unlock (app.py ~6076) so new coach registrations are NOT auto-unlocked (they go through resale approval). Few/no existing coaches, so safe.

### 3. Self-serve contact toggle (folds in item-1's deferred piece)
- Add `show_contact` to `GET/POST /api/practitioner/settings` and a checkbox in `practitioner-settings.html` ("Show my contact info in the public finder — default off"). Writes `practitioners.show_contact`. (Column + finder honoring already shipped in #141.)

### 4. Portal landing UI (`practitioner-portal.html`)
- Cert participant view: cert level (N/12), wallet, a **personal-order catalog** (open), and a **Reselling** section that is either locked (→ resale-license form) or unlocked (→ drop-ship link). Reuse existing portal styling.

### 5. Onboard the 11 (+ Mona)
- A console-gated `POST /api/cert/portal-invite {email}` (or a small batch script) that mints a magic link via `create_magic_link_token` and emails it via `_send_practitioner_magic_link`. Send to the cohort once the portal changes are live.

## Pricing / tax notes
- Personal orders: cert-level price + tax (reuse the checkout's existing tax computation; `resale_ok` is false for them → taxed).
- Resale-approved orders/drop-ship: `resale_ok` true → GET-exempt (existing behavior).
- Confirm with Rae/CPA before the first reseller goes live (existing `TAX_ENABLED` posture).

## Out of scope (later)
- Sophisticated anti-resale quantity policy beyond the v1 ceiling.
- White-label branding polish; client-page MAP tuning (already exists).
- Automated resale-license verification (manual approval stays).

## Open questions for Glen
1. Personal-order **quantity ceiling** — any per-order cap you want (or none in v1)?
2. Should personal orders **earn/redeem wallet credit**, or keep the wallet tied to reselling only? (Recommend: redeem allowed, earn only on reselling — matches the current wallet model.)
