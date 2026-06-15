# Biofield Checkout + Readiness Gate — Design (Upgrade Ladder Phase 2 / Mechanic 4)

**Date:** 2026-06-15
**Status:** Design approved in brainstorm (Glen). Spec for review → plan → build.
**Related:** [[tasks_upgrade_incentive_ladder]] (Mechanic 4 = points → Biofield), [[project_pricing_rewards_engine]] (points ledger + engine redemption path), the merged Mechanic 1 group-bundle, E4L voice scans, Practice Better intake.

## Problem

"Points → Biofield" can't reuse the product redemption path because the $300 Biofield isn't an in-app purchase — it's an external consultation booking (Truly.VIP/Biofield → Practice Better). And the real challenge: after payment (our checkout **or** Practice Better), we cannot run the Biofield analysis / program design / report until we have three things — **the client's photo, a completed intake form, and a fresh voice scan** — so booking the Zoom consult must be **gated** on those, then a ~48-hour fulfillment window starts.

## Decisions (locked in brainstorm)

- **Sell Biofield as a $300 checkout item in our funnel** (Option A), points-redeemable via the existing engine path (`points_to_redeem_cents`). Flow becomes pay-then-book. Practice Better remains a valid alternate payment path.
- **Post-payment readiness gate** with **hybrid confirmation**: auto-check what we can read server-side, otherwise the customer provides it in the gate. Booking unlocks only when all three items are green.
- **Three gate items** (each: green if confirmed on file, else an action):
  - **Photo** — green if a photo is already on file for the email; else **upload in the gate**.
  - **Intake** — green if present in `inbound_leads` (source practice-better/scoreapp/concierge); else a "complete your intake" link + self-confirm.
  - **Fresh voice scan** — green if a scan **within 7 days** is detectable (E4L, best-effort); else a scan link (Truly.VIP/E4L) + self-confirm.
- **Payment paths:** our Stripe checkout (clean trigger → gate). A **Practice Better payer** reaches the same gate via a magic link and confirms payment by uploading the PB receipt (the Studio.com receipt-upload pattern) or Rae marks them paid. No Practice Better integration required.
- **Booking unlock:** when all three are green (and payment confirmed), the gate reveals a **configurable booking link** (PB scheduler / Zoom). On booking, drop a **team task** ("Biofield prep due in 48h for X") so fulfillment doesn't slip.

## Architecture / components

### 1. Biofield checkout item + points redemption
- A `$300` Biofield item purchasable in the funnel (a SKU/line, `info_only=false`, not a Pure Powder). Reuse the existing card checkout + `_price_cart`/engine so `points_to_redeem_cents` already applies (points → Biofield, the Mechanic 4 ask). Points floor still protects the price.
- On payment success (checkout-return), mark a **Biofield order** + create/seed the readiness record for the email, then redirect to the readiness gate.

### 2. Readiness gate (`/biofield/ready` + `/api/biofield/ready`)
- Magic-link / member-session identified (reuse the reorder/coaching magic-link pattern so it's reachable post-payment and by PB payers).
- `GET /api/biofield/ready` returns the three item states for the email: `{photo, intake, scan, paid, booking_unlocked, booking_url?}` where each item is `{status: 'green'|'needed', action}`.
  - **photo:** green if a stored photo-on-file flag is set for the email; else `needed` → upload.
  - **intake:** green if `inbound_leads` has an intake row (practice-better/scoreapp/concierge); else `needed` → intake link + self-confirm.
  - **scan:** green if a scan within 7 days is detectable (best-effort E4L lookup; if E4L is not reachable from the server, fall back to self-confirm); else `needed` → scan link + self-confirm.
- Actions: `POST /api/biofield/photo` (upload → store + set photo-on-file), `POST /api/biofield/confirm` (self-confirm intake/scan/payment), all consent-gated.
- When `paid && photo && intake && scan` → `booking_unlocked=true`, return the booking link.

### 3. Practice Better payment entry
- `paid` becomes true via: (a) our Stripe Biofield checkout, or (b) a PB-receipt upload confirmed (reuse the chat OCR upload), or (c) Rae marks paid in the console. A magic-link email gets a PB payer into the gate.

### 4. Booking + 48h fulfillment
- On the customer hitting "Book" (booking link revealed), record `booked_at` and create a `todos` task for Glen's team: "Biofield prep due 48h — <email>" (reuse the BOS `todos`/`append_event` spine). Optionally kick the existing `dr-glen-swartwout-e4l-scan-remedy-matcher` agent to draft the analysis (deferred).

## Data / storage
- A `biofield_readiness` table (sqlite, DATA_DIR): `email`, `paid_at`, `paid_via`, `photo_on_file`, `intake_confirmed`, `scan_confirmed`, `booked_at`, `order_ref`, timestamps. Auto-checks (intake/scan) are computed live; self-confirms + photo persist here.
- **PHI handling (important):** client photos and scans are health data. The chat attachment infra deliberately does NOT persist image bytes; the Biofield photo DOES need to persist for fulfillment, so store it deliberately and access-controlled (DATA_DIR private path or object storage; never in a public/static dir), and record only a reference + the on-file flag. Flag for review before go-live.

## Phasing
- **Phase 2a (build):** the $300 Biofield checkout + points redemption + the readiness gate (hybrid, 3 items) + booking unlock + the 48h team task. Self-contained, ships the whole flow for the Stripe-paid path with PB-receipt/self-attest for the PB path.
- **Phase 2b (later):** tighter auto-verification (a real E4L scan-freshness API/db read on the server; Practice Better intake/photo/payment API), and auto-drafting the analysis via the matcher agent.

## Flags
Behind `BIOFIELD_CHECKOUT_ENABLED` (default off) — the funnel item, the gate, and the routes ship dark until the flow + PHI storage are reviewed.

## Open items / deferred
- **E4L server reachability** for the scan-freshness auto-check — confirm whether the deployed funnel can query E4L (`e4l.db` is likely local; portal.e4l.com API?). If not, the scan item is self-confirm in 2a; auto in 2b.
- Exact **photo spec** (face? eyes? how many) + upload instructions — Glen to specify; the gate upload supports one or more images.
- The **booking link/tool** (PB scheduler vs Zoom vs Calendly) — a configurable URL for now.
- Whether intake/photo are truly required for **every** Biofield or only some.
- Refunds, expiry of an un-booked paid Biofield, re-scan reminders.

## Testing
- Checkout: Biofield item priced at $300; `points_to_redeem_cents` reduces it (floor-protected); payment success seeds readiness + redirects to the gate.
- Gate: each item's green/needed logic (intake present vs absent; scan within/over 7 days or self-confirm; photo on-file vs upload); booking unlocks only when all green + paid; PB-receipt path flips `paid`.
- Booking: creates the 48h team task; idempotent.
- Points: redeem against the Biofield order recorded + deducted on paid (mirror the existing points settle path).
