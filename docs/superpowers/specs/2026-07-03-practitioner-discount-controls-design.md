# Practitioner Discount Controls — Design

**Date:** 2026-07-03
**Status:** Approved (brainstormed with Glen 2026-07-03)
**Repo:** deploy-chat
**Builds on:** `2026-07-02-discount-controls-design.md` (the 3-type console engine, PR #506/#523), `dashboard/pricing.py`, practitioner portal (PR #545), cohort/per-client pricing (PR #505–#516).

## Summary

Push control of the three product-discount types down from owner-only (`/console/pricing-settings`) to **each practitioner**, for their own portal's clients, bounded by ranges the owner already sets globally. A practitioner opts each type in/out and sets its amount, always clamped so it can never exceed our current global curve. A practitioner's clients price against *that practitioner's* config — it replaces the global config for them, it does not stack on top.

This lets a practitioner tune the discounts their own patients receive (trading their margin for depth) without any owner intervention, and without ever undercutting the direct channel or the public store at remedymatch.com.

## The three types and their ceilings

The engine's three types (see `dashboard/pricing.py`: `same_sku_pct`, `program_total_pct`, `open_total_pct`) are unchanged. Each gets a per-practitioner **ceiling equal to our live global ramp** for that type:

| Type | Practitioner ceiling (live global ramp) | Notes |
|------|------------------------------------------|-------|
| same-SKU | our `same_sku` ramp (currently `[[1,0],[12,29]]`) | per-line, same product |
| program-total | our `program_total` ramp (currently `[[1,0],[18,29]]`) | order-total, gated on our program membership |
| open-total | our **`program_total`** ramp (currently `[[1,0],[18,29]]`) | order-total, mix-and-match; ceiling is deliberately program-total's curve, NOT our global open-total (which is 0/off) |

**Why open-total's ceiling is program-total's curve, not 0:** we keep global open-total OFF because it conflicts with MAP on the public GrooveKart store at remedymatch.com. A practitioner's portal is a **private, gated channel** (their own clients, not the public web), so that conflict does not bind there. Glen's call: let practitioners open it up to the same depth as program-total.

**Ceilings are dynamic.** They track the live global `discounts` block in `pricing_settings`. If the owner retunes a global curve in `/console/pricing-settings`, every practitioner's ceiling for that type moves with it automatically. There is no separate admin surface to maintain, and a practitioner can never exceed the current global curve.

**Clamp mechanic:** pointwise. At every quantity `q`, the practitioner's effective discount for a type = `min(practitioner_ramp(q), global_ramp(q))`. A practitioner sets a single dial per type (how much of the available discount to pass through, 0 → the ceiling); the dial produces a ramp that is clamped to the global ramp before use. A disabled (opted-out) type contributes 0.

## Two schedules per practitioner

Each practitioner has:

- **Standard schedule** — three dials (0 → ceiling, each with an on/off toggle). Applies to **all** their patients by default.
- **Program schedule** (optional, one master toggle) — a second, deeper set of the same three dials, same ceilings. Applies **only to patients in a paid program**.

Leave the Program schedule off → every patient gets Standard (one price for all). Turn it on → program patients get the deeper rate; everyone else stays on Standard. Both schedules clamp to the same global ceilings, so neither can exceed our curves.

**What counts as "in a paid program" (v1):** our existing paid-program / membership signal — the same `program_member` flag that already gates `program_total` (`_is_paid_member(email)`). No new enrollment surface in v1.

- Consequence, accepted by Glen: because `program_total` is itself gated on our program, the Program tier as built keys on **our** programs. A practitioner running their *own* independent program is **out of scope for v1** (see Future).

## Resolution: replace, not stack

A practitioner's client resolves pricing against **that practitioner's** effective settings, not the global settings in parallel. Concretely: build a practitioner-effective `settings` object (global `settings` with its `discounts` block replaced by the practitioner's clamped, schedule-selected anchors + toggles) and pass it to the existing `pricing.compute(...)`. Everything downstream — the non-additive `line_pct = max(same_sku, program/open, subscriber/coupon)` — is unchanged.

- Direct-channel (non-practitioner) clients are unaffected: no practitioner context → global `settings` as today.
- The clamp guarantees a practitioner's client never beats our own direct curve, protecting the direct channel and MAP.

## Data & plumbing

- **New table `practitioner_pricing`** (`dashboard/practitioner_pricing.py`): `practitioner_id` → `{standard: {same_sku, program_total, open_total}, program: {enabled, same_sku, program_total, open_total}}`, each type = `{enabled, dial}` where `dial` expresses pass-through (0 → ceiling). Store the dial, derive the clamped ramp at read time against the live global anchors (keeps ceilings dynamic).
- **Effective-settings builder** (`practitioner_pricing.effective_settings(practitioner_id, program_member, global_settings)`): returns a `settings` dict with a clamped, schedule-selected `discounts` block. Pure, unit-testable in isolation.
- **`pricing.compute()` integration:** callers that price a practitioner's client resolve `settings` via the builder before calling `compute()`. No new parameter on `compute()` itself — the practitioner context is folded into `settings` upstream, mirroring how the existing engine is fed. Identify the practitioner-owning-client relationship the same way the portal/dispensary attribution already does (wholesale/portal linkage).
- **Practitioner-portal surface:** a Pricing panel in `static/practitioner-portal.html` (under Partner Program or Clients tab), backed by `dashboard/practitioner_portal.py` + a `POST /api/practitioner/pricing` endpoint. Shows the two schedules, per-type toggle + dial, and each type's live ceiling so the practitioner sees the bound.
- **Read path for portal-data:** `portal_data()` returns the practitioner's current config + the live ceilings.

## Testing

- `effective_settings` clamp: practitioner dial above/below/at ceiling → pointwise `min` at sampled quantities; disabled type → 0; open-total clamped to program-total curve.
- Schedule selection: program_member True → Program schedule when enabled, else Standard; program_member False → always Standard.
- Replace-not-stack: a practitioner client with a shallower dial than global gets the shallower price (global does not leak in); direct client unchanged.
- Dynamic ceiling: retune global anchors → practitioner effective ramp tracks.
- Route tests for `GET portal-data` (config + ceilings present) and `POST /api/practitioner/pricing` (validation rejects dials above ceiling / unknown types).
- Guard the deploy-safety invariant: no practitioner path can enable global open-total on the public store; practitioner open-total lives only in the practitioner-effective settings.

## Out of scope / Future (separate brainstorm)

- **Practitioner-run own programs** — a practitioner enrolls/tags their own patients into *their* program (rides the cohort system as a practitioner-scoped enrollment), independent of our membership. Deferred because v1's Program tier keys on our program membership and that is acceptable to Glen.
- **Turnkey: run *our* programs inside a practitioner's patient portal**, with fee-share to the practitioner via the already-wired affiliate/commission engine (Ambassador tab, PR #545). A significant standalone feature — expands a practitioner's offerings and outcomes with no added time, staff, space, or overhead. **Gets its own brainstorm→spec next (Glen 2026-07-03); resolved economic model below.**

  Resolved model (2026-07-03 brainstorm):
  - A patient buys our continuity-support program (recurring service fee, e.g. $99/mo) **through their doctor's portal**.
  - The patient receives **the product discounts the practitioner set** (this spec's feature is the pricing half).
  - The practitioner earns a **recurring share of the service fee**, **service portion only** (never bundled product/fulfillment).
  - **Retention is the job.** Because the share is on a *recurring* fee, retention is self-enforcing: **certification level sets the rate, retention sets the duration** — the share pays each month only while the patient stays enrolled; churn stops it automatically. No touchpoint logging.
  - **Rate = f(ASH certification level)**, base ~30% → up to **50%** at full certification. Point this ladder at the **same** cert signal that already sets the wholesale floor (`certification_floor_cents` / `modules_completed`) so a practitioner has one "level," not two. Illustrative: early/uncertified ~30%, mid ~40%, full ASH cert 50%.
  - This makes ASH certification the ladder for *service income* too (not just the wholesale-floor unlock), compounding the incentive to certify → earn more → deliver better continuity.
