# Attributed Prepay-Term Continuous Care Enrollment (v1) — Design

**Date:** 2026-07-04
**Status:** Approved (brainstormed with Glen 2026-07-04)
**Repo:** deploy-chat
**Fast-follow of:** #565 (turnkey fee-share — monthly-only attributed enrollment), #572 (continuity tooling C). Builds on the public prepay ladder (`dashboard/prepay.py`, `_fulfill_prepay_term`).

## Summary

Today a doctor can only enroll a patient in Continuous Care at the **monthly $99** rate through their dispensary (#565). The public prepay ladder (6mo $546 / 12mo $990) has no *attributed* entry point — so the highest-commitment, highest-retention, biggest-upfront-fee-share enrollments are off the table for doctors. This adds **attributed prepay-term enrollment**: a doctor enrolls a patient on a 6- or 12-month prepaid term through their dispensary, earns a cert-scaled fee-share on the lump, and the prepay patient becomes visible in the doctor's continuity tooling — first-class, uniform with monthly.

Approach **A2** (chosen): attribute the **prepay grant** and fire the fee-share directly on the lump; do NOT create a shadow `subscriptions` membership (which would risk existing monthly-membership logic misreading a prepaid term as a monthly member).

## Terms offered

The dispensary "Start Continuous Care" card (from #565) offers the ladder's commitment tiers: **Monthly ($99)** (existing #565 path), **6 months ($546)**, **12 months ($990)** — reusing `prepay.TIERS` (keys `1mo` / `6mo` / `12mo`; `3mo` stays public-only, out of scope here).

## Flow

- The dispensary CC card gains a term selector; the enrollment POST carries `tier_key`.
- The dispensary CC enrollment endpoint (`POST /dispensary/<code>/continuous-care`) branches on `tier_key`:
  - `1mo` → the existing #565 monthly path (`continuous_care_monthly` Stripe kind, unchanged).
  - `6mo` / `12mo` → a **`prepay_term` Stripe checkout** (mirroring the public `/continuous-care/checkout` session build) carrying the additional `dispensary_pid` + `share_consent` metadata (exactly like #565 threads them on the monthly path).
- Fulfillment extends `_fulfill_prepay_term` (already called from both the return redirect and the webhook, idempotent via `prepay_term_grants(session_id)`).

## Fee-share on the lump

Per #565's settled model (prepay lump → one credit at the purchase-time rate, base = the full charged amount), at fulfillment of an **attributed** prepay term fire the doctor's cert-scaled share **once**:

```
share = care_share.share_cents(tier["price_cents"], care_share.modules_for_practitioner(pid))
wallet.earn_care_share(pid, share, event_ref=f"care_share:prepay:{session_id}")
```

- Base = the full lump (`tier["price_cents"]` — $54600 / $99000). Rate read live from the doctor's `modules_completed`. A 12-month enrollment at full cert = 50% × $990 = **$495** upfront.
- The `care_share:prepay:<session_id>` event_ref is distinct from the monthly cron's `care_share:<sub_id>:<order_count>`, so it cannot collide; the wallet's own `_apply` idempotency (dedup on the event_ref) makes the redirect+webhook double-fire safe.
- Fires only when `dispensary_pid` is present and resolves to a practitioner (via `modules_for_practitioner` → None means not-a-practitioner → no credit) AND the payment succeeded. No attribution → no credit (the public prepay path is unaffected).

## First-class attribution + C-visibility (A2)

- **`prepay_term_grants` gains** `attributed_practitioner_id TEXT`, `practitioner_share_consent INTEGER NOT NULL DEFAULT 0`, and `term_end TEXT` (computed at grant time via `prepay.term_end_date(granted_at, tier_months)` — needed for the "still active" check). Set from the Stripe metadata at fulfillment.
- **`dashboard/continuity_view.py`** — `authorized_patient` and `roster` extend their predicate to a UNION: a patient is a doctor's consented continuity patient if EITHER
  1. a `subscriptions` row matches the existing gate (`attributed_practitioner_id==pid AND practitioner_share_consent==1 AND kind=='membership' AND status!='cancelled'`), OR
  2. a `prepay_term_grants` row matches `attributed_practitioner_id==pid AND practitioner_share_consent==1 AND term_end >= <today>` (still within the prepaid term — the prepay analogue of "not cancelled").

  So a prepay patient shows up in the doctor's continuity roster and per-patient view exactly like a monthly patient, and the whole C recommend loop works for them. The gate stays the single access boundary — it just reads two sources.

## Consent

The `share_consent` checkbox from #565's card applies to all terms (its value flows into the prepay grant's `practitioner_share_consent`), so C's access gate holds for prepay patients.

## Data model

- `prepay_term_grants` + `attributed_practitioner_id TEXT`, `practitioner_share_consent INTEGER NOT NULL DEFAULT 0`, `term_end TEXT` (guarded ALTER / `CREATE TABLE` addition; the table is created lazily in `_fulfill_prepay_term`).
- No new table. Reuses `care_share` (pure share + modules resolver), `wallet.earn_care_share`, `prepay.TIERS`/`term_end_date`.

## Integration points

- **Dispensary CC card** (`static/practitioner-client.html`): term selector (Monthly / 6mo / 12mo, from `prepay.tiers_public()` filtered to `1mo/6mo/12mo`); the POST carries `tier_key` + the existing `share_consent`.
- **Dispensary CC endpoint** (`app.py`): branch on `tier_key`; prepaid → build the `prepay_term` Stripe session with `dispensary_pid` + `share_consent` metadata.
- **`_fulfill_prepay_term`** (`app.py`): read `dispensary_pid`/`share_consent`; on the claimed grant, stamp attribution + consent + `term_end` on the `prepay_term_grants` row and fire the care-share credit on the lump.
- **`continuity_view`** (`dashboard/continuity_view.py`): UNION attributed prepay grants into `authorized_patient` + `roster`.

## Testing

- **Fee-share:** an attributed 12mo fulfillment credits the doctor `share_cents(99000, modules)` once; no credit without `dispensary_pid`; no credit if the owner isn't a practitioner; idempotent per `session_id` (redirect+webhook fire once); base = full lump.
- **Public prepay unaffected:** a non-dispensary `_fulfill_prepay_term` (no `dispensary_pid`) grants the term and fires NO care-share credit (unchanged behavior).
- **Attribution stamped:** the `prepay_term_grants` row carries `attributed_practitioner_id`, `practitioner_share_consent`, `term_end`.
- **C-visibility:** a consented attributed prepay patient within term is `authorized_patient` True and appears in `roster`; an EXPIRED term (`term_end < today`) → False; unconsented → False; another doctor → False. The existing subscriptions-based gate cases still hold (UNION doesn't regress them).
- **Endpoint routing:** the dispensary card's `1mo` still hits the monthly path (#565 untouched); `6mo`/`12mo` build a `prepay_term` session carrying `dispensary_pid` + `share_consent`.

## Out of scope / Future

- **Renewal attribution:** when a prepaid term ends and the patient renews, keeping the doctor attributed on the new term (retention). v1 covers the initial attributed enrollment; renewal attribution is a fast-follow.
- The `3mo` tier (public-only).
- Any change to the public prepay flow's behavior for non-dispensary buyers.
