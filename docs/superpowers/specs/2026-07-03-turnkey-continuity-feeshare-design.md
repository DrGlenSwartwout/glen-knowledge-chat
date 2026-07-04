# Turnkey Continuity — Attributed Enrollment + Recurring Fee-Share (v1) — Design

**Date:** 2026-07-03
**Status:** Approved (brainstormed with Glen 2026-07-03)
**Repo:** deploy-chat
**Relates to:** `2026-07-03-referral-based-attribution-design.md` (#560, the referral-graph patient↔doctor link), `2026-07-03-practitioner-discount-controls-design.md` (the pricing half of the practitioner partnership; PR #561). Continuous Care billing: `dashboard/subscriptions.py` / `dashboard/prepay.py`. Wallet payout rail: `dashboard/wallet.py`.

## Summary

Let a patient enroll in Continuous Care ($99/mo) **through their doctor**, permanently bond the patient to that doctor, and pay the doctor a **cert-scaled recurring share** of every successful membership payment, credited to their existing wallet. This turns a practitioner into a turnkey continuity provider — they expand their offerings and outcomes with no added time, staff, space, or overhead, and earn recurring income scaled to their certification.

This is the **revenue half** of the practitioner partnership; the discount-controls feature (PR #561) is the pricing half.

## Scope

**v1 = A (attributed enrollment) + B (recurring fee-share).** Deferred to their own specs: **C** (doctor continuity tooling — the patient's scans/outcome trends + next-step prompts surfaced in the doctor's portal) and **D** (branded patient experience). Program in scope: **Continuous Care at $99/mo only** (`MONTHLY_ANCHOR_CENTS = 9900`), plus its prepay terms (6mo $546 / 12mo $990).

## Economic model (settled)

- The share base is the **full charged amount**. Continuous Care $99/mo is **pure service** ("live group coaching with Dr. Glen" + premium access; product is bought separately at member pricing), so there is no product/fulfillment carve-out.
- **Rate scales linearly with certification**, mirroring the wholesale-floor design (`certification_floor_cents = base − modules × $1.25`, 12 modules): `rate(m) = 0.30 + m × (0.20/12)` where `m = modules_completed` (0–12), clamped to [0,12]. So 0 modules → **30%**, 6 → **40%**, 12 (full cert) → **50%**. One mental model shared with the wholesale floor: every module completed raises both.
- Read `m` **live at each charge**, so the doctor's share climbs as they certify.
- **Retention is self-enforcing:** the share fires on the *recurring* payment, so it stops automatically when the patient churns. Certification sets the rate; retention sets the duration.

## A — Attributed enrollment

**Entry point.** Reuse the doctor's attributed channel: add a **"Start Continuous Care"** option to the dispensary surface (`/dispensary/<code>`), which already resolves `practitioner_id` from the code and already carries the consent gate + Stripe checkout. Enrolling creates the membership via `subscriptions.create_membership(...)`.

**Durable attribution (isolated stamp).** Add a nullable column **`subscriptions.attributed_practitioner_id`** (TEXT). Set it at enrollment to the enrolling doctor's `practitioner_id`:
- primary source = the dispensary code's `practitioner_id` (explicit, when enrolled via `/dispensary/<code>`);
- fallback = `referrals.owner_of_referee(cx, patient_email)` resolved to a practitioner (when the patient arrived via the doctor's referral link but not the dispensary path) — see alignment note.

Attribution is **permanent for the life of that membership** (the doctor who holds the patient keeps earning as long as they retain them). Direct (non-doctor) enrollments leave it NULL → no share. Backward-compatible: every existing membership is NULL.

**Alignment with #560 (referral graph).** #560 blessed the referral graph (`referral_redemptions`, `owner_of_referee`, first-touch single-owner) as the durable patient↔doctor link and reframed attribution as "affiliate referral." We deliberately keep the care-share on its own **membership-scoped stamp** rather than writing into `referral_redemptions` at enrollment, for two reasons: (1) the care-share is a **service-fee share**, a distinct reward from the product L1/L2 commission/points policy Glen tuned in #560 (drop-ship product sale = markup only, no L1, L2 points only) — writing a redemption row could trigger unintended points effects; (2) a membership-scoped stamp is *more precise* for a per-membership reward. The stamp may be **seeded from** the referral owner (read-only) but never writes to the points ledger. A future unification (single attribution model across product + service rewards) is out of scope here.

## B — Recurring fee-share

On **each successful charge** — the enrollment/prepay charge *and* every monthly renewal — if the subscription has a non-NULL `attributed_practitioner_id`, credit that doctor's wallet:

```
share_cents = round(charge_cents × rate(modules_completed(pid)))
```

- `charge_cents` = the amount actually charged ($9900 monthly, or the prepay lump $54600 / $99000).
- `rate(m)` per the economic model; `m` resolved live from the practitioners table at charge time.
- **Per-payment-event** (Glen's decision): monthly subs → a credit each month (rate climbs as the doctor certifies); a prepay term → one credit on the lump sum at the purchase-time rate.

**New wallet primitive `wallet.earn_care_share(practitioner_id, share_cents, *, event_ref)`** — mirrors `earn_dropship_margin` (credit-only via `_apply`; there is no path from credit to cash except the existing payout). **Idempotent per `event_ref`**, where `event_ref = "care_share:<sub_id>:<charge_seq>"` (`charge_seq` = the subscription's `order_count` at the charge, which `advance_after_charge` increments — unique per successful charge). A cron retry with the same `event_ref` is a silent no-op.

## Data model changes

- `subscriptions` gains `attributed_practitioner_id TEXT` (nullable; lazy `ALTER TABLE ADD COLUMN` guarded like the module's other migrations). Set at `create_membership`.
- `dashboard/wallet.py` gains `earn_care_share(pid, share_cents, *, event_ref)` + its reversal counterpart `reverse_care_share(pid, share_cents, *, event_ref)` (a debit keyed to the same `event_ref`, no-op if the original credit is absent).
- A pure rate helper `dashboard/care_share.py: rate(modules_completed) -> float` and `share_cents(charge_cents, modules_completed) -> int` (pure, unit-tested in isolation — the economic heart).
- A resolver `care_share.modules_for_practitioner(pid) -> int` (reads `SELECT modules_completed FROM practitioners WHERE id=%s`, mirroring the existing `modules_completed_for_email`).

## Integration points

1. **Enrollment surface** — `/dispensary/<code>` gains a "Start Continuous Care" checkout that calls `create_membership(..., attributed_practitioner_id=pid)` and fires the **enrollment charge** credit (the first payment is a successful charge → same credit path).
2. **Renewal cron** — in `cron_charge_subscriptions()` (app.py ~23509), immediately after a successful charge + `advance_after_charge(cx, sid)` (~23639/23741), resolve the sub's `attributed_practitioner_id`; if set, compute `share_cents` and call `earn_care_share(pid, share_cents, event_ref="care_share:<sid>:<order_count>")`. Fires only on a confirmed successful charge.
3. **Refund / chargeback** — see below.

## Refund / chargeback

There is **no existing Stripe refund/chargeback webhook handler** in the app today (confirmed by grep). So automatic reversal is **not wired** and v1 does not invent a full webhook path. v1 delivers:
- `reverse_care_share(...)` (the primitive to undo a credit by `event_ref`), and
- a **console action** (owner-only) to reverse a care-share credit for a given subscription+charge when a refund is processed manually.

Automatic reversal on a Stripe `charge.refunded` / dispute webhook is a **fast-follow** (noted in Out of scope). This is called out so the reversal path is not mistaken for automatic.

## Doctor visibility (v1, minimal)

Care-share credits land in the **existing wallet / earnings surface** (Ambassador tab, over `wallet.py`), labeled as a continuity share (e.g. "Continuity care — <month>"). No new dashboards — richer doctor-facing continuity views are part of C.

## Edge cases

- No attribution (NULL) → no credit.
- Attributed owner is not (or no longer) a practitioner → no credit (resolver returns None).
- Failed charge → no credit (the credit is inside the success branch, after `advance_after_charge`).
- Cron retry / double-fire → idempotent by `event_ref`.
- Patient already has an active Continuous Care subscription → do not double-enroll (reuse the existing single-active-membership guard).
- `modules_completed` out of range → clamped to [0,12] by `rate`.

## Testing

- `care_share.rate(m)` at m = 0 / 6 / 12 → 0.30 / 0.40 / 0.50; clamps below 0 and above 12.
- `share_cents(9900, m)` → 2970 / 3960 / 4950 at those tiers.
- Credit fires with attribution; **no** credit when `attributed_practitioner_id` is NULL.
- **Idempotency:** two calls with the same `event_ref` credit once.
- **Rate climbs:** two successive monthly charges with `modules_completed` 6 then 12 credit 3960 then 4950.
- **Prepay:** a 12mo lump ($99000) credits once = `share_cents(99000, m)`.
- `reverse_care_share` undoes exactly the credit for that `event_ref`; no-op if absent.
- Enrollment via `/dispensary/<code>` stamps `attributed_practitioner_id` and fires the enrollment-charge credit.

## Out of scope / Future (fast-follows)

- **C** — doctor continuity tooling (patient scans/outcome trends + next-step prompts in the doctor's portal); this is what most deeply justifies the up-to-50%.
- **D** — branded patient experience (doctor-branded portal for the program).
- **Automatic refund/chargeback reversal** via a Stripe webhook (v1 is manual/console).
- **Monthly-drip accrual** for prepay terms (v1 credits the lump once at purchase-time rate).
- **Programs beyond Continuous Care** (Biofield, etc.).
- **Unifying attribution** onto the #560 referral graph across both product and service rewards.
