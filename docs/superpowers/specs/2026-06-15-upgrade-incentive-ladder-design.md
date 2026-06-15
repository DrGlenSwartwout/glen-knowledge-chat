# Upgrade-Incentive Ladder — Design

**Date:** 2026-06-15
**Status:** Design approved in brainstorm (Glen). Spec for review → plan(s) → build.
**Related:** [[project_pricing_rewards_engine]] (points/subscribe-and-grow it builds on), the ascend ladder (`/begin/ascend`, `begin_funnel.TIER_CATALOG`), client-nurture sequence (`04 Copy/client-nurture/`), Group Coaching subscription (`_MEMBERSHIP_TIERS`, `/admin/membership`).

## Problem

There is machinery to *acquire* free members and to *sell* high-ticket tiers, but nothing that deliberately walks a member **up** the ladder, and nothing that converts product spend into coaching engagement. Goal: a coherent set of incentives that move a member free → app → live group → Biofield → certification, while lifting AOV and retention.

## The ladder (prices locked)

| Rung | Billed by | The pull up to it |
|---|---|---|
| Free member | us | the on-ramp (ToS opt-in) |
| Studio.com app, $99/yr | **Studio.com** (creator rev-share pays us on every participant, including referrals) | two-way bridge (Mechanic 2); positioned **right after the free offers** |
| Live group coaching, $99/mo founders ($149 standard) | us | bundled-free-while-on-program (Mechanic 1) |
| Biofield Analysis, $300 one-time | us | clinical entry; unlocks the group bundle; reachable via points (Mechanic 4) |
| Certification, ~$3,600+ (then the existing $8.5k–$50k tiers) | us | commit / milestone bonus (Mechanic 3) |

Studio.com is a premium creator platform for AI coaching apps; Glen's app is live at `studio.com/drglen`. Creator payout / subscriber-data / referral-attribution specifics live in Glen's creator dashboard + agreement (not public) — see Mechanic 2 confirmation.

**Pricing reconciliation:** the live-group rung is **$99 founders / $149 standard** (matches `_MEMBERSHIP_TIERS`). The client-nurture copy's "Pay-It-Forward $97/mo" must be updated to this. The legacy CLIENT-PROFILE "$299/mo AI app" is superseded by Studio.com $99/yr.

---

## Mechanic 1 — Program-bundled live group (the engine)

A Biofield + designed remedy program **includes live-group access: 1 month per program month purchased.**

- **Biofield alone includes 0 months** — the patient must implement with a designed program to earn support.
- **Cap ~3 months** (Glen's max recommended length for one program). Past 3 months the patient re-Biofields and a fresh program is designed, which **starts a new support window**.
- **Auto-continue:** with a card on file, when the included window ends the live-group membership **auto-continues at $99/mo (founders) unless cancelled.** Clear disclosure + easy cancel + a reminder before the first charge.
- **Net effect:** live group is effectively free while the patient stays on the quarterly Biofield → 3-month-program → re-Biofield cycle. The $99/mo auto-charge monetizes the gap when they lapse out of the cycle. Rewards staying on protocol AND captures lapse revenue.

**Reuse:** the included-months count reads off the pricing engine's existing `volume_months` (one source of truth, reinforces the "more program = bigger discount AND more coaching" AOV story). Auto-continue reuses the Subscribe-and-Grow **Stripe card vault + own scheduler** and the existing Group Coaching subscription (`_MEMBERSHIP_TIERS`). Window-stacking: a new program purchased mid-window extends the included window.

**Value framing (marketing):** "6-month program → 3 months of live group included, a $297 value, then $99/mo."

---

## Mechanic 2 — Studio.com two-way bridge (monetized both directions)

Studio.com pays us creator rev-share on every participant, including ones we refer — so both flows earn.

### Flow B — us → Studio.com
"Join at **studio.com/drglen**, get your **first month of live group free.**" Pays us twice: rev-share on the Studio.com signup + the live-group month costs ~$0 marginal.

**Confirmation** (Glen to check his Studio.com dashboard for which is possible; design supports either):
1. **Email-match (preferred):** if the creator dashboard exports subscriber emails, match our member's email against the Studio.com participant feed → auto-grant the free month.
2. **Receipt screenshot:** member uploads their Studio.com receipt via our chat (image/PDF OCR already built) → confirm → grant.
3. **Self-attest + spot-check:** they tell us; bonus is ~$0 so abuse downside is trivial.

The grant = one free month of the live-group subscription (Mechanic 1 infra), then normal $99/mo auto-continue.

### Flow A — Studio.com → us (capture their users; positioned just after the free offers)
Layer our **clinical wedge** — the thing a phone app can't do — on Studio.com users, free:
- **Biofield voice scan + remedy match (the wedge):** leads to a remedy program purchase → which triggers Mechanic 1's group bundle.
- **Deeper AI Q&A:** our concierge (KB + catalog aware) beyond the app's on-phone AI.
- **Free membership + courses:** the container + nurture content toward Biofield.

Funnel placement: a "claim your free voice scan + remedy match" path offered **right after the free offers** (Studio.com is low-ticket, so this lives in the early/free onboarding, not a deferred phase). The flywheel: Studio.com users arrive pre-paying-us via rev-share; the free clinical wedge sells Biofield + remedies + group.

---

## Mechanic 3 — Certification commitment / milestone bonus

The certification tier (~$3,600) earns **monthly Biofield Analysis** as a bonus, to reward commitment to the full program over module-at-a-time:
- **Pay-in-full OR a 12-month payment commitment** both qualify for the monthly-Biofield bonus.
- **Plus completion of a certification level** earns a Biofield — tying the bonus to progress milestones, not only the payment structure.

Monthly Biofield is the right bonus: it is Glen's signature high-value diagnostic and matches the Accelerated Self-Healing™ monthly cadence (one bio-age year per cycle). Keep the monthly-Biofield bonus at the certification tier so it does not erode the $300 standalone lower on the ladder.

---

## Mechanic 4 — Points → Biofield converter

Let **loyalty points / a fixed % of past product spend bank as credit toward the next $300 Biofield.** This keeps loyal product buyers spinning the quarterly Biofield → program → re-Biofield cycle (which keeps both product revenue and free group access flowing) and rewards repeat buyers. Builds on the existing points ledger ([[project_pricing_rewards_engine]]); Biofield becomes a redemption target alongside products. (Points cannot apply to the Studio.com $99/yr — external billing.)

---

## Economics / guardrails

- **Group is near-zero marginal cost,** so bundling/giving live-group months is cheap; the discipline is making them **convert** (auto-continue) rather than train "coaching is free."
- **Anchoring:** always *show* the standalone value of included months; keep monthly-Biofield-as-bonus at certification only.
- **Cannibalization:** the $99/yr Studio.com (external) and $99/mo live group are differentiated — Studio.com = daily AI app on the phone; live group = live with Glen + cohort. They are complementary rungs, not substitutes; Flow B deliberately uses one to feed the other.
- **Stacking:** included-months + the volume discount + loyalty points all ride the same order — verify the combined economics per order size before launch (reuse the pricing preview).

## Buildable vs marketing split + phasing

**Phase 1 (build, highest leverage):** Mechanic 1 — program-bundled live group with auto-continue (reuses Stripe vault + scheduler + Group Coaching sub + `volume_months`). This is the engine and the biggest AOV/retention lever.

**Phase 2 (build):** Mechanic 4 — points → Biofield redemption (small, builds on the points ledger).

**Phase 3 (build, depends on the Studio.com dashboard answer):** Mechanic 2 Flow B confirmation + grant (email-match or receipt-OCR), and Flow A's "free voice scan" onboarding path for Studio.com users.

**Marketing/ops (not code):** nurture-copy updates (fix $97 → $99/$149, add the ladder offers), the certification commitment/milestone offer terms, and the studio.com/drglen push.

## Open items / deferred

- **Confirmation mechanism** for Flow B depends on what the Studio.com creator dashboard exposes (Glen to check: subscriber-email export?). Design supports email-match or receipt-OCR either way.
- Affiliate/rev-share exact terms with Studio.com (Glen has the agreement) — affects how hard to push Flow B (already clearly worth it).
- Whether Studio.com app should be a hard gate or a soft suggestion after the free offers.
- Refund/clawback handling for an auto-continued group membership (mirror the existing subscription cancel path).

## Testing considerations (buildable parts)

- Mechanic 1: included-months = f(program months) with the 0-for-Biofield-alone rule and the ~3 cap; auto-continue schedules the first $99 charge after the window; cancel stops it; window-stacking on a mid-window program purchase; idempotent on the paid event.
- Mechanic 4: points redeemable toward a Biofield line; floor/validation; idempotency.
- Mechanic 2: confirmation grant is idempotent (one free month per participant); receipt-OCR path; email-match path.
