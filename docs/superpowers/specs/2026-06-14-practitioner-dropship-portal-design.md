# Practitioner Drop-Ship + White-Label Portal — Design Spec

**Date:** 2026-06-14
**Status:** Draft for Glen's review
**Reworks:** the existing dispensary/wallet/wholesale system (`dashboard/wholesale_pricing.py`,
`dashboard/wallet.py`, `dashboard/practitioner_portal.py`, `/dispensary/<code>`,
`_record_dispensary_sale`). Builds alongside the customer pricing engine
([2026-06-13-recurring-orders-and-discount-engine-design.md]).

Lets a wholesale-approved practitioner serve their patients three ways from one branded
portal — send them to a branded **client order page**, place a **practitioner-paid
drop-ship**, or **stock their own dispensary** — with practitioner-set pricing on
Functional Formulations, a single 33% service fee on drop-ships, MAP protection on the
advertised page, and the practitioner's margin flowing to their existing wallet credit.

---

## Decisions locked with Glen (2026-06-14)

| Decision | Choice |
|---|---|
| Drop-ship base price | The **blended volume curve** (same as wholesale stocking): $50 @ 1 bottle → cert floor ($40 uncertified → $25 fully certified) reached at 40 bottles. Volume + certification lower it |
| Service fee | **Single flat 33%** of the practitioner's markup (selling price − blended base). **Drop-ship only** (no fee on stocking). No floor needed (MAP keeps the advertised price high) |
| Run modes | **(a) Practitioner-paid:** pays "drop-ship wholesale" = base + 33% fee (one number), collects retail from the patient privately → cash margin. **(b) Patient-paid:** patient pays the practitioner's retail; margin (selling − base − fee) accrues to the practitioner **wallet credit** |
| MAP (Minimum Advertised Price) | **$67** default, **per-SKU, console-settable**. Applies to the **advertised client page only**; private (practitioner-paid drop-ship + in-office) has **no minimum** |
| Quantity pricing on patient order | **None** (flat per-bottle at the practitioner's price ≥ MAP). Practitioners structurally can't match RM's direct volume curve (their wholesale base at volume already exceeds RM's direct volume price). v2 may add optional practitioner-set tiers |
| Practitioner price input | Set in **$ or %** (UI computes the other). **Functional Formulations only**; Pure Powders stay flat $40 (no practitioner pricing) |
| Reward currency | The existing practitioner **wallet** (dollar-cents, credit-only, no cash-out, spend ≤50% on orders / ≤100%/mo on $297 modules). **Replaces the flat $20/bottle dispensary credit** |
| Portal pages | **3 white-label pages** — client / drop-ship / wholesale. Patients access only the client page |
| Branding | Client page: photo + logo + practice name + contact + web link + **2 colors**. Drop-ship + wholesale pages: logo + practice name + 2 colors |

---

## A. Pricing model

Definitions (per bottle, Functional Formulations): **W** = blended wholesale base, **S** =
practitioner's selling price, **R** = RM retail (the product `price_cents`), **F** =
certification floor (`4000 − modules×125`, $40→$25).

### A.1 Drop-ship base (W) — the blended volume curve
`W = wholesale_pricing.blended_unit_price_cents(total_bottles_in_order, modules_completed, B)`
— $50 at q=1, interpolating to F at q=2B (40 bottles, B=20). Same curve as a wholesale
stocking order; **small drop-ships sit near $50** and the discount earns in with volume +
certification. (Note: q=1 = $50 for everyone — certification gives no benefit at a single
bottle. Acceptable: drop-ships are typically small and this protects RM margin on tiny orders.)

### A.2 Service fee — flat 33% (drop-ship only)
`service_fee = round(0.33 × (S − W))` per bottle. No floor (MAP ensures S ≥ ~retail, so the
markup is always healthy). The fee is **not** applied to wholesale stocking orders.

### A.3 The two run modes
- **Practitioner-paid drop-ship:** the practitioner pays RM `drop_ship_wholesale = W + fee`
  (shown to them as one number), we ship to the patient, the practitioner collects S from
  the patient out-of-band → their cash margin is `S − W − fee`. No minimum (private).
- **Patient-paid (client page):** the patient pays **S** to RM. RM keeps `W + fee`; the
  practitioner's margin `S − W − fee` accrues to their **wallet** (credit-only), idempotent
  per invoice. S ≥ MAP (advertised).

Worked example (FF, uncertified, single bottle, S = $80, R = $70): W = $50, fee = 0.33×$30 =
$9.90, practitioner take = $20.10 (cash or wallet), RM nets W + fee − COGS.

### A.4 MAP — advertised floor
`map_cents` per SKU, console-settable, default **$67**. The **client page** rejects S < MAP.
The **practitioner-paid drop-ship** and in-office sales have no minimum (private). MAP also
makes the fee a clean flat 33% (no need for a markup-floor on the fee).

### A.5 No quantity pricing on the patient order (v1)
The client page charges a **flat per-bottle S** regardless of quantity. Rationale: a
practitioner's blended wholesale base at volume (~$47 uncertified / ~$43 certified at 12
bottles) already exceeds RM's direct retail volume price ($40 at 12) — so the channel cannot
profitably match RM's volume curve. Bulk/price-sensitive patients are served via the
**private drop-ship** (no minimum), where the practitioner flexes their own margin.

### A.6 Practitioner price input ($ or %)
On FF products, the practitioner sets either a dollar price **S** or a markup **% over R**;
the UI computes and displays the other. Stored per practitioner per SKU (with a single
default markup % that applies to all FF unless a SKU is overridden). The input is rejected
if it resolves below MAP. Pure Powders are not practitioner-priceable (flat $40).

---

## B. The three portal pages (white-label)

All three live inside the authenticated practitioner portal. Wholesale-approval
(`wholesale_unlocked_at IS NOT NULL`) gates the drop-ship + wholesale pages.

### B.1 Client order page (patient-facing)
- The **only** page patients reach — via the practitioner's existing `/dispensary/<code>`
  link (extended). Patient picks FF at the practitioner's price (≥ MAP; defaults to R if the
  practitioner hasn't set one), enters their own shipping address, pays RM.
- On paid: margin → practitioner wallet (§A.3b); order recorded `source="dispensary"`.
- **Branding:** practitioner **photo, logo, practice name, contact details, web link, 2
  brand colors** — a true white-label storefront for the practice.

### B.2 Practitioner drop-ship order page
- The practitioner builds an order **for a specific patient**: picks FF, enters the
  **patient's shipping address** and the **selling price** (or accepts their default), and
  pays the **drop-ship wholesale (W + fee)**. We ship to the patient. No MAP (private).
- Order recorded `source="dropship"`, shipped to the patient address.
- **Branding:** logo + practice name + 2 colors.

### B.3 Practitioner wholesale (stocking) order page
- Largely the **existing** `/api/practitioner/checkout` flow: bulk order at the blended
  wholesale price (no fee), **ships to the practitioner** for in-office dispensing.
- **Branding:** logo + practice name + 2 colors.

---

## C. White-label settings
A portal settings section stores, per practitioner: `photo_url`, `logo_url`,
`practice_name`, `contact_details` (phone/email/address), `web_link`, `brand_color_1`,
`brand_color_2`. Client page uses all; drop-ship + wholesale pages use logo + practice name +
the two colors. (Image uploads reuse the existing attachment/storage path; never block a page
render if an asset is missing — fall back to RM default branding.)

---

## D. Data model (new vs existing)
- **Practitioner record** (`practitioners`, Supabase): add white-label fields (§C) + a
  per-SKU price/markup map (`practitioner_pricing` — practitioner_id, slug, price_cents OR
  markup_pct). Existing: `modules_completed`, `wallet_balance_cents`, `dispensary_code`,
  `wholesale_unlocked_at`, `portal_role`.
- **MAP**: `map_cents` per SKU (console setting — pairs with the pending pricing-settings
  console editor; default $67 for FF, n/a for Pure Powders).
- **Wallet** (`dashboard/wallet.py`): change drop-ship earning from flat $20/bottle to the
  **margin** `S − W − fee` (idempotent per invoice). Spend rules unchanged.
- **Orders**: `source` in {`dispensary` (patient-paid client page), `dropship`
  (practitioner-paid), `wholesale` (stocking)}; record the practitioner id, the W/fee/margin
  breakdown, and the ship-to (patient vs practitioner).

## E. Reworks / migration
- `/dispensary/<code>` + `_record_dispensary_sale`: patient now buys at the **practitioner's
  price** (not RM retail), and the wallet earns the **margin** (not flat $20/bottle).
- Preserve the existing dispensary-code, attribution-idempotency, and wallet-spend patterns.

## F. Edge cases & risks
- **Price below MAP** on the client page → rejected with a clear message (private path
  suggested). Practitioner-paid drop-ship has no floor.
- **Practitioner hasn't set a price** → client page defaults to RM retail R.
- **Margin ≤ 0** (e.g., S at MAP, high base on a single bottle): allowed; wallet earns 0,
  never negative. Surface the thin margin to the practitioner in the drop-ship UI.
- **Asset missing** (logo/photo) → fall back to RM branding; never break the page.
- **Certification at q=1** gives no base discount (q1 knot fixed) — make this visible in the
  drop-ship quote so practitioners understand volume matters.
- **MAP / private leakage**: the system enforces MAP only on the client page; a practitioner
  publishing private prices elsewhere is a wholesale-agreement matter, not a system one.
- **GET / shipping**: reuse the customer engine's recorded-not-charged GET + the actual-USA
  shipping (bottle-type keying + qty fallback) for all three pages; US ship-to only.

## G. Scope
- **v1:** the 3 pages; FF practitioner pricing ($ or %); 33% drop-ship fee; blended base;
  MAP on the client page; wallet margin earning; white-label settings + branding; dispensary
  rework. Pure Powders flat.
- **v2:** optional practitioner-set **volume tiers** on the client page (their margin to
  give, each tier ≥ MAP); per-SKU MAP overrides UI; richer storefront theming.

## H. Testing
Pure pricing/fee/MAP/margin functions table-driven (every cert × qty × $/% × MAP-clamp
case); the three checkout flows (stub QBO/Stripe, wallet, dispensary attribution); white-label
settings round-trip; MAP rejection; the dispensary rework (margin vs old $20/bottle). Run via
the documented `doppler … pytest` invocation.

## I. Open items
1. **Console MAP/pricing editor** — MAP and the practitioner-pricing defaults should be
   editable in the console (pairs with the separately-pending pricing-settings console editor).
2. **Per-SKU FF price overrides vs a single markup %** — default to one markup % with optional
   per-SKU override; confirm the UI scope.
3. **Image storage** — confirm the upload/storage path for logo/photo (reuse existing
   attachment handling).

---

## Decomposition (for writing-plans)
Sizable; likely several plans: **(1)** pricing/fee/MAP/margin core + wallet rework (pure +
tested); **(2)** the practitioner-paid drop-ship page + flow; **(3)** the client page rework
(branded, practitioner-priced, MAP) on `/dispensary/<code>`; **(4)** white-label settings +
branding; the wholesale stocking page is mostly existing.
