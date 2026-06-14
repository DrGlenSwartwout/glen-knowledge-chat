# Practitioner drop-ship pricing core (`dashboard/practitioner_pricing.py`)

Pure pricing layer for the practitioner drop-ship system. The three portal pages
(client / drop-ship / wholesale, built in later plans) all price through `quote_line`.

## The model
- **Base (W)** — `drop_ship_base_cents(qty, modules)` = the **blended wholesale curve**
  (`wholesale_pricing.blended_unit_price_cents`): $50/bottle at 1 bottle, declining to the
  certification floor ($40 uncertified → $25 fully certified) at 40 bottles. Same curve as a
  wholesale stocking order; small drop-ships sit near $50. (q=1 = $50 for every cert level —
  certification only helps at volume.)
- **Service fee** — `service_fee_cents(selling, base, settings)` = flat **33%** of the
  markup (`selling − base`), never negative. **Drop-ship only** (no fee on stocking).
- **Selling price (S)** — `resolve_selling_cents({price_cents} | {markup_pct} | {}, *,
  retail_cents, map_cents)`: practitioner sets a dollar price or a markup % (UI shows the
  other via `price_for_markup` / `markup_pct_for`); defaults to retail; **raises
  `MapViolation` below MAP** ($67 default, per-SKU console-settable). FF only.
- **Margin** — `quote_line(...)` returns `{base_cents, fee_cents, margin_cents (=selling −
  base − fee, ≥0), dropship_wholesale_cents (=base + fee), line_selling_cents}`.
  `quote_line` trusts a selling price that already passed `resolve_selling_cents` (MAP is
  enforced at resolution, not re-checked here).

## Two run modes (later plans wire the pages)
- **Practitioner-paid:** pays `dropship_wholesale_cents` (= base + fee), collects S from the
  patient privately → cash margin. No MAP (private).
- **Patient-paid:** patient pays S; the practitioner's `margin_cents` is credited to their
  **wallet** via `wallet.earn_dropship_margin(pid, margin_cents, qbo_invoice_id=...)`
  (idempotent per invoice) — **replacing the old flat $20/bottle** `earn_dropship`.

## Settings
`load_settings(overrides)` → `fee_pct` (0.33), `map_default_cents` (6700). Pairs with the
pending pricing-settings console editor (per-SKU MAP lives there).
