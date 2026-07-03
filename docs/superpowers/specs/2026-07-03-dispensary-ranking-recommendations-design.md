# Dispensary Tab — Product-Dispense Ranking + Practice-Type Recommendations

**Date:** 2026-07-03
**Status:** design approved, pending spec review → writing-plans
**Surface:** the practitioner portal **Clients** tab (`#dispensary-panel` area of `static/practitioner-portal.html`), building on PR #545 (portal tabbed shell).

## Goal

Turn the Clients/Dispensary tab into a practice-intelligence surface: (1) a prominent table of the products the practitioner actually moves, ranked by total units and split by fulfillment channel, and (2) below it, a lighter, curated list of the next Functional Formulations to add, chosen by their type of practice, each with a short structure/function blurb. Every product name links to its detail page in a new tab.

## Section 1 — "Products you dispense" (prominent ranked table)

- One row per product the practitioner has moved, **ranked by total units descending**.
- Columns: **Product** · Dispensed · Drop-shipped · Patient portal · **Total**.
- Channel definitions (confirmed with Glen 2026-07-03):
  - **Dispensed** — units on the practitioner's own wholesale/personal orders (stock they buy to hand to patients).
  - **Drop-shipped** — units on patient orders placed through the practitioner's dispensary link (shipped direct from Remedy Match).
  - **Patient portal** — digital sales made to the practitioner's patients through their own personal ordering page (the per-patient client portal with reorder), distinct from the practitioner's shared drop-ship link. The link from a patient's ordering page back to the practitioner who gave it to them **is not established in the data today** (portal publish is console/owner-only, no practitioner attribution on portal tables), so this column **ships as a placeholder (0), rendered but de-emphasized ("coming soon")**, until that practitioner↔patient-portal attribution exists. Building it is out of scope for the first plan.
- **Product** cell is a link to `/begin/product/<slug>`, `target="_blank" rel="noopener"`.
- Empty state: if the practitioner has moved nothing yet, show a friendly line ("Your dispensing history will appear here") instead of an empty table.

### Data flow (Section 1)

New pure aggregator, its own module `dashboard/dispensary_stats.py`:

```
dispense_stats(practitioner_id, *, db_path=None) -> list[dict]
# -> [{"slug","name","url","dispensed","dropshipped","patient_portal","total"}, ...]
#    sorted by total desc, then name. Reads orders.items_json; never raises (defensive).
```

Sources it reads (per-product `qty` comes from `orders.items_json`, which is `[{slug, qty, ...}]`):
- **Dispensed** = the practitioner's own orders (their account as buyer — personal + wholesale checkout orders).
- **Drop-shipped** = orders behind the practitioner's `dispensary_orders` rows, joined to the main `orders` table by `invoice_id` to reach `items_json`.
- **Patient portal** = 0 for now (attribution deferred).

Exact join columns (buyer match for Dispensed; `dispensary_orders.invoice_id` → `orders` for Drop-shipped) are verified against the live schema during the implementation plan. Product `name` resolves from `data/products.json`; unknown slugs fall back to the slug.

## Section 2 — "Recommended to add next" (below, lighter styling)

- A curated, ranked list of Functional Formulations to suggest next, keyed to the practitioner's **practice type** (their `credentials` field: Health Coach / OD / ND / DC / LAc / … with a `default` fallback).
- Products the practitioner **already dispenses** (appear in Section 1) are **excluded** so the list is genuinely "next."
- Each row renders: **linked product name → a brief structure/function blurb** (use / benefit / function-structure effect). Styling is lighter/less prominent than Section 1 (smaller cards, muted border) but names are still active links (`/begin/product/<slug>`, new tab).
- Compliance: blurbs use **structure/function language only** (supports / promotes / helps maintain), never disease claims; a one-line disclaimer sits under the section.

### Data flow (Section 2)

- Curated mapping file `data/practice_recommendations.json`:
  ```json
  {
    "default": [{"slug": "…", "blurb": "…"}, …],
    "Health Coach": [{"slug": "…", "blurb": "…"}, …],
    "OD": [...], "ND": [...], "DC": [...], "LAc": [...]
  }
  ```
  Blurbs are **seeded from existing formulation data** (the `specific-formulations` Pinecone namespace / product pages) as a one-time authoring aid, then Glen edits. The file is the single source of truth; no live Pinecone call at render time.
- New pure function in `dashboard/dispensary_stats.py`:
  ```
  recommended_ffs(practice_type, *, exclude_slugs=(), db_path=None) -> list[dict]
  # -> [{"slug","name","url","blurb"}, ...] in curated order, minus exclude_slugs.
  #    Resolves practice_type -> its list (case-insensitive), else "default".
  ```

## Wiring & placement

- Both blocks added to the practitioner payload: `data["dispense_stats"]` and `data["recommended_ffs"]`, computed in `portal_data()` (recommendations use `credentials` + the Section-1 slugs as `exclude_slugs`).
- Front-end (`static/practitioner-portal.html`, Clients pane): keep the existing drop-ship share link + credit total; add the ranked table (Section 1) and the recommendations grid (Section 2) below. New `renderDispenseStats(d)` / `renderRecommended(d)` in `render()`.

## Testing

- `dispensary_stats.py` is pure → unit tests without a DB: `dispense_stats` ranking/channel-bucketing from a synthetic orders fixture; `recommended_ffs` practice-type resolution, `default` fallback, and `exclude_slugs` filtering.
- Front-end: headless-render verification (ranked table, links open new tab, recommendations lighter styling, empty states) via injected payloads.

## Out of scope (future)

- Patient-portal channel attribution + real numbers (ships as placeholder now).
- Data-driven recommendations ("what similar practices dispense") — the approved basis is curated-per-practice-type; the auto version is a later enhancement once volume exists.
