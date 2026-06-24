# Bottle-Type Population + Console Editor

**Date:** 2026-06-24
**Author:** Glen Swartwout (design w/ Claude)
**Status:** Draft for review
**Repo:** deploy-chat — stacks on branch `sess/59a2725d` (geometric packer, PR #253)

## Problem

The geometric packer (PR #253) is built but dormant: no product has a `bottle_type`, so every order falls back to the qty rule. Auto-inference from product names failed (294/338 unresolved — names are formulations, not packaging). FileMaker holds real packaging data, and Glen has given the family rules. We need to (1) populate `bottle_type` from FileMaker + family rules, (2) let Glen/Rae see and adjust assignments and the bottle catalog in the console, and (3) add the one missing bottle type the catalog needs.

## Bottle types (final — keys are fill-content based)

| Key | Bottle | Ø×H cm | Ø×H mm |
|---|---|---|---|
| `5ml` | dropper — eye drops | 3×8 | 30×80 |
| `15ml` | dropper | 3×10 | 30×100 |
| `30ml` | dropper — infoceuticals **(new; est., Glen to verify)** | 3.5×11 | 35×110 |
| `50ml` | dropper | 4×14 | 40×140 |
| `100ml` | dropper | 5×16 | 50×160 |
| `30roll` | roll-on (rare) | 4×10 | 40×100 |
| `30g` | 100 ml cosmetic jar — powder **(renamed from `100cos`)** | 7×7 | 70×70 |
| `30cap` | 100 ml wide-mouth — 30 caps | 5×9 | 50×90 |
| `120cap` | 250 ml wide-mouth — 120 caps / pure powder | 8×10 | 80×100 |

## Data source

FileMaker export at `/tmp/fmp-export/newapp/products.csv` (1,194 rows; refreshed by the existing ODBC extract). Relevant fields: `product_name`, `type`, `category`, `sold_size`, `sold_measurement`, `zc_sold_display` (e.g. "50ml", "30pullulan", "30g"), `doses_per_bottle`. Storefront `data/products.json` (338) is the subset to populate; join is by normalized name (no FMP id is stored).

## Family / packaging → bottle-type rules

Applied by the populator, in priority order, writing only where `bottle_type` is unset:
1. **Infoceuticals** — storefront `source == "infoceutical-catalog"` OR name matches `^(EI|ED|ES|ET|MB|MR)\d` (with/without space) → `30ml`.
2. **Eye drops** — name/description contains "eye drop"/"eyedrop" → `5ml`.
3. **FMP packaging join** (normalized-name match into the FMP export), by `zc_sold_display` / `sold_size`+`sold_measurement`:
   - liquid `5ml`→`5ml`, `15ml`→`15ml`, `50ml`→`50ml`, `100ml`→`100ml`
   - capsules (`pullulan`/`enteric`/`vegicaps`/`gelcaps`/`capsules`): count ≤40 → `30cap`; 41–140 → `120cap`
   - `g` powder: FMP `type == "Pure Powders"` → `120cap` (250 ml wide-mouth); else (Functional-Formulation powder, 30–45 g) → `30g`
4. **Everything else** (no name match; 30 ml liquids; bulk 1000 ml/4 L; >140 caps; essences that don't match; books/services/`ea.`) → left unset, emitted to a **review list** for manual assignment in the console.

The populator is re-runnable, never overwrites an existing assignment, prints a review report, and `--write` patches `data/products.json` (committed baseline).

## Runtime override (console edits)

`products.json` is read-only on the server. Per-product console edits write to a new table on `chat_log.db`:

```sql
CREATE TABLE product_bottle_types (
    slug         TEXT PRIMARY KEY,
    bottle_type  TEXT NOT NULL,
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
```

**Resolution at checkout** (in `_price_cart`, app.py:3412): override table → `products.json` `bottle_type` → `"default"`. A `default`/unknown type still routes to the qty fallback (unchanged, safe).

## Schema / CRUD changes (`dashboard/shipping.py`)

- `_STANDARD_BOTTLES`: rename `100cos`→`30g`, add `30ml` (35×110). Idempotent migration for already-seeded DBs: rename row `100cos`→`30g` if present; insert `30ml` if absent.
- `add_bottle_type(name, diameter_mm=None, height_mm=None, notes=None)` and `update_bottle_type(id, name, diameter_mm, height_mm, notes)` — carry dimensions.
- Reuse `get_packing_settings`/`set_packing_setting` (already built) for the padding knobs.
- New: `list_product_bottle_overrides()`, `set_product_bottle_override(slug, bottle_type)`, `clear_product_bottle_override(slug)`, and `resolve_bottle_type(slug, products_dict)`.

## Console UI (extend `/admin/shipping`)

Same page, auth, and look as today (CONSOLE_SECRET / `X-Console-Key`). Add:
1. **Bottle catalog** — list the 9 types with **editable** `diameter_mm`/`height_mm` + notes; **add** a new type (key + dims); delete. (Box-fit matrix + rates sections stay as-is.)
2. **Padding** — two number inputs (`wrap_mm`, `box_margin_mm`) with save.
3. **Products** — every storefront product with its resolved bottle type + a dropdown (options = current catalog, so newly-added types appear) to set/override; unassigned/`default` products sorted to the top and badged. Saving writes the override table.

Endpoints follow the existing JSON-action pattern at `/admin/shipping` (e.g. `bottle_dims`, `packing_settings`, `product_bottle`).

## Testing

- `dashboard/shipping.py`: rename+add migration idempotent; dims on add/update; override CRUD; `resolve_bottle_type` precedence.
- `scripts/populate_bottle_types.py`: each family rule, FMP join classification, review list for gaps, never-overwrite, dry-run read-only.
- `_price_cart`: resolution precedence (override beats products.json beats default) — a focused unit test on the resolver.
- pytest; reuse `seeded_db`/`geo_db` fixtures.

## Non-goals

- No change to flat-rate pricing, `usps_rates`, or the rate watcher.
- No change to the geometric packer engine (`packing.py`) — it reads dims from the catalog, so new/edited types just work.
- Bulk/oversized containers (1000 ml/4 L) and non-bottle items (books/services) are left unset → qty fallback; not modeled as bottle types now.

## Open items

- `30ml` dropper real dimensions (seeded at 35×110 estimate; Glen verifies via the console).
- Essences: classified by FMP packaging where they name-match, else review — no dedicated family rule unless Glen specifies one.
