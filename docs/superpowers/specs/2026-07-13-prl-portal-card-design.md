# PRL Supplement Portal Card — Design Spec

**Date:** 2026-07-13
**Status:** Draft for review
**Author:** Claude (with Glen Swartwout)
**Repo:** deploy-chat (glen-knowledge-chat)

## 1. Purpose

Add a **Premier Research Labs (PRL)** recommendation card to the client portal,
parallel to the existing infoceutical scan-recommendations and FF-matches cards.
For a client's scan it shows, per key focus area, the recommended PRL products,
each paired with the client's equivalent Functional Formulation (FF), plus a
single **"Go to Premier Research Labs"** button so the client can order PRL
products through Dr. Glen's practitioner account.

This mirrors, in the portal, the "PRL Supplement" page E4L shows the practitioner.

## 2. Background / context

- The PRL Supplement data is **not** in the E4L scan PDFs; it lives only in the
  E4L portal UI/endpoints, which require Glen's logged-in session (single-session,
  console capture). Therefore the portal cannot fetch it live per scan.
- We have built three reusable data assets (vault `02 Products/PRL-catalog/`):
  - `prl_catalog_enriched.json` — 143 PRL products + ingredients + focus tags.
  - `e4l-map/e4l_prl_focus_area_map.json` — E4L focus area → infoceuticals (43/43),
    names (35/43), PRL products (6/43, growing).
  - `prl_ff_map_full.json` — all 143 PRL products → FF counterpart + relation
    (substitute / complement / consider).
- Existing portal precedents to mirror (from deploy-chat recon):
  - Payload assembly: `api_client_portal(token)` at `app.py:16680`; rec blocks
    added ~`app.py:17026-17086`, each **flag-gated + best-effort** (a failure
    never breaks portal load).
  - Data sources: `dashboard/scan_recommendations.py` (prod-mirrored
    `scan_recommendations` table in `chat_log.db`); `dashboard/condition_programs.py`
    (`resolve_program_items`); `dashboard/ff_matcher.py`.
  - Console-gated prod sync: `api_console_scan_recommendations_sync` (`app.py:11182`);
    seed pattern `data/condition_programs_seed.json` + `seed_if_empty`.
  - Card rendering: `static/client-portal.html` — scan-recs card ~line 1263,
    FF-matches ~1297, support-program ~1317, practitioner ~1351.

## 3. Architecture — derive-first, mirror override

Compute the PRL card **at portal render time** inside `api_client_portal`, from
reference data already pushed to the portal DB, with an optional authoritative
per-scan override.

```
scan priority items (scan_recommendations)
        │  item_code → focus areas            (prl_focus_area_items, inverted)
        ▼
   rank focus areas by scan-item overlap → top N (~6)
        │  focus_area → PRL products          (prl_focus_area_products)
        ▼
   PRL products  ── product → FF counterpart  (prl_products.best_ff / relation)
        │
        ▼
   IF prl_scan_mirror[scan_id] exists → use that captured E4L set verbatim instead
```

**Derive** is the default and works automatically for every parsed scan.
**Mirror** overrides the derived set for scans where Glen has captured the
authoritative `GetScanPatternsWithPRL` output.

## 4. Data model

Four tables, added to the portal DB (`chat_log.db`) and populated by
console-gated sync from the vault JSON assets (same push pattern as
`scan_recommendations` / `condition_programs`).

### 4.1 `prl_products`
The 143-product catalog + FF crosswalk.
| col | type | notes |
|---|---|---|
| name | TEXT PK | canonical PRL product name |
| external_id | TEXT | PRL's own product id (from `GetScanPatternsWithPRL`) |
| url | TEXT | prlabs.com product page |
| focus_tags | TEXT | JSON list of PRL focus-area tags |
| product_type | TEXT | supplement/topical/food/bundle/equipment |
| best_ff | TEXT | client's FF counterpart (from crosswalk) |
| relation | TEXT | substitute / complement / consider |
| ff_alts | TEXT | JSON list of secondary FF matches |

Source: `prl_catalog_enriched.json` + `prl_ff_map_full.json`.

### 4.2 `prl_focus_area_products` (the derive FA→PRL map)
| col | type | notes |
|---|---|---|
| focus_area_id | INTEGER | E4L pattern id |
| focus_area_name | TEXT | e.g. "Nervous System" |
| prl_product_name | TEXT | FK → `prl_products.name` |
| rank | INTEGER | order within the focus area |

Source: `e4l_prl_focus_area_map.json` (6/43 now; grows as scans accumulate).

### 4.3 `prl_focus_area_items` (item → focus area)
| col | type | notes |
|---|---|---|
| focus_area_id | INTEGER | |
| item_code | TEXT | e4l infoceutical code (ED4, EI1, …) |

Source: `e4l_prl_focus_area_map.json` `items[]` (43/43 complete). Queried
inverted: given the scan's item codes, find their focus areas.

### 4.4 `prl_scan_mirror` (authoritative override)
| col | type | notes |
|---|---|---|
| scan_id | TEXT PK | |
| payload | TEXT | captured `GetScanPatternsWithPRL` JSON |
| captured_at | TEXT | |

Empty until Glen captures a scan's authoritative data.

## 5. Compute — `_prl_supplement_for(scan)` (new, in a `dashboard/prl_supplement.py` module)

Best-effort; returns `None` on any error so the portal never breaks.

1. If `_prl_enabled()` flag is off → return None.
2. If `prl_scan_mirror[scan_id]` exists → parse it into the payload shape (§6),
   attach FF counterparts, return (mark `source: "mirror"`).
3. Otherwise **derive**:
   a. Get the scan's priority item codes (reuse `_scan_recommendations_for` /
      `scan_recommendations` rows for this scan).
   b. Map each item_code → focus_area_ids via `prl_focus_area_items`.
   c. Rank focus areas by **count of the scan's items falling in each** (tie-break
      by best item priority_rank); take the **top 6**.
   d. For each focus area → its `prl_focus_area_products` (ordered by rank);
      for each PRL product attach `best_ff` + `relation` from `prl_products`.
   e. Drop focus areas with no PRL products in the table (uncovered → simply
      absent, no error).
   f. Mark `source: "derived"`.

**Coverage note:** with 6/43 focus areas populated, derive yields a card whenever
a scan's priority items hit one of those six (the common organ systems). Uncovered
focus areas are silently skipped; coverage grows as the FA→PRL table is filled.

## 6. Payload shape

Added to the `api_client_portal` payload (~`app.py:17062`) behind the flag:

```json
"prl_supplement": {
  "enabled": true,
  "source": "derived",              // or "mirror"
  "prl_link": "https://truly.vip/prl",
  "focus_areas": [
    {
      "name": "Nervous System",
      "items": ["ED4 - Nerve", "EI1 - Large Int."],   // the scan's matching items
      "products": [
        {"name": "NeuroVen", "url": "...",
         "ff": {"name": "Neuroprotect", "relation": "substitute", "slug": "neuroprotect"}}
      ]
    }
  ]
},
"prl_supplement_enabled": true
```

`prl_link` is Glen's practitioner link **`https://truly.vip/prl`** (code 021a1a),
NOT E4L's default `prlabs.com/customer/account/login`.

## 7. Card rendering (`static/client-portal.html`)

New card **"Premier Research Labs options"**, sibling to the FF-matches card
(insert after ~line 1297). Body builder `prlSupplementBodyHtml(sp)`:

- One block per focus area: heading = focus area name; the scan's matching items
  as small tags.
- Per PRL product row: **PRL product name** (links to its prlabs.com page) +
  **"Your formula: <FF> · <relation>"** (links to Glen's product via slug).
- Footer: single **"Go to Premier Research Labs"** button → `sp.prl_link`
  (`truly.vip/prl`), plus microcopy: "Budget-friendly option — order these
  directly from PRL, or use your matched formulas above."
- Relation styling: `substitute` (neutral), `complement` (subtle), `consider`
  (softest — "worth considering").

## 8. Gating, curation, sync

- **Flag** `_prl_enabled()` — mirrors `_support_programs_enabled()` /
  `ff_matches_enabled`. Dark by default; card omitted entirely when off.
- **Console sync**: a `api_console_prl_sync` endpoint (mirroring
  `api_console_scan_recommendations_sync`, `app.py:11182`) that (re)loads the four
  tables from the vault JSON assets into `chat_log.db`. Idempotent, console-gated.
- **Curation**: the FA→PRL and PRL↔FF maps are proposed data; Glen reviews the
  vault JSON before sync. No new curation UI in v1 (edit JSON + resync).

## 9. Testing

- Unit: `_prl_supplement_for` — derive path (item→FA→product ranking, top-6,
  uncovered-skip), mirror-override path, flag-off returns None, empty-scan returns
  None. Follow `tests/test_formulation_map_routes.py` style; run under
  `doppler run -p remedy-match -c dev -- python3 -m pytest` (app-importing tests
  skip without keys — see memory `deploy_chat_test_doppler_skip`).
- Render check: portal loads with card present (headless browser) for a scan that
  hits a covered focus area; card absent when flag off; portal still loads if the
  PRL block raises (best-effort).
- Guard: full suite sends real email unless mocked — use focused tests +
  `PYTEST_CURRENT_TEST` guard (memory `pytest_floods_live_email`).

## 10. Out of scope (v1)

- Automated per-scan E4L capture (mirror is manual capture + sync).
- Completing FA→PRL coverage beyond what's captured (accumulates over time).
- Naming the 9 unnamed E4L edge focus areas.
- A console curation UI for the maps (edit JSON + resync for now).
- Per-product PRL help text (`GetPRLProductHelpFile`) display.

## 11. Resolved decisions

- Derive-first, mirror override where captured (Glen, 2026-07-13).
- Show top ~6 focus areas ranked by scan-item overlap.
- FF parallel shows the single best FF per PRL product with its relation tag.
- Relation taxonomy: substitute / complement / consider.
- Button → `truly.vip/prl` (practitioner code 021a1a).
- Flag-gated + best-effort, dark until published; console-gated sync from vault JSON.

## 12. Registry

On ship, register the surface in `06 Business Ops/PROGRAMS-REGISTRY.md`
(memory `programs_registry`): program → output surface (portal PRL card) + flag
(`_prl_enabled`) + data source (vault PRL-catalog JSON → console sync).
