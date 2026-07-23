# Portal — My Healing Oasis rollout + pre-flip verification

Companion to `2026-07-22-portal-my-healing-oasis.md`. The tile is built on branch `sess/f9b73eaa-oasis` and ships DARK. This note is the gate before flipping it on.

## The flag

- **`PORTAL_OASIS_ENABLED`** — default OFF; truthy set `("1","true","yes","on")` (mirrors `_PORTAL_HUB_ENABLED`). Off = `oasis` block returns `{"enabled": False}`, endpoints 404, no tile. The tile only renders when the hub itself is on (`_hub && v.oasis && v.oasis.enabled`), so `PORTAL_HUB_ENABLED` must also be on for the tile to appear.
- To flip in prod: `doppler set PORTAL_OASIS_ENABLED=on -c prd` (prod secret write may be classifier-blocked — see the finder-flag note in the hub plan). Merge = self-deploy + a brief illtowell 502 (dark, low-risk).

## BLOCKERS before flip — Glen must confirm three data/clinical decisions

These are curated data the build seeded to be functional; each is Glen's to confirm because they encode clinical/product judgment, not mechanics.

1. **Terrain phase → roadmap-key mapping (positional).** The client's terrain phase is a number 1-5 (`terrain_phase.phase_num`). The roadmap keys its terrain-specific tools by `energize / rejuvenate / regenerate / cleanse / balance`, mapped **positionally**: phase 1 → energize, ... phase 5 → balance (`_OASIS_TERRAIN_PHASE_KEYS` in app.py). Confirm **phase 1 == Energize** in the same order Glen numbers the 5 phases. If uncertain, the mapping degrades safely (a client whose phase can't be resolved gets hero + general tools only, no terrain block). Note: `terrain_phase.py`'s own `PHASE_NAMES` uses a DIFFERENT R-name set (Revive/Repair/Renew/Refresh/Relief) — the positional-by-number mapping deliberately sidesteps that naming discrepancy; confirm the numbering aligns.

2. **The tool lists — HERO_TOOLS / TERRAIN_TOOLS / GENERAL_TOOLS (`dashboard/oasis_roadmap.py`).** These are the actual product recommendations the roadmap makes. HERO_TOOLS lead for nearly everyone: `harmony` (Harmony Laser), `water-ionizer`, `kloud`. TERRAIN_TOOLS + GENERAL_TOOLS are seeded but shallow, and several entries are **off-catalog** aspirational recommendations (e.g. red-light-panel, grounding-mat, infrared-sauna) that Glen makes but does not sell directly. Review/extend all three lists against Glen's actual recommendations and do-not-recommend rules before flip.

3. **Device family map for ownership exclusion (`_DEVICE_FAMILY_PREFIXES` in `dashboard/oasis_block.py`).** So the roadmap does not recommend a device the client already owns, real catalog device slugs are mapped to the roadmap's simplified hero slugs by prefix: `water-ionizer*` → water-ionizer, `harmony-laser*` → harmony, `kloud*` → kloud. Confirm these cover every real variant family (e.g. all water-ionizer plate counts, both Kloud PEMF sizes). As the catalog grows, this map must grow with it — otherwise a newly-slugged variant would fail to exclude its hero.

## How consumable vs device is decided (no `type` field in the catalog)

The catalog has no product-type field, so Replenish/Build-Out split by:
- **Consumable** (Replenish): `shipping.is_shippable()` True AND `bottle_type` in the dosed-supplement allowlist (`_CONSUMABLE_BOTTLE_TYPES` in `oasis_replenish.py`) or unset. Verified against the live catalog: 978 consumable / 107 excluded, all excluded being devices/books/services.
- **Device/tool** (Build Out › owned_from_us): the non-consumable ordered products.
If Glen adds a new dosed bottle_type, add it to `_CONSUMABLE_BOTTLE_TYPES` or those products will silently drop out of Replenish (fail-closed).

## Pre-flip render-verify checklist (headless, not just parse)

Drive an actual browser against a portal token with the flag on. Confirm:

1. "My Healing Oasis" tile appears in the **Act** group; clicking drills into the panel with a Back-to-hub control and a Replenish / Build Out sub-toggle.
2. **Replenish** lists the client's owned consumables (from order history), shows a "Running low" pill on items older than the threshold, and the **Reorder** button reaches the existing checkout (`/api/portal/<token>/checkout`) for that slug.
3. **Build Out › Tools you own with us** lists device/tool orders; **Tools you own elsewhere** lists self-reported tools; add a tool (name + brand) and remove it.
4. **Recommended to complete your Oasis** leads with Harmony / Water Ionizer / Kloud (hero, emphasized) above terrain/general items, and a device the client owns (from an order OR a self-reported real variant slug like `water-ionizer-9plate`) is NOT recommended.
5. "Add to wishlist" on a roadmap item persists and shows under **Wanted** in Build Out.
6. A "My Remedies › Add to my Oasis" item (once that feature is also live) shows under the same **Wanted** list (shared wishlist).
7. Copy scan: no em dashes, no ALL CAPS in any client string; light and dark themes both render.

## Known non-blocking follow-ups (from task reviews)

- New oasis routes use raw `sqlite3.connect(LOG_DB)` (matching the pre-existing `/recommendations` + `/remedies` route families; `LOG_DB` is sqlite-backed).
- The wishlist "Added" toast is ephemeral; the persistent state is the Wanted list.
- Replenish Reorder always sends `qty:1` (no quantity stepper on those rows).
- `build_block` return-shape docstring omits the `wanted` field (cosmetic drift).

## Relationship to My Remedies

My Healing Oasis and My Remedies (PR #1144) share one wishlist as the "Add to my Oasis" ↔ "Wanted" handoff surface. They are independent PRs (this branch is off `main` incl. the merged Health Profile #1143); both add a tile to `buildHubHtml`, so expect a small additive merge touch-up in `static/client-portal.html` when the second of the two merges.
