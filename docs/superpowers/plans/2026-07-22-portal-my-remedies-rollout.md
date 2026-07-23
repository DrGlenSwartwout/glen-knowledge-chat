# Portal — My Remedies rollout + pre-flip verification

Companion to `2026-07-22-portal-my-remedies.md`. The tile is built on branch `sess/f9b73eaa` and ships DARK. This note is the gate before flipping it on.

## The flag

- **`PORTAL_REMEDIES_ENABLED`** — default OFF; truthy set `("1","true","yes","on")` (mirrors `_PORTAL_HUB_ENABLED`). Off = byte-identical portal (no tile, endpoints 404). The tile only renders when the hub itself is on (`_hub && v.remedies && v.remedies.enabled`), so `PORTAL_HUB_ENABLED` must also be on for the tile to appear.
- **Independent of `SUPPLEMENT_REVIEW_ENABLED`.** The external-list request-review sub-flow reuses the `supplement_reviews` pipeline (requested -> ai_draft -> confirmed) and its per-client access gate, which have their own flag/console approval. `PORTAL_REMEDIES_ENABLED` gates the tile + the `/remedies/*` endpoints; a review only becomes visible once Glen confirms it in the console, regardless of this flag.
- To flip in prod: `doppler set PORTAL_REMEDIES_ENABLED=on -c prd` (prod secret write may be classifier-blocked — see the finder-flag note in the hub plan). Merge = self-deploy + a brief illtowell 502 (dark, low-risk).

## BLOCKER before flip — Glen must authorize the upgrade map

`dashboard/remedy_upgrades.py._UPGRADE_MAP` encodes **clinical product-swap recommendations** (external product -> our equivalent). It was seeded by a subagent and needs Glen's sign-off, because a swap is a clinical/commercial claim, not a mechanical mapping. Current seed:

| External (normalized) | Suggested our product | Note |
|---|---|---|
| magnesium glycinate | `neuro-mag` | dead slug in catalog -> safely returns None today |
| fish oil | `wholomega` | |
| vitamin d | `vitamin-d-syntropy` | |
| turmeric | `curcumin` | |
| coq10 | `coq10` | self-map; suppressed by the own-product guard |
| zinc | `zinc-syntropy` | |
| b12 | `sublingual-b12` | |

Do NOT flip until Glen has reviewed/edited this list against his do-not-recommend and preferred-swap rules. The guard already suppresses a swap when the client's product IS ours (by brand or by matching our product name), so the map only ever *adds* a swap where Glen has explicitly listed one.

## Pre-flip render-verify checklist (headless, not just parse)

Drive an actual browser against a portal token with the flag on. Confirm:

1. "My Remedies" tile appears in the **Understand** group, between My Analysis and the rest; clicking it drills into the panel with a Back-to-hub control.
2. **Top recommended** section lists the same items (same order) that `/api/portal/<token>/recommendations` returns for that client — it is a read-through, so they must match.
3. Ranked-item `reason` shows only the client's own note (never an operator note).
4. **External stack:** add a product (brand, product, reason, importance 1-10); it appears as "On your list". Edit the reason only -> importance is preserved (and vice versa). Remove it -> gone.
5. **Request a review** on a listed item -> status flips to "Review in progress", and the item shows in the console review queue. After Glen confirms in the console, the review text becomes visible on the portal.
6. **Upgrade pointer** renders only on external items that map to one of our products, and never on an item that already IS our product.
7. **Add to my Oasis / order** on a ranked item -> the product lands on the client's wishlist (idempotent; a second click keeps it, never removes it) and records a `my-remedies` engagement event.
8. Copy scan: no em dashes, no ALL CAPS in any client-visible string; light and dark themes both render.

## Known non-blocking follow-ups (from task reviews)

- `/remedies/to-oasis` returns `{"ok": true}` even when `slug` is missing from the body (only the trusted button calls it today).
- `"my-remedies"` is not registered in `dashboard/recommendation_sources.RECOMMENDATION_SOURCES`, so its engagement events render with the default bullet icon (cosmetic only; `record_click` stores the event fine).
- `esc()` does not scheme-validate `href` values (file-wide pre-existing pattern; the ranked/upgrade urls are server-sourced from the products catalog, so low risk here).
- Separate pre-existing issue surfaced during this build: `/api/portal/<token>/recommendations` already emits console-only `operator_note` to clients verbatim. Out of scope for this tile, needs its own decision.
