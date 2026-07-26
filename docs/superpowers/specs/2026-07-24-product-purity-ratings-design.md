# Product purity ratings — red/yellow/green excipient screen

**Date:** 2026-07-24
**Status:** approved design, not yet planned
**Owner:** Glen

## Problem

The Fullscript dispensary channel (PR #1173) surfaces third-party professional products Glen
does not manufacture. Whether a product is worth recommending is not decided by its actives — it
is decided by its **excipients**, exactly as in the Pam Schreur supplement analysis (stearates cut
bioavailability, gelatin carries Seneff's glyphosate concern, dicalcium phosphate is avoided).

Nothing screens those excipients today. A client could be shown a stearate-laden product with no
signal that it fails Glen's own standard, and with no counterpoint to his cleaner formulation.

The opportunity, in Glen's framing: screen each relevant product, flag it red / yellow / green, and
let the aggregate show **how many "professional quality" products fail his standard** — an authority
signal that doubles as the argument for his own formulations.

## Decision

A two-tier rating built as a shared, product-keyed cache that composes three systems Glen already
shipped, plus one new core.

- **Tier 1 — excipient screen** against a Glen-authored avoid-list → a red / yellow / green color.
- **Tier 2 — the formulation-analyzer** (`dr-glen-swartwout-formulation-analyzer`) runs only on
  non-red products, ranking efficacy / dose / form / mechanism among what passed.

Both cache into a new product-keyed `product_ratings` table. Two consumers read it: the Fullscript
seed-gate (what the client sees, paired with Glen's `best_ff` formula), and the aggregate stat
(the public "% fail" authority number).

### Primary use (decided)

**Internal gate, aggregate stat public.** Only non-red products are eligible to appear to a client;
the aggregate failure *rate* is the public authority signal. Individual products are only ever
color-flagged privately, to the one client who requested that product's rating — never as a public
wall naming competitors. This threads the brand relationship: Fullscript is a brand-listing target
(b2b playbook), and Glen dispenses some of these brands, so publicly red-badging named competitors
is avoided while the honest failure rate still does the authority work.

### Reuse (confirmed against code)

- **On-request pipeline + gating + confirm-before-client** — `dashboard/supplement_reviews.py`
  state machine (requested→ai_draft→confirmed, never downgrade), `supplement_review_access`
  gating pattern, `/api/console/product-review/draft` analyzer hand-off, `product_review.confirm`
  action. This new system mirrors that shape; it does not replace it.
- **Tier 2** — the existing formulation-analyzer.
- **Green pairing + channel** — the Fullscript channel's `best_ff` (PR #1173) is literally
  "red product → Glen's clean formula".

### Non-goals

- No big upfront batch screen of the whole catalog. Processing is **on-request / gated**, and the
  ratings DB fills organically and caches.
- No public per-product red badge on named competitors. Only the aggregate rate is public.
- No change to the Fullscript matcher pool or Glen's own-formulation hardening.

## Section 1 — The avoid-list and the role-aware screen

### The avoid-list is a versioned data asset, not code

A file Glen authors and edits without a developer (JSON, in the repo; also the source of a
publishable "our standard" artifact). Two lists:

- **Red list** (disqualifiers): stearates, gelatin, dicalcium phosphate, titanium dioxide,
  artificial colors (FD&C lakes), hydrogenated oils, carrageenan.
- **Yellow list** (tolerated fillers / flow agents): silicon dioxide / silica and its kin.

Each entry carries:

| field | purpose |
| --- | --- |
| `canonical` | the canonical excipient name |
| `aliases` | every label wording that means the same thing (magnesium stearate = vegetable stearate = stearic acid; silicon dioxide = silica; hypromellose = HPMC) |
| `rationale` | short text in Glen's voice — the authority copy |

The file carries a **version stamp**. Every rating records the avoid-list version that produced it,
so a list change marks older ratings stale for re-screen rather than silently diverging.

### The screen is role-aware

The screen operates on a label parsed into `{actives, other_ingredients}` and reads **only
`other_ingredients`**. The same substance's role decides everything: silica in the Supplement Facts
is a nutrient and never counts; silica in Other Ingredients is a filler and goes yellow.

Algorithm, per product:

1. Normalize each Other Ingredient (lowercase, strip descriptors like "(vegetable source)").
2. Alias-match against the red list. Any hit → **red**, record which items hit.
3. Else alias-match against the yellow list. Any hit → **yellow**, record which items hit.
4. Else → **green**.

Red beats yellow: a product with both a red and a yellow item is red.

### The non-negotiable safety rule

**Absence of excipient data is never green.** If a product's Other Ingredients cannot be obtained,
its color is **unrated** — it never defaults to clean. A product shown green merely because we lack
its label is the same class of failure as an invented `best_ff` reaching a client. Unknown stays
unknown until a human supplies the label. This is enforced by the state machine (Section 2), not
left to a caller's discipline.

## Section 2 — The `product_ratings` database and the on-request flow

### Product-keyed shared cache

`product_ratings` holds **one row per product**, shared across all clients — the key difference
from `supplement_reviews`, which is one row per (client, product). Screen a stearate product once;
every client who asks reads the same verdict and it counts once toward the stat.

| column | notes |
| --- | --- |
| `product_key` | canonical brand+name key; primary identity, UNIQUE |
| `brand`, `product_name` | display |
| `fullscript_slug`, `fullscript_external_id` | set when it is a Fullscript catalog product |
| `other_ingredients_raw` | the verbatim label text, always stored so a human can verify the parse |
| `other_ingredients_parsed` | JSON array, normalized |
| `color` | red / yellow / green / unrated |
| `red_hits`, `yellow_hits` | JSON arrays of the avoid-list entries that triggered |
| `avoidlist_version` | the version that produced this color |
| `tier2_score`, `tier2_json` | null until the analyzer runs (never on reds) |
| `best_ff` | the paired Functional Formulations product |
| `status` | state machine below |
| `requested_at`, `screened_at`, `drafted_at`, `confirmed_at`, `updated_at` | |

### State machine (mirrors the shipped product-review flow)

`requested → screened (tier-1 color set) → ai_draft (tier-2 done on non-reds) → confirmed`, never
downgrading. **Reds skip tier-2**: a red goes `screened → confirmed` (no analyzer run, since a red
is already excluded), while a yellow/green goes `screened → ai_draft → confirmed`. Every color still
passes through Glen's confirm before it counts toward the public stat or reaches a client — the
screen is deterministic, but the aggregate is public, so a human ratifies before publication. A row
whose excipients could not be obtained rests in **unrated** and cannot advance to a color — the
Section 1 safety rule made structural.

### Gated and cached

A client reaches a rating on **paid membership** (e.g. `membership_category == 'full'`) or an
**explicit request**, reusing the `supplement_review_access` gating shape. Flow:

1. Confirmed row exists → return instantly.
2. Else create the row (`requested`), acquire the Other Ingredients (the two-step cascade below),
   run the screen, run tier-2 on anything not red, land in `ai_draft`.
3. Glen confirms in the console → cached + surfaced.

### Acquisition — a two-step cascade (decided 2026-07-26)

Excipient acquisition is the fuzzy, failure-prone step, so it is isolated, **async/operator-triggered
(a deferred `requested → screened` transition), not inline** — web search + fetch + extract is slow
(tens of seconds), so it never blocks the client's request behind a spinner.

- **Step 1 — online, automatic (no client burden).** Search for the product's Other Ingredients on a
  product listing or a label image online, fetch the candidate source, and extract. Prefer this so
  most products resolve without ever bothering the client.
- **Step 2 — client photo, only if Step 1 fails.** If nothing verifiable is found online, ask the
  client (in the portal) to upload a clear photo of the facts panel, then extract from the image.

**The fabrication guard binds BOTH steps and is the whole safety story.** Extraction accepts an
ingredient line only if it is a verified quote from the fetched source (the page text, or the vision
model's verbatim transcription of the image) — never a model guess. This reuses the shipped
`dashboard/document_extract.verify_quotes` + fails-closed discipline from the document-ingestion
feature (#1172). An online scrape that *guessed* ingredients is the worst outcome — it could
green-light a stearate product — so unverifiable Step-1 output is treated as **"not found"** and
cascades to Step 2; if Step 2 is also unavailable or fails, the row rests **`unrated`**, never a
color, never green. Acquisition failure never crashes the request.

**Reuse boundary:** Step 2 (image) reuses #1172's vision-extraction path directly (with a
supplement-specific prompt). Step 1 (usually HTML text, not an image) reuses the *guard pattern*
(`verify_quotes`, fails-closed, draft store) but needs its own text-source extractor. If a Step-1
source is itself a label image, the image path is reusable there too.

## Section 3 — The two readers

Only two consumers read `product_ratings`; nothing else does.

**Fullscript seed-gate.** The portal card shows a product only once it has a **confirmed, non-red**
rating, annotated with its color, each yellow/green paired with the `best_ff` formula. Reds are
suppressed at render. This is decoupled from the static seed file: the seed holds candidates, the
rating decides whether and how one appears, so a color change never regenerates the seed. Ties into
PR #1173.

**Aggregate stat.** A group-by-color count over confirmed rows — "of N professional products
screened, X% red, Y% filler-only, Z% fully clean." The public authority artifact; feeds
skepticalreviews and the content engine; honest, growing denominator framed as "of the N we have
screened."

## Error handling

- **Unrated-never-green** (Section 1), enforced by the state machine.
- **Re-screen on version bump**: a rating stores its `avoidlist_version`; a newer avoid-list marks
  older ratings stale for re-screen rather than silently diverging.
- **Red beats yellow** precedence.
- **Acquisition failure isolated**: leaves the row `unrated`, request still returns.

## Testing

Each guard written to actually fail against a broken implementation (the discipline that held all
through PR #1173):

- **Alias matching** — "vegetable stearate" and "stearic acid" both hit the stearate red entry.
- **Role distinction** — silica in `actives` is ignored; silica in `other_ingredients` goes yellow.
- **Red precedence** — a product carrying both a red and a yellow item resolves red.
- **Unrated-never-green** — empty or unobtainable Other Ingredients resolves to `unrated`, not
  green. (The safety guard; must bite under mutation.)
- **Aggregate math** — group-by-color over a known fixture.
- **Gating** — an uncached request triggers the pipeline; an unconfirmed rating never reaches a
  client; a cached confirmed rating returns without re-analysis.
- **Seed render** — a red product is suppressed; a yellow/green shows with its color and its
  `best_ff` pairing.

## Phasing (mirrors the Fullscript A1 build)

- **Phase 1 — the engine.** The avoid-list asset, the role-aware screen, the `product_ratings`
  table and state machine, all unit-tested with **manual excipient entry**. No UI, no acquisition
  at scale. Ships the core and its guards.
- **Phase 2 — the on-request flow.** Gating (membership/request), excipient acquisition, the
  analyzer hand-off, and the confirm console. Reuses the product-review infrastructure.
- **Phase 3 — the two readers.** The Fullscript seed-gate integration and the aggregate stat.

Each phase produces something testable on its own.

## Relationship to existing work

Composes: the free product review pipeline (`project_free_product_review`), the
formulation-analyzer, and the Fullscript channel (PR #1173). The only genuinely new core is the
avoid-list asset + the role-aware screen + the product-keyed `product_ratings` cache.
