# Fullscript dispensary channel

**Date:** 2026-07-23
**Status:** approved design, not yet planned
**Owner:** Glen

## Problem

Glen holds a verified Fullscript practitioner dispensary. Nothing in deploy-chat knows about it.
Four jobs go unserved today:

1. **Gap filling.** A client needs something outside the ~150 Functional Formulations SKUs and the
   answer is currently "I don't have that."
2. **Transition.** A client arrives on a shelf of Designs for Health / Thorne / Klaire. The free
   product review pipeline (#1138 / #1140 / #1141 / #1142, live) audits it and produces a verdict,
   then stops. There is no buy path for anything Glen does not formulate, so the re-order margin
   stays with whoever recommended it first.
3. **Practitioner convenience.** Practitioners in Glen's orbit dispense through Fullscript.
4. **Revenue.** Dispensary margin on third-party orders is meaningful on its own.

## Decision

Build Fullscript as **another channel listed separately in the portal, parallel to E4L and PRL.**
Not a change to the remedy matcher, whose pool was deliberately hardened to Functional Formulations
only on 2026-07-14 and stays that way.

Phase A ships the curated channel. Phase B wires the supplement-review loop. A full OAuth and
treatment-plan API integration is considered and **deliberately not built** (see Rejected below).

### Non-goals

- No change to `reveal` matcher pool membership. Fullscript products never enter remedy matching.
- No runtime calls to Fullscript from the app. The catalog is a committed seed file.
- No Fullscript OAuth, no treatment-plan creation, no partner integration in this scope.

## Prior art this copies

`dashboard/prl_supplement.py` plus `_prl_supplement_for` (app.py:21642) and the card at
`static/client-portal.html:2680` are the working template for a third-party channel:

- reference tables of third-party products, each carrying `best_ff` / `relation` / `ff_alts` so
  Glen's own formula always renders beside the third-party one
- matched off the client's E4L scan focus areas
- one portal card behind a flag, and when the flag is off the payload key never appears at all
- relation vocabulary is `complement` or `substitute`, defaulting to `consider`
  (`_prl_ff_view`, app.py:21632)

Fullscript reuses all of it, including the relation vocabulary verbatim so both cards render
consistently.

## Architecture

`dashboard/fullscript.py`, a direct sibling of `prl_supplement.py`. Schema and queries only, pure
sqlite, caller passes `cx`. No Flask, no app imports.

Payload builder `_fullscript_for(email, scan_date)` sits beside `_prl_supplement_for` in app.py.
Best effort: any error returns `None` and never raises.

Card rendered in `static/client-portal.html` as a sibling of the PRL card.

All tables live in `LOG_DB`, same as PRL and `recommendation_events`.

### Postgres note

Prod runs on Postgres since the P06 cutover. `cur.lastrowid` raises on the PG adapter, so any insert
needing its new id must use `RETURNING id`. Reuse `dashboard/dbwrite.insert_or_replace` the way
`prl_supplement.sync_from_seed` does rather than hand-rolling upserts.

## Data model

Six tables, created by `init_tables(cx)`: the catalog, four driver tables, and `fullscript_clicks`
(defined under Click tracking below).

### `fullscript_products`

The curated catalog slice. One row per dispensable product.

| Column | Notes |
| --- | --- |
| `name` | primary key |
| `brand` | e.g. "Jarrow Formulas" |
| `external_id` | Fullscript node id, e.g. `U3ByZWU6OlByb2R1Y3QtMTA3Njc2` |
| `product_slug` | Fullscript product slug |
| `url` | dispensary product URL, the only redirect destination ever used |
| `focus_tags` | JSON array |
| `product_type` | mirrors PRL's field |
| `best_ff` | best Functional Formulations equivalent, nullable |
| `relation` | `complement` or `substitute` |
| `ff_alts` | JSON array of alternative FF products |
| `source` | `seed` or `api`, so a later documented-API sync can augment rows without a schema change |
| `active` | soft delete |

`external_id` and `source` exist from day one specifically so an API-sourced row and a hand-seeded
row are the same shape.

### Driver tables

| Table | Driver | Columns |
| --- | --- | --- |
| `fullscript_focus_area_products` | E4L scan | `focus_area_id`, `focus_area_name`, `fs_product_name`, `rank` |
| `fullscript_condition_products` | condition / intake | `condition_key`, `fs_product_name`, `rank` |
| `fullscript_client_pins` | operator | `email`, `fs_product_name`, `note`, `pinned_by`, `pinned_at` |
| `fullscript_review_links` | supplement review | `review_id`, `fs_product_name`, `rank`, `created_at` |

`fullscript_focus_area_products` mirrors `prl_focus_area_products` exactly. `fullscript_condition_products`
keys on the same condition keys that drive Eye Support Programs.

`fullscript_review_links` is keyed by `review_id` deliberately: `supplement_reviews` is live in prod
and its `review_text` is prose with no structured replacement field. A join table adds the mapping
without altering a live schema.

## Resolver

```
candidates_for(cx, email, scan_date=None) -> [candidate]
```

Runs the four drivers, each returning `(product_name, origin, reason)`. Union, dedupe by product name
keeping the highest-priority origin, then rank.

**Origin priority:** `pinned` > `review` > `scan` > `condition`.

A pin is an explicit clinical decision by Glen, so it outranks anything derived. Rationale for the
rest: the more specific the evidence about this particular client, the higher it sorts.

Every candidate carries its FF-equivalent view via the same `_prl_ff_view` shape, so Glen's own
formula renders beside the third-party product in every case.

Drivers are separate functions with one signature so adding a fifth later is a function plus a
registry entry, not a resolver rewrite.

## Portal surface

One "Fullscript" card, sibling to the PRL card, grouped by origin with a plain-language heading:

| Origin | Heading |
| --- | --- |
| `pinned` | Chosen for you |
| `review` | Replaces something you're taking |
| `scan` | Matched from your scan |
| `condition` | For what you're working on |

Each row shows product name, brand, the FF-equivalent chip, and a buy button.

Payload key is `fullscript`, gated by `fullscript_enabled`, mirroring `prl_supplement` /
`prl_supplement_enabled`. When the flag is off the key is absent and responses stay byte-identical.

## Click tracking and the redirect

### Why this does not use `recommendation_events`

The obvious move is to register `fullscript` in `dashboard/recommendation_sources.py` and call
`record_click`. That is wrong for phase A, for two reasons.

`record_click(cx, email, product_key, source_key)` does not validate keys itself, but
`product_sources()` groups by `product_key`, and both the portal recommendations block and the
console 360 hub resolve those keys against the storefront catalog. The canonical `product_key` is a
storefront slug. A Fullscript product has no storefront slug, so a `fs:`-namespaced key would render
as an unresolvable product in two live client-facing surfaces.

Second, PRL is the channel this design is explicitly modeled on, and PRL does not write to
`recommendation_events` at all. A separately-listed channel stays self-contained.

So phase A owns its own table and leaves the unified view alone.

### `fullscript_clicks`

| Column | Notes |
| --- | --- |
| `email` | resolved from the portal token only |
| `fs_product_name` | FK to `fullscript_products.name` |
| `origin` | which driver surfaced it, for measuring which driver earns clicks |
| `clicked_at` | |

Folding Fullscript into the unified recommendations view is a deliberate later question (see A3),
because it requires teaching both display surfaces to resolve non-storefront keys. Not free, not in
phase A.

### The redirect

The existing `/r/<token>/<source>/<slug>` redirect cannot carry this. It validates the slug against
`_get_product` and always 302s **internally** to `/begin/product/<slug>`. Fullscript needs an
outbound hop.

New route: `GET /fs/<token>/<product_slug>`, keyed on the Fullscript product slug, which is unique,
stable and readable. `fullscript_products.name` stays the primary key so the driver tables reference
`fs_product_name` exactly as PRL's do.

1. Resolve client email from the **portal token only**, never from a request field.
2. Look up the row in `fullscript_products` by `product_slug`. Unknown or inactive means 302 to `/`.
3. Insert a `fullscript_clicks` row, failure-isolated.
4. 302 to `row["url"]`.

**The destination is read from the database row, never from the request.** Additionally the URL is
host-checked against a Fullscript allowlist before redirecting. Together these make the route
structurally incapable of becoming an open redirect. Failure to record a click must not block the
redirect.

## Seed generation

`data/fullscript_seed.json`, same three-part shape as `data/prl_seed.json`
(`products`, `focus_area_products`, `focus_area_items`), loaded by an idempotent
`sync_from_seed(cx, seed)` that full-replaces the reference tables.

The seed is generated **once, offline** by a script in `scripts/`, not by the app. It queries the
public unauthenticated catalog endpoint at browsing volume, maps results to FF equivalents, and
writes the JSON for Glen to review and correct. The committed file is the source of truth. The
running app never contacts Fullscript.

Coverage target for the first seed is the same 15 focus areas PRL already covers, so the two
channels reach parity on day one.

### On the endpoint used

`POST https://fullscript.com/api/fs-graphql` backs the public open catalog, needs no authentication,
and returns products, brands and ingredients with brand name and product slug. It expects `variables`
as a JSON-encoded **string**, not an object.

It is undocumented and unversioned, and Fullscript is the company PLAYBOOK.md names as the
highest-priority brand-listing target. Building a production dependency on their internal endpoint is
a bad trade against that relationship. Hence: one-time offline seed generation only, and the
documented `catalog` OAuth scope is the path for anything that ever needs to run on a schedule.

## Configuration

| Name | Purpose |
| --- | --- |
| `FULLSCRIPT_ENABLED` | master flag, off means the payload key never appears |
| `FULLSCRIPT_DISPENSARY_URL` | fallback "browse the whole dispensary" link on the card |

Flags are read at startup, so flipping one in Doppler needs a Render restart, and a merge plus a
flag flip is two deploys.

## Phasing

**A1, the spine.** `dashboard/fullscript.py`, six tables, seed loader, resolver with the `pinned`
and `scan` drivers, portal card, `FULLSCRIPT_ENABLED`, the `/fs/` redirect, `fullscript_clicks`.
Ships dark.

**A2.** Condition driver on the Eye Support Programs condition keys. Console UI to pin a product to a
client.

**B.** Review driver plus a console control to attach Fullscript products when confirming a
supplement review, closing the loop the review pipeline is missing.

**A3, optional and explicitly deferred.** Fold Fullscript into the unified recommendations view by
registering the source and namespacing keys as `fs:<product_slug>`. Requires teaching the portal
recommendations block and the console 360 hub to resolve non-storefront keys. Only worth doing if
Glen wants third-party products in the single per-client picture; the channel works fully without it.

## Testing

Mirrors PRL's tests, plus three specific guards:

1. **Flag off is byte-identical.** Payload with `FULLSCRIPT_ENABLED` unset contains no `fullscript`
   key at all.
2. **Redirect cannot be abused.** A row whose `url` points off-host is refused; no request-supplied
   URL is ever honored.
3. **Cross-client isolation.** Client A's token cannot surface or click client B's candidates.

Plus: resolver dedupe and origin-priority ordering; seed sync idempotency; every driver returning
empty yields no card rather than an empty one.

Note the standing hazards: `$DATA_DIR` strips `products.json` in the full suite, so pin
`load_products` to the repo file; and a bare full-suite run sends real email.

## Rejected: full OAuth and treatment-plan API

Fullscript's "treatment plan dynamic link" reads as if it generates patient-facing plans. It does
not. Per their own docs the redirect lands **the practitioner** in their own dispensary to add
products by hand, and it requires OAuth plus partner onboarding through `integrations@fullscript.com`.
It is built for an EHR's "Recommend on Fullscript" button.

High cost, external dependency on someone else's approval timeline, and a low ceiling because a human
still works the dispensary by hand. The curated channel delivers the four jobs without it.

## Parallel track, no code

PLAYBOOK.md ranks getting Functional Formulations **listed on** Fullscript as highest priority for
Q3 2026: third-party cGMP certification, MAP acceptance, product data and photography for the first
20 to 30 SKUs, 3 to 6 month timeline. That runs alongside this build and shares nothing with it
except the relationship. Keeping this build off their undocumented endpoints protects it.
