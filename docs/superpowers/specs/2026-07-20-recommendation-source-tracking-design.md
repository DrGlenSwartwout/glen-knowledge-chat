# Recommendation Source Tracking & Client Recommendation Portal — Design

**Date:** 2026-07-20
**Status:** Design — brainstormed & validated with Glen; ready for phased implementation plans
**Repo:** `deploy-chat`
**Supersedes:** the lightweight "Option B" follow-up card (`00 System/client-360-option-b-followup.md`) and the "Deferred (Option B)" section of the client-360 spec.

## Purpose

Give every product a per-client **provenance**: which sources brought it onto that
client's radar, how many *actions* each source has driven, and in what order those
sources first appeared. Surface it two ways:

- **Client-facing portal** — the client sees their recommended/discovered products
  grouped by source, each product carrying an icon row (one icon per source, in
  first-touch order, each with an action count), ranked so the products they keep
  reaching for rise to the top, each with an order button.
- **Operator side** — the client-360 process strip (`/console/client`) reads the
  resulting per-line sources as multi-badges, replacing the Option A email/latest-cycle
  heuristic with hard data captured at the point of ordering.

This replaces *inference* (client-360 Option A) with *capture at the ordering surface*:
the recommendation lists carry order buttons, and clicking one stamps the resulting
order line with its source — a true hard link, no guessing.

## Background

The client-360 hub (`/console/client`, shipped #1092) tags the process strip's
Recommendation stage heuristically because `orders` has no source link. This design
supplies that link at the source, and generalizes it into a full cross-source
recommendation-provenance system.

## Non-goals

- Not a re-architecture of ordering/checkout. Order buttons feed the existing order
  pipeline; we add a per-line `source`, not a new order system.
- Not marketing-attribution analytics dashboards. We capture the event data that would
  power those later; building reports is out of scope.
- No deletion of history. "Remove" is a client-set visibility flag; data is always kept.
- Marketing-channel instrumentation (email/newsletter/ads/social click capture) is
  Phase 3 — designed here, built later.

## Core model

### Source registry (extensible)

Sources are a **registry**, not a hardcoded enum. Each entry:

```
{ key, label, icon, kind, counting_rule }
```

- `kind` ∈ `clinical` (recommendation-generated) | `engagement` (client-action) — see
  counting rules.
- Current registry: `biofield`, `scan`, `intake`, `chat`, `self`, `email`, `newsletter`,
  `ads`, `social`, `purchased`. Adding a source later is a registry row + an icon, not a
  schema change.
- The `icon` is the badge shown in the portal; it carries a count at its center.

### Event log (append-only, never deleted)

One row per counted action:

```
recommendation_events(
  id, client_email, product_key, source_key,
  occurred_at, origin_ref, created_at
)
```

- `product_key` = a stable product identity (see Product identity below).
- `origin_ref` = the originating record (scan id, reveal id, order id, campaign/message
  id, …) for traceability. Nullable where not applicable.
- Append-only. Nothing is updated or deleted; corrections are new rows.

### Counting rules (what is an "action" per source)

An action is an **intentful** event; passive exposure (opens, impressions, views) never
counts.

| Source | `kind` | +1 when |
|---|---|---|
| biofield | engagement | the client ACTS on a reveal for the product — clicks to learn about it, or orders it (NOT merely that a reveal was generated) |
| intake | clinical | the product is recommended from an intake |
| scan | engagement | client opens the product page and interacts / adds to list / orders (NOT when the AI match is generated) |
| chat | engagement | client opens the product page and interacts / adds to list / orders from a chat CTA |
| self | engagement | client adds it from a product page ("add to my portal") |
| email | engagement | client clicks the product's CTA link (not opens) |
| newsletter | engagement | client clicks the product's CTA link (not opens) |
| ads | engagement | client clicks through to the product |
| social | engagement | client clicks through to the product |
| purchased | engagement | an order for the product is placed (re-orders increment) |

Rationale: biofield/intake are deliberate and scarce, so the recommendation itself is
the signal; scan/chat are AI and can generate many matches, so only client engagement
counts.

### Derived aggregates (per client × product × source)

Computed from the log (materialized or on read — decide in the plan):

- `count` = number of qualifying events.
- `first_touch` = min(occurred_at) — drives icon ORDER.
- `last_touch` = max(occurred_at) — drives the recency tie-break in sort.

### Client / product preferences

A per-(client, product) preferences record (folds in Phase 1's `recommendation_hidden`):

- **hide** flag — client-set; removes the product from the client's portal view but keeps
  all event data (it still counts on the operator side). Un-hideable.
- **operator note** — free text, authored by the operator in the console, READ-ONLY to the
  client, shown next to the remedy in the portal (personalized guidance / dosing / "why
  this one for you").
- **client note** — free text, authored and edited BY the client on the portal (their own
  note on the remedy).

Plus a per-(client, section) **section-collapse** state: each portal category can be
shown/hidden by the client and the choice is remembered across visits (persisted per client
off the portal token, so it survives across devices — not just localStorage).

### Product identity

`product_key` must be a **stable** identifier, not a display name — repo bottle names ≠
prod names (two naming universes). Resolve to the product slug/SKU used by the store and
by `items_json` lines. The plan must confirm the one canonical key and a resolver from
each source's own product reference to it.

## Order-line source capture

The hard link lives on the **order line**, captured at ordering — not inferred.

- Storage: a `source` key on each `items_json` line dict. **It MUST be added to the
  invoice line-item whitelist** or it is silently dropped between save and render (see
  `reference_invoice_line_item_render`); verify by RENDERING the invoice/portal, not just
  by inspecting the saved payload.
- **Portal/product-page order buttons** carry the source of the list/button they were
  clicked from → the created line's `source` = that source.
- **Phone/email (manual) orders**: the console order-entry (`/orders/new`) gets a per-line
  `source` picker, defaulting to `self`, changeable. Example: Bobbi's biofield-recommended
  items stay `biofield`; the extra products she piled on = `self`.
- **Every order placement also emits a `purchased` event** for each product (re-orders
  increment its purchased count).
- Legacy lines (pre-capture) have no `source`; leave them `unknown` or hand-code the
  important ones. Order-level inference (`FFINV-{email}-{scan_date}` external_ref;
  `items_json LIKE '%biofield-analysis%'`) remains available as a fallback but is never
  used to fabricate per-line provenance.

## Presentation — client portal

- Products grouped by source **category**. Each category is a **collapsible section** with
  a show/hide control whose state is **remembered per client** across visits (see
  section-collapse above).
- Within a category: sorted **count DESC, then last_touch DESC** (recency tie-break);
  show **top 5** with a **"show more"** expander.
- Each product row shows:
  - an **icon row** — one icon per DISTINCT source, in first_touch order, each icon
    carrying its `count` at center (the row reads as the product's journey, e.g. self →
    scan → email → biofield);
  - an **order button** (carries this list's source);
  - a **hide** control;
  - the **operator note** (read-only guidance) and the client's own editable **client
    note**.
- Purchased is a source like any other: a product bought repeatedly shows a purchased
  icon with its re-order count and can rank in a "Purchased / your staples" grouping.
- Client-facing surface uses the existing portal token identity (writes/reads key off the
  portal token, not an email in the URL).

## Presentation — operator (client-360 process strip)

The `dashboard/client_360.py::process_strip` Recommendation stage renders the DISTINCT
set of sources as **multi-badges**, replacing the single-source heuristic. Data source is
phase-dependent: in **Phase 1** it reads the client's recommendation events (biofield /
scan / purchased ingested); from **Phase 2** the authoritative source is the current
order's per-line `source` values, with events as the fallback for legacy lines. No change
to the Invoice→Sent→Paid→Fulfilled stages.

## Phasing

### Phase 1 — the spine (backend + operator; lowest risk)
- Source registry + `recommendation_events` log + derived aggregates (count, first_touch,
  last_touch) + client hide flag.
- Ingest ONLY `purchased` (paid `orders`) — the one source with a real recorded ACTION
  today (a paid order line). **biofield, scan, intake, chat are all deferred**: every one
  of them now counts a client ACTION (biofield = acted-on reveal; scan/chat = engagement;
  intake = its own signal), and none of those actions are instrumented until Phase 2's
  reveal-click tracking + order-line source capture. Ingesting reveal/draft *generation*
  would count exposure, not action — exactly what this system rejects.
- The operator Recommendations section therefore shows purchased staples (with re-order
  counts) in Phase 1; the presence-based multi-badge strip still surfaces biofield / scan /
  intake / chat *presence* (not counts).
- Surface the aggregates on the OPERATOR client-360 hub (a "Recommendations" section:
  products with per-source icon+count), driven by lazy ingest-on-read — a Phase-1 preview
  of what the client portal (Phase 2) will show, exercising ingest→aggregate→display
  end-to-end.
- Wire the client-360 process strip to multi-badge (present sources), via presence
  detection in Phase 1 (authoritative per-line source arrives in Phase 2).
- No new client UI, no order-creation changes, no marketing integrations.

### Phase 2 — client portal + ordering (the payoff; touches money path)
Multi-subsystem — built as four sub-slices, each its own plan/PR:

- **2a — order-line source capture (foundation, money path):** a `source` key on each order
  `items_json` line, captured at order time — the manual per-line source picker (default
  `self`) in `/orders/new`, the portal add-to-invoice buttons stamping their source
  (FF→`scan`, support→`intake`), and the render whitelist (`_invoice_line_view`) carrying
  `source`. On order placement (`upsert_order` choke point), emit an "acted-on"
  `recommendation_events` event per sourced line (idempotent, failure-isolated so it can
  never break order creation). This is distinct from the later paid `purchased` event.
  - *Edit caveat (append-only):* if an operator changes a saved line's `source` (or slug)
    and re-saves, a new acted-on event is emitted while the prior one remains (events are
    never deleted). Two rows for one line is expected; 2b's counting reads the log as-is.
    Server-side validation of the `source` value (registry-only) is deferred to a later
    registry-validation slice — today the console picker offers only the 5 valid values.
- **2b — client portal UI:** the categorized portal (collapsible sections with remembered
  state, icon rows + counts, top-5 + show-more, per-product hide control, operator note +
  client note), reading `product_sources` + the per-product prefs. Wire the client-360
  process strip to prefer per-line `source` (2a) over the presence heuristic.
- **2c — product-page "add to my portal":** a product-page button → `self` source (adds the
  product to the client's portal for future); requires tying a product-page visitor to a
  client identity (portal token / logged-in email).
- **2d — reveal / engagement click capture:** a `biofield` event when the client clicks a
  reveal's product link or orders from the biofield list; `scan`/`chat` events on the same
  click/add/order actions. Where the deferred clinical/AI sources start accruing on real
  client actions rather than generation.

### Phase 3 — marketing channels (separate integrations)
- Email + newsletter CTA-click capture (GHL / email-content-engine tracked links) → events.
- Then ads and social click-through capture — each its own external-instrumentation effort.

## Open questions (resolve in planning)

- **Product identity key** — confirm the one canonical `product_key` (store slug/SKU) and
  a resolver from each source's product reference (reveal remedy name, FF item, order line
  slug, campaign product ref).
- **Aggregates materialized vs on-read** — event log is source of truth; decide whether
  counts are computed on read (simpler, fine at current scale) or materialized (needed only
  if the portal query gets slow).
- **Where the client-facing portal lives** — extend the existing client portal
  (`client_portals` / portal token) vs a new surface; must reuse portal-token identity.
- **Icon set** — one icon per source; define the glyphs (Phase 2 UI detail).
- **"Add to my portal" without an account** — how a product-page visitor is tied to a
  client identity for the `self` write (portal token vs logged-in email).

## Testing strategy (per phase)

- Phase 1: pure ingestion + aggregate functions unit-tested against synthetic
  `biofield_reveals`/`ff_match_drafts`/`orders` rows (count, first/last touch, dedup by
  source, hide flag); process-strip multi-badge read tested with mixed-source fixtures.
- Phase 2: order-line `source` survives the whitelist end-to-end (render, don't just
  inject); manual picker default + override; `purchased` event emitted on order; portal
  sort (count DESC, recency tie-break) + top-5/show-more + hide.
- Phase 3: each channel's click event maps to the right `(client, product, source)` and
  excludes opens/impressions.
- Respect the deploy-chat CI known_failures ratchet; no bare full-suite runs (sends live
  email).
