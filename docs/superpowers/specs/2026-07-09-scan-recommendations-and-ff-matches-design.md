# Scan recommendations + Functional Formulations matches — design

**Date:** 2026-07-09
**Status:** design approved, ready for an implementation plan
**Repo:** deploy-chat (Flask, Render service `glen-knowledge-chat`)
**Builds on:** family plan (#747), household member view (#750), in-funnel sales pages
(PR #171, `SALES_PAGES_ENABLED` live), scan history A+B (#605/#610)

---

## Problem

A client's E4L voice scan already produces a ranked set of matched infoceuticals. E4L
shows those results free on the client's own E4L account, so we charge nothing for the
information. Today they are invisible in our portal: production has never seen them.

At the same time, the one thing we *can* sell — Dr. Glen's clinical judgment, and the
fulfilment that follows it — has no surface in the portal at all.

This design puts the scan's recommendations on every client's portal, free, orderable;
generates a Functional Formulations match on request; and draws the paid line at
**review and fulfilment**, not at information.

## The line we are drawing

> Free members get the algorithm. Paid members get Dr. Glen.

Concretely: E4L's infoceuticals and the AI's FF matches are visible to everyone. What a
Family Plan or paid membership buys is (a) Dr. Glen reviewing and editing that FF list,
and (b) one-button add-to-invoice, which routes into the monthly consolidated shipment at
member pricing.

**This is NOT the Causal Biofield Analysis.** The $300 analysis — the causal chain, the
narrative, the audio, the PDF — remains gated by `_portal_biofield_unlocked` exactly as it
is today. The scan-recommendations card is a separate, always-visible surface that was
never behind that gate. `PORTAL_PAID_GATE_ENABLED` is unaffected and stays off.

> **Copy consequence:** the Family Plan's benefit list currently says "access to scan
> results (full)". That must be reworded to "your Causal Biofield Analysis", or a member
> will read it as something the free tier already gets.

## Decisions (resolved with Glen, 2026-07-09)

1. **Queue for review, but show the AI matches immediately**, with a disclose note.
   Rationale: the automated matches are mostly good now; latency is the bigger cost.
2. **Infoceuticals are orderable immediately.** They are E4L's deterministic output.
3. **The add-to-invoice affordance activates only after Dr. Glen publishes.** A retail
   purchase through the product page is available to anyone at any time; what waits for
   review is *our invoice for a list we recommended*. Nobody is ever invoiced for a
   remedy he replaced, so a review edit never triggers a refund or a change-of-scope
   conversation.
4. **Free members may order too** — every order button targets the product page. For a
   free member no review is coming, so the product page is their only order path, and the
   disclose copy must say so.
5. **All product pages are the new-style in-funnel pages.** Never remedymatch.com.
6. **The FF request button appears on animal pages as well.** Species does not gate it.
7. **Free-tier FF matches carry no dosing.** Names and meanings only. Dosing is a clinical
   instruction and arrives with the review.

## What we verified rather than assumed

- **Every sellable product already has a new-style page.** `/begin/product/<slug>` renders
  from catalog data for all 966 sellable products, including the 644 flagged
  `no_groovekart` and the ones with no `url`. Render-verified in a headless browser:
  correct name and price on an infoceutical with no storefront URL, on an "orphan"
  hologram, and on an FF. `SALES_PAGES_ENABLED` is live in prod.
  → There is no missing-product-page project. The `url` field is the *old* GrooveKart page.
- **`BFA` is never captured.** `item_code = 'BFA'` appears in **0 of 5,764** rows of
  `e4l_scan_results`. On the scan page it reads "Big Fields (BFA)" — no digit — and the
  extractor keys on letter+digit codes. The same blind spot hides every `BFA-*` and
  `ENV-*` item in `e4l_items`.
- **The match-strength colour is never captured.** `score` is `NULL` in all 5,764 rows.
- **`protocol_days = 15` is a faithful proxy for the purple band.** Only `ED`, `ET`, `EI`,
  `ES`, `MB` ever carry it; `ER` and `MR` (Glen: *"ER's are not infoceuticals"*) are always
  2-day. Across all scans, **269 have five items** in that band and **205 have four** —
  consistent with `BFA` being dropped whenever it is matched.
- **`priority_rank` alone does NOT reproduce E4L's top five.** Top-5-by-rank pulls in an
  `ER`, which is not an infoceutical.
- **The portal's existing order button charges a card immediately** (Stripe Checkout). So
  "add to invoice" cannot reuse it; it must create an order row instead.

## Architecture

Four slices. Each is independently shippable and flag-gated.

### Slice 0 — the parser (local, `e4l.db`)

Nothing downstream is correct until an animal's five infoceuticals are five.

- Teach the scan parser the `BFA` / "Big Fields" row, and the non-numeric `BFA-*` /
  `ENV-*` codes.
- Capture the colour band into a new `e4l_scan_results.match_band` column
  (`purple|yellow|green`). Keep `protocol_days` as the fallback proxy.
- Backfill: re-parse the ~205 scans whose infoceutical band holds only four items.

**Test:** a fixture scan whose page lists "Big Fields (BFA)" yields six rows, not five;
a scan already holding five in the 15-day band is unchanged by the backfill.

### Slice 1 — push recommendations to production

Production cannot read `e4l.db`. Today a bare manifest (`scan_date`, `scan_id`) is pushed
into `client_scans` via `POST /api/console/client-scans/sync`. This adds a sibling.

**New table `scan_recommendations`** (prod, `chat_log.db`):

| column | notes |
|---|---|
| `email` | lowercased; the scan owner, not the caregiver |
| `scan_id`, `scan_date` | |
| `item_code` | `ED6`, `ES7`, `BFA`, … |
| `rank` | `priority_rank` |
| `band` | `purple` \| `yellow` \| `green` |
| `category` | `ED`/`ES`/`ET`/`EI`/`MB`/`BFA`/`ER`/`MR` |
| `label` | from `e4l_items` |

- **New endpoint** `POST /api/console/scan-recommendations/sync` (console-key gated,
  `_db_lock`, upsert by `(email, scan_id, item_code)`).
- **New pusher** `02 Skills/e4l-scan-recommendations-push.py`, wired into the daily
  `e4l-daily-watch` beside the existing manifest push.
- Only genuinely newly-inserted rows are returned, mirroring `client_scans.upsert_scans`,
  so a backfill can never trigger a mass email.

**Test:** upsert is idempotent; a re-push of an unchanged scan inserts nothing; a scan for
an unknown email is stored and simply never displayed.

### Slice 2 — the scan-recommendations card

`api_client_portal` gains `payload["scan_recommendations"]` for the *currently selected*
scan date, honouring `?member=` exactly as the report does (the member's scan, not the
caregiver's). Best-effort; a failure must never break the portal load.

Rendered as a card in `static/client-portal.html`: the infoceuticals for this scan, each
with its label, its band, and an order button.

**`dashboard/order_destination.py`** — one function, one job, no Flask, no DB:

```python
def destination_for(slug: str) -> str:
    """The new-style in-funnel page for a product. Never the old storefront."""
    return f"/begin/product/{slug}"
```

It exists as a named seam rather than an inline f-string so that (a) it is unit-testable,
(b) the "new style only" rule has one enforcement point, and (c) a future change of page
route touches one line. `_resolve_buy_slug(name)` already maps a remedy name to a slug.

Gate: `SCAN_RECOMMENDATIONS_ENABLED`, default off; flag-off payload is byte-identical.

**Test:** the card lists exactly the scan's infoceuticals, ordered by rank; `?member=`
shows the member's, never the primary's; every order link is `/begin/product/<slug>` and
no link points at `remedymatch.com`; flag off → key absent.

### Slice 3 — FF matches, and the paid actions

**New table `ff_match_drafts`**: `(email, scan_date)` primary key, `items_json`,
`status` (`draft` | `published`), `created_at`, `updated_at`, `published_at`.

**`POST /api/portal/<token>/ff-matches`** — generate once, then return the cache.

- Rate-limited per email (a free member clicking a token-spending button is an obvious
  hole). `dashboard/analysis_quota.py` is the nearest existing pattern — free 1/month,
  paid unlimited — but its period is a calendar month keyed to a different action, so
  confirm the shape rather than reuse the table blindly.
- **Generated once per scan and cached.** A second view must return the same list; a
  member who reads five formulations, refreshes, and reads five different ones has lost
  all confidence in both lists.
- Writes a row into the existing `analysis_requests` rail (a new `kind='ff_match'`) so the
  local worker and `/console/biofield-reveals` pick it up with no new queue.
- **Animals and clients flagged "cannot take herbal/nutritional remedies"** receive the
  scan's infoceuticals as their recommendation, per Glen's rule, rather than an FF set.
  The button is still present (decision 6); its result differs.

**Display rules:**

| | free member | paid member / family |
|---|---|---|
| E4L infoceuticals | visible, orderable now | visible, orderable now |
| FF matches | visible, **no dosing** | visible, dosing after review |
| disclose copy | "matched automatically from your scan — Dr. Glen reviews these for members" | "AI-generated, pending Dr. Glen's review" |
| product-page order link | always | always |
| add-to-invoice | never | after Dr. Glen publishes |

Everyone can buy at retail through the product page at any time. **`add-to-invoice` is the
only affordance that waits for review**, because it is the only one where *we* bill for a
list *we* recommended. That is what makes decision 3 cost nothing: an edit before publish
cannot strand an invoice, and a retail purchase was never ours to change.

The free-tier disclose copy must not promise a review that is never coming. This is the
single most important string in the feature.

**Add-to-invoice** (paid only, gated on `family_plan.covers(email) or _is_paid_member(email)`):

- Creates an `orders` row, `pay_status='unpaid'`, `portal_published=0`. **Not** a Stripe
  Checkout session. Nothing is charged.
- Glen reviews and edits the FF list, publishes; Rae invoices the reviewed list through
  the existing composer.
- Because nothing is billed before publish, an edit is free. This is why decision 3 makes
  decisions about refunds unnecessary.

Gate: `FF_MATCHES_ENABLED`, default off.

**Test:** a second POST returns the cached draft, never regenerates; a free member's
payload carries no `dosing` key on any FF item; add-to-invoice 403s for a free member and
creates exactly one unpaid, unpublished order for a covered member; a covered member is
covered via their caregiver's plan (`family_plan.covers`), not only via their own payment;
double-clicking add-to-invoice does not stack two orders.

### Slice 4 — species

E4L's `Species` column (human / cat / dog / horse) is the durable signal. It is a fact E4L
already holds; a `client_prefs` toggle would be a fact somebody must remember to set.

- Sync `Species` into `e4l_clients.species` on the existing E4L pull.
- Push it alongside the recommendations (Slice 1).
- An animal's portal header reads **"Give our Aloha to <name>"** rather than "Aloha
  <name>", because the reader is the caregiver, not the animal.
- Drives the infoceutical-only recommendation rule in Slice 3.

**Test:** a human's header is unchanged; an animal's header reads "Give our Aloha to …";
a client with no species recorded is treated as human (fail-safe, never "Give our Aloha to"
a person).

## Non-goals

- Creating product pages. They already exist for all 966 sellable products.
- Linking the 65 infoceutical "twins" to their old GrooveKart URLs. We do not link there.
- Retiring the inert `no_groovekart` flag (worth doing; unrelated).
- Changing `PORTAL_PAID_GATE_ENABLED` or the Causal Biofield Analysis paywall.
- The monthly family shipment, member product discounts, and group-coaching access. Those
  are the Family Plan's other three benefits and remain unbuilt; they hang off the same
  `family_plan.covers()` predicate.

## Risks

**Unreviewed clinical content reaches free members at scale.** A free member sees an
AI-matched remedy list that no clinician will ever look at. Mitigations: no dosing, honest
disclose copy, and the fact that E4L's infoceuticals — the deterministic part — are what
carries the order button most prominently.

**The `covers()` predicate now gates money.** It already governs report blur; Slice 3 makes
it govern invoicing. It is one function, unit-tested, and it fails closed.

**Cache staleness.** A new scan for the same client produces a new `(email, scan_date)`
row, so drafts never collide. But if Glen edits the catalog under a published draft, the
stored item names can drift from the catalog. The draft stores slugs, not just names.

**Backfilling `match_band` touches 5,764 rows** of a live local database. Run on a copy
first; the column is additive and defaults to `NULL`, which reads as "unknown band" rather
than as "green".

## Open questions

None blocking. The Family Plan benefit copy needs rewording before anyone is sold the plan
(see "The line we are drawing"), and that is a content task, not a code one.
