# In-House Testimonial Collection — Phase 1 Design

**Date:** 2026-06-26
**Status:** Approved (brainstorm) — ready for implementation plan
**Owner:** Glen

## Problem

Testimonials are collected today through **Boast.io**, reached via the vanity short link
`Truly.VIP/Results` → `remedy-match-llc.boast.io/form/csat-customer-satisfaction-score`. Boast is a
paid SaaS that owns the collection form, the response dashboard, the on-site display widgets, and
social auto-publish (e.g. approved testimonials → the business Facebook page).

Glen wants to bring the **whole testimonial process in-house** — own the form, the data, moderation,
and (later) the display and social-sharing — on the existing deploy-chat / illtowell.com stack so
PHI-adjacent client praise stays under his control and the Boast subscription can be retired.

## Vision (full, for context) vs. this spec

Glen confirmed the destination includes all four capabilities:

1. **Submitter self-share** — after leaving a testimonial, one-tap share to their own social with a
   link back (viral "pay it forward" loop).
2. **Practitioner-branded collection** — a practitioner sends the form to *their* clients;
   testimonials attribute to that practitioner.
3. **Approved → branded social post** — an approved testimonial becomes a branded graphic/post that
   Glen/Rae/practitioners publish to Healing Oasis channels (Boast's Facebook auto-publish, in-house).
4. **Embeddable proof widget** — approved testimonials feed a widget embedded on the site / product /
   practitioner pages.

**This spec covers Phase 1 only: the in-house collection form that replaces the Boast form.** It
bakes in the two fields the later phases need *at collection time* (practitioner attribution + public
-use consent) so submitters are never re-asked.

### Phasing

- **Phase 1 (this spec):** branded standalone form → ingest → moderate in console; repoint short link.
  Video via **upload / paste-link** only (reuses existing engine handling).
- **Phase 1b:** add **record-in-browser** video capture (MediaRecorder) to the same form — the one
  genuinely new frontend piece, deliberately split out.
- **Phase 2:** embeddable proof widget (approved + `consent_public` testimonials).
- **Phase 3:** submitter self-share loop + approved→branded social post / Facebook auto-publish.

## Decision: extend the existing review engine (sentinel campaign slug)

We extend the existing **`product_reviews`** engine rather than build a standalone testimonials
module (Glen's choice). The engine partitions everything by `product_slug` — it is the media folder
name, the `UNIQUE(product_slug, email)` key, and the console label source. A general testimonial has
no product, so:

**Testimonials are stored in `product_reviews` under a reserved campaign key `product_slug = "_results"`**,
distinguished by a new `kind` column. This is the maximal-reuse approach: media path, console queue,
video transcription pipeline, and AI compliance scoring all reuse as-is. `_get_product("_results")`
returns `None`; the console already falls back to the raw slug string, so we only add a friendly
label.

Rejected alternative (B): add a nullable `campaign` column and relax the `UNIQUE` constraint —
cleaner semantically but touches the unique key and multiple queries on the working verified-buyer
product path for no Phase-1 benefit.

The `UNIQUE("_results", email)` constraint caps a person to one testimonial per campaign. This
matches Boast (re-submitting updates the prior entry via the existing `ON CONFLICT … DO UPDATE`
upsert) and is acceptable.

## Data model

Reuse `product_reviews`. Add three additive columns (same `ALTER TABLE … ADD COLUMN` /
`OperationalError`-swallow pattern already in `product_reviews.init_table`):

| Column | Type / default | Meaning |
|---|---|---|
| `kind` | `TEXT DEFAULT 'product'` | `'testimonial'` for this path; existing rows stay `'product'`. |
| `practitioner_id` | `INTEGER DEFAULT 0` | Attribution from a `?p=<token>` form-link param. `0` = unattributed. Resolved against existing practitioner records; stored as id so Phase 2/3 can group by practitioner. |
| `consent_public` | `INTEGER DEFAULT 0` | `1` only if the submitter ticks the public-use checkbox. Nothing is ever displayed or shared without this. |

`rating` (existing) carries the CSAT/star score. `body`, `video_kind`, `video_ref`,
transcription/AI columns are reused unchanged.

## Ingestion endpoint

**New `POST /api/testimonials`** — a sibling to `/api/reviews`; does **not** modify the existing
review endpoint.

- **No product lookup, no verified-buyer gate** (open submission, like Boast).
- **Required:** `rating` 1–5, `name`, `consent_public` checked (to submit a *public* testimonial).
  Email is **optional** but encouraged. A submission without consent may still be accepted but is
  stored `consent_public=0` and is never shareable/displayable.
- Reuses the existing **video upload / paste-link handling** and saves media under
  `/review-media/_results/...`. (Record-in-browser is Phase 1b.)
- Reuses the existing **video transcription job queue** verbatim when `REVIEWS_VIDEO` is on and a
  video was uploaded.
- Runs **`review_scoring.score_review`** for compliance with a synthetic product context
  `{"name": "Dr. Glen Swartwout — Biofield Analysis & Functional Formulations"}` so the
  no-diagnose/treat/cure compliance check still fires (this matters *more* for open testimonials than
  for product reviews).
- **No points / store-credit and no review-gift** on this path — it is ungated, and paying for
  ungated submissions invites abuse. Reward stays exclusive to the verified-buyer product path. The
  form's incentive is the existing "pay it forward, help us help others" framing.
- Inserts with `kind='testimonial'`, `product_slug='_results'`, `status='pending'`.

## Collection form (Boast replacement)

**Route:** `GET /results` — public standalone page; returns `404` when `TESTIMONIALS_ENABLED` is off.
Optional `?p=<practitioner_token>` captures attribution. This is the page `Truly.VIP/Results`
repoints to.

**Look:** reuse the existing dark-green/gold brand styling from `static/practitioner-share.html`
(`--bg:#0a150d`, gold `#d4a843`, Raleway / Open Sans), with the current Boast header line:
*"Biofield Analysis™ and Functional Formulations™ for Accelerated Self-Healing™ — Please pay it
forward… Help us help others!"*

**Fields (single screen, mirrors the Boast CSAT form):**

1. **Star rating** 1–5 (required).
2. **Your story** — free-text testimonial (`body`). Microcopy nudges structure/function language and
   away from disease claims, so more submissions pass compliance.
3. **Video (optional)** — upload **or** paste a link (Phase 1). Reuses existing size/length limits +
   transcription pipeline.
4. **Name** (required) + **email** (optional) + optional **photo**.
5. **Consent checkbox** (required for a public testimonial): *"I give Dr. Glen / Healing Oasis
   permission to share my words, photo, and video publicly (website, social media, marketing)."* →
   sets `consent_public`.

**On submit:** `POST /api/testimonials` → friendly thank-you state (reuse the `.confirmed` block
already in `practitioner-share.html`). No social-share buttons yet (Phase 3). If `?p=` was present,
the thank-you may note "shared with [practitioner]."

## Moderation & console

Testimonial rows land in the **same `/console/reviews` pending queue** (`pending_queue` is not
filtered by slug), so they appear for moderation automatically. Three small touches:

- **Label:** for `_results` rows, show **"Testimonial"** (or the practitioner's name when
  `practitioner_id` is set) instead of the raw `_results` slug.
- **Surface the new fields** on the moderation card: `consent_public` (never feature a non-consented
  testimonial) and attribution.
- **Media URL** already resolves via `/review-media/_results/<file>` — no change.

Approve / reject / feature reuse the existing `dashboard/reviews_actions.py` actions unchanged. No new
moderation UI in Phase 1; featured testimonials sit ready for the Phase 2 widget.

## Gating, short-link repoint, rollout

- **Flag:** new `TESTIMONIALS_ENABLED` (default off), independent of `REVIEWS_ENABLED`. Ships dark;
  flipped in Doppler `remedy-match/prd` when ready.
- **Repoint:** `Truly.VIP/Results` → `https://illtowell.com/results`. This is a redirect Glen
  controls on the Truly.VIP side — done manually at flag-flip, **not** code. Boast stays live until
  the switch → zero downtime.

## Testing

- `POST /api/testimonials` happy path: testimonial row created with `kind='testimonial'`,
  `product_slug='_results'`, `consent_public` stored, `status='pending'`.
- Compliance rejection: a disease-claim body scores `compliance_ok=false` (synthetic product context
  still triggers the check).
- **No-reward assertion:** the ungated testimonial path credits **no** points and creates **no** gift.
- Practitioner attribution captured from `?p=<token>` into `practitioner_id`.
- `/results` returns `404` when `TESTIMONIALS_ENABLED` off, `200` when on.
- **Whole-branch regression pass** confirming the verified-buyer **product-review** path is untouched
  — specifically the `UNIQUE(product_slug, email)` key, `_REVIEW_PAID_STATUSES`, points/gift credit,
  and `/api/reviews`. (Per the enum/unique-coupling lessons in memory.)

## Out of scope (Phase 1)

- Record-in-browser video capture (Phase 1b).
- Embeddable display widget (Phase 2).
- Submitter self-share buttons and approved→branded social post / Facebook auto-publish (Phase 3).
- Any change to the existing verified-buyer product-review flow, points, or gifts.
