# Affiliate ↔ Funnel Integration — Design

**Date:** 2026-06-04
**Domain:** illtowell.com (live, serving the deploy-chat app)
**Branch:** sess/ec0e1f15

## Goal

Integrate the affiliate system into the `/begin` funnel at illtowell.com. Four
independent workstreams, all surgical (no schema/commission/GHL changes):

- **A.** Explore page gains a "Recommended Tools & Partners" section + an "Earn by
  Sharing" affiliate on-ramp door.
- **B.** Partner links surface contextually where relevant (reuse existing chat
  trusted-links mechanism; confirm Blushield flows through it).
- **C.** Visual/brand consistency pass on the three affiliate pages.
- **D.** Customer-facing URLs point at illtowell.com instead of the onrender domain.

## A. Explore sections

**Source of partner links:** `data/trusted-links.json`. Add `"affiliate": true`
to the entries that are external partner/affiliate links (Blushield + the six
`amzn.to` entries). Internal links (E4L → truly.vip) stay unflagged so they do
NOT appear in the partner section.

**New helper** `begin_funnel.partner_links()`:
- Reads `_TRUSTED_LINKS` (passed in or imported), returns a list of
  `{name, url, note}` for entries where `affiliate is True`.
- Stable order = insertion order of the JSON.

**`_EXPLORE_LAYOUT` additions** (rendered by existing `explore_sections(ref)`):
- `"Recommended Tools & Partners"` — cards built from `partner_links()`. All
  external (`target=_blank`). Hrefs are bare (Amazon strips unknown params;
  Blushield `/heal` is already Glen's affiliate path). A disclosure line renders
  at the foot of this section: *"As an Amazon Associate, Healing Oasis earns from
  qualifying purchases."*
- `"Earn by Sharing"` — single card → `/affiliate`, ref-threaded
  (`?ref=<slug>` when a ref cookie/param is present), giving the affiliate
  program its own directory door (today it only lives inside `/begin/path`).

`explore_sections()` returns sections of cards; partner cards need a `disclosure`
field on the section (or a sentinel) so the template can render the Amazon line.
Template `static/begin-explore.html` renders `section.disclosure` when present.

## B. Contextual surfacing

The chat/match system prompt already injects every `trusted-links.json` name so
the model recommends them by exact name when they fit, and the whitelist
auto-opens them. Blushield was added last commit, so it already flows through
this. Work here is: (1) confirm Blushield resolves via the existing
name→URL resolver, (2) no new mechanism required. The affiliate on-ramp already
surfaces contextually via the `pay_forward` card.

## C. Visual/brand consistency

Chrome-align `static/affiliate.html`, `affiliate-portal.html`,
`affiliate-hub.html` with the `begin-*` pages:
- A "← back to funnel" header link to `/begin/explore` (ref-preserving).
- Consistent footer + the education-spirit + Amazon disclosure where partner
  links appear.
- Same container max-width / spacing tokens. Fonts already match; this is
  alignment, not a rebuild.

## D. illtowell.com everywhere

- `PUBLIC_BASE_URL` default → `https://illtowell.com` (env still overrides).
- `app.py:2806` share-link and `app.py:4659` `recruit_url`: derive from
  `PUBLIC_BASE_URL` instead of the literal onrender host.
- **Leave** the two `RENDER_EXTERNAL_URL` / `RENDER_EXTERNAL_HOSTNAME` fallbacks
  (lines ~7133, ~10309): they are Render's internal self-call URLs, not
  customer-facing.

## Testing

- `partner_links()` unit: Blushield present, an `amzn.to` entry present, E4L
  absent; shape `{name,url,note}`.
- Route test: `/begin/explore` HTML contains both new section titles and the
  Amazon disclosure string.
- Guard test: no customer-facing source emits a hardcoded
  `glen-knowledge-chat.onrender.com` (allow-list the two RENDER_* fallbacks).

## Out of scope

Commission logic, affiliate DB schema, GHL workflows.
