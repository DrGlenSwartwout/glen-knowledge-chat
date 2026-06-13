# Unified People Hub — Phase 1 (Aggregation)

**Date:** 2026-06-12 · **Status:** implemented (Phase 1)

## Problem

The Console **People tab** should be one comprehensive directory of every contact
(prospects, clients, practitioners, PR/media, affiliates, partners, vendors), and
(Phase 2) every contact should also exist in GoHighLevel, tagged, so GHL can drive
automated communications. Today the data is fragmented: the `people` table already
dedupes funnel leads + Practice Better clients by email, but **practitioners** live
only in Supabase and **PR/media** contacts only in vault CSVs, and there is no
consistent contact-type tagging.

## Decisions

- **Approach A, phased**: keep `people` as the hub; add one-way feeders; mirror to GHL later.
- **Consent policy**: store everyone, automate only opted-in. Cold/scraped get a suppression tag.
- **Taxonomy** (multi-valued, additive on merge), stored as prefixed tags in `people.tags`:
  `type:{prospect,client,practitioner,practitioner-cold,pr-media,affiliate,partner,vendor}`
  and `consent:{opted-in,cold-no-consent}`.

**Phase 1 = aggregation + a People-tab contact-type filter. No GHL push, no sends.**

## Implementation

All in `app.py` + `static/console.html` + one local script + tests.

1. **Additive upsert** — `_upsert_person_additive(cx, person, ts)`: idempotent upsert by
   email; JSON-array fields (tags/roles/…) UNION with existing; scalars only overwrite
   when non-blank; counts take max. Shared by the feeders and the endpoint.
2. **`POST /api/people?merge_tags=1`** — routes each item through the additive helper.
   Default (no flag) keeps the prior overwrite behavior so the console editor is unaffected.
3. **Practitioner feeder** — `_practitioner_to_person(row)` (pure tag rule: `portal_role`
   or `wholesale_unlocked_at` ⇒ engaged `type:practitioner`+`consent:opted-in`, else
   `type:practitioner-cold`+`consent:cold-no-consent`) + `sync_practitioners_to_people()`
   reading the Supabase `practitioners` base table (`email` non-null) + route
   `POST /admin/sync-practitioner-tags` (auth/params mirror `/admin/sync-pb-tags`).
4. **PR/media feeder** — `sync-media-contacts.py` (local Mac, like `sync-ghl-leads.py`):
   reads `~/AI-Training/04 Copy/pr-pitches/media-contacts.csv`, POSTs to
   `/api/people?merge_tags=1` tagged `type:pr-media`+`consent:cold-no-consent`. Rows
   without email are skipped (the vault CSV stays the full record).
5. **People tab** — `pf-type` `<select>` folded into the existing `tags` query param
   (LIKE filter already supports it); `type:`/`consent:` chip colors via `_renderPtag` + CSS.

## Reuse

`sync_pb_to_people_and_ghl()` (additive-upsert template), `_merge_two_people()` +
`_PEOPLE_JSON_UNION` (dedup union), `_people_search_query()` (tag LIKE filter, unchanged),
`scrapers/practitioner_finder/ghl_sync.py` (email-nonnull filter), `db_supabase.py`,
`sync-ghl-leads.py` (local-script pattern).

## Tests

`tests/test_people_feeders.py` (10 tests, all green): engaged/cold/no-email mapping;
`merge_tags` union vs default overwrite; idempotency; no-email skip; type-tag filter;
cross-source tag union. Full suite: 1233 passed (1 pre-existing unrelated failure).

## Out of scope / follow-ups

- GHL mirror + consent **enforcement** (Phase 2); automation sequences (Phase 3).
- Cron-schedule `/admin/sync-practitioner-tags` (mirror the PB cron).
- Contacts without email; the not-yet-merged `application_status` approval column.

## Verification

`POST /admin/sync-practitioner-tags?dry_run=1` → counts; live `&limit=50` → inspect tags;
`sync-media-contacts.py --dry-run` then live; People tab filter by contact type shows the
right segments and dual-role people show both chips.
