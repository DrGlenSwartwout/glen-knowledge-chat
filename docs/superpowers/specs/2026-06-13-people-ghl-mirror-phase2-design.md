# People Hub — Phase 2: GHL Mirror (opted-in) + classification

**Date:** 2026-06-13 · **Status:** approved, implementing

## Context

Phase 1 put all contact sources into the local `people` hub tagged `type:*` /
`consent:*`. Phase 2 mirrors the **opted-in** subset into GoHighLevel so GHL can
drive automated communications, with cold/no-consent contacts enforced *out* of
GHL entirely.

**Decisions (with Glen):**
- **Opted-in only** reaches GHL. The 4,148 cold practitioners stay local-only (saves GHL quota; enforcement = they're simply not there).
- **Explicit opt-in signal** required for `consent:opted-in`. Bare funnel leads stay `consent:cold-no-consent` until they opt in.

A dependency surfaced: only 2 people are `consent:opted-in` today, because the
existing funnel/PB population was never classified. So Phase 2 = **classify
existing people**, then **mirror the opted-in ones**.

## A. Classifier (`classify_people`)

Server-side scan of `people`, additive tags only (never overrides feeder tags):

- **type:** `order_count > 0` or `pb_id` set → add `type:client`. A person with no `type:*` tag and no commerce signal (bare funnel lead) → `type:prospect`. Practitioner / practitioner-cold / pr-media keep their tags; a commerce signal can still add `type:client` on top.
- **consent (opted-in on an explicit signal only):** opted-in if **any** of —
  `order_count > 0`, `pb_id` set (PB/Healing Oasis member), `journey_state.tos_agreed_at` set (joined by email), or an opt-in tag (`_OPTIN_TAG_PATTERNS`: "opted in", "opt-in", "email list opted"). Else `consent:cold-no-consent`.
- **Hard-suppression (compliance):** any email-negative tag (`_SUPPRESS_TAG_PATTERNS`: "email bounced", "reengagement:bounced", "do not email", "spam complaint", "email unsubscribed") forces `consent:cold-no-consent` regardless of opt-in signals. SMS-only unsubscribes are excluded.
- Additive only — never downgrades an existing `consent:` tag (unsubscribe reconciliation from GHL is out of scope).

Reuses the additive upsert so re-runs are idempotent.

## B. GHL mirror (`sync_people_to_ghl`)

For each person carrying `consent:opted-in`, **enqueue a `tag_add` op** (their
`type:*` tags) to `ghl_write_queue` via `dashboard.ghl_queue.enqueue(...)`. The
existing Mac drainer (`sync-ghl-writes.py`, on launchd `com.remedymatch.ghl-write-drain`)
executes it through `ghl_upsert_contact`, which finds-or-creates the GHL contact
and merges tags additively.

- **Cold contacts are never enqueued** → never reach GHL → cannot be automated. That is the consent enforcement.
- **No workflow enrollment** this phase. Phase 2 only gets correctly-tagged contacts into GHL; Glen builds the tag-triggered automations (Phase 3).
- Queue path chosen over direct-from-Render because Render→GHL is WAF-blocked; the queue + Mac drain is the purpose-built safe path.

## Surface

- `POST /admin/sync-people-to-ghl` — runs A then B; auth/params mirror `/admin/sync-pb-tags` (X-Cron-Secret/X-Console-Key, `dry_run=1`, `limit=N`).
- Render daily cron `glen-people-ghl-sync-daily` + `scripts/run_people_ghl_sync_cron.py` (mirrors `glen-pb-tag-sync-daily`).
- Idempotent: classifier tags are additive; queued tag_adds are additive; re-running is safe (the drainer also dedups against existing GHL tags).

## Reuse
`_upsert_person_additive` (additive tagging), `dashboard.ghl_queue.enqueue` +
`ghl_write_queue` + `sync-ghl-writes.py` (WAF-safe push), `ghl_upsert_contact`
(find-or-create + additive tags), the PB-sync route/cron shape.

## Out of scope
Pushing cold contacts (deferred); GHL workflow enrollment + the automation
sequences (Phase 3); two-way consent reconciliation from GHL unsubscribes.

## Tests (`tests/test_people_ghl_phase2.py`)
- Classifier: orderer/PB → `type:client`+`consent:opted-in`; bare lead → `type:prospect`+`consent:cold-no-consent`; ToS-agreed lead → opted-in; idempotent re-run; doesn't strip feeder tags.
- Mirror: only `consent:opted-in` people get queued; cold people produce no queue rows; queued op is `tag_add` with the `type:*` tags; re-run doesn't duplicate pending rows.

## Verification
`POST /admin/sync-people-to-ghl?dry_run=1` → counts (classified client/prospect, opted-in, would-enqueue). Live `&limit=50`; inspect `ghl_write_queue` for `tag_add` rows only for opted-in; confirm a cold practitioner produced none. After the Mac drains, spot-check the contact's tags in GHL.
