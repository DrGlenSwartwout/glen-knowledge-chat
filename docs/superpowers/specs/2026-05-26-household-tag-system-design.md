# Household Tag System — Design

**Date:** 2026-05-26
**Owner:** Glen Swartwout
**Operators:** Glen + Shaira (both can create/edit households via Console)
**Target deploy:** `~/deploy-chat` (Flask app, Render service `glen-knowledge-chat`)

## Context

GHL contacts don't have a native "household" concept. Today, multiple people from the same household appear as independent contacts — sometimes sharing an email (Savant family: mother + daughter), sometimes sharing a phone (Perdomo household: 3 members), sometimes spread across separate emails. Three real operational problems result:

1. **Campaigns triple-touch the same household.** A "send to all e4l clients" email lands at the same hotmail inbox three times if three Savants are in the list.
2. **Clinical context fragments.** Family medical patterns matter (genetic conditions, shared environment), but each member's record is isolated.
3. **Dedup bugs masquerade as household merges.** A naive deduper keyed on email+phone would collapse the Savant mother and daughter (which we just discovered today is the wrong move).

Resolved earlier today: 4 pre-existing GHL duplicate clusters tagged with `relationship:family-shared-email` as an interim marker. That tag is a narrow band-aid — this spec is the proper system.

**Goal:** Give Glen + Shaira a low-friction way to group GHL contacts into households so campaigns can target a single member per household, clinical workflows can surface related members, and future automation (a real dedup tool, family-pattern queries) has a stable substrate.

## Scope

**v1 (this spec):**
- Households only — biological family / shared dwelling. Practices, masterminds, orgs deferred.
- Two member roles: `head` (campaign target) + member.
- Tags as the membership representation; one person in at most one household.
- Two creation flows: from a person's detail view, and from multi-select in the People list.
- Detection-and-suggest (never auto-tag) for shared-email, shared-phone+lastname, shared-address+lastname signals.
- Immediate GHL sync on every household change.
- Daily cron re-sync to recover from drift.

**Out of scope (v2 or later):**
- Practice / mastermind / organization grouping (separate namespace, separate UX surface).
- Family role taxonomy (parent/child/spouse/sibling tags).
- One person in multiple households (divorced parents splitting custody, dual-residence scenarios).
- Auto-suggestions for "household has grown" when a new PB contact matches an existing household.
- Cross-channel sync beyond GHL (BNSN, email, etc.).
- Per-household permission ACLs.

## Data Model

### New SQLite tables in `LOG_DB`

```sql
CREATE TABLE households (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  slug            TEXT UNIQUE NOT NULL,
  name            TEXT NOT NULL,
  head_person_id  INTEGER,
  address         TEXT DEFAULT '',
  notes           TEXT DEFAULT '',
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  created_by      TEXT NOT NULL
);

CREATE TABLE household_candidates (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  detected_at     TEXT NOT NULL,
  signal          TEXT NOT NULL,
  person_ids      TEXT NOT NULL,
  status          TEXT NOT NULL DEFAULT 'pending',
  resolved_at     TEXT DEFAULT '',
  resolved_by     TEXT DEFAULT '',
  household_id    INTEGER
);

CREATE INDEX idx_household_candidates_status ON household_candidates(status);
CREATE INDEX idx_households_head ON households(head_person_id);
```

Field semantics:
- `households.slug` — URL-safe identifier; immutable after creation (renames change `name`, never `slug`, to keep GHL tag references stable across renames).
- `households.head_person_id` — nullable. If the head's `people` row is later hard-deleted (the only delete mode the people table supports), the orphaned `head_person_id` becomes invalid; the UI detects this in `renderPersonDetail` for any household member and prompts the operator to designate a new head before further edits.
- `households.created_by` — `'glen'` | `'shaira'` | `'rae'` | other workspace user name.
- `household_candidates.person_ids` — JSON array of `people.id` integers, sorted ascending. Sorting is required for dedup matching across runs.
- `household_candidates.status` — `'pending'` | `'confirmed'` | `'dismissed'`. Confirmed candidates link to the resulting `households.id`. Dismissed candidates stay in the table so the same cluster isn't re-suggested.

### Tag conventions on `people.tags` JSON

Two tag namespaces, mirrored to GHL:

- `household:<slug>` — every member, including the head
- `household-head:<slug>` — only on the designated head

**Invariants enforced at write time:**
- A person can carry at most one `household:*` tag. Attempting to add them to a second household returns HTTP 409 with the current household slug in the response; UI prompts the operator to confirm a move.
- Exactly one member per household carries the `household-head:<slug>` tag (validated on insert/update; head changes are a single atomic op that removes from old head and adds to new head).

**Slug generation rule:**
```python
def household_slug(name, head_first_name=""):
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if slug_exists(base) and head_first_name:
        return f"{base}-{re.sub(r'[^a-z0-9]+','-',head_first_name.lower()).strip('-')}"
    return base
```

The existing `relationship:family-shared-email` tag (added by this morning's cleanup of the 4 pre-existing dupes) is purely informational — nothing in the codebase filters on it. When household creation runs against members who carry this tag, the tag is automatically removed from each member (DB + GHL) since the new `household:<slug>` tags supersede it. It remains on contacts that haven't been household-tagged yet (e.g., a future shared-email cluster awaiting human confirmation).

## API Surface

All routes live in `app.py`. Auth on every route accepts `X-Console-Key` (admin) or a workspace token (matches the existing `/api/people` and `/api/leads` pattern via `_auth()` at `app.py:4503`).

### Reads

| Route | Returns |
|---|---|
| `GET /api/households` | `{households: [{slug, name, member_count, head: {id, name}, updated_at}, ...]}` |
| `GET /api/households/<slug>` | Full household record + members array with each member's `{id, name, email, phone, is_head}` |
| `GET /api/people/<id>/household` | The person's household (or `{household: null}`); used by the Overview tab section |
| `GET /api/household-candidates?status=pending` | `{candidates: [{id, signal, persons: [{id, name, email}], detected_at}, ...]}` |

### Writes (transactional: DB then GHL, per-member error reporting)

| Route | Body | Behavior |
|---|---|---|
| `POST /api/households` | `{name, head_person_id, member_person_ids[], address?, notes?}` | Insert households row → write `household:<slug>` to each member's `people.tags`, plus `household-head:<slug>` to the head → push the same tags to each member's GHL contact via `ghl_update_tags`. `address` and `notes` are accepted but not surfaced in the v1 dialog (reserved for v2 UI when the household section grows into its own tab) |
| `PATCH /api/households/<slug>` | `{name?, address?, notes?, head_person_id?}` | Slug never changes. If `head_person_id` changes, transactionally move the `household-head:<slug>` tag from old → new head (DB + GHL) |
| `POST /api/households/<slug>/members` | `{person_id}` | Add member. Rejects with `409 Conflict` `{error, current_household: {slug, name}}` if person is already in another household |
| `DELETE /api/households/<slug>/members/<person_id>` | — | Remove `household:<slug>` tag (DB + GHL). Returns `409` if person is current head |
| `DELETE /api/households/<slug>` | — | Disband: remove `household:` and `household-head:` tags from every member (DB + GHL), mark related candidate rows resolved, **hard-delete the households row** |
| `POST /api/households/<slug>/resync-ghl` | — | Idempotent re-push of household tags to all members' GHL contacts. Used to clear `⚠ GHL sync incomplete` warnings after a transient failure |

### Candidate actions

| Route | Body | Behavior |
|---|---|---|
| `POST /api/household-candidates/<id>/confirm` | `{name, head_person_id}` | Creates a household from the candidate; sets candidate row to `confirmed` with link to new household |
| `POST /api/household-candidates/<id>/dismiss` | — | Sets candidate row to `dismissed` |

### Admin / cron

| Route | Auth | Behavior |
|---|---|---|
| `POST /admin/detect-household-candidates` | `X-Cron-Secret` or admin key | Runs detection algorithm, upserts new pending candidates, returns counts |
| `POST /admin/resync-all-households` | `X-Cron-Secret` or admin key | Re-pushes every household's tags to GHL (full drift recovery) |

Detection is chained to the existing PB sync cron (`glen-pb-tag-sync-daily` in `render.yaml`) — after `sync_pb_to_people_and_ghl` completes successfully, the same handler calls `detect_household_candidates()` so new PB contacts get household-checked within 24h.

## UI Integration in `static/console.html`

### 1. Household section in the Overview tab

In `renderPersonDetail` at `static/console.html:1151`, the Overview pane gains a new section between Profession/Source and the Roles list. It renders from a new fetch in `loadPersonDetail` to `/api/people/<id>/household`.

Layout (per the approved mockup B from brainstorming):
- Header row: `👥 <Household name>` + `<Head firstname> is head` badge.
- One member card per person: avatar circle (initials), name, role/context line, `Head` badge when applicable, the current person's card visually marked as "this person".
- `+ Add member` button at the bottom of the section (opens the create dialog in member-add mode).
- ⚙ icon at the top right of the section header → opens the edit dialog (rename, change head, disband).
- If person is in no household: the section shows only a `+ Mark as household` button (entry point A).

**Future expansion:** when the household section grows enough (shared address, household notes, family medical pattern fields), it migrates to its own `Household` tab between Health and Activity (per approved mockup C). The data model already supports this — only the render layer changes.

### 2. Entry point A — from person detail

`+ Mark as household` button opens `<dialog id="household-dialog">` (new dialog modeled on the existing `new-task-dialog` at `static/console.html:1887`):
- Household name input (auto-filled with person's last name).
- The triggering person pre-added as head.
- Member search input → hits `/api/people?q=...` (existing endpoint).
- Search results show a small "same email" / "same phone" / "same lastname + city" badge when applicable as a visual cue.
- Submit → `POST /api/households`.

### 3. Entry point B — multi-select in People list

- Each `.person-card` in the list (built around `static/console.html:1118`) gets a small checkbox in the top-left. The checkbox is visible only on row hover by default; once at least one row is checked, all checkboxes become sticky-visible.
- A floating toolbar slides down from the People section header when ≥1 selected: `<N> selected` + `Group as household` + `Clear`.
- `Group as household` opens the same `household-dialog`, prefilled with the selected people. Head defaults to the first selected; radio buttons let the operator switch.
- Submit → same `POST /api/households`.

### 4. "Possible households" review banner

Above the People search bar (around `static/console.html:501`):
- Renders `👥 <N> possible household(s) detected · Review →` only when `GET /api/household-candidates?status=pending` returns non-empty.
- No live polling — fetch once on People page load, refresh after the user takes a candidate action.
- Click → expands an inline panel listing each candidate cluster: people names + the signal that flagged them (`shared email`, `shared phone + lastname`, etc.) + `Confirm` (opens create dialog prefilled) and `Dismiss` buttons.
- Clusters with > 5 members render with a yellow `⚠ Unusually large — review carefully` warning above the cluster.

### 5. Editing an existing household

The ⚙ icon in the household section opens the same `household-dialog` populated with current state. Supports: rename, change head (radio), add/remove member, disband (red button with `"Disband <name> household? Members remain as individuals."` confirmation).

### 6. GHL sync status surface

If `POST /api/households` or `PATCH .../households/<slug>` returns `ghl_errors`, the Overview household section shows a yellow `⚠ GHL sync incomplete` badge with a `Retry` button. Click → `POST /api/households/<slug>/resync-ghl`.

## Candidate Detection Algorithm

Implemented as a single function `detect_household_candidates()` in `app.py`. Runs independently against the `people` table.

### Signals

| Signal | Trigger | Notes |
|---|---|---|
| `shared-email` | 2+ people with the same `email` (case-insensitive, exact match) | Rare after today's GHL dedup fix; strongest signal when it fires |
| `shared-phone-lastname` | 2+ people with same normalized `phone` AND same `last_name` (case-insensitive) | Phones already stored normalized in the people table |
| `shared-address-lastname` | 2+ people with same `city` + same `state` + same `last_name` (case-insensitive); skip when `city` is empty | Weakest of the three; address fields are inconsistently populated |

### Cluster build rules

- Skip any cluster where ≥1 person already carries a `household:*` tag.
- Skip clusters that exactly match a previously-confirmed or previously-dismissed candidate (dedup keyed on sorted `person_ids` JSON).
- **No hard cap on cluster size** — clusters > 5 are still surfaced but render with a UI warning so operators review them carefully (a phone system or a typo will manifest as an unusually large cluster).
- A cluster that grows over time (a confirmed 3-person household where a 4th person now matches the signal) is treated as a *different* candidate type and deferred to v2 ("Add to existing household?" suggestions).

### Storage

Each new pending candidate writes one `household_candidates` row with `status='pending'`. The detection function returns `{detected: N, new_pending: N, skipped_already_household: N, skipped_dedup: N}`.

### Run cadence

- Chained onto `sync_pb_to_people_and_ghl` after it completes successfully (so new PB contacts trigger detection within 24h of the daily PB sync cron at 3 AM HST).
- Plus on-demand via `POST /admin/detect-household-candidates` for manual runs from the Console.

## GHL Sync + Error Handling

### New helper

```python
def ghl_update_tags(email, add=None, remove=None):
    """Fetch contact by email via /contacts/lookup, add+remove tags as set
    operations, PUT the merged set. Returns (contact_id, err).

    Uses the corrected lookup endpoint from commit 69f61c3 so this always lands
    on the established contact rather than creating duplicates."""
```

Used for all household-related tag mutations. `ghl_upsert_contact` remains the additive-only path used by the PB tag sync.

### Transactional behavior on writes

1. SQLite transaction commits first (`people.tags` JSON + `households` table row). This is the authoritative state.
2. GHL push happens after the commit, per-member, in a try/except. Per-member failures are collected.
3. Response shape: `{ok: true, household_id: <int>, ghl_errors: [{email, error}]}`.

### Drift recovery

- Daily Render cron `glen-pb-tag-sync-daily` chains a household tag re-sync after the PB sync. Idempotent: every member's GHL tags get re-pushed. Any manual GHL tag edits get clobbered back to canonical state next morning.
- `POST /admin/resync-all-households` provides on-demand full re-sync.

### Failure modes covered

| Mode | Handling |
|---|---|
| GHL API transiently down | DB writes succeed, UI shows ⚠ warning, daily cron retries |
| Member doesn't yet have a GHL contact | `ghl_update_tags` falls through to `ghl_upsert_contact` to create them first |
| DB write fails | Return 500, no GHL calls attempted, no state change |
| Two operators editing same household | SQLite locking serializes; second write replays atop first |
| Invalid head_person_id (person doesn't exist) | Return 400 before any writes |
| Invalid member_person_ids (some don't exist) | Skip invalid IDs, return 200 with `skipped_invalid_ids` in response |

## Permissions

- All household read/write routes accept either `X-Console-Key` (admin = Glen) or a workspace token (Shaira, Rae) via the existing `_auth()` helper at `app.py:4503`.
- No per-household ACLs — anyone with console access can edit any household. Matches the existing tag-everywhere convention used by `/api/people` and `/api/leads`.
- `/admin/detect-household-candidates` and `/admin/resync-all-households` accept `X-Cron-Secret` (cron) or admin key.

## Testing

Test files in `tests/`, following the `tmp_db` fixture pattern from `tests/conftest.py`:

| File | Coverage |
|---|---|
| `tests/test_household_model.py` | Pure unit tests: slug generation + collision handling, candidate dedup key, cluster build logic |
| `tests/test_household_api.py` | Integration tests for all API endpoints against a temp SQLite DB. Mocks `ghl_update_tags` and `ghl_upsert_contact` so tests don't hit real GHL |
| `tests/test_household_detection.py` | Feeds known people fixtures into `detect_household_candidates()`, asserts the right clusters surface with the right signals, validates dedup against existing rows |

### Manual verification checklist (post-deploy)

1. Create a test household via entry point A → confirm GHL contact for each member has `household:<slug>` and head has `household-head:<slug>`.
2. Create a test household via entry point B with 3 selected people → same verification.
3. Edit head via the ⚙ dialog → confirm old head loses `household-head:`, new head gains it (DB + GHL).
4. Add a member via the dialog → confirm tag added (DB + GHL).
5. Try to add a member who's already in another household → confirm 409 with the current household slug surfaces the "Move them?" confirmation UI.
6. Remove a non-head member → confirm tag removed.
7. Try to remove the head → confirm UI blocks with "change head first".
8. Disband the test household → confirm all members lose their household tags (DB + GHL), `households` row hard-deleted.
9. Trigger `POST /admin/detect-household-candidates` manually → confirm the Savant, Perdomo, Brenda clusters do **not** re-surface (they already have household tags from the migration in step 11 below).
10. Inject a fake unrelated 3-person cluster with the same lastname + phone → confirm it surfaces in the review banner.
11. **Migration of existing pre-tagged contacts:** Run a one-time migration script that creates households for the four cases already tagged today (`Savant` mother+daughter, `Perdomo` 3-member, `Schulman` single — actually skip Schulman since single-person, `Montecalvo` single — also skip). Net: two real households (Savant, Perdomo) get formalized; existing `relationship:family-shared-email` tags stay in place.

## Render Cron Updates

No new cron service needed. The existing daily 3 AM HST cron `glen-pb-tag-sync-daily` (defined in `render.yaml`) curls `POST /admin/sync-pb-tags`. The handler for that endpoint in `app.py` (`admin_sync_pb_tags`) gets two additional steps appended after `sync_pb_to_people_and_ghl` returns successfully:
1. Call `detect_household_candidates()` so new PB contacts get household-checked within 24h.
2. Call `resync_all_households_to_ghl()` so any GHL drift gets corrected nightly.

Both new steps run inside the existing cron-triggered request. The cron container itself is unchanged.

## Migration

One-off Python script at `scripts/migrate_existing_households.py` (run once, post-deploy):

1. Lookup the Savant contacts (`lotikasavant@hotmail.com`) → `POST /api/households` with name="Savant", head=Lotika, members=[Lotika, Omika].
2. Lookup the Perdomo contacts (`restorealoha@gmail.com`) → `POST /api/households` with name="Perdomo", head=Kauilani, members=[all 3].
3. Log counts.

This formalizes the two genuine households we identified today. The single-person cases (Schulman, Montecalvo) don't need household records.

## What we explicitly chose not to do

(Documented so the next reader doesn't re-litigate.)

- **One person in multiple households.** Tags enforce one-household membership at write time (409 with explicit error). Multi-membership would require a junction table; deferred unless real demand surfaces.
- **Auto-tag obvious matches.** Today's Savant lesson — mother and daughter share email + phone but are not a "merge" case — proves auto-tagging is dangerous. All household creation requires human confirmation.
- **Practice / mastermind / org grouping.** Same machinery would apply (`practice:<slug>`, `cohort:<slug>`), but the v1 scope is households only. Adding more namespaces later is non-breaking.
- **Family role taxonomy (parent/child/spouse).** Just head + members for v1. The data model accommodates a future `role` field on a join table without breaking the tag layer.
- **Live polling for new candidates.** "Possible households" banner reads on People-page load only, not via polling.
- **Per-household permissions.** Console-wide access; matches existing patterns.

## Critical files

- `app.py` — new helpers (`ghl_update_tags`, `detect_household_candidates`, `sync_household_tags_to_ghl`), new routes (`/api/households/*`, `/api/household-candidates/*`, `/admin/detect-household-candidates`, `/admin/resync-all-households`), new SQLite schema initialization, hook into existing `/admin/sync-pb-tags` handler
- `static/console.html` — new Overview household section in `renderPersonDetail`, new `household-dialog`, new multi-select toolbar in People list, new candidate review banner, new fetch in `loadPersonDetail`
- `tests/test_household_model.py`, `tests/test_household_api.py`, `tests/test_household_detection.py` — new test files
- `scripts/migrate_existing_households.py` — one-off migration

## Open follow-ups (out of scope for this spec)

- v2: "Add to existing household?" suggestions when a confirmed household grows.
- v2: practice / mastermind / org namespace.
- v2: family role taxonomy.
- v2: address-aware detection signals (more accurate than current city+state heuristic).
- v3: shared family history fields surfaced in the household section.
