# People Tag Editor — Design Spec

**Date:** 2026-06-19
**Status:** Approved (brainstorm), pending plan
**Author:** Claude + Glen

## Problem

The console People directory has no way to edit a tag on an **existing** person. Tags can only be set when *creating* a person (the "Add Person" dialog has a Tags field); the detail card renders tags read-only. This blocks a concrete need: tagging people `tier:pro-influencer` so the rewards engine pays them cash commission instead of points (`dashboard/rewards.py:91` reads `"tier:pro-influencer" in tags` from `people.tags`).

## Goal

Let an operator add and remove tags on any existing person directly from that person's detail card in the console, behind the existing CONSOLE_SECRET gate.

## Non-goals (YAGNI)

- Bulk tagging across many people.
- Tag autocomplete / suggestion list.
- Immediate push of tag edits to GHL (the existing daily sync handles propagation — see GHL Sync below).
- Any change to how tags are *read* by the rewards engine or other consumers.

## Data model (existing, unchanged)

Tags live in the `people` table, `tags` column — a TEXT field holding a JSON array of strings (e.g. `["type:client","tier:pro-influencer","OD"]`). Case is preserved (tags mix forms: `type:client`, `OD`, `hawaii-island`). No schema change.

## Backend

### New endpoint
`POST /api/people/<int:person_id>/tags`

- **Auth:** CONSOLE_SECRET, same mechanism as the sibling `POST /api/people/<id>/note` and `POST /api/people` (accepts `X-Console-Key` header or `?key=`). Returns `401` if missing/invalid.
- **Body (JSON):** `{"add": ["tier:pro-influencer"], "remove": ["typo-tag"]}`. Both keys optional; each defaults to `[]`. Delta-based (not full-list replace) so concurrent edits on different tags don't clobber each other.
- **Behavior:** load person by id → `404` if not found → compute new tag list via the pure helper → persist → return `200 {"tags": [<updated list>]}`.
- **Validation:** body must be a JSON object; `add`/`remove` must be lists of strings → else `400`. Non-string / non-list inputs rejected.

### Pure helper (testable core)
`set_person_tags(current_tags: list[str], add: list[str], remove: list[str]) -> list[str]` in `dashboard/people.py`.

- Order: start from `current_tags`, **remove** first, then **add** (so an item in both ends up present).
- Normalization on each incoming tag: `strip()` whitespace; drop empties; drop tags longer than 64 chars; de-duplicate (preserve first-seen order; case-sensitive match so `OD` and `od` are distinct, consistent with current data).
- Remove matches are exact (post-strip).
- Returns the resulting list (stable order: existing order minus removals, then new adds appended).

A thin DB wrapper (e.g. `update_person_tags(cx, person_id, add, remove)`) reads the row's current tags, calls `set_person_tags`, writes the JSON back, and returns the new list (or `None`/raises a not-found signal if the id is absent). The route translates not-found into `404`.

## Frontend (`static/console.html`, People detail view)

The existing `.detail-tags` block (renders `tags.map(_renderPtag)`) becomes editable:

- **Chips:** each tag renders as a chip with a trailing `×` (text glyph / SVG — no emoji, per portal copy conventions) that removes that tag.
- **Add control:** below the chips, a text input (placeholder `e.g. tier:pro-influencer`) + an **Add** button. Enter key or button click adds the typed tag. Input clears and refocuses after a successful add.
- **Wiring:** add/remove each POST the delta to `/api/people/<id>/tags` with the console key (same auth header the People tab already sends for its other writes). On `200`, re-render the chips from the returned `tags` array (server is the source of truth — avoids client/server drift). On error, show the existing console error affordance and leave chips unchanged.
- **No confirm dialog** on remove — a mis-removed tag is one click to re-add.

## GHL sync

Local-only. The editor writes `people.tags`; the existing daily `/admin/sync-people-to-ghl` cron already mirrors opted-in contacts' tags to GHL. No immediate GHL write from this editor. `tier:pro-influencer` is read locally by the rewards engine, so it needs no GHL propagation for its primary purpose.

## Error handling

| Condition | Response |
|---|---|
| Missing/invalid console key | `401` |
| Person id not found | `404` |
| Malformed body (`add`/`remove` not lists of strings) | `400` |
| Tag too long / empty / duplicate | silently normalized (trimmed/dropped/deduped), not an error |

## Testing (TDD)

**Pure helper `set_person_tags`:**
- add a new tag to an empty list and to a populated list
- remove an existing tag; remove a tag that isn't present (no-op)
- add + remove in one call (incl. same tag in both → present)
- dedupe (adding an existing tag is a no-op; duplicate inputs collapse)
- normalization: whitespace trimmed, empty dropped, >64-char dropped
- case sensitivity: `OD` and `od` treated as distinct

**Endpoint `POST /api/people/<id>/tags`:**
- `401` without the console key
- happy path: returns `200` with the updated `tags` array and the DB row reflects it
- `404` for a non-existent person id
- `400` for a malformed body

Run via the project invocation: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest` (DATA_DIR set after `--`).

## Rollout

Additive, no migration, no feature flag — it's a console-only, CONSOLE_SECRET-gated editing affordance with no public surface. Ships on merge to main.
