# Tag Autocomplete (People detail card) — Design Spec

**Date:** 2026-06-19
**Status:** Approved (brainstorm), pending plan
**Author:** Claude + Glen
**Builds on:** `2026-06-19-people-tag-editor-design.md` (the add/remove chip editor, shipped PR #183)

## Problem

The tag-add input on the People detail card is free-text only. To reuse an existing tag (e.g. `tier:pro-influencer`) you must remember and retype it exactly — easy to mistype and create a near-duplicate. Operators want to type a few letters and pick from a dropdown of existing tags that contain that substring.

## Goal

As the operator types in the add-tag input, show a dropdown of existing tags (across the whole People directory) that **contain** the typed substring (case-insensitive) and that the person doesn't already have. Clicking one adds it immediately. Typing a brand-new tag still works.

## Non-goals (YAGNI)

- Keyboard arrow-key navigation of the dropdown (click + type only).
- Fuzzy/ranked matching (plain case-insensitive substring).
- Reusing the autocomplete on the People list `pf-tag` filter.
- Server-side per-keystroke querying (we fetch the list once and filter client-side).

## Approach

Custom dropdown, **not** a native `<datalist>`: datalist substring matching is prefix-only in Safari (the primary client is macOS), and we want contains-matching plus the ability to exclude already-applied tags and add-on-click. The candidate list is small enough to fetch once and filter in the browser.

## Backend

### New endpoint
`GET /api/people/tags`

- **Auth:** CONSOLE_SECRET, same pattern as the sibling people routes (`X-Console-Key` header or `?key=`). `401` if missing/invalid.
- **Response:** `200 {"tags": [<distinct, case-sensitive, sorted ascending>]}` — the union of every tag currently present in `people.tags`.
- **Implementation:** `SELECT tags FROM people`, parse each row's JSON array, union, via the pure helper below.

### Pure helper (testable core)
`distinct_tags(tag_lists) -> list[str]` in `dashboard/people.py` (same module as `set_person_tags`).

- Input: an iterable of per-person tag values, each either a `list[str]` or a JSON string (the helper tolerates both, since callers may pass raw DB strings).
- For each entry: if `str`, `json.loads` it (on failure, skip that entry); if `list`, use as-is; otherwise skip. Collect string items, `strip()`, drop empties.
- Returns the de-duplicated set sorted ascending (case-sensitive, stable `sorted()`).

## Frontend (`static/console.html`)

Augment the existing tag-add input (built in PR #183). New globals/functions near the tag editor helpers:

- **Cache:** `_allTagsCache` (array or null). `_fetchAllTags()` fetches `/api/people/tags` once with the console key, stores the array, returns it; on fetch error returns `[]` (dropdown silently absent — freeform add still works).
- **On input:** an `oninput` handler on `#tag-add-input` reads the current value; if empty, hide the dropdown; else filter `_allTagsCache` to tags that (a) contain the value case-insensitively and (b) are not already on this person (compare against the person's current tags), cap to the first 10, and render a dropdown positioned under the input.
- **Suggestion click:** calls `addPersonTag(id)`-style flow with the chosen tag (immediate add) — reuses `_postTagDelta` → `_rerenderTags`. After any successful add, set `_allTagsCache = null` so a newly-created tag appears on the next lookup.
- **Dismiss:** dropdown hides on Escape, on blur (with a small delay so a click registers), and after a selection.
- The person's current tags are available to the handler from the render context (the same `tags` array used to draw the chips).

No emoji; markup escaped via the existing `_esc`.

## Error handling

| Condition | Behavior |
|---|---|
| `/api/people/tags` missing/invalid key | `401` (server); client treats a failed fetch as empty list → no dropdown |
| Malformed `tags` JSON in a row | that row skipped by `distinct_tags`, others still counted |
| Empty filter input | dropdown hidden |
| No matches | dropdown hidden (or empty — hidden preferred) |

## Testing (TDD)

**Pure helper `distinct_tags`:**
- mix of list and JSON-string inputs → union
- de-duplication across people
- sorted ascending, case-sensitive (`OD` before `od` per ASCII)
- malformed JSON string entry skipped, others retained
- empty / whitespace tags dropped
- non-list/non-str entries skipped

**Endpoint `GET /api/people/tags`:**
- `401` without the console key
- returns sorted distinct union of seeded people's tags (isolated temp-db, monkeypatched `LOG_DB`/`CONSOLE_SECRET`, same pattern as `tests/test_people_tags_api.py`)

**Frontend:** JS syntax check (`node --check` on extracted scripts) + manual verify in the running console.

## Rollout

Additive, console-only, CONSOLE_SECRET-gated. No migration, no feature flag. Ships on merge to main (Doppler-synced, auto-deploy).
