# CTI-2 Slice 2 — canonical attributes reach the biofield analysis — Design

**Date:** 2026-07-26
**Status:** Approved in brainstorm with Glen 2026-07-26
**Repo:** deploy-chat
**Branch:** `sess/72339d9d-cti23` (off `main`)

## Problem

`canonical_tags.person_attributes` (the in-house clinical attribute store) is written by document approval (`source='document:<id>'`) and is read by exactly two places today: the eye-condition **support program** ([[project_cti2_canonical_conditions]] Slice 1) and `eye_vision_report.reported_eye_history`. It is NOT read by the flagship AI analysis — the **biofield causal-chain narrative** Glen reviews and sends. So a doc-approved condition/terrain/body-system never appears in that narrative.

Confirmed by grep: `canonical_tags.rebuild_people_columns` and `import_from_people` are **never called** (non-test), so nothing projects `person_attributes` into the `people` columns; and `/api/people` (which feeds the narrative's profile) reads only `people` columns. The canonical data is stranded from the analysis.

## The data path (grounded)

```
narrative generate (biofield_local_app.py:1627)
  → prof = fetch_profile(email)                     [_default_fetch_profile, biofield_local_app.py:129]
      → GET {PUBLIC_BASE_URL}/api/people?q=<email>  [get_people, app.py:34063]
          → SELECT ... conditions, terrain_concerns, body_systems, challenges, goals, tags ... FROM people
  → generate_narrative(..., profile=prof)
      → _profile_block(profile)                      [biofield_narrative.py:134] renders _PROFILE_FIELDS
        = (conditions, challenges, goals, tags, terrain_concerns, body_systems)
```

The narrative prompt **already has the slot** (`_PROFILE_FIELDS` are exactly the canonical fields). Nothing needs to change in `biofield_local_app` or `biofield_narrative`. The single seam is `/api/people` GET.

## Approach — read-through merge in `/api/people` GET (approved)

A helper `_merge_canonical_into_person(cx, person)` unions `canonical_tags.get_person(email)` into each returned person's clinical fields, preserving each field's serialized shape so no downstream consumer breaks.

**Fields merged:** `conditions`, `terrain_concerns`, `body_systems` (discrete), `challenges`, `goals` (scalar).

**`tags` is NOT merged.** `/api/people` also feeds the console people list; `people.tags` is the CRM/GHL bucket with many writers. Merging canonical `tags` into it would muddy that view. Same "never touch `people.tags`" rule as Slice 1. (`_PROFILE_FIELDS` includes `tags`, but its value stays the people-column CRM tags — unchanged.)

**Serialized shapes (must be preserved — verified in code):**
- Discrete columns are stored/returned as a **JSON-string of a list** (`json.dumps(["glaucoma", ...])`). Merge = parse the people-column JSON → list, union with `get_person[field]` (already a list), **case-insensitive dedup preserving first-seen casing**, re-serialize with `json.dumps`. Output stays a JSON string → the console people view and `_profile_block` (which `str()`s it) both see the same shape, now including canonical items.
- Scalar columns (`challenges`, `goals`) are stored/returned as **plain strings**. Merge = canonical value wins if non-empty, else the people value. Output stays a plain string.

**Best-effort:** any failure resolving/reading canonical for a person returns that person unchanged. `get_people` must never break because of the merge (wrap per-person in try/except).

## Where it applies

- `get_people` (`app.py:34063`, `GET /api/people`) — the list endpoint `fetch_profile` calls with `?q=<email>`. Apply the helper to each returned row that has an email.
- `get_person` by id (`app.py:34100`, `GET /api/people/<int:person_id>`) — apply the same helper for consistency, since it's the other people-detail reader. (Its `SELECT *` returns the same clinical columns.)

Both are console-gated (existing `CONSOLE_SECRET`/owner-token check) — unchanged.

## Behavior change to be explicit about

This makes canonical/doc-approved clinical attributes appear in `/api/people` for **every** person the endpoint returns — including the console people list, not only the one being narrated. That is correct (the field genuinely reflects the person's record) and is the intended effect, but it is a visible change on the console people surface: doc-approved conditions will now show there, merged into the clinical columns. Noted, not hidden.

## Performance

The merge does one `canonical_tags.get_person` (a single indexed `SELECT` on `person_attributes`) per returned person. `get_people` caps `limit` at 200, so worst case is 200 cheap indexed reads per request. Acceptable; not gated. If it ever matters, the merge can be limited to `q`-filtered (single-person) queries — noted, not built.

## Scope guards / non-goals

- Only `/api/people` GET routes change (helper + two call sites). `biofield_local_app`, `biofield_narrative`, `_profile_block` are all untouched.
- **Read-through, not projection** — `rebuild_people_columns`/`import_from_people` stay un-called; no `people`-column writes; no denormalization drift; `people.tags` never written.
- Does NOT change how the narrative renders the profile (the JSON-string-in-prompt rendering is pre-existing and out of scope — this slice only ensures canonical items are *present* in the field).
- The matcher (Slice 3, the separate gated signal) is a different spec.

## Testing

**Helper `_merge_canonical_into_person`**
- discrete field: people `["glaucoma"]` + canonical `["ocular hypertension"]` → JSON string `["glaucoma","ocular hypertension"]` (union, JSON-string shape preserved).
- case-insensitive dedup: people `["Glaucoma"]` + canonical `["glaucoma"]` → one item, first-seen casing kept.
- scalar field: canonical `challenges="fatigue"` wins over people `""`; people value kept when canonical empty.
- **`tags` is never modified** — people `tags` passes through byte-identical even when canonical has tags.
- best-effort: `get_person` raising (monkeypatched) → person returned unchanged, no raise.
- a person with no canonical row → returned identical to input.
- malformed people-column JSON → degrades to the canonical list alone (or unchanged), never raises.

**Endpoint integration (`GET /api/people`)**
- seed a `person_attributes` condition for an email + a `people` row → `GET /api/people?q=<email>` returns that condition merged into `conditions`; `tags` unchanged.
- canonical failure path → response equals the un-merged people row.
- console-key gate still enforced (401 without).

**CI conventions:** pin the product catalog per the `$DATA_DIR`-strips-catalog rule; dummy `OPENAI`/`PINECONE` keys for the app-importing test module.

## Deferred to Slice 3

The remedy matcher (`ff_matcher`) is scan-driven. Glen chose a **separate, gated canonical signal** at matcher review (canonical conditions shown as a distinct, clearly-labeled signal Glen sees, NOT blended into the scan query or the scan-driven ranking). That is CTI-2 Slice 3 — its own spec after this ships, and it needs recon on the matcher-review surface (where `generate_ff_matches` output is presented).
