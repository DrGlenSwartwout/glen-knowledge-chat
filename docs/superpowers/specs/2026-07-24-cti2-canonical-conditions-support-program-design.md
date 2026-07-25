# CTI-2 Slice 1 — canonical conditions drive the eye-condition support program — Design

**Date:** 2026-07-24
**Status:** Approved in brainstorm with Glen 2026-07-24
**Repo:** deploy-chat
**Branch:** `sess/72339d9d-cti2` (off `main`, independent of the document-ingestion PR #1172)

## Problem

`dashboard/canonical_tags.py` (`person_attributes`) is the in-house authoritative store for a person's clinical attributes — conditions, tags, terrain concerns, body systems. It was built dark in CTI-1: its docstring says "no caller is wired yet (CTI-2 makes it authoritative)." Confirmed by grep — the only non-test writer is the document-ingestion approval route (PR #1172), and **nothing reads it** for any decision.

So a canonical condition — including one Glen approves off an uploaded medical record — reaches no clinical surface. This slice gives it its first reader: the **eye-condition support program**.

## Why this slice, and why independent

CTI-2 in full would wire canonical attributes into three reader systems (support programs, the AI analysis, the remedy matcher). That is too much for one spec. This is **Slice 1 of that decomposition**: support programs only. The AI analysis and the matcher are later slices, deliberately deferred until this proves the plumbing on the lowest-risk reader.

It is **independent of PR #1172**: `canonical_tags`, `_client_condition_for`, and the console condition API all exist on `main` today. This slice makes canonical conditions *count* regardless of what writes them. Document ingestion is simply one writer; the two features light each other up once both merge, but neither depends on the other to be correct.

## The two readers, today (grounded in code)

Both resolve a client's eye-condition support-program key with identical logic: **operator override wins; else auto-detect** by running `_condition_key_from_tags` over the client's `people.conditions` then `people.tags`.

- **`_client_condition_for(email)`** — `app.py:21555`. The resolver used by the support-program surfaces (called at three sites). Override (`client_conditions`) first, else auto-detect from `people`.
- **`api_console_client_condition_get`** — `app.py:22356`. The `/api/console/client-condition` GET. Inlines the *same* people-read + `_condition_key_from_tags`, and returns `{resolved, override, auto_detected, tags}` for Glen's console.

`_condition_key_from_tags` (`app.py:21540`) returns the first unambiguous map hit; bare `glaucoma`/`cataract` are in `_AMBIGUOUS_CONDITION_TERMS` and are skipped, not guessed.

## Key integration fact

A canonical condition drives the program **only if its normalized string hits `_CONDITION_TAG_MAP`** — exactly the rule tags already follow. `_normalize_condition_tag` lowercases, strips brackets/quotes, drops a `pb:` prefix, and spaces out `-`/`_`. So:
- doc-approved `"ocular hypertension"` → `glaucoma-elevated-iop` ✓
- doc-approved `"wet AMD"` → `wet-amd` ✓
- doc-approved bare `"glaucoma"` → **ambiguous, skipped** — still needs Glen's override, identical to a `glaucoma` tag today.

**This slice adds no special-casing.** Canonical conditions flow through the same `_condition_key_from_tags` with the same ambiguity semantics. The only change is that they are *added to the detection input, ranked ahead of `people` data*.

## Precedence (the whole design decision)

1. **Operator override** (`client_conditions`) — wins outright. **Untouched.**
2. **Canonical conditions** (`person_attributes` via `canonical_tags.get_person`) — new; fill the auto-detect layer.
3. **Existing `people.conditions` then `people.tags`** — fallback, unchanged.

Canonical conditions rank ahead of `people` data (they are Glen-approved and vocabulary-canonical), but never ahead of the explicit operator override. They only ever matter when Glen has set no override — precisely the state where auto-detect runs today.

## Approach — read-through via one shared helper

Both readers duplicate the same "gather the auto-detect input from `people`" step. Unify it, and have the unified version consult canonical first.

### New helper (in `app.py`, beside `_client_condition_for`)

```python
def _condition_detect_tags(cx, email):
    """The ordered auto-detect input for a client's eye-condition program:
    canonical conditions first (person_attributes, Glen-approved + vocabulary-
    canonical), then people.conditions, then people.tags. This is ONLY the
    detection input — the operator override is applied by the caller, above this.
    Best-effort: a canonical read error degrades to the people-only input,
    never raises. Does NOT read or write people.tags as a canonical target."""
```

Order is load-bearing: `_condition_key_from_tags` returns the *first* unambiguous hit, so canonical conditions must be prepended to outrank tag-derived hits.

Canonical read is wrapped so any failure (missing table, bad row) falls back to the people-only list — the resolver's existing "best-effort, never raises" contract is preserved.

### `_client_condition_for` — `app.py:21555`
Replace its inline `SELECT conditions, tags FROM people` + list-building with a call to `_condition_detect_tags(cx, email)`. Override-first logic unchanged. Net behavior: identical when a client has no canonical conditions; canonical-driven when they do and no override is set.

### `api_console_client_condition_get` — `app.py:22356`
Replace its inline people-read with the same helper so the console diagnostic matches the resolver exactly (they must never diverge). Its `tags` response field now includes canonical conditions — this is honest (they *are* part of the detection input) and useful to Glen, but it **is a response-shape change**: the array may contain entries not present in `people`. Documented, not hidden.

## What this slice does NOT touch

- **`people.tags`** — never written or owned by this work. It has many writers (GHL, manual tagging); canonical never clobbers it. (Canonical conditions are *read alongside* it, not merged into it.)
- **The operator override** — `client_conditions` precedence is unchanged.
- **`rebuild_people_columns` / `import_from_people`** — no projection/denormalization. The store is read as authoritative, not copied into `people`. (This is the deliberate rejection of the project-through alternative, which would depend on a projection trigger firing and risk re-clobbering `people.tags`.)
- **The AI analysis and the remedy matcher** — later CTI-2 slices.
- **`canonical_tags.py` itself** — read-only consumer; no store changes.

## Data flow (end state, both features merged)

```
upload → extract → Glen approves a condition
      → canonical_tags.set_attr(email, "conditions", "ocular hypertension", source="document:<id>")
      → person_attributes row
      → _condition_detect_tags reads it FIRST
      → _condition_key_from_tags → "glaucoma-elevated-iop"
      → (no operator override) → client's support program = glaucoma-elevated-iop
```

Without PR #1172, the same flow works from any other canonical write (import, a future console attribute editor).

## Error handling

- Canonical read failure → helper returns the people-only input; resolver behaves exactly as today. Never raises (preserves `_client_condition_for`'s `try/except → None` contract).
- No `people` row and no canonical conditions → empty input → `None` (no program), as today.
- No `people` row but canonical conditions exist → **canonical still drives the program** (the read-through does not depend on a `people` row — a concrete advantage over project-through).
- Ambiguous canonical condition (bare `glaucoma`) → skipped, override still required, as today.

## Testing

**The helper `_condition_detect_tags`**
- Canonical conditions are returned **first**, ahead of `people.conditions` and `people.tags`.
- A canonical read error (monkeypatch `get_person` to raise) → returns the people-only list, does not raise.
- No `people` row + canonical conditions present → returns the canonical conditions.
- Empty everything → `[]`.

**`_client_condition_for` (behavior)**
- Operator override still wins even when a *different* canonical condition is present (pins precedence #1).
- No override + canonical `"ocular hypertension"` → `glaucoma-elevated-iop`.
- No override + canonical bare `"glaucoma"` → `None` (ambiguous, unchanged).
- Canonical condition outranks a conflicting `people.tags` entry (pins ordering): e.g. canonical `"wet amd"` + `people.tags` `["dry amd"]`, no override → `wet-amd`.
- A client with only `people` data and no canonical row → identical result to today (regression guard).

**`api_console_client_condition_get`**
- `resolved` matches `_client_condition_for` for the same client (the two must never diverge — assert equality across override / canonical / people-only cases).
- `tags` includes the canonical condition when one is present.

**CI conventions:** pin the product catalog per the `$DATA_DIR`-strips-catalog rule; dummy `OPENAI`/`PINECONE` keys for the app-importing test module.

## Explicitly out of scope

- AI analysis and remedy-matcher wiring (later CTI-2 slices).
- Any change to how canonical conditions are *written* (that's document ingestion / import / a future editor).
- Terrain concerns, body systems, tags, challenges, goals as program inputs — only **conditions** feed the eye-condition program.
- Backfilling or projecting `person_attributes` into `people` columns.
- Widening `_CONDITION_TAG_MAP` or the `canonical_vocab` aliases — the feature uses them as they stand.
