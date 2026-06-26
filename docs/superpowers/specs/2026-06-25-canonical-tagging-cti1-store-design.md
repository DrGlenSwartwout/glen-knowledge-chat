# Canonical Tagging In-House — CTI-1: Canonical Store + Vocabulary (Foundation)

**Date:** 2026-06-25
**Status:** Approved (design)
**Author:** Glen + Claude
**Parent:** Canonical Tagging In-House (CTI) — see `2026-06-25-canonical-tagging-inhouse-NOTES.md`. This is the first increment (foundation). CTI-2 (GHL flip), CTI-3 (writers/AI auto-tag), CTI-4 (console curation) follow.

## Problem

Today GHL owns the clinical fields: `console_push_cron.py:sync_people_from_ghl()` pulls GHL custom fields → `people.{tags, conditions, terrain_concerns, body_systems, challenges, goals}` hourly and overwrites the local copy. The app has no in-house authoritative store for these, no controlled vocabulary, and AI-extracted health content never persists to the person. To make the app the source-of-truth (Glen's decision), we first need the authoritative store + vocabulary + an import of current values — without breaking anything.

## Goal

Establish the in-house canonical store, a controlled-vocabulary/alias mechanism, a one-time import of current `people.*` values, and a function that regenerates the `people.*` columns from the store — all **dark and non-breaking** (no behavior change yet; CTI-2 wires it as authoritative and flips GHL).

## Non-goals (later increments)

- Flipping the GHL sync / pushing out to GHL (CTI-2).
- New writers — manual console edit, AI auto-tag-from-comms, intake/scan derivations (CTI-3).
- Curation console for the vocabulary/aliases (CTI-4) — CTI-1 seeds the vocab table empty.
- Any `/api/people` endpoint change, route, or prod-deploy wiring — CTI-1 is a pure, offline-testable module + tests. (CTI-2 runs the import on prod and makes the store authoritative.)

## Design

Mirror the canonical remedy-meanings pattern (`dashboard/biofield_meanings.py`): a pure module, `cx`-based, none-raising.

### Field model

- **Discrete (multi-value, JSON-array columns):** `tags`, `conditions`, `terrain_concerns`, `body_systems` — vocabulary/alias-controlled.
- **Scalar (free-text columns):** `challenges`, `goals` — canonical-owned but free-text (normalized only, no alias vocabulary).

`tags` holds BOTH clinical and operational (`type:`/`consent:`) tags; operational tags simply have no alias and resolve to themselves.

### Tables (chat_log.db)

- `canonical_vocab(field TEXT, alias_norm TEXT, canonical TEXT, PRIMARY KEY(field, alias_norm))` — maps a normalized alias → curated canonical display term, per discrete field. Seeded EMPTY in CTI-1 (curation = CTI-4). When empty, `resolve` falls back to the cleaned value, so behavior = plain normalization until aliases are curated.
- `person_attributes(id INTEGER PK, email TEXT, field TEXT, value TEXT, value_norm TEXT, source TEXT, added_at TEXT, UNIQUE(email, field, value_norm))` — per-person canonical values. Discrete fields: many rows per (email, field), deduped by `value_norm`. Scalar fields: exactly one row per (email, field) (replace on change). `source ∈ manual|ai|ghl|rule|scan|import`.

### Module `dashboard/canonical_tags.py`

- `DISCRETE_FIELDS = ("tags","conditions","terrain_concerns","body_systems")`, `SCALAR_FIELDS = ("challenges","goals")`, `ALL_FIELDS = DISCRETE_FIELDS + SCALAR_FIELDS`.
- `init_tables(cx)`.
- `_norm(s)` — lowercase, collapse whitespace, strip surrounding non-word chars (same convention as biofield_stress).
- `resolve(cx, field, value) -> str` — discrete: return `canonical_vocab[(field,_norm(value))]` if present, else the trimmed value (whitespace-collapsed). Scalar: return the trimmed value (no vocab).
- `set_attr(cx, email, field, value, *, source) -> bool` — `email` lowercased; `value` resolved. Discrete: `INSERT OR IGNORE` keyed on `(email, field, value_norm=_norm(resolved))` (dedup; first source wins, never clobbers). Scalar: replace the single `(email, field)` row with the new value. Returns whether a row was written. No-op on empty value.
- `get_person(cx, email) -> dict` — reconstruct: each discrete field → sorted list of its canonical values; each scalar field → its single value (or ""). Always returns all 6 keys.
- `rebuild_people_columns(cx, email)` — `UPDATE people SET tags=?, conditions=?, terrain_concerns=?, body_systems=? (json.dumps lists), challenges=?, goals=? (text)` from `get_person`. This is how the store stays authoritative while everything keeps reading the legacy columns. (Used by CTI-2; in CTI-1 it's covered by tests, proving it round-trips.)
- `import_from_people(cx) -> {"persons": n, "attrs": m}` — one-time seed: for each `people` row, parse current `tags/conditions/terrain_concerns/body_systems` (JSON arrays) and `challenges/goals` (scalars), and `set_attr(..., source='import')` each. Idempotent via dedup. `source='import'` marks legacy-sourced values (provenance for CTI-2's flip).

### Why nothing breaks in CTI-1

No endpoint/route/cron/writer changes. The module is additive; the hourly GHL sync and `/api/people` are untouched. The store is built and provably regenerates the existing columns, but is not yet wired as the authoritative writer — that's CTI-2.

### Components / files

- `dashboard/canonical_tags.py` (new) — the whole module above. (No app.py change in CTI-1.)

### Testing (TDD, offline; tmp sqlite)

1. **set_attr / dedup** — discrete: two normalized-duplicate values collapse to one row; distinct values both stored; scalar: a second value replaces the first (one row). Empty value → no-op. source recorded.
2. **resolve / vocabulary** — with a seeded `canonical_vocab` alias ("adrenal exhaustion" → "Adrenal Fatigue") for `conditions`, `set_attr`/`resolve` store the canonical form; an unknown value falls back to the trimmed input; scalar fields ignore vocab.
3. **get_person** — reconstructs all 6 keys; discrete → sorted canonical lists, scalar → value; missing person → all-empty.
4. **rebuild_people_columns** — writes the 6 columns (JSON arrays for discrete, text for scalars) from the store; round-trips (import → rebuild reproduces the original column values, modulo normalization/sort).
5. **import_from_people** — seeds the store from a tmp `people` table (JSON-array + scalar columns), `source='import'`, idempotent on re-run; bad/empty JSON tolerated (no raise).

## Rollout

CTI-1 ships the module + tests only — **no prod behavior change, no deploy-time wiring**. CTI-2 will: run `import_from_people` on prod (console-gated trigger, dry-run first), stop `sync_people_from_ghl` from overwriting the 6 fields, make `rebuild_people_columns` the writer, and add the push-out to GHL.
