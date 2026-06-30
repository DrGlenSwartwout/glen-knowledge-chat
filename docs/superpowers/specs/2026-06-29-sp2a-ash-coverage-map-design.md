# SP2a — ASH Coverage Map + Ally Memory (design)

**Date:** 2026-06-29
**Status:** Design / spec (pending review)
**Repo:** deploy-chat (illtowell.com)
**Parent:** the ASH voice-first health ally (foundation doc
`2026-06-27-ash-health-ally-foundation.md`). SP1/1.5/1.6 (voice doorway + voice-out) shipped.

## Context

SP2 is the persistent, multi-turn ally that talks with a person across sessions, quietly filling a
private 12-dimension ASH coverage map and following the threads they open. It is decomposed:
- **SP2a (THIS spec):** the durable per-person state — the ASH coverage map + ally memory + the
  per-turn updater that fills it. Self-contained and testable with no chat surface.
- **SP2b (later):** the persistent ally conversation surface that reads/writes this state.
- **SP2c (later):** the "owner's manual" surfacing + always-reach (Terrain/Tissue/Regulation).

Decided with Glen 2026-06-29: the ally is **email-keyed** (lives for anyone opted in through the
doorway; remembers across devices/sessions; not gated on a purchase). The memory holds **coverage +
substance** (state + verbatim opening excerpt + notes + a rolling "who they are" summary), at the
**12-dimension** level (notes can hold any five-fold detail that surfaces).

The existing chat infra (mapped 2026-06-29) is reused by SP2b, not SP2a. SP2a only needs: the
Anthropic client + Haiku model + forced-tool-use structured-output pattern (as in
`journal_blueprint._haiku_analyze`), and the sqlite `LOG_DB` / `_db_lock` storage pattern.

## The 12 ASH dimensions (canonical keys + definitions for the updater prompt)

| key | name | one-line meaning (for classification) |
|---|---|---|
| `body` | Body / States of Matter | the physical body's substance, density, structure |
| `mind` | Mind / 5 C's | mental focus, emotional patterns, how they connect and communicate |
| `spirit` | Spirit / 5 Elements | meaning, purpose, emotional-elemental balance |
| `inheritance` | Inheritance / 5 Generations | family, genetic, lineage health patterns |
| `personal_history` | Personal History / 5 Penetration | their own health history and how deep issues have gone |
| `epigenetics` | Epigenetics / 5 Infoceuticals | bioenergetic / informational regulation (terrain, organs, meridians, systems) |
| `symptoms` | Symptoms / 5 Cardinal Signs | active symptoms: pain, heat, swelling, redness, loss of function |
| `terrain` | Terrain / 5 R's | the body's vitality and capacity to heal |
| `diagnosis` | Diagnosis / 5 Pathology Types | diagnosed conditions or tissue changes |
| `treatment` | Treatment / 5 Therapy Levels | treatments they use and how invasive vs. supportive |
| `regulation` | Regulation / 5 Levels | how the body responds when they try to heal |
| `prognosis` | Prognosis / 5 Stages | seriousness or trajectory of their main concern |

These live as a constant in `dashboard/ash_map.py` (`ASH_DIMENSIONS`: ordered list of
`{key, name, meaning}`). The updater prompt renders them so Haiku can map turn content → dimensions.

## Data model

One new table (init-on-use, mirroring `journal_store`):

```sql
CREATE TABLE IF NOT EXISTS ash_ally_memory (
  email           TEXT PRIMARY KEY,
  summary         TEXT NOT NULL DEFAULT '',   -- short rolling "who they are"
  dimensions_json TEXT NOT NULL DEFAULT '{}',  -- the 12-dimension map (below)
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL
);
```

`dimensions_json` decodes to a dict keyed by the 12 canonical keys; each value:

```json
{
  "state": "untouched | opened | explored | deep",
  "opened_excerpt": "their verbatim words that first opened this door (set once)",
  "notes": "accumulated free-text: what we have learned in this dimension",
  "last_touched_at": "ISO8601 | null"
}
```

A freshly-read map has all 12 keys present with `state="untouched"`, empty excerpt/notes, null
timestamp (the getter fills any missing keys so callers always see all 12). Email is stored
lowercased/stripped (`_norm_email`).

**State ladder (monotonic forward-only):** `untouched(0) < opened(1) < explored(2) < deep(3)`.

## Module interface — `dashboard/ash_map.py`

Pure functions over a caller-supplied `sqlite3.Connection`, plus one thin LLM call.

- `ASH_DIMENSIONS: list[dict]` — the 12 `{key, name, meaning}` above. `DIM_KEYS: list[str]`.
- `init_table(cx) -> None` — create table if absent.
- `get(cx, email) -> dict` — returns `{email, summary, dimensions:{<all 12>...}, created_at,
  updated_at}`; a never-seen email returns an all-`untouched` skeleton (not persisted until written).
- `_blank_map() -> dict` — the 12-key all-untouched skeleton (pure).
- `merge_turn(memory: dict, updater_output: dict) -> dict` — **PURE.** Applies one updater result:
  for each entry in `updater_output["dimensions"]`, bump `state` forward-only
  (`max(current, proposed)` by the ladder), set `opened_excerpt` only if currently empty, append the
  `notes` delta (newline-joined, deduped on exact-line), stamp `last_touched_at=now`. Replace
  `summary` with `updater_output["summary"]` when non-empty. Untouched dims untouched. Returns a new
  dict (does not mutate input).
- `_haiku_extract(memory: dict, user_text: str, ally_text: str) -> dict` — the LLM call. Builds a
  Haiku `messages` request with `tool_choice` forcing an `emit_coverage` tool whose schema is the
  updater output below; returns the parsed tool input (or `{"dimensions": {}, "summary": ""}` on any
  error — never raises). Uses the shared Anthropic client + `claude-haiku-4-5-20251001`.
- `update_from_turn(cx, email, user_text, ally_text="") -> dict` — orchestrator: `get` → `_haiku_extract`
  → `merge_turn` → persist (upsert) → return the merged memory. Designed to be called
  **fire-and-forget after** an ally reply.
- `context_block(memory: dict) -> str` — formats the memory for an ally system prompt:
  `Who they are: <summary>` / `Already explored (do not re-ask): <dim: notes ...>` /
  `Opened, go deeper when they return to it: <dim: excerpt>` / `Not yet touched: <dim list>`.
  An empty or all-untouched memory yields a short "first conversation, nothing covered yet" line.

**Updater structured-output schema (`emit_coverage` tool input):**
```json
{
  "dimensions": {
    "<dim_key>": { "state": "opened|explored|deep", "excerpt": "verbatim user words or ''",
                   "notes": "what was learned this turn" }
  },          // ONLY dimensions actually touched this turn; omit the rest
  "summary": "refreshed 1-2 sentence 'who they are' (or '' to keep the prior summary)"
}
```
The updater prompt instructs: include a dimension ONLY if the turn genuinely touched it; `excerpt`
= the person's own words that opened/deepened it; never invent; default to fewer dimensions.

## Reuse (no new infra)

- Anthropic client + `claude-haiku-4-5-20251001` + forced `tool_choice` structured output — same
  shape as `journal_blueprint._haiku_analyze` / its `ANALYSIS_TOOL`. (Import or mirror the client
  construction used there; keep `_haiku_extract` self-contained so tests mock one `requests.post`.)
- Storage: `LOG_DB` + `with _db_lock, sqlite3.connect(LOG_DB) as cx:` pattern. SP2a's functions take
  `cx`, so the module imports neither app nor begin_funnel (pure-module testable).

## Testing — `tests/test_ash_map.py` (pure-module, plain pytest; no app import)

- `merge_turn` (pure, the core): state bumps forward-only and never downgrades; `opened_excerpt`
  set once then preserved; notes accumulate without duplicating an identical line; untouched dims
  stay untouched; non-empty summary replaces, empty summary preserves; input not mutated.
- `get` returns all 12 keys as `untouched` for an unseen email; round-trips after an upsert.
- `context_block`: all-untouched → "first conversation" line; a populated map lists explored
  (with notes), opened (with excerpt), and not-yet-touched correctly.
- `_haiku_extract`: monkeypatch the HTTP call (the `test_journal_haiku` pattern) to return a
  forced-tool-use body → assert it returns the parsed `{dimensions, summary}`; malformed/no-tool
  response → returns the empty default, does not raise.
- `update_from_turn`: with `_haiku_extract` monkeypatched to a fixed output, assert the persisted
  memory equals `merge_turn(blank, output)` and a second turn accumulates.

## Verification (end-to-end, local)

- `python3 -m pytest tests/test_ash_map.py -v` (pure-module — no doppler needed).
- A scripted 2-turn walk (updater mocked): turn 1 opens `symptoms` + `terrain`; turn 2 deepens
  `symptoms` and opens `inheritance` → assert states, the symptoms excerpt is the turn-1 wording,
  notes from both turns present, untouched dims still untouched, summary updated.

## Out of scope (SP2a)
- The ally conversation surface, streaming, and where it lives (SP2b). SP2a writes/reads state only;
  nothing calls `update_from_turn` yet.
- The owner's-manual surfacing + always-reach trio (SP2c).
- Migrating prior `query_log` history into the map (a later backfill if wanted).
- Five-fold (12×5) sub-category cells — notes hold any five-fold detail at the dimension level.
