# CTI-2 Slice 3 — canonical record as a gated signal at remedy-match review — Design

**Date:** 2026-07-26
**Status:** Approved in brainstorm with Glen 2026-07-26
**Repo:** deploy-chat
**Branch:** `sess/72339d9d-cti3` (off `main`)

## Problem

The remedy matcher (`ff_matcher` / `_make_ff_items_for`) is deliberately **scan-driven**: a client's voice-scan infoceutical labels → Pinecone query → LLM rank → `{name, slug, url, meaning}` match items. It reads nothing about a client's stored/canonical conditions, and Glen wants to keep that ranking pure (a fresh scan is the source of truth). But when Glen reviews the matches, he has no view of the client's canonical record (e.g. a doc-approved diagnosis) that might inform his review.

CTI-2 Slice 1 (support program) and Slice 2 (biofield narrative) wired canonical into those readers. Slice 3 completes CTI-2 by surfacing canonical at the **remedy-match review**, as Glen chose: **a separate, clearly-labeled signal Glen sees at review, NOT blended into the scan query or the scan-driven ranking.**

## The surface (grounded)

- **Client card:** `POST /api/portal/<token>/ff-matches` (`app.py:22370`) — token-scoped, client-facing; dosing hidden unless covered+reviewed. **Untouched by this slice.**
- **Glen's review list:** `GET /api/console/ff-match-drafts` (`api_console_ff_match_drafts_list`, `app.py:22512`) — console-gated (`_portal_console_ok`), returns `{"drafts": [...]}` via `ff_match_drafts.list_by_status`. Each draft row (`ff_match_drafts._row`) = `{email, scan_date, items, status, updated_at, published_at}`.
- **Review UI:** `static/console-ff-drafts.html` — `boot()` fetches `?status=draft`, `card(d, di)` renders each draft (header with email/scan_date, then the match `items` via `itemBlock`, then a Publish button). `collectItems(di)` reads the `.item` nodes and `publish` POSTs them to `/api/console/ff-match-drafts/publish`.

The signal attaches to **the console review list + its card**, nowhere else.

## Design

### Backend — enrich each draft with a `canonical` block

`api_console_ff_match_drafts_list` adds a per-draft `canonical` object sourced from `canonical_tags.get_person(email)`, **with `tags` dropped** (the CRM/GHL bucket — not a clinical signal). A small best-effort helper:

```python
_FF_SIGNAL_FIELDS = ("conditions", "terrain_concerns", "body_systems", "challenges", "goals")

def _ff_draft_canonical(cx, email):
    """The client's canonical record for the FF-review signal: get_person minus
    `tags`. Best-effort -- {} on any failure. Read-only."""
```

Applied to each draft in the list endpoint: `draft["canonical"] = _ff_draft_canonical(cx, draft["email"])`. Discrete fields come back as lists (as `get_person` returns them), scalar as strings. **Never merged into `items`.**

Glen chose **all canonical fields except `tags`**: `conditions`, `terrain_concerns`, `body_systems` (discrete lists), `challenges`, `goals` (scalar strings).

### Frontend — render `d.canonical` as a distinct block in `card()`

`card(d, di)` in `static/console-ff-drafts.html` renders a `.canon` section **above the match items** (so Glen reads the record context before the recommendations), labeled (e.g. "From the client's records"), styled distinctly from the match items, every value escaped via the existing `esc()`. **Rendered only when non-empty** (at least one field has content) — a client with no canonical record shows no block, exactly as today.

Each field renders as its own labeled line: discrete lists comma-joined, scalar strings as-is. Empty fields are omitted.

## Why the ranking and the published draft stay pure

This is the load-bearing property of Glen's choice:

- The `canonical` block is **separate from `items`** — it has no `slug`/`url`, is never passed to `ff_matcher`, `_make_ff_items_for`, the Pinecone query, or `_ff_llm_rank`. The scan-driven ranking is byte-for-byte unchanged.
- `collectItems(di)` reads only `.item` DOM nodes; the `.canon` block is NOT an `.item`, so it can never be collected into the published items. `publish` sends only `items`. The signal is **display-only, review-only** — it never becomes part of the draft, the published matches, or anything the client sees.

## Gated to Glen

Only the console review surface changes (the list endpoint is `_portal_console_ok`-gated; the page is the console FF-review page). The client card `/api/portal/<token>/ff-matches` and everything it feeds are untouched. The client never sees this signal — that is the "gated" in Glen's choice.

## Scope guards / non-goals

- **`ff_matcher.py`, `_make_ff_items_for`, `_ff_llm_rank`, `_ff_query_specific_formulations`, the published `items` — all untouched.** No change to what is matched or how it ranks.
- **Read-through** — no writes; `get_person` is read-only; no `people`/`person_attributes` write.
- **`tags` is not surfaced** (CRM bucket).
- The client-facing FF card is not changed.
- No new persistence — the block is computed at read time, not stored on the draft.

## Testing

**Backend helper `_ff_draft_canonical`**
- returns get_person's fields minus `tags`; a client with canonical conditions → `{conditions:[...], ...}` with no `tags` key.
- best-effort: `get_person` raising (monkeypatched) → `{}`, no raise.
- blank email → `{}`.

**Endpoint `GET /api/console/ff-match-drafts`**
- a draft whose email has canonical attributes → the returned draft has a `canonical` block with those fields and NO `tags`; `items` is byte-identical to before (unchanged).
- a draft whose email has no canonical row → `canonical` is `{}` (or empty fields); `items` unchanged.
- console-gate still 401s without the key.
- canonical failure path → `items` and the rest of the draft returned unchanged, 200.

**Frontend `card()` (node render test, mirroring the existing static-JS test pattern)**
- a draft with a `canonical` block renders a distinct labeled section containing the conditions/terrain/etc., ABOVE the item blocks.
- a draft with an empty/absent `canonical` renders NO canon block (no empty section).
- values are escaped (a `<script>` in a canonical value is not injected).
- the canon block is NOT an `.item` (so `collectItems` never picks it up) — assert its class is distinct from `item`.

**CI conventions:** pin the product catalog per the `$DATA_DIR`-strips-catalog rule; dummy `OPENAI`/`PINECONE` keys for the app-importing test module. Note: `.js` render tests are not CI-gated (pytest only) — run by hand; the Python endpoint tests gate the behavior that matters.

## Completes CTI-2

With this, all three CTI-2 readers see canonical: the eye-condition support program (Slice 1), the biofield narrative (Slice 2), and — as a gated review signal, ranking untouched — the remedy matcher (Slice 3).
