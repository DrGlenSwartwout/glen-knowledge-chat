# Stress-Pattern Glossary — Design

**Date:** 2026-07-14
**Status:** Approved (design direction)
**Builds on:** [[2026-07-13-report-entity-hover-linkout-design]] — finishes its deferred "stress-pattern glossary page" so portal pop-up patterns become click-through.

## Problem

Stress patterns on the client portal report show a hover pop-up (their `e4l_description`) but have no full page to click through to — the one entity type left pop-up-only. There is also no browsable glossary of the E4L stress patterns.

## Data reality (grounds the scope)

- `e4l.db` `e4l_items`: **224 patterns, 85 with `e4l_description`, 0 with `clinical_notes`** (empty column). So per-pattern prose = the same description the pop-up already shows.
- `e4l.db` `e4l_pattern_structures`: **866 rows over 141 patterns** mapping each pattern → body structures/functions, `stype ∈ {organ, system, function, emotion, substance, immune_cell}`, `is_primary` flag. This is the value a full page adds over the pop-up.
- Focus-area pattern→remedy maps exist but are **deferred** (messier mapping; second pass).

## Goal

A public glossary: an index of the E4L stress patterns plus a detail page per pattern showing its description and the body structures/functions it involves. Portal pop-up patterns gain a click-through to their detail page (new tab).

## Scope (this slice)

- **In:** per-pattern detail page (description + structures/functions grouped by type), a browsable index grouped by category, and wiring the portal pop-up patterns to link out.
- **Out (deferred):** a "What may help / remedies" section (focus-area maps); authoring descriptions for the ~139 patterns that lack one; structure/function detail pages (structures stay plain text on the pattern page — no page exists for them yet).

A pattern is considered to **have a page** (and thus is linkable / listed) when it has a description **or** ≥1 mapped structure. Patterns with neither stay plain text.

## Architecture

Mirror the `/begin/ingredient/<slug>` pattern (static HTML shell + JSON data endpoint), reading straight from `e4l.db` read-only — NOT the AI-authored `topic_pages`/`/learn/<slug>` infra, since pattern data is already structured.

### `dashboard/pattern_glossary.py` (new, pure, read-only)

Uses `biofield_e4l._connect_ro(_db_path())` for a read-only `e4l.db` connection.

- `slug_for(code) -> str` — `slugify(code)` (code is the `e4l_items` PK; unique).
- `_slug_map(cx) -> {slug: code}` — built from all `e4l_items.code` (224 rows; cheap).
- `get_pattern(cx, slug) -> dict|None` — `{code, name, full_name, category, subcategory, description, structures: [{structure, stype, is_primary}], has_page}` where `structures` is ordered `is_primary DESC, stype, structure`. Returns `None` when the slug maps to no code.
- `page_exists(cx, slug) -> bool` — the code has a description or ≥1 structure.
- `list_patterns(cx) -> [{category, patterns: [{slug, name, full_name, has_desc, n_structures}]}]` — only patterns that `page_exists`, grouped by category in a fixed category order, patterns sorted by `sort_order, name`.
- `STYPE_LABELS = {"organ":"Organs","system":"Body systems","function":"Functions","emotion":"Emotions","substance":"Substances","immune_cell":"Immune cells","other":"Other"}`.

### Routes (`app.py`)

- `GET /learn/patterns` → `send_from_directory(STATIC, "patterns-index.html")`.
- `GET /learn/patterns-data` → JSON `{groups: list_patterns(...)}`.
- `GET /learn/pattern/<slug>` → `send_from_directory(STATIC, "pattern-page.html")`.
- `GET /learn/pattern-data/<slug>` → `get_pattern(...)` or `404 {"state":"unknown"}`.

Flask prefers static rule segments over `/learn/<slug>`, so `/learn/patterns` and `/learn/pattern/<slug>` do not collide with the existing single-segment `/learn/<slug>`.

### Static pages (mirror `begin-ingredient.html` styling + theme-toggle)

- `static/pattern-page.html` — fetches `/learn/pattern-data/<slug>`; renders title (`full_name` or `name`), a category badge, the description, and a "Body structures & functions involved" block grouped by `stype` using `STYPE_LABELS`, primary items first. `notfound` state when data 404s.
- `static/patterns-index.html` — fetches `/learn/patterns-data`; renders category sections, each a list of pattern links (`/learn/pattern/<slug>`), with a small "N structures" / "described" hint.

### Portal wire-up

- `biofield_e4l._findings` attaches `pattern_href` per finding: `"/learn/pattern/" + slug_for(code)` when the pattern `page_exists`, else `""`. (Resolved against the same read-only `e4l.db` connection already open in `_findings`.)
- `static/client-portal.html` pattern chip: when `f.pattern_href` is present, render the described chip as an `<a class="pat-chip entity-ref" href=… target="_blank" rel="noopener">` (link-out); otherwise keep the pop-up-only `<span>` as today. Undescribed findings stay plain `<span>`.
- `entity_refs.pattern_ref(name, description, href="")` gains an optional `href` param so the resolver stays consistent for any other caller (defaults to pop-up-only).

## Error handling / edge cases

- Unknown/empty slug → detail page shows a "not found" state; data endpoint returns 404. Never a 500.
- `e4l.db` unavailable (read-only open fails) → glossary endpoints return an empty index / 404 detail; `_findings` degrades to `pattern_href=""` (patterns stay pop-up-only). No crash.
- Slug collision (two codes slugify equal) → `_slug_map` keeps the first; deterministic by `code` order. (Codes are distinct PKs; collisions are unlikely but must not error.)
- A pattern with structures but no description → page renders the structures block and omits the description paragraph (no empty heading).
- HTML injection: all DB text is escaped on render (textContent / escaped template), same discipline as the entity-ref component.

## Testing

- **Unit (`dashboard/pattern_glossary.py`)** against a tiny in-memory/temp `e4l.db`: `get_pattern` shape + structure ordering (primary first); `page_exists` true for described-only, structures-only, false for neither; `list_patterns` groups + excludes empty patterns; `slug_for` round-trips via `_slug_map`.
- **Render-verify (headless Chrome)**: `/learn/pattern/<slug>` for a described+structured pattern shows description + grouped structures; a structures-only pattern omits the description; an unknown slug shows not-found. `/learn/patterns` lists grouped links that navigate to detail pages. On the portal, a linkable pattern chip opens its detail page in a new tab while still showing the hover pop-up.

## Out of scope (restated)

Remedies/"what may help", authoring missing descriptions, structure/function detail pages, and any write to `e4l.db`.
