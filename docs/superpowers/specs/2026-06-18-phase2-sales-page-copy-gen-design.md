# Phase 2 — In-Funnel Sales Pages: AI Narrative Copy Generation

**Date:** 2026-06-18
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render service `glen-knowledge-chat`)
**Builds on:** Phase 1 (merged PR #171, live behind `SALES_PAGES_ENABLED`). Parent spec: `2026-06-18-funnel-sales-pages-design.md`.

---

## Problem

Phase 1 renders each product's *existing* data (description, ingredients, benefits, how-it-works). The factual panels are solid, but the **narrative/persuasion** copy is only as good as the often-thin existing `description`. Phase 2 generates rich, grounded narrative copy per product so every page reads like a real sales page — without touching the factual data.

## Scope

**Generate (narrative only):** the **intro** ("what this does for you"), the **overview/description**, and the **research section's how-it-works framing**. Grounded in the product's real ingredient stack + Glen's clinical lens + voice.

**Do NOT generate / leave as real data:** ingredient panel + doses, comparison table (stays the generic archetype — product-specific comparison facts are a later phase), images, video, CTA.

**Gate:** new flag **`SALES_PAGES_AI_COPY`** (default OFF). Ships dark on top of the live Phase 1; generation turns on independently.

**Out of scope (later phases):** console approval + the "now reviewed" viewer email (Phase 5); product-specific comparison facts; image generation (Phase 3); image feedback/credit (Phase 4).

---

## Architecture

### Storage — `sales_pages` table (`chat_log.db` under `DATA_DIR`)

`sales_pages(product_slug TEXT PRIMARY KEY, state TEXT, content_json TEXT, model TEXT, generated_at TEXT, created_at TEXT, updated_at TEXT)`

- `content_json` accretes generated copy per section: `{"intro": "...", "description": "...", "research": "..."}`.
- `state` is `draft` throughout Phase 2 (the `approved` transition arrives with the Phase-5 console). Follows the `client_portals(content_json)` pattern in `dashboard/client_portal.py`.
- New data-layer module **`dashboard/sales_pages.py`**: `init_table(cx)`, `get_page(cx, slug) -> dict|None`, `get_section(cx, slug, section) -> str|None`, `upsert_section(cx, slug, section, text, model)`.

### Generation endpoint (per-section, streamed)

New route **`GET /begin/product-page-gen/<slug>/<section>`** (section ∈ {intro, description, research}):
- If a cached draft exists → stream it back immediately (no regeneration) and finish.
- Else build the section prompt (below), stream via `sse()` + `_cl.messages.stream(model=…)` (in-request on Render — same pattern as the live chat endpoint, so no web-worker risk), accumulate tokens, `upsert_section(...)` the full text, send a `done` frame.
- On Claude error → send an `error` frame; the frontend renders the Phase-1 fallback copy for that section.
- Flag-gated: with `SALES_PAGES_AI_COPY` off, the route 404s (frontend never calls it).

### Page-data integration (`/begin/product-page-data/<slug>`)

For the three narrative sections, when `SALES_PAGES_AI_COPY` is on, mark each:
- `"ai": "cached"` + the stored draft text as the body, if a draft exists; or
- `"ai": "pending"` (body = the Phase-1 fallback text, used until/if generation runs).
When the flag is off, behavior is exactly Phase 1 (no `ai` field). Non-narrative sections are unchanged.

### Frontend (`static/begin-product.html`)

- On render, a narrative section marked `ai: "pending"` shows its fallback text but, **on first open** (intro = on load), opens an `EventSource` to `/begin/product-page-gen/<slug>/<section>`, clears the fallback, and streams tokens in live ("watch it write"). On `done`, leaves the generated copy; on `error`, restores the fallback text.
- `ai: "cached"` → render the stored draft immediately (no stream).
- No `ai` field (flag off) → Phase-1 rendering, untouched.
- The **AI caveat banner** renders at the top whenever the page carries draft AI copy: "Generated from Dr. Glen's knowledge base — pending his personal review for final approval."

### Generation prompt + compliance

Each section call receives: product name; the ingredient stack (names + doses via the existing `dashboard/product_content` / `products.json`); Glen's copy conventions (clinical lens, no fluff, no AI-pleasantry filler, supplement-label conventions); and a **section-specific brief**:
- **intro** — one warm paragraph: what this does for you.
- **description** — fuller plain-language overview of the formula and who it's for.
- **research** — the mechanism / how-it-works framing in lay language.

**Hard compliance constraints (in every prompt):** structure-function language only ("supports / promotes / helps maintain"); **no disease treat/cure/prevent claims**; no diagnosis; inject Glen's verified authority only where natural and true; no invented studies. Default model `claude-haiku-4-5`; a per-section model override allows upgrading to Sonnet later. Phase-5 console review + the `authority-injected-copy-guardian` skill are the human backstop before `approved`.

---

## Data flow

1. Viewer opens `/begin/product/<slug>` → page-data returns sections; narrative ones tagged `ai: cached|pending`.
2. Intro open by default: if `pending`, EventSource streams the intro live → server caches it.
3. Other narrative sections: stream on first open; cached after first generation (every later viewer gets it instantly).
4. Caveat banner shown while draft.

## Error handling

- Generation failure (Claude error / timeout) → `error` SSE frame → frontend keeps the Phase-1 fallback copy. Page never breaks.
- `upsert_section` is idempotent on `(slug, section)` — re-generation overwrites; concurrent first-opens may both generate, last write wins (harmless, copy is equivalent).
- DB unavailable → endpoint streams without caching (degrade to generate-every-time) rather than erroring.

## Testing

- **Data layer:** `init_table` idempotent; `upsert_section` then `get_section` round-trips; `get_page` returns accreted `content_json`.
- **Gen endpoint:** mock `_cl`; absent draft → streams + persists (assert row written); present draft → returns cached, `_cl` NOT called; flag off → 404.
- **Page-data:** flag on + no draft → narrative sections `ai: "pending"` with fallback body; flag on + draft present → `ai: "cached"` with draft body; flag off → no `ai` field (byte-identical to Phase 1).
- **Compliance smoke:** a generated-copy unit test asserting the prompt includes the structure-function / no-disease-claim constraint string (guards the guardrail's presence).
- **Fallback:** endpoint error frame path covered.
- Follow deploy-chat test isolation (mock Supabase; tmp `$DATA_DIR/chat_log.db`; `pytest.importorskip` playwright). SSE-into-section is the manual visual check.

## Flag summary

- `SALES_PAGES_ENABLED` (live) — the Phase-1 pages exist + are linked.
- `SALES_PAGES_AI_COPY` (new, default OFF) — narrative AI generation on top. Both must be on for generated copy to appear; with only the first on, pages render Phase-1 copy exactly as today.
