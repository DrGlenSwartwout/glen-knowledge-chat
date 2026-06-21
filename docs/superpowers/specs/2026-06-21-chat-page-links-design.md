# Chat Page-Links — surface existing pages as active links in chat

**Date:** 2026-06-21
**Status:** Approved design, ready for implementation plan
**Sub-project of:** the on-request content arc (the surfacing half of the chat sub-project)
**Builds on:** [[project_topic_pages]] (the `/learn` pages) and the existing `begin_funnel.surface_for_chat` card machinery

---

## 1. Purpose

When a `/begin` chat conversation is **about a topic that already has a published page**, surface
an **active link** to that page as a card below the answer. This connects the conversational
funnel to the growing library of `/learn` topic pages, ingredient pages, and product pages —
turning "the AI mentioned methylation" into a one-tap link to the methylation guide.

This is purely **surfacing existing, already-approved pages**. It does NOT create pages
(that is the separate, still-deferred chat-creates-a-page sub-project).

## 2. Decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| Eligible pages | **Approved** topic pages (`/learn/<slug>`), ingredient pages (`/begin/ingredient/<slug>`), product/sales pages (`/begin/product/<slug>`) |
| Match method | **Deterministic name/alias match** over the user query **and** the AI answer text (no embeddings, no extra LLM call) |
| Surface as | Cards in the existing `surfaced_cards` payload (reuses the `{key,title,sub,href}` card the client already renders — no client change) |
| Card cap | Up to **2 link cards + 1 journey/quiz card = 3 max** (today's cap is 2; bump to 3 only when link cards fire) |
| Gated ingredient pages | **Link to everyone** (the page shows its own teaser/gate — itself a conversion surface) |
| Chat surface | The `/begin` funnel `/chat` endpoint only (other chat surfaces are future) |
| Rollout | **DARK behind `CHAT_PAGE_LINKS_ENABLED`** (default off) |

## 3. Why deterministic, and why match the answer text

The existing `begin_funnel._match_card_keys` already matches the query against keyword sets
deterministically. We follow that pattern. The key recall insight: **matching the AI answer
text on canonical page names is high-recall for free**, because the model naturally uses the
exact terms ("methylation", "magnesium", "brain fog") that are the page names. Curated aliases
mainly help the query side (e.g. "can't focus" → Brain Fog) and are added over time.

## 4. Architecture

### New module: `dashboard/page_links.py` (pure, testable, no Flask import)

- `build_index(pages, *, alias_map=None) -> dict`
  - `pages`: a list of `{slug, name, kind, href, gated}` records for approved pages.
  - Produces a phrase index: `{phrase_lower: {"title", "href", "kind", "gated"}}`.
  - Phrases per page = the page **name**, the **slug-as-words** (`"low-energy"` → `"low energy"`),
    and any curated aliases pointing at that slug.
  - On a phrase collision, the longer/more-specific source wins; ties keep first-seen.

- `match_page_links(text, index, *, limit=2) -> list[dict]`
  - Lowercases `text`, finds index phrases present as **whole words/phrases** (word-boundary,
    so `"iron"` does not match inside `"environment"`).
  - **Longest-phrase-first**: `"magnesium glycinate"` matches before `"magnesium"`, and once a
    span is claimed, shorter sub-phrases inside it are not double-matched.
  - Returns ordered, deduped (by `href`) link-card dicts: `{"key", "title", "sub", "href"}`,
    capped at `limit`. `key = f"{kind}:{slug}"`; `sub` is a type label
    (`"Read the guide"` topic, `"See the ingredient"` ingredient, `"View product"` product).
  - Pure and synchronous; never raises on normal input.

### Data: `data/page-aliases.json`

`{ "alias phrase": "slug", ... }` — hand-maintained, seeded with a handful of obvious
paraphrases (e.g. `"can't focus": "brain-fog"`, `"trouble sleeping": "poor-sleep"`,
`"tired": "low-energy"`). No schema change, no AI call. Missing file = no aliases (not an error).

### Wiring in `app.py`

- Flag `CHAT_PAGE_LINKS_ENABLED` (default off), near the other funnel flags.
- A cached index builder `_chat_page_link_index()` that enumerates approved pages once and
  caches with a short TTL (rebuilt at most every N seconds), wrapped so any failure yields an
  empty index. Sources:
  - topic pages: `dashboard.topic_pages.list_pages(cx)` filtered to `state == "approved"`,
    `href = /learn/<slug>`, `kind = "topic"`, `gated = False`.
  - ingredient pages: approved rows from `dashboard.ingredient_pages`,
    `href = /begin/ingredient/<slug>`, `kind = "ingredient"`, `gated = True`.
  - product/sales pages: approved rows from `dashboard.sales_pages`,
    `href = /begin/product/<slug>`, `kind = "product"`, `gated = False`.
- In the `/chat` SSE done handler (where the full answer text and `surfaced_cards` already
  exist): when the flag is on, call
  `page_links.match_page_links(query + " " + answer, _chat_page_link_index(), limit=2)`,
  then merge: **link cards first**, then the existing `surface_for_chat` cards, deduped by
  `href`, capped at **3 total**. Flag off → matcher not called; behavior unchanged.

## 5. Error handling & safety

- Index build and matching are each wrapped: any failure → zero link cards, the chat stream is
  never broken.
- Hrefs are constructed only from known approved slugs — no model-generated URLs, no
  hallucinated links.
- Dedupe by `href` so a link card and an existing journey card never point to the same place.
- Cap enforced (≤2 link cards, ≤3 cards total).
- Flag off → the matcher is never invoked.

## 6. Testing

Unit tests (no `import app` — Pinecone-at-import blocks boot in sandbox; test `dashboard/*`
helpers directly and verify `app.py` via `py_compile`):

- **`match_page_links`:** a query naming a page surfaces its card; the **answer-text** path
  surfaces a page named only in the answer; **word-boundary** (`"iron"` must NOT match inside
  `"environment"`); **longest-phrase-first** (`"magnesium glycinate"` beats `"magnesium"`, no
  double match); **dedupe by href**; **cap** at `limit`; correct `sub`/`key`/`href` per kind.
- **`build_index`:** includes name + slug-words + alias phrases; alias file present and absent;
  collision resolution.
- **gated flag** carried through for ingredient pages (links still produced — they show to
  everyone — but the `gated` field is correct for future use).
- **`app.py`** edits verified with `python3 -m py_compile`.

## 7. Rollout

- Ships **DARK behind `CHAT_PAGE_LINKS_ENABLED`** (default off).
- Go-live: flip the flag in Doppler `remedy-match/prd` after a quick live check that a chat
  message about a published topic surfaces its link card. Reversible by flipping back.

## 8. Out of scope (future)

- Chat *creating* a page when none exists (the other deferred sub-project).
- Semantic/embedding matching (deterministic + aliases is v1; embeddings are a later recall
  upgrade).
- Other chat surfaces beyond the `/begin` `/chat` endpoint.
- Inline hyperlinks inside the answer text (cards only, to avoid hallucinated URLs).
- Session-level "don't repeat a link already shown" suppression (per-message dedupe only in v1).
