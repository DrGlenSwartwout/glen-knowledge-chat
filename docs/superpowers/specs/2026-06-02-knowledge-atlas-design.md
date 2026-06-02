# Knowledge Atlas — Design Spec

**Date:** 2026-06-02
**Status:** Approved (brainstorm) — ready for implementation planning
**Project:** `~/deploy-chat` (Flask app, Pinecone `remedy-match-llc`, funnel = `static/begin.html`)

---

## 1. Goal

A reusable **Knowledge Atlas** module: a keyword/topical index of Glen's clinical
knowledge base that a visitor can explore three ways — a visual **meaning-map**
(default), an **A–Z index**, and a **hierarchical index** — with a **chat spine**
that answers questions in prose *and* drives the visual (highlight + answer). Every
concept carries active links to videos (YouTube/Rumble), articles, and products.

It is **one embeddable module with two display sizes**, mounted in three places:

1. **Standalone hub page** (`/atlas`) — the public "knowledge atlas," full size.
2. **Embedded** on the funnel and other pages — compact widget via a `widget.js`-style
   snippet, with a **⤢ expand-to-fullscreen** toggle.
3. **Inside the chat experience** — the chat panel can surface the Atlas; chat and map
   are the same backend made visible.

Audience: **public knowledge hub** — educational, articles & videos first, products
secondary. No login. Doubles as an SEO topical-authority surface.

---

## 2. Core principle: chat is the spine

The existing `/chat` endpoint already does semantic retrieval over all 11 Pinecone
namespaces (`query_all_namespaces`). The Atlas is **that same vector space made
visible and clickable**:

- The **meaning-map** = concepts laid out by vector similarity (2-D projection).
- The **A–Z index** = the same concepts, sorted alphabetically.
- The **hierarchy** = the same concepts, grouped by cluster/parent.
- The **chat** = the same retrieval, returned as prose with concept terms linked back
  to the map.

One concept store, one selection model, one chat backend → three faces.

---

## 3. Data model

### 3.1 Concept store — `data/atlas-concepts.json`

Curated/approved concepts (the live set). Follows the existing `data/*.json` pattern
(`product-aliases.json`, `trusted-links.json`).

```jsonc
{
  "version": "2026-06-02",
  "concepts": [
    {
      "id": "biofield",                 // stable slug
      "label": "Biofield",              // display
      "aliases": ["bioenergetic field", "energy field"],
      "summary": "The body's organizing energy field; basis of BEV measurement.",
      "namespaces": ["clinical-qa", "glen-authored-works"],  // where it lives in KB
      "cluster": "energetic-medicine",  // for hierarchy + map coloring
      "parent": "energetic-medicine",   // hierarchy parent (nullable → top level)
      "coords": { "x": 0.42, "y": 0.55 }, // precomputed 2-D map position (0..1)
      "neighbors": ["light-therapy", "detox", "structured-water"], // map edges (top-k)
      "links": [
        { "type": "video",   "source": "youtube", "url": "https://youtu.be/…", "title": "BEV explained" },
        { "type": "video",   "source": "rumble",  "url": "https://rumble.com/…", "title": "Field testing" },
        { "type": "article", "url": "/begin/learn/…", "title": "Biofield basics" },
        { "type": "product", "url": "https://remedymatch.com/…", "title": "…", "slug": "…" }
      ],
      "status": "live"                  // live | pending | hidden
    }
  ]
}
```

**Why precomputed `coords`/`neighbors`:** the map must render instantly client-side
without a per-load embedding/projection call. The build pipeline computes them offline.

### 3.2 Video catalog — `data/atlas-videos.json`

A flat list of known YouTube + Rumble videos (url, title, description, platform). The
seed pipeline matches videos to concepts by embedding title+description and scoring
against concept vectors. Manually correctable. (Source list provided by Glen / pulled
from channel exports — gathering it is a Phase-1 task.)

### 3.3 Pending queue — `data/atlas-pending.json`

Auto-seeded concept/link proposals awaiting Glen's approval (hybrid model). Same shape
as a concept entry plus `proposed_from` provenance. Approval moves an entry into
`atlas-concepts.json` with `status: "live"`.

---

## 4. Components & boundaries

Each unit has one purpose, a defined interface, and is independently testable.

### 4.1 `atlas_seed.py` — offline build/seed pipeline (CLI, not web)
- **Does:** reads the KB (Pinecone namespaces), extracts candidate concepts, embeds
  them, computes a 2-D projection (UMAP/PCA → `coords`), top-k `neighbors`, cluster
  labels, and matches videos/articles/products to each. Writes proposals to
  `data/atlas-pending.json`.
- **Depends on:** `embed()`, `_idx` (Pinecone), `data/products.json`,
  `data/trusted-links.json`, `data/atlas-videos.json`.
- **Interface:** `python atlas_seed.py [--namespaces …] [--limit N]` → writes pending.
- **Note:** projection/clustering libs (e.g. `umap-learn`/`scikit-learn`) added to
  `requirements.txt`; pipeline runs offline so heavy deps never touch request path.

### 4.2 Admin approval view — `/admin/atlas` (gated, mirrors `/admin/membership`)
- **Does:** lists pending concepts/links; Glen approves/edits/rejects; approved entries
  written to `data/atlas-concepts.json`. Edit a concept's links, label, cluster, parent.
- **Depends on:** existing console/admin gate (`@require_console_key` pattern).
- **Interface:** HTML page + `POST /admin/atlas/approve`, `/admin/atlas/reject`,
  `/admin/atlas/edit`.

### 4.3 Atlas API — read endpoints (public)
- `GET /atlas/data` → the live concept graph (concepts + coords + neighbors + links +
  hierarchy). Cached in-memory, served as JSON. This is what the frontend loads once.
- `POST /atlas/ask` → wraps the existing chat retrieval; returns `{answer, concept_ids,
  highlight}` so the frontend can render prose **and** know which nodes to light up.
  Reuses `/chat` retrieval internals; does not duplicate them.

### 4.4 Frontend module — `static/atlas.js` + `static/atlas.css`
- **Does:** renders the three views from `/atlas/data`, manages the **single shared
  selection**, the **detail drawer**, the **chat spine**, the **⤢ expand/⤡ collapse**
  toggle, and the **bidirectional linking**. Self-contained; no framework (matches the
  app's vanilla-JS pages).
- **Map rendering:** lightweight canvas/SVG force-free layout using precomputed
  `coords` (no physics engine needed for v1; nodes are positioned, edges drawn to
  `neighbors`). Pan/zoom/click.
- **Interface (embed):** `<div id="rm-atlas"></div><script src="…/atlas.js" data-mode="compact">`
  — mirrors the existing `widget.js` embed contract.

### 4.5 Standalone page — `static/atlas.html` (served at `/atlas`)
- Thin shell that mounts `atlas.js` in `full` mode with funnel chrome. SEO `<meta>` +
  server-rendered concept list in `<noscript>`/hidden markup for crawlers (topical
  authority) — concepts and their links present in initial HTML.

---

## 5. The three live links (interaction model)

Single shared **selected concept** + single **detail drawer** across all three views.

| Trigger | Effect |
|---------|--------|
| **Type in chat** (`/atlas/ask`) | Matching nodes glow + zoom on map; A–Z and Hierarchy auto-filter to the same set; prose answer rendered with concept terms underlined. |
| **Click a node** (or A–Z / hierarchy row) | Detail drawer opens its links; concept passed to chat as context for follow-ups. |
| **Hover/click a highlighted term in the chat answer** | Its node pulses on the map and the drawer opens. Map ⇄ answer reference the same concept objects by `id`. |

Switching Map / A–Z / Hierarchy **preserves the current selection**.

---

## 6. Display modes

- **`full`** — standalone `/atlas` page and the ⤢-expanded overlay. Map hero + drawer +
  chat spine + view toggle.
- **`compact`** — embedded form: smaller map, view toggle (⊞ ⋔) + **⤢ expand** in the
  header, chat as a single ask bar. ⤢ opens the `full` overlay in place; ⤡ collapses.
- Palette (both modes) = **funnel**: bg `#0a150d`, surface `#111f16`, green `#3d8a52`,
  gold `#d4a843`, cream `#fdf4d8`, border `#21472d`, accents `#e2ba5c`/`#e08a8a`.

---

## 7. Video links

- v1: links resolve to existing **YouTube + Rumble** URLs from `atlas-videos.json`,
  matched to concepts by the seed pipeline (Glen approves matches).
- **＋ generate a video** action in the drawer is a **Phase 2** hook — calls the
  ai-talking-head / NotebookLM pipeline on demand for concepts with no video. v1 shows
  the affordance but it may be disabled/"coming soon" until wired.

---

## 8. Error handling

- `/atlas/data` missing/empty → frontend shows a friendly "atlas is being built" state;
  page still renders chrome.
- `/atlas/ask` failure (chat backend error) → drawer/map stay usable; chat shows a
  retry message (mirror existing `/chat` error handling).
- Dead/uncertain links never auto-published: product links go through the existing
  `trusted-links.json` whitelist (never AI-guessed URLs); video/article URLs are only
  those Glen approved.
- Seed pipeline is offline and idempotent — re-running regenerates pending without
  touching the live `atlas-concepts.json`.

---

## 9. Testing

- **Pipeline:** unit tests for concept extraction shape, projection determinism (fixed
  seed), video-matching scoring, and idempotent pending writes (fixture KB sample).
- **API:** `/atlas/data` schema test; `/atlas/ask` returns `{answer, concept_ids}` and
  only references known concept ids; admin approve/reject mutate JSON correctly (tmp
  copy). Follow existing `tests/` patterns + `dashboard.CONSOLE_SECRET` dual-patch for
  the gated admin routes.
- **Frontend:** a small Playwright check (repo already uses Playwright) — load `/atlas`,
  switch all three views with selection preserved, ask a question → a node gains the
  highlight class, click a node → drawer shows links, ⤢ expands / ⤡ collapses.

---

## 10. Scope & phasing

**Phase 1 (this spec's implementation plan):**
1. `data/atlas-videos.json` gathered (YouTube + Rumble export) + `atlas-concepts.json`
   schema + a **small hand-seeded starter set** (~20–40 concepts) so the UI is testable
   before the full pipeline.
2. `atlas_seed.py` pipeline (extract → embed → project → neighbors → cluster → match
   links → write pending).
3. `/admin/atlas` approval view.
4. `/atlas/data` + `/atlas/ask` endpoints.
5. `atlas.js` + `atlas.css`: three views, shared selection, drawer, chat spine,
   bidirectional linking, full/compact + ⤢ expand.
6. `/atlas` standalone page (funnel chrome + SEO markup).
7. Embed snippet (`widget.js`-style) + mount on `begin.html` funnel.

**Phase 2 (noted, not planned yet):**
- `＋ generate a video` wired to ai-talking-head/NotebookLM.
- In-chat surfacing (Atlas view inside the existing chat panel / `embed.html`).
- "Seed then learn": refine concepts/links from real Atlas usage logs.
- Force-directed/physics map layout if the static projection feels flat.

**Out of scope:** auth/personalization, editing KB content from the Atlas, multilingual.

---

## 11. Open inputs needed from Glen at build time
- The YouTube + Rumble video list/export (or channel URLs to pull from).
- Confirm the hand-seeded starter concept set (Phase 1.1) once proposed.
