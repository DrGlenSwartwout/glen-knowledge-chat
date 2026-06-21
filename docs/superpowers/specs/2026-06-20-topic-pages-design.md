# Topic Pages — Public, SEO-First Symptom / Condition / Function Pages

**Date:** 2026-06-20
**Status:** Approved design, ready for implementation plan
**Sub-project of:** the on-request content arc (extends the ingredient-page engine, PR #196)

---

## 1. Purpose

Build the **backbone** of an on-request content system: public, SEO-indexed pages for
**symptoms, conditions, and functions**. These pages are the top-of-funnel net — a stranger
searching "low energy" or "methylation" lands on an educational, wellness-framed page that
links to the related remedies and ingredient pages already in the catalog, and pulls the
reader into the `/begin` funnel via an optin CTA.

This spec covers **only** the public page backbone + a hard compliance gate. The following
are explicitly **out of scope** (each a later sub-project, already sequenced):

- The chat-detects-gap *offer* (chat asks "want me to create a page for that?")
- Additional content formats: article / video / course / book
- Paid generation ($29.99 book, $49.99 course) + fulfillment
- Traffic-driven auto-creation of high-interest pages

## 2. Decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| First slice | Pages for symptoms / conditions / functions (the backbone) |
| Visibility | **Public, SEO-first** (open web, indexable, recruits new traffic) |
| Compliance posture | **Wellness/educational framing + automated hard gate** before publish |
| Page content | AI-drafted educational overview **+ auto cross-links to existing catalog entities** + optin CTA |
| Architecture | **New `topic_pages` module** (sibling of `ingredient_pages` / `sales_pages`), server-rendered for SEO |

## 3. Architecture

Mirrors the proven content-module spine used by `ingredient_pages`, `sales_pages`,
`biofield_reveals`, and `product_reviews`:
`draft → console approve → notify requesters → serve approved-only → regenerate`.

Two deliberate departures from the ingredient-page pattern:

1. **Server-rendered public HTML** (not a JS shell that fetches JSON). The ingredient page
   at `/begin/ingredient/<slug>` is a static HTML player that fetches
   `/begin/ingredient-page-data/<slug>` — Google sees an empty shell. SEO-first **requires**
   real server-rendered markup (`<title>`, `<h1>`, `<meta name="description">`, JSON-LD).
2. **Compliance hard gate** — a structural block on publish, not a human-vigilance task.

### New files

| File | Responsibility |
|---|---|
| `dashboard/topic_pages.py` | Data layer: schema, `get_page`, `upsert_section`, `record_request`, `notify_on_approve` |
| `dashboard/topic_copy.py` | AI drafting (`build_section_prompt`, `propose_links`) + compliance check (`compliance_scan`) |
| `dashboard/topic_page_actions.py` | Console actions on the dispatch spine: `topic_page.approve` (gated), `.edit`, `.regenerate` |

### Touch points in `app.py`

- Public route `GET /learn/<slug>` — server-rendered, approved-only.
- Public route `GET /learn` — index of approved pages.
- Public route `GET /learn/sitemap.xml` (or extend existing sitemap) — approved slugs only.
- Public `POST /learn/<slug>/request` — record email-when-ready + kick off background build.
- Console API: `GET /api/console/topic-pages` (list), `GET /api/console/topic-page/<slug>` (detail).
- Console page route `/console/topic-pages` (static HTML, mirrors `/console/ingredient-pages`).
- Module wiring: `import dashboard.topic_page_actions as _tpa; _tpa.register(); _tpa.configure(...)`
  alongside the existing `ingredient_page_actions` registration block.
- Feature flag check `TOPIC_PAGES_ENABLED` (default off) guards all public routes.

## 4. Data model

### `topic_pages`

| Column | Type | Notes |
|---|---|---|
| `slug` | TEXT PK | URL slug, e.g. `low-energy`, `methylation` |
| `kind` | TEXT | `symptom` \| `condition` \| `function` (discriminator — one table, identical lifecycle) |
| `name` | TEXT | Display name |
| `state` | TEXT | `draft` \| `gated` \| `approved` (see state machine §6) |
| `content_json` | TEXT | Ordered sections `[{section, text}, ...]` |
| `links_json` | TEXT | Validated related slugs: `{ingredients: [...], products: [...], topics: [...]}` |
| `compliance_json` | TEXT | Gate result: `{passed: bool, flags: [{phrase, reason}], scanned_at, model}` |
| `seo_json` | TEXT | `{title, meta_description, jsonld}` |
| `model` | TEXT | Generating model id |
| `generated_at` | TEXT | |
| `approved_at` | TEXT | |
| `approved_by` | TEXT | actor email/role |
| `created_at` / `updated_at` | TEXT | |

### `topic_page_requests`

| Column | Type | Notes |
|---|---|---|
| `slug` | TEXT | composite PK part |
| `email` | TEXT | composite PK part |
| `requested_at` | TEXT | |
| `emailed_at` | TEXT | NULL until notified; set atomically on approve |

Exactly mirrors `ingredient_page_requests`.

## 5. Draft + cross-link flow

1. **Seed or request.** Glen seeds a list of topic names (`kind` + `name`), OR a public
   visitor hits `POST /learn/<slug>/request` (records `topic_page_requests` row, kicks off a
   background daemon-thread build — same pattern as `_ingredient_kickoff_build`).
2. **AI drafts sections.** `topic_copy.build_section_prompt(section, topic)` → `(system, user)`,
   run through `_cl` haiku (`claude-haiku-4-5-20251001`). System prompt carries the
   wellness/structure-function COMPLIANCE block (no treat/cure/diagnose; "people exploring X
   often look into Y"; observation not probability).
3. **AI proposes links.** `topic_copy.propose_links(topic, catalog)` → candidate related
   ingredient / product / topic slugs. **Every candidate is validated against the real
   catalog before storage** — any slug that does not resolve to an existing
   ingredient/product/topic is dropped (no hallucinated links). Validated set → `links_json`.
4. **SEO metadata.** AI proposes `title` + `meta_description`; server builds `jsonld`
   (Article + optional FAQ schema) at render time from `content_json`. Stored in `seo_json`.
5. Draft written via `upsert_section`, `state = draft`.

## 6. Compliance hard gate (the new structural piece)

- `topic_copy.compliance_scan(content_json, client)` runs the draft through the
  authority-injected-copy-guardian compliance logic: detect disease-claim / treat / cure /
  diagnose / prevent language and any phrase asserting a medical outcome. Returns
  `{passed, flags:[{phrase, reason}], scanned_at, model}`. Never raises; on error returns
  `passed=False` (fail-closed) so an errored scan cannot publish.
- **State machine:**
  - `draft` — freshly generated or edited; not yet scanned/approvable.
  - `gated` — scan ran and **failed**; flagged phrases visible in console; cannot be approved.
  - `approved` — scan passed **and** Glen approved; publicly served.
- **`topic_page.approve` refuses to publish unless `compliance_json.passed == true`.**
  If the latest scan failed (or is missing), approve returns an error listing the flags;
  Glen must `edit` (which resets to `draft` and clears the stale scan) then `regenerate`
  (re-draft + re-scan) until the scan passes. Approve then flips to `approved`.
- This makes "no disease claims reach the public web" a **structural guarantee**, enforced by
  the action layer, not a thing Glen has to remember to check.

## 7. Public rendering + CTA (SEO)

- `GET /learn/<slug>`:
  - Flag off (`TOPIC_PAGES_ENABLED` false) → 404.
  - Approved → fully server-rendered HTML: semantic `<h1>`/headings from sections,
    `<meta name="description">` from `seo_json`, JSON-LD `<script type="application/ld+json">`,
    a "Related" block rendering `links_json` as links to the ingredient pages / product
    sales pages / sibling topic pages, and a footer **optin CTA into `/begin`**.
  - Not approved (`draft`/`gated`/absent) → a lightweight "being prepared" page with the
    email-when-ready request form. **Never** emits draft content (same isolation rule as every
    other content module — public path checks `state == "approved"`).
- `GET /learn` — index listing approved pages grouped by kind, internally linked.
- Sitemap — approved slugs only, so Google crawls the graph but never a draft.

## 8. Console

Cloned from `/console/ingredient-pages`:

- `/console/topic-pages` — static HTML review queue.
- `GET /api/console/topic-pages` — list draft/gated/approved pages.
- `GET /api/console/topic-page/<slug>` — full page incl. compliance flags + live URL.
- Actions on the dispatch spine (`@action`, `risk_tier=LOW_WRITE`, `permission=(OWNER, OPS)`,
  audit-logged):
  - `topic_page.approve` — **compliance-gated** (see §6); on success sets `approved`, calls
    `notify_on_approve`.
  - `topic_page.edit` — update a section / name / SEO field; forces `state=draft`, clears
    stale `compliance_json`.
  - `topic_page.regenerate` — re-run AI draft + link proposal + compliance scan; stays
    `draft`/`gated` per scan result.
- `notify_on_approve(slug)` — emails everyone in `topic_page_requests` with unset `emailed_at`
  the link `{base_url}/learn/{slug}`, marks each emailed atomically. Mirrors
  `ingredient_pages.notify_on_approve`.
- Add `/console/topic-pages` to the hand-maintained `static/console-search-index.json` so it
  is discoverable in the console Pages search.

## 9. Testing

Unit tests (no `import app` — Pinecone-at-import blocks boot in sandbox; test the
`dashboard/*` helpers directly, verify `app.py` via `py_compile`):

- **Data layer:** upsert/get round-trip; request record + atomic `emailed_at` marking; only
  approved pages returned by the public getter.
- **Compliance gate (critical):** a draft containing a planted disease-claim phrase scans
  `passed=False` → `topic_page.approve` is refused and returns the flags; a clean draft scans
  `passed=True` → approve succeeds. An errored scan fails closed.
- **Link validation:** a proposed slug that is not in the catalog is dropped from `links_json`;
  a real one is kept.
- **Public isolation:** the `/learn/<slug>` render path returns no section text for a
  `draft`/`gated`/absent page.
- **State machine:** edit resets to `draft` and clears compliance; regenerate re-scans.

## 10. Rollout

- Ships **DARK behind `TOPIC_PAGES_ENABLED`** (default off) — same as every `/begin`
  sub-project. No public route is live while dark.
- Go-live: flip `TOPIC_PAGES_ENABLED` in Doppler `remedy-match/prd` after (a) a CSS/visual
  pass on the public template + console queue, and (b) one real seeded page drafted →
  compliance-passed → approved → rendered → verified indexable end-to-end.
- Fully reversible by flipping the flag back.

## 11. Open follow-on sub-projects (not this spec)

1. Chat-detects-gap offer (surface "create a page for that?" in `/begin` chat).
2. Article / video / course / book formats.
3. Paid generation + Stripe fulfillment ($29.99 book, $49.99 course).
4. Traffic-driven auto-creation from demand signals.
