# Ingredient Page

**Date:** 2026-06-20
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`, illtowell.com)
**Origin:** During Begin #4a, Glen asked that each ingredient on a Functional Formulation page link to that ingredient's own page - same layout as the formulation page (familiar/comfortable), with a heavier research section. Plus his 1-10 research and traditional-use rating scales and related-forms (superior/inferior) comparison.

---

## Problem

Formulation pages (`/begin/product/<slug>`, `static/begin-product.html`) render each ingredient as a plain, unlinked list item (name + dose). There is no ingredient page, no way to learn about a single ingredient, and the richest per-ingredient data (the Pinecone `ingredients` research studies) is never surfaced. We want a per-ingredient page that mirrors the formulation page's feel, leans heavily on research, carries Glen's 1-10 rating scales and a superior/inferior related-forms comparison, and is reachable by clicking any ingredient name on a formulation page.

## Goal

A `GET /begin/ingredient/<slug>` page that mirrors the formulation-page accordion; renders the ingredient's structured details, a heavy research section (AI lay-summary + the raw study citations), Glen's two 1-10 gauges (research + traditional use), a related-forms (superior/inferior) comparison, and the formulations that contain it; with AI-proposed narrative + ratings + traditional-use + related-forms that Glen approves/edits in a console (draft banner until approved). Each ingredient name on a formulation page links to its ingredient page.

## Scope

Reuse the sales-page subsystem (generation + storage + console approve + draft banner) in an ingredient variant; a new ingredient resolver; the new page + route; and the ingredient-name link on `begin-product.html`. Built in 3 increments.

**Out of scope:** building a new research corpus (we use the existing Pinecone `ingredients` namespace); the formulation-only sections (Watch / How it compares / Help shape this); any change to the sales-page (product) subsystem itself (we mirror it, not modify it).

---

## Confirmed decisions (Glen, 2026-06-20)

- **Same accordion layout as the formulation page** (familiar/comfortable), ingredient-appropriate sections.
- **Sections:** (1) What it is (AI), (2) Details (structured data), (3) The research (HEAVY: AI lay-summary + raw study citations), (4) In these formulations (links back). An Order link only when the ingredient is also a standalone product.
- **Two 1-10 gauges:** a **research rating** and a **traditional-use rating** (Glen's scales), rendered near the top with the green->gold gauge style used on the journey cards.
- **Traditional use detail:** a list of the **traditional medicine systems and formulas that use the ingredient** - the system (e.g. TCM, Ayurveda, Western herbalism), the named formula where applicable, what it is traditionally used for, and **in what form(s)** the ingredient appears (raw, decoction, tincture, powder, etc.). Pairs with the traditional-use gauge. AI-proposed, Glen-VERIFIED (classical formulas must be real; Glen's curation is the accuracy gate).
- **Related forms, rated superior / inferior** (Glen's clinical verdict), each linking to that form's ingredient page.
- **Content model = the sales-page model:** AI proposes the narrative + the two scores + the traditional-use + related-forms lists; Glen edits/approves/verifies in the console; draft banner until approved; generate-on-view (lazy).
- **Each ingredient name on a formulation page links to `/begin/ingredient/<slug>`** (new tab).
- Compliance: structure/function language only, no disease claims, no em dashes, no emoji (reuse the sales-copy guardrails). Live, no feature flag.

---

## Architecture

### Slug + resolver - `dashboard/ingredients.py` (new)
- `slugify(name)` - the same rule as `_slugify_product` (`re.sub(r"[^a-z0-9]+","-", name.lower()).strip("-")[:40]`).
- `resolve(slug) -> dict | None`: maps a slug back to a canonical ingredient name and returns `{slug, name, fmp}` where `fmp` is the `dashboard/ingredient_content.get(name)` record (scientific, label_form, percent, active, rda_content, rda_mg) or `{}`. Name<->slug is resolved by slugifying the `fmp-ingredient-content.json` keys (built once into a `{slug: name}` index) plus the names that appear in `data/products.json` ingredient lists. Returns None only when the slug matches no known ingredient.
- `formulations_with(name) -> [{slug, name}]`: scans `data/products.json` for products whose `ingredients[]` contains this ingredient (normalized-name match), returning the linking list for the "In these formulations" section.
- `research_studies(name, k=12) -> [{study_title, publication, year, url, text}]`: thin wrapper over the existing Pinecone `ingredients`-namespace query (`dashboard/product_content._research_sources` logic) but PER-INGREDIENT and with a higher k (heavy research). Returns the raw, citable study list. Degrades to `[]` if Pinecone is unavailable.

### Storage - `dashboard/ingredient_pages.py` (new, mirrors `sales_pages.py`)
Table `ingredient_pages` in `chat_log.db`:
```
ingredient_slug TEXT PRIMARY KEY,
name TEXT,
state TEXT,            -- 'draft' | 'approved'
content_json TEXT,     -- {"what_it_is": str, "research": str}  (the AI narrative sections)
research_score INTEGER,        -- Glen's 1-10 (AI-proposed, Glen-editable)
traditional_score INTEGER,     -- Glen's 1-10
traditional_use_json TEXT,     -- [{system, formula, uses, forms}]  systems/formulas using the ingredient
related_forms_json TEXT,       -- [{name, slug, verdict:'superior'|'inferior'|'comparable', note}]
model TEXT, generated_at TEXT, approved_at TEXT, approved_by TEXT,
created_at TEXT, updated_at TEXT
```
Functions mirror `sales_pages.py`: `init_table`, `get_section`/`upsert_section`, `get_page`, `set_state`, plus `set_scores(cx, slug, research, traditional)`, `set_related_forms(cx, slug, forms)`, and `set_traditional_use(cx, slug, entries)`. Getters use a per-cursor Row factory.

### Content generation - `dashboard/ingredient_copy.py` (new, mirrors `sales_copy.py`)
- `NARRATIVE_SECTIONS = ("what_it_is", "research")` (the two AI text sections).
- `build_section_prompt(section, ingredient)` -> `(system, user)`: grounded in the ingredient's `fmp` details + its `research_studies` list. `what_it_is` = one warm structure/function paragraph; `research` = a heavy lay-summary that synthesizes the studies (it cites the mechanisms; the raw study list is rendered separately by the page). Same compliance system prompt as `sales_copy`.
- `propose_curation(ingredient) -> {research_score, traditional_score, related_forms, traditional_use}`: a synchronous AI call (haiku) that proposes the two 1-10 scores (research = strength/volume of the studies found; traditional = historical/traditional-use evidence), a related-forms list (other forms of the same nutrient with a superior/inferior/comparable verdict and a one-line note, each form slugged via `ingredients.slugify`), and a traditional-use list (`[{system, formula, uses, forms}]` - the traditional medicine systems/formulas using the ingredient and the forms used). These are PROPOSALS for Glen to VERIFY/edit/approve (the prompt instructs the model to omit anything it is not confident is a real classical formula rather than invent one). Returns safe defaults (scores null, lists []) on failure.

### The gen endpoint + page-data
- SSE: `GET /begin/ingredient-page-gen/<slug>/<section>` (mirror `/begin/product-page-gen`): cache-first; else stream haiku from `ingredient_copy.build_section_prompt`, write to `ingredient_pages` on completion. Gated by the same `SALES_PAGES_AI_COPY`-style enablement (reuse the existing flag; no new flag).
- `GET /begin/ingredient-page-data/<slug>` (mirror `begin_product_page_data`): returns `{slug, name, sections[], research_score, traditional_score, traditional_use, related_forms, formulations[], standalone_product_slug|null, ai_state}`. On first request for a slug with no row, it triggers (best-effort, in-process) `propose_curation` to seed the scores + traditional-use + forms as a draft, and marks the narrative sections `ai:"pending"` so the page streams them. `ai_state` drives the draft banner (suppressed when `state=approved`).

### The page - `static/begin-ingredient.html` (new, models `begin-product.html`)
Accordion mirroring the product page. Top of page: the ingredient name + the **two gauges** (research N/10, traditional N/10) rendered with the green->gold fill style (`fill = score/10`). Sections:
1. **What it is** (AI narrative; default open).
2. **Details** (the `fmp` fields: scientific name, label form, % RDA / mg, active).
3. **The research** (the AI lay-summary, then the **study list**: each `study_title` is a link to its `url` in a new tab, with publication + year). HEAVY.
4. **Traditional use** (the traditional-use gauge context, then the `traditional_use` list: per entry the system + formula + uses + the form(s) the ingredient appears in).
5. **Related forms** (each `{name, verdict, note}` with a superior/inferior badge; the name links to `/begin/ingredient/<slug>` in a new tab).
6. **In these formulations** (each formulation links to `/begin/product/<slug>`).
Plus a draft banner while `ai_state != approved`. All dynamic text via `textContent`; links set `.href` to server-built `/begin/...` or the study `url` with `target=_blank rel=noopener`.

### Console review - `/console/ingredient-pages` (mirror `/console/sales-pages`)
A new `dashboard/ingredient_page_actions.py` on the dispatch spine (RBAC OWNER/OPS), actions:
- `ingredient_page.edit` - save edited section text AND/OR the two scores AND/OR the traditional-use list AND/OR the related-forms list (stays draft).
- `ingredient_page.approve` - set state `approved` (drops the draft banner).
- `ingredient_page.regenerate` - re-run the narrative + `propose_curation` for review (stays draft).
A new `static/console-ingredient-pages.html` (model `console-sales-pages.html`): edit the two narrative sections, the two 1-10 scores, the traditional-use list, and the related-forms list; approve. A "Ingredient Pages" console nav sub-tab.

### The formulation-page link (the original ask)
`static/begin-product.html` `renderIngredientsBody` (lines ~341-370): render each ingredient's **name as a link** to `/begin/ingredient/<slugify(name)>` (`target=_blank rel=noopener`), keeping the dose text. Compute the slug client-side with the same rule, OR have `begin_product_page_data` include a `slug` per ingredient (preferred - one source of truth). No other product-page change.

### Reuse / untouched
- Pinecone `ingredients` namespace + `product_content._research_sources` logic; `dashboard/ingredient_content.py` (label/RDA); `data/fmp-ingredient-content.json`; `data/products.json`; the dispatch spine + console-auth + draft-banner pattern; haiku `claude-haiku-4-5-20251001`.
- Untouched: the sales-page (product) subsystem (`sales_pages.py`/`sales_copy.py`/`sales_pages_actions.py` and the product page), the journey/funnel, pricing/Stripe.

---

## Data flow
1. Visitor clicks an ingredient name on a formulation page -> `/begin/ingredient/<slug>`.
2. The page calls `/begin/ingredient-page-data/<slug>` -> resolver returns details + formulations + study list; if no stored row, `propose_curation` seeds the two scores + traditional-use + related-forms (draft) and the narrative sections stream via the gen endpoint.
3. The page renders: two gauges, What-it-is, Details, The research (+ study links), Related forms (linked), In these formulations (linked), with a draft banner.
4. Glen opens `/console/ingredient-pages`, edits the scores / forms / narrative as his clinical judgment dictates, approves -> banner drops.

## Error handling
- Unknown slug -> a friendly "ingredient not found" page (no 500); a slug that resolves to a name with no FMP record still renders (name + research + formulations; details section shows what is available).
- Pinecone unavailable -> the research study list is empty and the AI research summary is skipped or generic; the page still renders.
- `propose_curation` / narrative gen failures -> safe defaults (null scores hide the gauges, empty forms hide that section); never 500.
- The two scores are clamped to 1-10 on store; a missing score hides its gauge rather than showing 0.
- Compliance guardrails identical to sales copy (the system prompt forbids disease claims / em dashes).

## Testing
- **Resolver:** `slugify` round-trips; `resolve(slug)` returns the FMP record for a known ingredient and None for a bogus slug; `formulations_with(name)` finds the products containing it; `research_studies` returns the study shape (mock Pinecone) and `[]` when unavailable.
- **Storage:** `ingredient_pages` init + `upsert_section`/`get_section`, `set_scores` (clamped 1-10), `set_related_forms`, `set_traditional_use`, `set_state` draft->approved, per-cursor Row factory (no connection leak).
- **Gen + page-data:** `/begin/ingredient-page-data/<slug>` returns the expected keys incl. `research_score`/`traditional_score`/`traditional_use`/`related_forms`/`formulations`; `ai_state` reflects state; the gen SSE caches and replays.
- **Console actions:** `ingredient_page.edit` updates section/scores/forms and stays draft; `approve` -> approved; RBAC OWNER/OPS.
- **Serve:** `/begin/ingredient/<slug>` 200 with the page scaffold (the two gauges, the section ids, the draft banner element); `begin-product.html` renders ingredient names as `/begin/ingredient/...` links (assert the link + `_blank`).
- Front-end (gauge fills, accordion, study links, related-form badges, the draft banner) = manual visual pass. deploy-chat test isolation (tmp `LOG_DB`; mock the Anthropic client + Pinecone; mock the dispatch actor). No emoji; no em dashes.

## Build order (increments)
1. **Increment 1 - resolver + data page + links:** `dashboard/ingredients.py`; `/begin/ingredient/<slug>` + `/begin/ingredient-page-data` serving Details + the study list + the formulation links; `static/begin-ingredient.html` (no AI yet, scores hidden); the ingredient-name links on `begin-product.html`. Deliverable: a working, navigable ingredient page from real data.
2. **Increment 2 - AI narrative + ratings + traditional use + related forms:** `ingredient_copy.py`, `ingredient_pages.py` store, the gen SSE endpoint, `propose_curation`, the two gauges + traditional-use + related-forms rendering, the draft banner.
3. **Increment 3 - console review:** `ingredient_page_actions.py` + `/console/ingredient-pages` + nav (edit narrative/scores/forms, approve).

## Notes
- **Live page, no flag.** `main` auto-deploys. Generate-on-view means a freshly-viewed ingredient shows a draft (AI proposals) until Glen approves; the draft banner communicates this.
- All copy + the proposed scores/forms are provisional until Glen approves - his clinical judgment is the final word on the 1-10 scales and the superior/inferior verdicts.
- The two 1-10 scales and the related-forms verdicts are Glen's IP; the AI seeds them to save blank-slate effort, but the console is where they become authoritative.
- Reuses the proven sales-page draft->approve machinery, so the ingredient subsystem is a sibling, not a rebuild.
