# Ingredient Page

**Date:** 2026-06-20
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`, illtowell.com)
**Origin:** During Begin #4a, Glen asked that each ingredient on a Functional Formulation page link to that ingredient's own page - same layout as the formulation page (familiar/comfortable), with a heavier research section. Plus his 1-10 research and traditional-use rating scales and related-forms (superior/inferior) comparison.

---

## Problem

Formulation pages (`/begin/product/<slug>`, `static/begin-product.html`) render each ingredient as a plain, unlinked list item (name + dose). There is no ingredient page, no way to learn about a single ingredient, and the richest per-ingredient data (the Pinecone `ingredients` research studies) is never surfaced. We want a per-ingredient page that mirrors the formulation page's feel, leans heavily on research, carries Glen's 1-10 rating scales and a superior/inferior related-forms comparison, and is reachable by clicking any ingredient name on a formulation page.

## Goal

A `GET /begin/ingredient/<slug>` page that mirrors the formulation-page accordion; renders the ingredient's structured details, a heavy research section (AI lay-summary + the raw study citations), Glen's two 1-10 gauges (research + traditional use), a related-forms (superior/inferior) comparison, and the formulations that contain it; gated for PAID members; built on request (a paid member triggers an AI draft, sees "preparing"); Glen VERIFIES/approves in a console; on approval the requester is emailed and then views the approved page. Each ingredient name on a formulation page links to its ingredient page.

## Scope

Reuse the sales-page subsystem (generation + storage + console approve) and the Phase-5b notify-on-approve pattern in an ingredient variant, behind a paid gate; a new ingredient resolver; the new page + route; and the ingredient-name link on `begin-product.html`. Built in 3 increments.

**Out of scope:** building a new research corpus (we use the existing Pinecone `ingredients` namespace); the formulation-only sections (Watch / How it compares / Help shape this); any change to the sales-page (product) subsystem itself (we mirror it, not modify it).

---

## Confirmed decisions (Glen, 2026-06-20)

- **Same accordion layout as the formulation page** (familiar/comfortable), ingredient-appropriate sections.
- **Sections:** (1) What it is (AI), (2) Details (structured data), (3) The research (HEAVY: AI lay-summary + raw study citations), (4) In these formulations (links back). An Order link only when the ingredient is also a standalone product.
- **Two 1-10 gauges:** a **research rating** and a **traditional-use rating** (Glen's scales), rendered near the top with the green->gold gauge style used on the journey cards.
- **Traditional use detail:** a list of the **traditional medicine systems and formulas that use the ingredient** - the system (e.g. TCM, Ayurveda, Western herbalism), the named formula where applicable, what it is traditionally used for, and **in what form(s)** the ingredient appears (raw, decoction, tincture, powder, etc.). Pairs with the traditional-use gauge. AI-proposed, Glen-VERIFIED (classical formulas must be real; Glen's curation is the accuracy gate).
- **Related forms, rated superior / inferior** (Glen's clinical verdict), each linking to that form's ingredient page.
- **Content model = AI proposes, Glen verifies (the #4a pattern), NOT instant-draft.** AI proposes the narrative + the two scores + the traditional-use + related-forms lists; Glen edits/approves/VERIFIES in the console. His verification is the accuracy gate (real classical formulas, his 1-10 verdicts), so a member NEVER sees the unverified draft.
- **Gated for PAID membership (at least for now).** Viewing/triggering an ingredient page requires an active PAID membership (`_active_membership_for_email`, app.py:5688 - the paid-coaching tier, distinct from the free ToS `is_member`). Behind a config knob `INGREDIENT_PAGES_PAID_ONLY` (default on; loosens later without a code change). Non-paid -> an upgrade prompt.
- **Built on request:** a paid member viewing an ingredient with no APPROVED page -> the server records the request (their email), triggers the AI draft in the background, and shows "We are preparing your deep-dive on <name> and will email you when it is ready." They do not see the draft.
- **Email when ready:** on Glen's approval, every requester of that ingredient is emailed "Your <name> deep-dive is ready" + link (reuse the Phase-5b sales-page notify-on-approve pattern). An already-approved page shows instantly to paid members (no wait, no email).
- **Each ingredient name on a formulation page links to `/begin/ingredient/<slug>`** (new tab); the page enforces the paid gate.
- Compliance: structure/function language only, no disease claims, no em dashes, no emoji (reuse the sales-copy guardrails). Live; no feature flag (the paid gate + config knob govern access).

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

A second table records who requested a page so they can be emailed when it is approved (mirrors the Phase-5b `sales_page_viewers` pattern):
```
ingredient_page_requests ( ingredient_slug TEXT, email TEXT, requested_at TEXT, emailed_at TEXT,
                           PRIMARY KEY(ingredient_slug, email) )
```
Functions: `record_request(cx, slug, email)` (INSERT OR IGNORE - one row per requester per ingredient), `requesters_to_email(cx, slug)` (rows with `emailed_at` null), `mark_emailed(cx, slug, email)`, and `notify_on_approve(cx, slug, name, base_url, *, send, strip)` (email each un-emailed requester once "Your <name> deep-dive is ready" + `{base}/begin/ingredient/<slug>`, then `mark_emailed`; at-most-once, never raises into approve).

### Content generation - `dashboard/ingredient_copy.py` (new, mirrors `sales_copy.py`)
- `NARRATIVE_SECTIONS = ("what_it_is", "research")` (the two AI text sections).
- `build_section_prompt(section, ingredient)` -> `(system, user)`: grounded in the ingredient's `fmp` details + its `research_studies` list. `what_it_is` = one warm structure/function paragraph; `research` = a heavy lay-summary that synthesizes the studies (it cites the mechanisms; the raw study list is rendered separately by the page). Same compliance system prompt as `sales_copy`.
- `propose_curation(ingredient) -> {research_score, traditional_score, related_forms, traditional_use}`: a synchronous AI call (haiku) that proposes the two 1-10 scores (research = strength/volume of the studies found; traditional = historical/traditional-use evidence), a related-forms list (other forms of the same nutrient with a superior/inferior/comparable verdict and a one-line note, each form slugged via `ingredients.slugify`), and a traditional-use list (`[{system, formula, uses, forms}]` - the traditional medicine systems/formulas using the ingredient and the forms used). These are PROPOSALS for Glen to VERIFY/edit/approve (the prompt instructs the model to omit anything it is not confident is a real classical formula rather than invent one). Returns safe defaults (scores null, lists []) on failure.

### The paid gate (shared)
A helper `_ingredient_paid_ok(email)` -> True when `INGREDIENT_PAGES_PAID_ONLY` is off OR `_active_membership_for_email(email)` is active. The visitor email is the authenticated/portal/member email (the same resolution other paid surfaces use). The route, page-data, and gen endpoint all enforce it; a failed check is fail-safe (gate, never bypass).

### The page-data endpoint (the state machine)
`GET /begin/ingredient-page-data/<slug>` (mirror `begin_product_page_data`). Resolve the slug (404 page if unknown). Then:
- **Not paid** (`_ingredient_paid_ok` false): return `{slug, name, state:"locked"}` - the page shows the upgrade prompt; nothing else generated.
- **Paid, an APPROVED row exists:** return the full payload `{slug, name, state:"approved", sections[], research_score, traditional_score, traditional_use, related_forms, formulations[], standalone_product_slug|null}` - shown instantly.
- **Paid, no approved row:** `record_request(cx, slug, email)`; if no row at all, kick off the background draft build (best-effort, in-process thread, same non-blocking pattern as the funnel onboard) - `propose_curation` + queue the two narrative sections - and return `{slug, name, state:"preparing"}`. The page shows "We are preparing your deep-dive on <name> and will email you when it is ready." The member never receives the draft content.

The DRAFT content (narrative, scores, traditional-use, forms) is only ever read by the console; the public page-data emits it solely when `state=="approved"`.

### The gen endpoint
SSE `GET /begin/ingredient-page-gen/<slug>/<section>` (mirror `/begin/product-page-gen`): used by the BACKGROUND build (and the console preview), NOT the public page. Cache-first; else stream haiku from `ingredient_copy.build_section_prompt`, write to `ingredient_pages` on completion. Also paid-gated (defense in depth) so it cannot be hit by a non-member to incur AI cost.

### The page - `static/begin-ingredient.html` (new, models `begin-product.html`)
Reads `/begin/ingredient-page-data/<slug>` and renders by `state`:
- `state:"locked"` -> an upgrade prompt ("Ingredient deep-dives are a member benefit") + a link to the membership/upgrade page; nothing personal/generated.
- `state:"preparing"` -> "We are preparing your deep-dive on <name> and will email you when it is ready."
- `state:"approved"` -> the full accordion mirroring the product page. Top: the ingredient name + the **two gauges** (research N/10, traditional N/10), green->gold fill (`fill = score/10`). Sections: (1) **What it is** (AI narrative; default open); (2) **Details** (the `fmp` fields); (3) **The research** - AI lay-summary, then the **study list** (each `study_title` links to its `url`, new tab, with publication + year), HEAVY; (4) **Traditional use** - the `traditional_use` list (system + formula + uses + form(s)); (5) **Related forms** (each `{name, verdict, note}` with a superior/inferior badge; name links to `/begin/ingredient/<slug>`, new tab); (6) **In these formulations** (links to `/begin/product/<slug>`).
All dynamic text via `textContent`; links set `.href` to server-built `/begin/...` or the study `url` with `target=_blank rel=noopener`. No draft banner is needed on the public page (members only see approved).

### Console review - `/console/ingredient-pages` (mirror `/console/sales-pages`)
A new `dashboard/ingredient_page_actions.py` on the dispatch spine (RBAC OWNER/OPS), actions:
- `ingredient_page.edit` - save edited section text AND/OR the two scores AND/OR the traditional-use list AND/OR the related-forms list (stays draft).
- `ingredient_page.approve` - set state `approved`, then `notify_on_approve(...)` emails every un-emailed requester "Your <name> deep-dive is ready" + the link (best-effort, at-most-once; the email never fails the approve). An already-approved page is now instantly viewable by any paid member.
- `ingredient_page.regenerate` - re-run the narrative + `propose_curation` for review (stays draft).
A new `static/console-ingredient-pages.html` (model `console-sales-pages.html`): edit the two narrative sections, the two 1-10 scores, the traditional-use list, and the related-forms list; approve. A "Ingredient Pages" console nav sub-tab.

### The formulation-page link (the original ask)
`static/begin-product.html` `renderIngredientsBody` (lines ~341-370): render each ingredient's **name as a link** to `/begin/ingredient/<slugify(name)>` (`target=_blank rel=noopener`), keeping the dose text. Compute the slug client-side with the same rule, OR have `begin_product_page_data` include a `slug` per ingredient (preferred - one source of truth). No other product-page change.

### Reuse / untouched
- Pinecone `ingredients` namespace + `product_content._research_sources` logic; `dashboard/ingredient_content.py` (label/RDA); `data/fmp-ingredient-content.json`; `data/products.json`; the dispatch spine + console-auth + draft-banner pattern; haiku `claude-haiku-4-5-20251001`.
- Untouched: the sales-page (product) subsystem (`sales_pages.py`/`sales_copy.py`/`sales_pages_actions.py` and the product page), the journey/funnel, pricing/Stripe.

---

## Data flow
1. A visitor clicks an ingredient name on a formulation page -> `/begin/ingredient/<slug>`.
2. The page calls `/begin/ingredient-page-data/<slug>`. **Not a paid member** -> `state:"locked"` -> upgrade prompt. **Paid + approved page exists** -> the full page renders instantly. **Paid + no approved page** -> `record_request(email)`, the background draft build kicks off (`propose_curation` + the two narrative sections), and `state:"preparing"` -> "we are preparing it, we will email you."
3. Glen opens `/console/ingredient-pages`, VERIFIES/edits the scores, traditional-use formulas, related-forms verdicts, and narrative, then approves.
4. On approve, `notify_on_approve` emails every requester "Your <name> deep-dive is ready" + the link (at-most-once).
5. The member clicks the link -> the approved page renders (paid-gated).

## Error handling
- Unknown slug -> a friendly "ingredient not found" page (no 500).
- Paid gate fail-safe: a `_active_membership_for_email` error -> treated as not-paid (locked), never bypassed.
- Pinecone unavailable -> the research study list is empty and the AI research summary is generic; the build still produces a draft.
- `propose_curation` / narrative gen failures (background) -> safe defaults (null scores, empty lists); the page stays in `preparing` and Glen sees a thin draft to fill in; never 500s the request.
- The two scores are clamped to 1-10 on store; a missing score hides its gauge.
- `notify_on_approve` is best-effort and at-most-once (`emailed_at`); it never fails the approve and never re-emails a requester.
- A member with no email (cannot happen for a paid member) -> no request recorded, no email; the page still shows preparing/locked appropriately.
- Compliance guardrails identical to sales copy (the system prompt forbids disease claims / em dashes / invented classical formulas).

## Testing
- **Resolver:** `slugify` round-trips; `resolve(slug)` returns the FMP record for a known ingredient and None for a bogus slug; `formulations_with(name)` finds the products containing it; `research_studies` returns the study shape (mock Pinecone) and `[]` when unavailable.
- **Storage:** `ingredient_pages` init + `upsert_section`/`get_section`, `set_scores` (clamped 1-10), `set_related_forms`, `set_traditional_use`, `set_state` draft->approved, per-cursor Row factory; `record_request` once-per-(slug,email), `requesters_to_email` excludes emailed, `mark_emailed`, `notify_on_approve` emails each once.
- **Paid gate + state machine:** `/begin/ingredient-page-data/<slug>` for a non-paid email -> `state:"locked"`, NO generation, NO draft content; for a paid email with no approved row -> `state:"preparing"` + a request recorded; for a paid email with an approved row -> `state:"approved"` + the full payload (scores/traditional_use/related_forms/formulations). The DRAFT content never appears in the page-data unless approved. Use a config monkeypatch to toggle `INGREDIENT_PAGES_PAID_ONLY`.
- **Console actions:** `ingredient_page.edit` updates section/scores/traditional-use/forms and stays draft; `approve` -> approved AND calls the injected send fn once per requester; RBAC OWNER/OPS.
- **Serve + link:** `/begin/ingredient/<slug>` 200 with the page scaffold (the state-driven render, the gauge + section ids); `begin-product.html` renders ingredient names as `/begin/ingredient/...` links (assert the link + `_blank`).
- Front-end (gauge fills, accordion, study links, related-form badges, the locked/preparing/approved states) = manual visual pass. deploy-chat test isolation (tmp `LOG_DB`; mock the Anthropic client + Pinecone; mock `_active_membership_for_email`; mock the dispatch actor + the send fn). No emoji; no em dashes.

## Build order (increments)
1. **Increment 1 - resolver + paid-gated page shell + links:** `dashboard/ingredients.py`; the paid gate (`_ingredient_paid_ok` + `INGREDIENT_PAGES_PAID_ONLY`); `/begin/ingredient/<slug>` + `/begin/ingredient-page-data` with the locked/preparing/approved state machine (approved renders Details + study list + formulation links from data); `static/begin-ingredient.html`; the ingredient-name links on `begin-product.html`. (No AI build yet; an approved page can be hand-seeded for the test.) Deliverable: a paid-gated, navigable ingredient page.
2. **Increment 2 - AI draft build + request/notify:** `ingredient_copy.py`, `ingredient_pages.py` store (+ requests table), the gen SSE endpoint, `propose_curation`, the background build on a paid request, `record_request`, and the gauges/traditional-use/related-forms rendering on the approved page.
3. **Increment 3 - console review + email-when-ready:** `ingredient_page_actions.py` + `/console/ingredient-pages` + nav (edit narrative/scores/traditional-use/forms, approve); approve fires `notify_on_approve`.

## Notes
- **Live page, no feature flag** (the paid gate + `INGREDIENT_PAGES_PAID_ONLY` govern access). `main` auto-deploys. In practice the page is only useful once Increments 2-3 land and Glen approves a draft; until then a paid member sees "preparing."
- **Built on request, paid-gated (at least for now):** generation only fires for a paid member's request; the gate can be loosened later via the config knob. The member never sees an unverified draft - only the approved page, delivered by the "ready" email.
- All copy + the proposed scores/traditional-use/forms are provisional until Glen approves - his clinical judgment is the final word on the 1-10 scales, the traditional-formula accuracy, and the superior/inferior verdicts.
- Reuses the proven sales-page draft->approve machinery + the Phase-5b notify-on-approve pattern, so the ingredient subsystem is a sibling, not a rebuild.
