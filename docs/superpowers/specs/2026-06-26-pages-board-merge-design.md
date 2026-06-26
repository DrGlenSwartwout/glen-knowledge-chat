# Pages Board Merge (Sub-project B2) — Design Spec

**Date:** 2026-06-26
**Status:** Design approved (de-dup into a shared core; board named "Pages"; Topic Suggestions as a 4th tab). Ready for implementation plan.

Second increment of sub-project **B** (console consolidation). B2 merges the four AI page-editor
boards into one **Pages** board. B1 (Money) shipped. B3 (Approvals queue) and B4 (Settings cleanup)
are separate cycles, out of scope here.

## Goal

Collapse the four near-identical AI content editors — **Sales Pages**, **Ingredient Pages**,
**Topic Pages**, **Topic Suggestions** — into one **Pages** board with a type switcher, and in doing
so **de-duplicate** the ~800 lines of triplicated editor code into one shared, config-driven core.
The three editors share the identical workflow (list → select → section blocks → Regenerate & review
→ Approve); only their endpoints, approve label, and a few type-specific section panels differ.
Suggestions is a sibling triage queue (Build / Dismiss) that hands its output to the Topic editor.
**Pure consolidation:** every existing view/action preserved exactly; backends untouched.

## Current state (reuse-first)

Four standalone pages, each loading `op-nav.js`, each with the same `#gate`/`unlock()`/`key()`
console-key pattern and `X-Console-Key` auth:

| Page | Route | List endpoint | Detail endpoint | Approve action / label |
|---|---|---|---|---|
| Sales | `/console/sales-pages` | `/api/console/sales-pages` (`.pages`) | `/api/console/sales-page/{slug}` | `sales_pages.approve` / "Approve" |
| Ingredient | `/console/ingredient-pages` | `/api/console/ingredient-pages` (`.pages`) | `/api/console/ingredient-page/{slug}` | `ingredient_page.approve` / "Approve and notify" |
| Topic | `/console/topic-pages` | `/api/console/topic-pages` (`.pages`) | `/api/console/topic-page/{slug}` | `topic_page.approve` / "Approve and publish" |
| Suggestions | `/console/topic-suggestions` | `/api/console/topic-suggestions` (`.suggestions`) | `/api/console/topic-page/{slug}` | — (Build/Dismiss, see below) |

The functions `key/unlock/api/act/setStatePill/boot/loadList/select/saveSection` and the section-
rendering loop (`#ed-sections` → per-section `<textarea id="sec-{id}">` + Save button) are
byte-identical across the editors modulo endpoint strings. Edit actions: `sales_pages.edit` /
`ingredient_page.edit` / `topic_page.edit`. Regenerate actions: `<type>.regenerate`.

**Type-specific divergences (all cleanly separable):**
- **Ingredient** renders three extra panels inside `select()` — Scores (`#ed-scores`: research_score
  + traditional_score number inputs), Traditional Use (`#ed-traditional-use`: a JSON textarea), Related
  Forms (`#ed-related-forms`: a JSON textarea). Each owns its own container, none touch `#ed-sections`,
  and all three save via `ingredient_page.edit` with different payload fields.
- **Topic** (and Suggestions) render a compliance panel (`#ed-compliance`, `renderCompliance(...)`)
  from `match.compliance`, after regen, and on an `approve` `compliance_failed` error.
- **Sales** uses innerHTML template literals (needs an `esc()` helper) where the others use DOM API,
  reads its live URL from the API (`data.live_url || '/begin/product/'+slug`), and has no extra panels.
- **Suggestions** replaces regen/approve with **Build** (`topic_page.regenerate` then reload the list,
  status "Built. Review in Topic Pages to approve.") and **Dismiss** (`topic_page.dismiss`, then
  clear + reload). Its list rows show `name (demand)` with a `kind` pill.

## Design

### New page: `static/console-pages.html` at `GET /console/pages`

**One editor, type-switched** — not four panels. Because the core is shared, the page has a single
editor UI (one `#list`, one `#editor` with `#ed-title`/`#ed-live`/`#ed-state`/`#ed-sections`, plus the
type-specific containers `#ed-scores`/`#ed-traditional-use`/`#ed-related-forms`/`#ed-compliance` and
the action buttons). Switching a tab sets the active type config and reloads. This sidesteps the
DOM-id collisions a 4-panel merge would create — there is only one set of editor ids.

**Tab bar:** Sales · Ingredient · Topic · Suggestions. The active tab is reflected in the URL hash
(`#sales`/`#ingredient`/`#topic`/`#suggestions`), deep-linkable; default = Sales.

**Shared core (one module, parameterized by `TYPES[active]`):** `key`, `unlock`, the `?key=` capture
IIFE, `api`, `act`, `setStatePill`, `boot`, `loadList`, `select`, the section-rendering loop,
`saveSection`, `regen`, `approve`. Plus `switchType(t)`: set active config, clear the editor, swap the
action buttons (Regenerate+Approve for editors; Build+Dismiss for suggestions), set the hash, reload
the list.

**Per-type config** (`TYPES`):
```
sales:      { key:'sales',      list:'/api/console/sales-pages',      respKey:'pages',
              detail:s=>'/api/console/sales-page/'+enc(s), live:(s,d)=>d.live_url||'/begin/product/'+s,
              edit:'sales_pages.edit', regen:'sales_pages.regenerate',
              approve:'sales_pages.approve', approveLabel:'Approve',
              approveMsg:'Approved. The live page no longer shows the draft banner.',
              labels:{intro,description,research} }
ingredient: { …list/detail/edit/regen/approve = ingredient_page.*; approveLabel:'Approve and notify';
              approveMsg:'Approved. Requesters have been notified.';
              live:(s)=>'/begin/ingredient/'+enc(s); labels:{what_it_is,research} }
topic:      { …topic_page.*; approveLabel:'Approve and publish';
              approveMsg:'Approved. Topic page is now public.';
              live:(s)=>'/learn/'+enc(s); labels:{overview,symptoms,causes,solutions,lifestyle,when_to_seek} }
suggestions:{ list:'/api/console/topic-suggestions', respKey:'suggestions',
              detail:s=>'/api/console/topic-page/'+enc(s); live:(s)=>'/learn/'+enc(s);
              edit:'topic_page.edit'; labels:(topic's); mode:'queue' /* Build/Dismiss */ }
```

**Per-type hooks:**
- `renderItem(p)` — list row. Default: name + state pill. Suggestions override: `name (demand)` + `kind` pill.
- `afterSections(match)` — ingredient: render the Scores/Traditional-Use/Related-Forms panels; topic +
  suggestions: `renderCompliance(match.compliance||{})`; sales: nothing. Always hide the containers a
  type doesn't use.
- `afterRegen(result)` — topic + suggestions: `renderCompliance((result||{}).compliance||{})`; else nothing.
- `onApproveFail(r)` — topic: if `r.json.error==='compliance_failed'` → `renderCompliance({passed:false,
  flags:r.json.flags})` + the gated message (return handled); else the generic approve-fail path.
- Suggestions' **Build** = `act('topic_page.regenerate',{slug})` then reload the list; **Dismiss** =
  `act('topic_page.dismiss',{slug})` then clear + reload. (Ingredient's three save functions, scoped to
  their containers, stay ingredient-only.)

The `esc()` helper is dropped — the section loop is the DOM-API version (the ingredient/topic original),
so no innerHTML escaping is needed.

### Backend

**Unchanged.** No `/api/console/*-page[s]`, `/api/console/topic-suggestions`, or `/api/action/*` edits.
A new route serves the page: `@app.route("/console/pages")` → `send_from_directory(STATIC,
"console-pages.html")`, gated/cached like its siblings.

### Nav (`static/op-nav.js`)

In `bosMods`, remove the four entries `sales` (Sales Pages), `ingredients` (Ingredient Pages),
`topic-pages`, `topic-suggestions` and add one `{id:"pages", label:"Pages", href:"/console/pages"+qs}`.
In `NAV_PROFILES`, replace `"sales","ingredients"` (and `"topic-pages"`,`"topic-suggestions"` wherever
they appear) with `"pages"` in `glen.bos` (Pages is a Glen-primary board); `rae.bos` does not list
these, so no change there. **Note:** keep the unrelated `ingredients-ops` entry (`/admin/ingredients`,
the production suite) — only the *page-editor* `ingredients` id is removed.

### Old routes redirect

`/console/sales-pages` → 302 `/console/pages#sales`; `/console/ingredient-pages` → `#ingredient`;
`/console/topic-pages` → `#topic`; `/console/topic-suggestions` → `#suggestions`. Replace the four
`send_from_directory` route bodies with redirects. The four old HTML files (`console-sales-pages.html`,
`console-ingredient-pages.html`, `console-topic-pages.html`, `console-topic-suggestions.html`) are
**deleted** once their logic is folded into the shared core.

## Out of scope

- Any new action or change to editor/approve/regen/build/dismiss behavior or the section model.
- Changes to `/api/console/*` or `/api/action/*`.
- B3/B4. The Settings sub-row, Money board, etc. are untouched.

## Risks / decisions (from the structural map)

- **sales `live`** reads `data.live_url`, so the `live(slug, data)` hook takes the detail object; other
  types ignore it. Keep the two-arg signature.
- **topic `onApproveFail`** is the one non-additive coupling — handled by the hook (a closure over
  `renderCompliance`), not by sharing approve's error branch across types.
- **suggestions Build reloads the list** (the suggestion leaves the queue when built) whereas editor
  regen does not — `mode:'queue'` drives Build/Dismiss + the reload.

## Dependencies

- Sub-project A's `op-nav.js` `NAV_PROFILES` map and `bosMods` (already merged) — B2 edits the
  `sales`/`ingredients`/`topic-pages`/`topic-suggestions` entries.
- The existing `/api/console/*-page[s]`, `/api/console/topic-suggestions`, `/api/action/*` endpoints.

## Testing (run via [reference_deploy_chat_local_tests])

- **Routes:** `/console/pages` 200; the four old routes 302 to the right hashes.
- **Render-verify (headless, per the render-verify lesson) — per type, zero JS console/page errors:**
  - **Sales:** tab active by default; list loads; selecting an item shows its 3 sections + Regenerate &
    Approve; no compliance/scores panels shown.
  - **Ingredient:** switching shows the ingredient list; selecting shows its 2 sections **plus** the
    Scores / Traditional-Use / Related-Forms panels; Approve label = "Approve and notify".
  - **Topic:** sections + the **compliance** panel; Approve label = "Approve and publish".
  - **Suggestions:** list rows show `name (demand)`; the action buttons are **Build / Dismiss** (not
    Regenerate/Approve); selecting shows sections + compliance.
  - Deep-link `/console/pages#ingredient` opens the Ingredient tab; type switch reflects in the hash.
  - (A fresh local DB may return empty lists / network errors from the AI endpoints — the gate is
    structure + correct per-type panels/buttons/labels + zero JS errors, not row counts.)
- `node --check static/op-nav.js`; BOS sub-row shows one **Pages** entry; `sales`/`ingredients`(page)/
  `topic-pages`/`topic-suggestions` absent; `ingredients-ops` still present.
- `grep -rn "console-sales-pages\.html\|console-ingredient-pages\.html\|console-topic-pages\.html\|console-topic-suggestions\.html" app.py static/` returns nothing after deletion.

## Rollout

Additive + redirect + delete: one new page + route, four route bodies → redirects, four old HTML files
deleted, a small `op-nav.js` edit. No backend/data change, no feature flag. Console-key gated (and
Rae's OWNER token via sub-project A).
