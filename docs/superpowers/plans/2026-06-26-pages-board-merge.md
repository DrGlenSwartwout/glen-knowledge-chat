# Pages Board Merge (Sub-project B2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the four AI page-editors (Sales/Ingredient/Topic pages + Topic Suggestions) into one config-driven **Pages** board at `/console/pages`, de-duplicating the triplicated editor code.

**Architecture:** One `static/console-pages.html` with a single type-switched editor (one set of editor DOM ids, re-pointed by a `TYPES[active]` config) and a 4-tab bar. The shared core (key/unlock/api/act/setStatePill/boot/loadList/select/section-loop/saveSection/regen/approve) is parameterized; per-type divergences live in hooks (`afterSections`/`afterRegen`/`onApproveFail`) + a `mode:'queue'` Build/Dismiss sibling for Suggestions. Then old routes redirect and op-nav collapses to one entry.

**Tech Stack:** Vanilla JS / static HTML, Flask route (`app.py`), headless Playwright render-verify. No backend/data change.

## Global Constraints

- **Pure consolidation:** every existing view/action preserved exactly; NO new actions; no change to `/api/console/*-page[s]`, `/api/console/topic-suggestions`, `/api/action/*`.
- **Board "Pages"**, BOS `data-sub="pages"`; tabs **Sales · Ingredient · Topic · Suggestions**; default Sales; active tab in the URL hash.
- **The shared core handles ONE editor instance** (one `#list`/`#editor`/`#ed-sections`/`#status`, etc.) — switching a tab re-points it; do NOT create four parallel DOM copies.
- **Per-type config** (exact — from the source files):

  | key | list | respKey | detail(slug) | edit action | regen action | approve action | approve label | live(slug,data) | section labels |
  |---|---|---|---|---|---|---|---|---|---|
  | sales | `/api/console/sales-pages` | `pages` | `/api/console/sales-page/`+enc | `sales_pages.edit` | `sales_pages.regenerate` | `sales_pages.approve` | `Approve` | `data.live_url \|\| '/begin/product/'+slug` | `intro,description,research` |
  | ingredient | `/api/console/ingredient-pages` | `pages` | `/api/console/ingredient-page/`+enc | `ingredient_page.edit` | `ingredient_page.regenerate` | `ingredient_page.approve` | `Approve and notify` | `'/begin/ingredient/'+slug` | `what_it_is,research` |
  | topic | `/api/console/topic-pages` | `pages` | `/api/console/topic-page/`+enc | `topic_page.edit` | `topic_page.regenerate` | `topic_page.approve` | `Approve and publish` | `'/learn/'+slug` | `overview,symptoms,causes,solutions,lifestyle,when_to_seek` |
  | suggestions | `/api/console/topic-suggestions` | `suggestions` | `/api/console/topic-page/`+enc | `topic_page.edit` | — | — | — (Build/Dismiss) | `'/learn/'+slug` | (topic's) |

- Approve success messages (verbatim): sales `Approved. The live page no longer shows the draft banner.`; ingredient `Approved. Requesters have been notified.`; topic `Approved. Topic page is now public.`
- Console-key gated (and Rae's OWNER token via sub-project A). Render-verify is the core gate, **per type**.
- **Test env:** run the local server via `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/<scratch> CONSOLE_SECRET=test-secret PORT=<p> python3 app.py` (mkdir scratch first). A fresh DB returns empty lists / AI-endpoint errors — the gate is **structure + correct per-type panels/buttons/labels + zero JS errors**, not row counts.

---

### Task 1: Shared-core `console-pages.html` with Sales / Ingredient / Topic + route

**Files:**
- Create: `static/console-pages.html`
- Modify: `app.py` — add `@app.route("/console/pages")` next to `/console/sales-pages` (search the file for `console-sales-pages.html`).
- Source files to lift FROM (read them): `static/console-topic-pages.html` (use as the BASE — it has the most general flow incl. compliance), `static/console-ingredient-pages.html` (the Scores/Traditional-Use/Related-Forms panels), `static/console-sales-pages.html` (sales config only).
- Verify: headless Playwright.

**Interfaces:**
- Consumes: existing `/api/console/*-page[s]` + `/api/action/*` (unchanged).
- Produces: `GET /console/pages`; page globals `TYPES`, `ACTIVE`, `switchType(t)`, `loadList`, `select`, `saveSection`, `regen`, `approve`, `afterSections`, `afterRegen`, `onApproveFail`, `renderCompliance`, `renderIngredientPanels`, `saveScores`, `saveTraditionalUse`, `saveRelatedForms`, shared `key/unlock/api/act/setStatePill/boot`.

- [ ] **Step 1: Page scaffold (HTML)**

Create `static/console-pages.html`. Use `console-topic-pages.html` as the visual base (copy its `<style>` block verbatim — it already styles `#list`/`#editor`/`#ed-sections`/`#ed-compliance`/`.pill`/`.btn`). Add the tab bar + the ingredient-only containers. Structure:

```html
<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pages · Console</title>
<style>
  /* paste console-topic-pages.html's <style> here verbatim, then add: */
  .pages-tabs{display:flex;gap:6px;padding:0 20px;border-bottom:1px solid var(--border,#21472d)}
  .ptab{background:transparent;border:0;border-bottom:2px solid transparent;color:var(--muted,#a89870);font:600 14px/1 inherit;padding:10px 16px;cursor:pointer}
  .ptab.active{color:var(--cream,#fdf4d8);border-bottom-color:var(--gold,#d4a843)}
</style>
</head><body>
<script src="/static/op-nav.js" data-active="bos" data-sub="pages"></script>
<div id="gate" style="display:none"><input id="key" type="password" placeholder="Console key"><button onclick="unlock()">Unlock</button></div>
<div id="app" style="display:none">
  <div class="pages-tabs">
    <button class="ptab active" data-type="sales"       onclick="switchType('sales')">Sales</button>
    <button class="ptab"        data-type="ingredient"  onclick="switchType('ingredient')">Ingredient</button>
    <button class="ptab"        data-type="topic"       onclick="switchType('topic')">Topic</button>
    <button class="ptab"        data-type="suggestions" onclick="switchType('suggestions')">Suggestions</button>
  </div>
  <div class="cols">
    <div id="list"></div>
    <div id="editor" style="display:none">
      <h2 id="ed-title"></h2>
      <a id="ed-live" target="_blank">View live ↗</a>
      <span id="ed-state" class="pill"></span>
      <div id="ed-sections"></div>
      <div id="ed-scores" style="display:none"></div>
      <div id="ed-traditional-use" style="display:none"></div>
      <div id="ed-related-forms" style="display:none"></div>
      <div id="ed-compliance" style="display:none"></div>
      <div class="row" id="ed-actions">
        <button class="btn" id="btn-regen" onclick="regen()">Regenerate &amp; review</button>
        <button class="btn primary" id="btn-approve" onclick="approve()">Approve</button>
        <button class="btn" id="btn-build" style="display:none" onclick="build()">Build</button>
        <button class="btn err" id="btn-dismiss" style="display:none" onclick="dismiss()">Dismiss</button>
      </div>
      <div id="status" class="sub"></div>
    </div>
    <div id="empty" class="sub">Select an item to edit.</div>
  </div>
</div>
```
(Match the exact class names the pasted `<style>` uses for `.cols`/`#list`/`#editor` from the topic source; if the topic source uses different wrapper classes, mirror them.)

- [ ] **Step 2: Shared core JS — parameterized from `ACTIVE`**

Add the script. Lift `key`, the `?key=` IIFE, `hdr`/`api`, `act`, `setStatePill`, `unlock` **verbatim** from `console-topic-pages.html` (they are byte-identical across all four files). Then write the parameterized core. `$ = id => document.getElementById(id)`. Globals `let ACTIVE=null, CURRENT=null, CURRENT_DATA=null;` and the `TYPES` object built exactly from the Global-Constraints config table, each value carrying: `key, list, respKey, detail(slug), edit, regen, approve, approveLabel, approveMsg, live(slug,data), labels{}` (+ `mode:'queue'` on `suggestions`).

- `boot()`: `if(!key()){ $('gate').style.display='flex'; return; } $('app').style.display=''; initType();`
- `initType()`: `var t=(location.hash||'').replace('#',''); switchType(TYPES[t]?t:'sales');`
- `switchType(t)`: set `ACTIVE=TYPES[t]`; toggle `.ptab.active` by `data-type`; `if(location.hash!=='#'+t) location.hash=t`; reset editor (`CURRENT=null; CURRENT_DATA=null; $('editor').style.display='none'; $('empty').style.display='';`); **swap action buttons by mode** — if `ACTIVE.mode==='queue'`: hide `#btn-regen`/`#btn-approve`, show `#btn-build`/`#btn-dismiss`; else: show regen/approve, hide build/dismiss, and set `$('btn-approve').textContent = ACTIVE.approveLabel`; then `loadList()`.
- `loadList()`: `const r = await api(ACTIVE.list); const items = (r.json||{})[ACTIVE.respKey]||[]; render rows into #list` — each row a clickable element calling `select(slug)`, built via `renderItem(p)`.
- `renderItem(p)`: default = name + a `.pill` of `p.state` (`.approved` class if `state==='approved'`). If `ACTIVE.mode==='queue'`: name + `<span class="demand"> ('+p.demand+')</span>` + a `.pill` of `p.kind` (never `.approved`). (Lift the exact row markup from the topic source's `loadList` for the default, and the suggestions source for the queue variant — but Suggestions is wired in Task 2; for Task 1 it's fine if the queue row path exists but the tab isn't exercised yet.)
- `select(slug)`: `const r=await api(ACTIVE.detail(slug)); const match=(r.json||{}).page||r.json; CURRENT=slug; CURRENT_DATA=match; $('editor').style.display=''; $('empty').style.display='none'; $('ed-title').textContent=match.name||slug; $('ed-live').href=ACTIVE.live(slug, match); setStatePill(match.state||'draft');` then render the section loop into `#ed-sections` (verbatim from the topic source: for each `s` in `match.sections`, a `<label>` using `ACTIVE.labels[s.id]||s.id`, a `<textarea id="sec-"+s.id>` with `s.text`, and a `Save` button calling `saveSection(s.id)`); then `afterSections(match); $('status').textContent='';`
- `saveSection(id)`: `await act(ACTIVE.edit, {slug:CURRENT, section:id, text:$('sec-'+id).value});` on ok → `setStatePill('draft'); $('status').textContent='Saved '+id+' (still draft).';` (lift the exact success/fail handling from topic source).
- `regen()`: disable `#btn-regen`; `const r=await act(ACTIVE.regen,{slug:CURRENT});` re-enable; on ok update each `#sec-*` textarea from `(r.json.result||{}).content`; `afterRegen(r.json.result||{}); $('status').textContent='Regenerated. Review the content, then Approve.';`
- `approve()`: `const r=await act(ACTIVE.approve,{slug:CURRENT});` if ok → `$('status').textContent=ACTIVE.approveMsg; loadList();` else `if(!onApproveFail(r)) $('status').textContent='Approve failed: '+((r.json||{}).error||r.status);`

- [ ] **Step 3: Per-type hooks + lifted type-specific code**

```javascript
function afterSections(match){
  ['ed-scores','ed-traditional-use','ed-related-forms','ed-compliance'].forEach(function(id){ $(id).style.display='none'; $(id).innerHTML=''; });
  if (ACTIVE.key==='ingredient') renderIngredientPanels(match);
  if (ACTIVE.key==='topic' || ACTIVE.key==='suggestions') { $('ed-compliance').style.display=''; renderCompliance(match.compliance||{}); }
}
function afterRegen(result){
  if (ACTIVE.key==='topic' || ACTIVE.key==='suggestions') renderCompliance((result||{}).compliance||{});
}
function onApproveFail(r){            // returns true if it handled the message
  if (ACTIVE.key==='topic' && (r.json||{}).error==='compliance_failed'){
    $('ed-compliance').style.display=''; renderCompliance({passed:false, flags:(r.json||{}).flags});
    $('status').textContent='Approve blocked: compliance failed. Resolve the flags and retry.';
    return true;
  }
  return false;
}
```
- **`renderCompliance(c)`** — lift **verbatim** from `console-topic-pages.html` (its `renderCompliance`, the 3-state Passed/Failed+flags/Not-scanned renderer writing into `#ed-compliance`).
- **`renderIngredientPanels(match)`** — lift from `console-ingredient-pages.html`'s `select()` body (the Scores / Traditional-Use / Related-Forms render blocks): set each container's `style.display=''` and render — Scores into `#ed-scores` (two number inputs `#research_score`/`#traditional_score` prefilled from `match.research_score`/`match.traditional_score` + a "Save scores" button → `saveScores()`); Traditional-Use into `#ed-traditional-use` (`<textarea id="traditional-use-json">` prefilled `JSON.stringify(match.traditional_use||[],null,2)` + Save → `saveTraditionalUse()`); Related-Forms into `#ed-related-forms` (`<textarea id="related-forms-json">` prefilled `JSON.stringify(match.related_forms||[],null,2)` + Save → `saveRelatedForms()`).
- **`saveScores()` / `saveTraditionalUse()` / `saveRelatedForms()`** — lift verbatim from the ingredient source: each `await act('ingredient_page.edit', {slug:CURRENT, ...})` with payload `{research_score, traditional_score}` / `{traditional_use: JSON.parse(...)}` / `{related_forms: JSON.parse(...)}` respectively; keep their try/catch on JSON.parse and status messages.

End the script with `boot();`.

- [ ] **Step 4: Route**

In `app.py`, immediately after the `/console/sales-pages` route, add:
```python
@app.route("/console/pages")
def bos_pages_page():
    resp = send_from_directory(STATIC, "console-pages.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp
```

- [ ] **Step 5: Render-verify the 3 editors (headless)**

`mkdir -p /tmp/pages-test`; start the app on PORT=5097 (see Global Constraints). Save `/tmp/pages-test/pv.py`:
```python
from playwright.sync_api import sync_playwright
B="http://127.0.0.1:5097/console/pages?key=test-secret"
def probe(pg):
    return pg.evaluate("""()=>({
      active:(document.querySelector('.ptab.active')||{}).dataset?.type,
      approveLabel:document.getElementById('btn-approve').textContent.trim(),
      regenShown:getComputedStyle(document.getElementById('btn-regen')).display!=='none',
      buildShown:getComputedStyle(document.getElementById('btn-build')).display!=='none'
    })""")
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1280,"height":900})
    errs=[]; pg.on("pageerror",lambda e:errs.append(str(e)))
    pg.on("console", lambda m: errs.append("CJS:"+m.text) if (m.type=="error" and "Failed to load resource" not in m.text) else None)
    pg.goto(B, wait_until="networkidle"); pg.wait_for_timeout(900)
    print("SALES default:", probe(pg))
    for t,lbl in [("ingredient","Approve and notify"),("topic","Approve and publish")]:
        pg.click(".ptab[data-type='%s']"%t); pg.wait_for_timeout(700)
        s=probe(pg); print(t.upper(), s, "hash", pg.evaluate("()=>location.hash"))
        assert s["active"]==t and s["approveLabel"]==lbl and s["regenShown"] and not s["buildShown"]
    # deep-link
    pg.goto(B+"#topic", wait_until="networkidle"); pg.wait_for_timeout(700)
    assert pg.evaluate("()=>(document.querySelector('.ptab.active')||{}).dataset?.type")=="topic"
    print("JS errs:", errs or "NONE"); assert not errs, errs
    b.close(); print("OK")
```
Run `python3 /tmp/pages-test/pv.py` → `OK`, `JS errs: NONE`; Sales default; Ingredient label "Approve and notify"; Topic label "Approve and publish"; regen shown / build hidden in all editor tabs; deep-link `#topic` works. Kill the server. (List rows may be empty on a fresh DB — fine; the gate is the per-type chrome + zero JS errors.)

- [ ] **Step 6: Commit**
```bash
git add static/console-pages.html app.py
git commit -m "feat(console): Pages board — shared-core editor for Sales/Ingredient/Topic + route"
```

---

### Task 2: Suggestions tab (queue mode — Build / Dismiss)

**Files:**
- Modify: `static/console-pages.html` (add `build`/`dismiss`, the queue list-row path, wire the suggestions config).
- Source: `static/console-topic-suggestions.html` (lift `build`/`dismiss`).
- Verify: headless Playwright.

**Interfaces:**
- Consumes: Task 1's `ACTIVE`/`switchType`/`loadList`/`select`/`renderItem`/`act`.
- Produces: `build()`, `dismiss()`; the suggestions tab functional.

- [ ] **Step 1: Wire the queue list rows + actions**

Confirm `renderItem(p)` handles `ACTIVE.mode==='queue'` (name + `(demand)` + `kind` pill) — if Task 1 stubbed it, complete it now by lifting the suggestions source's list-row markup. Add:
```javascript
async function build(){
  $('btn-build').disabled=true;
  const r = await act('topic_page.regenerate', {slug:CURRENT});
  $('btn-build').disabled=false;
  if (r.ok){ const c=(r.json.result||{}); /* update #sec-* from c.content */ ; setStatePill('draft'); renderCompliance(c.compliance||{}); $('status').textContent='Built. Review in Topic Pages to approve.'; loadList(); }
  else $('status').textContent='Build failed: '+((r.json||{}).error||r.status);
}
async function dismiss(){
  const r = await act('topic_page.dismiss', {slug:CURRENT});
  if (r.ok){ CURRENT=null; $('editor').style.display='none'; $('empty').style.display=''; loadList(); }
  else $('status').textContent='Dismiss failed: '+((r.json||{}).error||r.status);
}
```
(Lift the exact `build`/`dismiss` bodies from `console-topic-suggestions.html` — update-section + status + reload logic — adapting DOM access to the shared `#sec-*`/`#status` ids, which are the same.)

- [ ] **Step 2: Render-verify Suggestions**

Restart the app (PORT=5097), then:
```python
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(); errs=[]
    pg.on("pageerror",lambda e:errs.append(str(e)))
    pg.on("console", lambda m: errs.append("CJS:"+m.text) if (m.type=="error" and "Failed to load resource" not in m.text) else None)
    pg.goto("http://127.0.0.1:5097/console/pages?key=test-secret#suggestions", wait_until="networkidle"); pg.wait_for_timeout(900)
    s=pg.evaluate("""()=>({active:(document.querySelector('.ptab.active')||{}).dataset?.type,
      buildShown:getComputedStyle(document.getElementById('btn-build')).display!=='none',
      regenShown:getComputedStyle(document.getElementById('btn-regen')).display!=='none'})""")
    print("SUGGESTIONS:", s, "errs:", errs or "NONE")
    assert s["active"]=="suggestions" and s["buildShown"] and not s["regenShown"] and not errs
    b.close(); print("OK")
```
Expected: Suggestions tab active, **Build shown / Regenerate hidden**, zero JS errors. Kill the server.

- [ ] **Step 3: Commit**
```bash
git add static/console-pages.html
git commit -m "feat(console): Pages board — Suggestions tab (Build/Dismiss queue mode)"
```

---

### Task 3: Cut over — redirects, nav collapse, delete old pages

**Files:**
- Modify: `app.py` — replace the 4 old route bodies with redirects.
- Modify: `static/op-nav.js` — collapse to one `pages` entry.
- Delete: the 4 old HTML files.
- Verify: curl redirects + headless nav render.

**Interfaces:**
- Consumes: `/console/pages` (Tasks 1–2).
- Produces: 4 redirects; op-nav shows one **Pages** board.

- [ ] **Step 1: Redirect the 4 old routes**

In `app.py`, replace the bodies of the routes serving `console-sales-pages.html`, `console-ingredient-pages.html`, `console-topic-pages.html`, `console-topic-suggestions.html` (find each via grep) with, respectively:
```python
    return redirect("/console/pages#sales", code=302)
    return redirect("/console/pages#ingredient", code=302)
    return redirect("/console/pages#topic", code=302)
    return redirect("/console/pages#suggestions", code=302)
```
(`redirect` is already imported in app.py.)

- [ ] **Step 2: Collapse op-nav**

In `static/op-nav.js` `bosMods`, remove the four entries with ids `sales`, `ingredients` (the **Ingredient Pages** one, label "Ingredient Pages", href `/console/ingredient-pages` — NOT `ingredients-ops`), `topic-pages`, `topic-suggestions`, and add one `{ id: "pages", label: "Pages", href: "/console/pages" + qs }` (place it where `sales` was). In `NAV_PROFILES.glen.bos`, replace `"sales","ingredients","topic-pages"` (they are consecutive in that array) with `"pages"`, and remove `"topic-suggestions"` if present (it lives in the owner-More group, not `glen.bos`). `rae.bos` does not contain any of these — leave it unchanged. Run `node --check static/op-nav.js`.

- [ ] **Step 3: Delete the old pages + render-verify**

```bash
git rm static/console-sales-pages.html static/console-ingredient-pages.html static/console-topic-pages.html static/console-topic-suggestions.html
```
Start the app (PORT=5098). Verify redirects:
```bash
for p in sales-pages:sales ingredient-pages:ingredient topic-pages:topic topic-suggestions:suggestions; do
  old="${p%%:*}"; frag="${p##*:}"
  curl -s -o /dev/null -w "$old -> %{http_code} %{redirect_url}\n" "http://127.0.0.1:5098/console/$old?key=test-secret"
done
```
Expected: each `302 …/console/pages#<frag>`. Then headless render `/console/pages?key=test-secret` and assert the BOS sub-row (incl. `#op-nav-more-bos` menu) contains `pages` and NOT `sales`/`ingredients`/`topic-pages`/`topic-suggestions`, but still contains `ingredients-ops`, zero JS errors:
```python
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(); errs=[]
    pg.on("pageerror",lambda e:errs.append(str(e)))
    pg.goto("http://127.0.0.1:5098/console/pages?key=test-secret", wait_until="networkidle"); pg.wait_for_timeout(900)
    ids=pg.evaluate("()=>[...document.querySelectorAll('.op-nav-sub a.op-nav-subtab, #op-nav-more-bos .op-nav-more-menu a')].map(a=>a.dataset.id)")
    print("BOS ids:", ids)
    assert "pages" in ids and "ingredients-ops" in ids
    assert not any(x in ids for x in ["sales","ingredients","topic-pages","topic-suggestions"])
    assert not errs, errs
    b.close(); print("OK")
```
Kill the server.

- [ ] **Step 4: Commit**
```bash
git add app.py static/op-nav.js
git commit -m "feat(console): cut over to Pages board — redirect old routes, collapse nav, drop old pages"
```

---

## Verification (whole sub-project)

- `/console/pages` 200; the 4 old routes 302 to the right hashes.
- Per-type render-verify (Tasks 1–2): Sales (3 sections, no panels), Ingredient (2 sections + Scores/Traditional-Use/Related-Forms, "Approve and notify"), Topic (sections + compliance, "Approve and publish"), Suggestions (Build/Dismiss, demand/kind rows); deep-links; zero JS errors throughout.
- `node --check static/op-nav.js`; BOS shows one **Pages**, the 4 page ids gone, `ingredients-ops` kept.
- `grep -rn "console-sales-pages\.html\|console-ingredient-pages\.html\|console-topic-pages\.html\|console-topic-suggestions\.html" app.py static/` returns nothing.

## Self-Review Notes

- **Spec coverage:** shared-core editor + 3 types (Task 1) ✓; Suggestions queue mode (Task 2) ✓; route (T1 S4) ✓; redirects + nav collapse + deletes (Task 3) ✓; per-type config + hooks incl. ingredient panels/topic compliance/onApproveFail (T1 S2–3) ✓; one-editor-not-4-panels (architecture) ✓; backends untouched (Global Constraints) ✓.
- **Type consistency:** `TYPES[key]` fields (`list/respKey/detail/edit/regen/approve/approveLabel/approveMsg/live/labels/mode`) referenced consistently; hooks `afterSections/afterRegen/onApproveFail` + `renderCompliance/renderIngredientPanels/saveScores/saveTraditionalUse/saveRelatedForms` + `build/dismiss` names consistent across tasks.
- **YAGNI:** drop sales' `esc()` (DOM-API loop); no new actions; redirects not dead pages.
