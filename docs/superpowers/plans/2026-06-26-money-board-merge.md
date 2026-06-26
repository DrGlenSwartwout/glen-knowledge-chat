# Money Board Merge (Sub-project B1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the Payments (Stripe ledger) and Finance (QBO receivables) console boards into one tabbed **Money** board at `/console/money`, backends unchanged.

**Architecture:** A new `static/console-money.html` with two tabs (Payments | Receivables). Each existing page's differing JS is lifted into a panel and wrapped in an IIFE that exposes a namespaced object (`MoneyPayments` / `MoneyReceivables`); the three identical helpers (`key`/`hdr`/`esc`) and the gate live once at page level; each panel's colliding DOM ids and CSS are prefixed. Then the old routes redirect and op-nav collapses to one Money entry.

**Tech Stack:** Vanilla JS / static HTML, Flask route (`app.py`), headless Playwright (Chromium) render-verify. No backend or data change.

## Global Constraints

- **Pure consolidation:** every existing view/action is preserved exactly; **no new actions** (failed-charge retry/contact, record-payment belong to sub-project C). Do not change `/api/payments`, `/api/finance/ar`, `/api/action/finance` or their behavior.
- **Board name = "Money"**; tabs = **Payments** and **Receivables**; BOS `data-sub="money"`.
- **No broken links:** `/console/payments` → 302 `/console/money#payments`; `/console/finance` → 302 `/console/money#receivables` (the dashboard "Money & Cash" card deep-links to `/console/finance`).
- **`fmt()` differs between the two pages** — Payments' takes **cents** (`/100`), Finance's takes **dollars**. They MUST stay separate (one per IIFE); never share a single `fmt`.
- Console-key gated as today (and Rae's OWNER token via sub-project A). Render-verify is the core gate (the render-verify lesson): the merged page must load **both** tabs' real data with **zero** JS console/page errors and no redeclaration/ReferenceError.
- **Test env:** app validates secrets at import; run the local server via `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/<scratch> CONSOLE_SECRET=test-secret PORT=<p> python3 app.py` (mkdir the scratch dir first).

---

### Task 1: Build `console-money.html` (the merged board) + its route

**Files:**
- Create: `static/console-money.html`
- Modify: `app.py` — add `@app.route("/console/money")` (next to the existing `/console/payments` route, app.py ~24676).
- Source files to lift FROM (read them): `static/console-payments.html`, `static/console-finance.html`.
- Verify: headless Playwright.

**Interfaces:**
- Consumes: existing endpoints `/api/payments`, `/api/finance/ar`, `/api/action/finance` (unchanged).
- Produces: `GET /console/money` serving the page; page globals `MoneyPayments = {load, setFilter}` and `MoneyReceivables = {load, act, doRefund}`; shared `key()`/`hdr()`/`esc()`/`unlock()`/`showMoneyTab(t)`.

- [ ] **Step 1: Create the page scaffold + shared script**

Create `static/console-money.html` with this exact skeleton. The two `<!-- LIFT … -->` panel bodies and the two IIFE bodies are filled in the next steps.

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Money - Business OS</title>
<style>
  /* Page chrome + tabs. The two source pages' <style> blocks are pasted below in
     Step 4, each rule prefixed with its panel selector to avoid CSS collisions. */
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg,#0a150d); color:var(--cream,#fdf4d8); }
  .money-head { padding:16px 20px 0; }
  .money-head h1 { margin:0 0 12px; font-size:20px; }
  .money-tabs { display:flex; gap:6px; padding:0 20px; border-bottom:1px solid var(--border,#21472d); }
  .mtab { background:transparent; border:0; border-bottom:2px solid transparent; color:var(--muted,#a89870); font:600 14px/1 inherit; padding:10px 16px; cursor:pointer; }
  .mtab.active { color:var(--cream,#fdf4d8); border-bottom-color:var(--gold,#d4a843); }
  .money-panel { padding:16px 20px; }
  #gate { display:flex; gap:8px; align-items:center; padding:20px; }
  #gate input { padding:8px 10px; border-radius:6px; border:1px solid var(--border,#21472d); background:#0d1117; color:var(--cream,#fdf4d8); }
</style>
</head>
<body>
<script src="/static/op-nav.js" data-active="bos" data-sub="money"></script>

<div id="gate">
  <input id="key" type="password" placeholder="Console key">
  <button onclick="unlock()">Unlock</button>
</div>

<div id="money-wrap" style="display:none">
  <div class="money-head"><h1>Money</h1></div>
  <div class="money-tabs">
    <button class="mtab active" data-tab="payments" onclick="showMoneyTab('payments')">Payments</button>
    <button class="mtab" data-tab="receivables" onclick="showMoneyTab('receivables')">Receivables</button>
  </div>
  <div id="panel-payments" class="money-panel">
    <!-- LIFT: console-payments.html body (lines 69–100), MINUS its #gate block, ids prefixed p- (Step 2) -->
  </div>
  <div id="panel-receivables" class="money-panel" style="display:none">
    <!-- LIFT: console-finance.html body (lines 56–84), MINUS its #gate block, ids prefixed f- (Step 3) -->
  </div>
</div>

<script>
  // ---- shared (identical in both source pages) ----
  function key(){ return localStorage.getItem('console_key') || ''; }
  (function(){ var u=new URLSearchParams(location.search).get('key'); if(u) localStorage.setItem('console_key', u); })();
  function hdr(){ return { 'X-Console-Key': key(), 'Content-Type':'application/json' }; }
  function esc(s){ return String(s==null?'':s).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];}); }

  function unlock(){ localStorage.setItem('console_key', document.getElementById('key').value); document.getElementById('gate').style.display='none'; reveal(); }

  var _moneyLoaded = { payments:false, receivables:false };
  function showMoneyTab(t){
    if (t!=='payments' && t!=='receivables') t='payments';
    document.querySelectorAll('.mtab').forEach(function(b){ b.classList.toggle('active', b.dataset.tab===t); });
    document.getElementById('panel-payments').style.display    = (t==='payments') ? '' : 'none';
    document.getElementById('panel-receivables').style.display = (t==='receivables') ? '' : 'none';
    if (location.hash !== '#'+t) location.hash = t;
    if (!_moneyLoaded[t]) { _moneyLoaded[t]=true; (t==='payments' ? MoneyPayments : MoneyReceivables).load(); }
  }
  function reveal(){ document.getElementById('money-wrap').style.display=''; var t=(location.hash||'').replace('#',''); showMoneyTab(t); }

  // ---- Payments module (Step 2 fills this IIFE) ----
  var MoneyPayments = (function(){
    /* LIFT console-payments.html script: fmt, when, curSrc, setFilter, tile, rowHtml,
       renderFailures, load — with getElementById ids prefixed p-. Omit its key/hdr/esc/
       unlock/URL-IIFE (shared above) and its final `if(key()) load()` init line. */
    return { load: /*load*/null, setFilter: /*setFilter*/null };
  })();

  // ---- Receivables module (Step 3 fills this IIFE) ----
  var MoneyReceivables = (function(){
    /* LIFT console-finance.html script: fmt, ageBadge, rowHtml, escAttr, load, tile, act,
       doRefund — with getElementById ids prefixed f-. Omit its key/hdr/esc/unlock/URL-IIFE
       and its final `if(key()) load()` init line. */
    return { load: /*load*/null, act: /*act*/null, doRefund: /*doRefund*/null };
  })();

  // ---- init: reveal if already unlocked, else show gate ----
  (function(){ if (key()){ document.getElementById('gate').style.display='none'; reveal(); } })();
</script>
</body>
</html>
```

- [ ] **Step 2: Fill the Payments panel + IIFE (lift from `console-payments.html`)**

Into `#panel-payments`, paste the body of `console-payments.html` (lines 69–100) **except its `#gate` block** (the gate is shared). Rename these DOM ids by adding a `p-` prefix: `failBanner→p-failBanner`, `filters→p-filters`, `summary→p-summary`, `err→p-err`, `empty→p-empty`, `board→p-board`, `tbl→p-tbl`. In the lifted markup, rewrite the filter buttons' `onclick="setFilter(this)"` → `onclick="MoneyPayments.setFilter(this)"`.

Into the `MoneyPayments` IIFE, paste the page's script functions **`fmt`, `when`, `curSrc`, `setFilter`, `tile`, `rowHtml`, `renderFailures`, `load`** (console-payments.html lines 107–173). Do NOT copy `key`/`hdr`/`esc`/`unlock`/the URL-IIFE (shared) or line 175 (init). Inside these functions, change every `document.getElementById('X')` to the `p-`-prefixed id (e.g. `getElementById('board')` → `getElementById('p-board')`, same for `err`/`empty`/`summary`/`tbl`/`failBanner`; `getElementById('filters')` in `setFilter` → `p-filters`). `load`'s 401 branch keeps `document.getElementById('gate')` (shared gate). End the IIFE with `return { load: load, setFilter: setFilter };`.

- [ ] **Step 3: Fill the Receivables panel + IIFE (lift from `console-finance.html`)**

Into `#panel-receivables`, paste the body of `console-finance.html` (lines 56–84) **except its `#gate` block**. Rename ids by adding `f-`: `err→f-err`, `empty→f-empty`, `summary→f-summary`, `board→f-board`. The refund-form ids (`r-invoice`, `r-amount`, `r-reason`, `r-result`, `r-stripe`) are already unique — keep them. Rewrite the inline `onclick="doRefund()"` (the refund button) → `onclick="MoneyReceivables.doRefund()"`.

Into the `MoneyReceivables` IIFE, paste functions **`fmt`, `ageBadge`, `rowHtml`, `escAttr`, `load`, `tile`, `act`, `doRefund`** (console-finance.html lines 93–196). Omit `key`/`hdr`/`esc`/`unlock`/URL-IIFE and line 198 (init). Inside, change `getElementById('err'|'empty'|'summary'|'board')` to the `f-` ids; keep `r-*` ids; keep the 401 branch's `getElementById('gate')`. **In `rowHtml`, the generated buttons call `act(...)` inside an onclick string — change those to `MoneyReceivables.act(...)`** (lines 118–119: `onclick="act(...)"` → `onclick="MoneyReceivables.act(...)"`). End the IIFE with `return { load: load, act: act, doRefund: doRefund };`.

- [ ] **Step 4: Merge the two `<style>` blocks, panel-scoped**

Paste `console-payments.html`'s `<style>` rules and `console-finance.html`'s `<style>` rules into the page `<style>`. To prevent CSS collisions (both define `.summary-tile`/`.label`/`.value` etc.), **prefix every lifted payments rule with `#panel-payments ` and every lifted finance rule with `#panel-receivables `** (e.g. `.summary-tile{…}` from payments → `#panel-payments .summary-tile{…}`). Leave the scaffold's own `.money-*`/`#gate` rules as-is.

- [ ] **Step 5: Add the route**

In `app.py`, immediately after the `/console/payments` route (`bos_payments_page`, ~line 24676), add:

```python
@app.route("/console/money")
def bos_money_page():
    resp = send_from_directory(STATIC, "console-money.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp
```

- [ ] **Step 6: Render-verify (headless) — the core gate**

`mkdir -p /tmp/money-test`, start the app:
```bash
doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/money-test CONSOLE_SECRET=test-secret PORT=5097 python3 app.py > /tmp/money-test/srv.log 2>&1 &
for i in $(seq 1 45); do curl -s -o /dev/null "http://127.0.0.1:5097/console/money?key=test-secret" && break; sleep 1; done
```
Save `/tmp/money-test/mv.py` and run with `python3`:
```python
from playwright.sync_api import sync_playwright
B="http://127.0.0.1:5097/console/money"
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1280,"height":900})
    errs=[]; pg.on("pageerror",lambda e:errs.append(str(e)))
    pg.on("console", lambda m: errs.append("C:"+m.text) if m.type=="error" else None)
    pg.goto(B+"?key=test-secret", wait_until="networkidle"); pg.wait_for_timeout(1200)
    s1=pg.evaluate("""()=>({
      tabs:[...document.querySelectorAll('.mtab')].map(b=>b.dataset.tab),
      activeTab:(document.querySelector('.mtab.active')||{}).dataset?.tab,
      filters:document.querySelectorAll('#p-filters button').length,
      payVisible:getComputedStyle(document.getElementById('panel-payments')).display!=='none'
    })""")
    pg.click(".mtab[data-tab='receivables']"); pg.wait_for_timeout(1200)
    s2=pg.evaluate("""()=>({
      recVisible:getComputedStyle(document.getElementById('panel-receivables')).display!=='none',
      hash:location.hash, refundBtn:!!document.querySelector('#panel-receivables [id^=\"r-\"]')
    })""")
    print("PAYMENTS:", s1); print("RECEIVABLES:", s2); print("errs:", errs or "NONE")
    assert s1["activeTab"]=="payments" and s1["payVisible"] and s1["filters"]>=4
    assert s2["recVisible"] and s2["hash"]=="#receivables"
    assert not errs, errs
    b.close(); print("OK")
```
Run `python3 /tmp/money-test/mv.py` → prints `OK`, `errs: NONE`. Then deep-link test: load `/console/money#receivables?key=test-secret` style isn't valid (hash after query); use `/console/money?key=test-secret#receivables` and assert the Receivables tab is active on load. Kill the server (`lsof -ti :5097 | xargs kill`).
Expected: Payments tab active by default with ≥4 filter buttons + the Stripe table; clicking Receivables lazy-loads its AR panel (with the refund-form `r-*` controls) and sets `#receivables`; **zero JS errors** (no `Identifier 'load' has already been declared`, no `ReferenceError`).

- [ ] **Step 7: Commit**

```bash
git add static/console-money.html app.py
git commit -m "feat(console): Money board — merge Payments + Finance into /console/money tabs"
```

---

### Task 2: Cut over — redirect old routes, collapse the nav, delete old pages

**Files:**
- Modify: `app.py` — replace the `/console/payments` and `/console/finance` route bodies with redirects.
- Modify: `static/op-nav.js` — collapse `payments`+`finance` to one `money` entry (bosMods + NAV_PROFILES).
- Delete: `static/console-payments.html`, `static/console-finance.html`.
- Verify: route redirects + headless nav render.

**Interfaces:**
- Consumes: `/console/money` (Task 1).
- Produces: `/console/payments` → 302 `/console/money#payments`; `/console/finance` → 302 `/console/money#receivables`; op-nav BOS shows one **Money** board.

- [ ] **Step 1: Redirect the old routes**

In `app.py`, replace the body of `bos_payments_page` (~24676) and `bos_finance_page` (~24841). Ensure `redirect` is imported (it is — used elsewhere in app.py). New bodies:

```python
@app.route("/console/payments")
def bos_payments_page():
    return redirect("/console/money#payments", code=302)
```
```python
@app.route("/console/finance")
def bos_finance_page():
    return redirect("/console/money#receivables", code=302)
```

- [ ] **Step 2: Verify the redirects**

`mkdir -p /tmp/money-test2`; start the app on PORT=5098 as in Task 1 Step 6. Then:
```bash
for path in payments finance; do
  curl -s -o /dev/null -w "$path -> %{http_code} %{redirect_url}\n" "http://127.0.0.1:5098/console/$path?key=test-secret"
done
```
Expected: `payments -> 302 …/console/money#payments`, `finance -> 302 …/console/money#receivables`. (Leave the server up for Step 4.)

- [ ] **Step 3: Collapse the nav in `op-nav.js`**

In `static/op-nav.js`, in the `bosMods` array, replace these two lines (83–84):
```javascript
    { id: "payments", label: "Payments",  href: "/console/payments" + qs },
    { id: "finance",  label: "Finance",   href: "/console/finance" + qs },
```
with one line:
```javascript
    { id: "money",    label: "Money",     href: "/console/money" + qs },
```
In the `NAV_PROFILES` map, change the `glen.bos` array (line ~284) entry `"payments","finance"` to `"money"`, and the `rae.bos` array (line ~290) entry `"payments","finance"` to `"money"`. (Each array currently contains `…"orders","payments","finance","crm"…`; make it `…"orders","money","crm"…`.) Run `node --check static/op-nav.js`.

- [ ] **Step 4: Delete the old pages + render-verify the nav**

```bash
git rm static/console-payments.html static/console-finance.html
```
Then render-verify (server from Step 2 still up, or restart on 5098): load `/console/money?key=test-secret` and assert the BOS sub-row shows a **Money** subtab and NOT Payments/Finance, with zero JS errors:
```python
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(); errs=[]
    pg.on("pageerror",lambda e:errs.append(str(e)))
    pg.goto("http://127.0.0.1:5098/console/money?key=test-secret", wait_until="networkidle"); pg.wait_for_timeout(900)
    ids=pg.evaluate("()=>[...document.querySelectorAll('.op-nav-sub a.op-nav-subtab, #op-nav-more-bos .op-nav-more-menu a')].map(a=>a.dataset.id)")
    print("BOS ids:", ids, "errs:", errs or "NONE")
    assert "money" in ids and "payments" not in ids and "finance" not in ids and not errs
    b.close(); print("OK")
```
Expected: `money` present, `payments`/`finance` absent, zero errors. Kill the server.

- [ ] **Step 5: Commit**

```bash
git add app.py static/op-nav.js
git commit -m "feat(console): cut over to Money board — redirect old routes, collapse nav, drop old pages"
```

---

## Verification (whole sub-project)

- Routes: `/console/money` 200; `/console/payments` + `/console/finance` 302 to the right tab.
- Render-verify: both tabs load real data with zero JS errors; deep-link `#receivables` activates the Receivables tab; the Refund/Send-reminder/Void actions render; the source filters work on Payments.
- `node --check static/op-nav.js` passes; BOS sub-row shows one **Money** board; a Rae-profile render keeps Money primary (not in More).
- `grep -rn "console-payments\.html\|console-finance\.html" app.py static/` returns nothing (no dangling references after deletion).

## Self-Review Notes

- **Spec coverage:** merged tabbed page (Task 1) ✓; backends untouched (Global Constraints; no `/api/*` edits) ✓; old-route redirects (Task 2 Step 1) ✓; nav collapse (Task 2 Step 3) ✓; old files deleted (Task 2 Step 4) ✓; deep-link hash (Task 1 scaffold `showMoneyTab`/`reveal`) ✓; lazy-load (`_moneyLoaded`) ✓; namespacing incl. separate `fmt` (Steps 2–3) ✓.
- **Type consistency:** `MoneyPayments.{load,setFilter}` and `MoneyReceivables.{load,act,doRefund}` are produced in Task 1 and referenced by the inline handlers + `showMoneyTab`; shared `key/hdr/esc/unlock/showMoneyTab/reveal` names consistent across steps.
- **YAGNI:** no new actions; redirects rather than keeping dead pages.
