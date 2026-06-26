# Console Cross-links & Affordances (Sub-project C3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Four pure-frontend affordances — Reveals→Portal link, CRM queue Cancel/Retry, CRM contact search, and a dashboard alert that lands on the Money board's Receivables tab (fixing the key-loss bug).

**Architecture:** Frontend-only edits across four static pages, reusing existing endpoints (`/api/console/biofield-portal?email=`, `/api/ghl/queue/result`, `/api/people?q=`, the Money board `#receivables` tab). No backend, no schema, no new endpoint.

**Tech Stack:** Vanilla JS / static HTML, headless Playwright (mocked) render-verify. No Python.

## Global Constraints

- **Pure frontend; no backend/schema/endpoint change.** Reuse the existing endpoints verbatim.
- No JSON-in-onclick of objects — queue buttons pass the numeric `q.id`.
- Render-verify (the render-verify lesson) is the gate for every piece: mocked endpoints, zero JS console/page errors.
- **Test env:** run the app via `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/<scratch> CONSOLE_SECRET=test-secret PORT=<p> python3 app.py` (mkdir scratch first); mock the relevant `/api/*` with Playwright `route`.

---

### Task 1: Reveals → Portal link

**Files:**
- Modify: `static/console-biofield-portal.html` (`boot()` ~line 141 — auto-load from `?email=`).
- Modify: `static/console-biofield-reveals.html` (`buildCard()` — add the "Open Portal →" anchor).
- Verify: headless Playwright (mocked).

**Interfaces:**
- Consumes: `GET /api/console/biofield-portal?email=` (existing), the reveal card's `d.email` + the page's `key()`.
- Produces: an "Open Portal →" anchor per reveal card; the Portal seeds `#email` + calls `loadExisting()` from `?email=`.

- [ ] **Step 1: Portal auto-loads from `?email=`**

In `static/console-biofield-portal.html`, in `boot()` (~line 141), after `loadCatalog();` add (before the listener wiring):
```javascript
  var _urlEmail = new URLSearchParams(location.search).get('email');
  if (_urlEmail) { $('email').value = _urlEmail; loadExisting(); }
```
(`$` is the page's `getElementById` helper; `loadExisting()` already fetches `/api/console/biofield-portal?email=<#email value>`.)

- [ ] **Step 2: "Open Portal →" anchor on the reveal card**

In `static/console-biofield-reveals.html` `buildCard()`, in the card's button/action row, append an anchor (use the card's `d.email` and the page's `key()`):
```javascript
  var portalLink = document.createElement('a');
  portalLink.href = '/console/biofield-portal?email=' + encodeURIComponent(d.email || '')
                  + '&key=' + encodeURIComponent(key());
  portalLink.target = '_blank';
  portalLink.rel = 'noopener';
  portalLink.textContent = 'Open Portal →';
  portalLink.style.cssText = 'margin-left:10px;color:var(--gold,#d4a843);text-decoration:none;font-size:13px';
  row.appendChild(portalLink);
```
(Match the actual variable name of the button-row element in `buildCard` — read it first; the reveal-card row is where the existing action buttons are appended.)

- [ ] **Step 3: Render-verify (headless, mocked)**

`mkdir -p /tmp/cl-test`; start the app on PORT=5097. Save `/tmp/cl-test/t1.py`:
```python
from playwright.sync_api import sync_playwright
import json
with sync_playwright() as p:
    b=p.chromium.launch()
    # (a) reveals card shows the Open Portal anchor with the right href
    pg=b.new_page(); errs=[]; pg.on("pageerror",lambda e:errs.append(str(e)))
    pg.on("console", lambda m: errs.append("CJS:"+m.text) if (m.type=="error" and "Failed to load resource" not in m.text) else None)
    pg.route("**/api/console/biofield-reveals", lambda r: r.fulfill(status=200, content_type="application/json",
        body=json.dumps({"ok":True,"data":[{"id":1,"email":"jo@x.com","scan_date":"2026-06-01","client_name":"Jo","tags":[],"interpretation":{"greeting":"","body":""},"layers":[],"first_approved":False,"notified_at":None}]})))
    pg.goto("http://127.0.0.1:5097/console/biofield-reveals?key=test-secret", wait_until="networkidle"); pg.wait_for_timeout(900)
    href=pg.evaluate("()=>{var a=[...document.querySelectorAll('a')].find(x=>/Open Portal/.test(x.textContent)); return a?a.getAttribute('href'):null}")
    print("reveal portal href:", href, "errs:", errs or "NONE")
    assert href and "email=jo%40x.com" in href and "key=" in href, href
    pg.close()
    # (b) portal ?email= seeds #email + fires the detail fetch
    pg2=b.new_page(); e2=[]; pg2.on("pageerror",lambda e:e2.append(str(e)))
    detail_hit={"n":0}
    def h2(route):
        if "/api/console/biofield-portal" in route.request.url and "email=" in route.request.url:
            detail_hit["n"]+=1; return route.fulfill(status=200, content_type="application/json", body=json.dumps({"ok":True,"data":{"email":"jo@x.com","layers":[],"order":[]}}))
        return route.continue_()
    pg2.route("**/api/console/biofield-portal**", h2)
    pg2.goto("http://127.0.0.1:5097/console/biofield-portal?key=test-secret&email=jo@x.com", wait_until="networkidle"); pg2.wait_for_timeout(900)
    seeded=pg2.evaluate("()=>document.getElementById('email').value")
    print("portal #email:", seeded, "detail fetches:", detail_hit["n"], "errs:", e2 or "NONE")
    assert seeded=="jo@x.com" and detail_hit["n"]>=1 and not e2
    b.close(); print("OK")
```
Run `python3 /tmp/cl-test/t1.py` → `OK`: the reveal card has an Open-Portal anchor with `email=jo%40x.com&key=…`; the portal page seeds `#email=jo@x.com` and fires the detail fetch; zero JS errors. (Adapt the mock envelopes to the real `/api/console/biofield-reveals` + `/api/console/biofield-portal` shapes — read them first.) Kill the server.

- [ ] **Step 4: Commit**
```bash
git add static/console-biofield-portal.html static/console-biofield-reveals.html
git commit -m "feat(console): Reveals -> Portal cross-link (Open Portal + ?email= auto-load)"
```

---

### Task 2: CRM queue Cancel/Retry + contact search

**Files:**
- Modify: `static/console-crm.html` (`loadQueue()` ~line 113 — per-item buttons; add `cancelQ`/`retryQ`; add a contact search type-ahead on `#email`).
- Verify: headless Playwright (mocked).

**Interfaces:**
- Consumes: `POST /api/ghl/queue/result` (existing), `GET /api/people?q=` (existing), the page's `hdr()`/`key()`/`esc()`.
- Produces: `cancelQ(id)`/`retryQ(id)`; a `#crm-ac` search dropdown filling `#email`.

- [ ] **Step 1: Cancel/Retry buttons + handlers**

In `static/console-crm.html` `loadQueue()`, change the per-item row markup to add two buttons (q.id is numeric — safe in onclick):
```javascript
        return '<div class="row"><span><span class="op">'+esc(q.op)+'</span>'+esc(q.email)+'</span>'
             + '<span class="meta">'+esc((q.created_at||'').slice(0,16).replace('T',' '))
             + ' <button onclick="cancelQ('+Number(q.id)+')">Cancel</button>'
             + ' <button onclick="retryQ('+Number(q.id)+')">Retry</button></span></div>';
```
Add the handlers (near `loadQueue`):
```javascript
  async function _setQ(id, status){
    await fetch('/api/ghl/queue/result', {method:'POST', headers:hdr(), body:JSON.stringify({id:id, status:status})});
    loadQueue();
  }
  function cancelQ(id){ _setQ(id, 'cancelled'); }
  function retryQ(id){ _setQ(id, 'pending'); }
```

- [ ] **Step 2: Contact search type-ahead on `#email`**

Add a dropdown element after the `#email` input (~line 59) in the HTML: `<div id="crm-ac" style="position:absolute;z-index:50;background:var(--surface,#111f16);border:1px solid var(--border,#21472d);border-radius:8px;display:none;max-height:240px;overflow:auto"></div>`. Add the type-ahead JS (clone of the Portal's `clientSearch`, adapted to CRM's `hdr()`/`esc()` and its single `#email` field):
```javascript
  var _crmAcTimer=null;
  function crmAcHide(){ var d=document.getElementById('crm-ac'); if(d){ d.style.display='none'; } }
  function crmSearch(ev){
    var input=ev.target, q=(input.value||'').trim();
    clearTimeout(_crmAcTimer);
    if(q.length<2){ crmAcHide(); return; }
    _crmAcTimer=setTimeout(async function(){
      var r=await fetch('/api/people?q='+encodeURIComponent(q)+'&limit=8', {headers:hdr()});
      var j=await r.json().catch(function(){return {};});
      var people=(j && j.people) || [];
      var d=document.getElementById('crm-ac');
      if(!people.length){ crmAcHide(); return; }
      d.innerHTML=people.map(function(p){
        var nm=esc(p.name || [p.first_name,p.last_name].filter(Boolean).join(' ') || '(no name)');
        return '<div class="ac-item" data-email="'+esc(p.email||'')+'" style="padding:7px 12px;cursor:pointer">'+nm+' <span class="meta">'+esc(p.email||'')+'</span></div>';
      }).join('');
      var rect=input.getBoundingClientRect();
      d.style.left=(rect.left+window.scrollX)+'px'; d.style.top=(rect.bottom+window.scrollY+2)+'px'; d.style.width=rect.width+'px'; d.style.display='block';
      d.querySelectorAll('.ac-item').forEach(function(it){ it.onclick=function(){ document.getElementById('email').value=it.dataset.email; crmAcHide(); }; });
    }, 250);
  }
  document.getElementById('email').addEventListener('input', crmSearch);
  document.addEventListener('click', function(e){ if(!e.target.closest('#crm-ac') && e.target!==document.getElementById('email')) crmAcHide(); });
```
(Confirm `esc()` exists in console-crm.html — it's used in `loadQueue`. If the `#email` input is inside a relatively-positioned container, the absolute dropdown will anchor under it via the `getBoundingClientRect` positioning above.)

- [ ] **Step 3: Render-verify (headless, mocked)**

Start the app (PORT=5097). Save `/tmp/cl-test/t2.py`:
```python
from playwright.sync_api import sync_playwright
import json
posts=[]
def h(route):
    u=route.request.url
    if "/api/ghl/queue/pending" in u:
        return route.fulfill(status=200, content_type="application/json", body=json.dumps({"queue":[{"id":3,"op":"tag_add","email":"x@y.com","status":"pending","created_at":"2026-06-26T10:00"}],"count":1}))
    if "/api/ghl/queue/result" in u:
        posts.append(route.request.post_data); return route.fulfill(status=200, content_type="application/json", body=json.dumps({"ok":True}))
    if "/api/people" in u:
        return route.fulfill(status=200, content_type="application/json", body=json.dumps({"total":1,"people":[{"id":1,"email":"found@z.com","name":"Found Person"}]}))
    return route.continue_()
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(); errs=[]
    pg.on("pageerror",lambda e:errs.append(str(e)))
    pg.on("console", lambda m: errs.append("CJS:"+m.text) if (m.type=="error" and "Failed to load resource" not in m.text) else None)
    pg.route("**/api/**", h)
    pg.goto("http://127.0.0.1:5097/console/crm?key=test-secret", wait_until="networkidle"); pg.wait_for_timeout(900)
    has=pg.evaluate("()=>({cancel:[...document.querySelectorAll('#queue button')].some(b=>/Cancel/.test(b.textContent)), retry:[...document.querySelectorAll('#queue button')].some(b=>/Retry/.test(b.textContent))})")
    pg.evaluate("()=>{var b=[...document.querySelectorAll('#queue button')].find(x=>/Cancel/.test(x.textContent)); if(b) b.click();}"); pg.wait_for_timeout(400)
    pg.evaluate("()=>{var b=[...document.querySelectorAll('#queue button')].find(x=>/Retry/.test(x.textContent)); if(b) b.click();}"); pg.wait_for_timeout(400)
    # search
    pg.fill('#email','fou'); pg.wait_for_timeout(600)
    ac=pg.evaluate("()=>{var d=document.getElementById('crm-ac'); return {shown:d&&getComputedStyle(d).display!=='none', items:d?d.querySelectorAll('.ac-item').length:0}}")
    pg.evaluate("()=>{var it=document.querySelector('#crm-ac .ac-item'); if(it) it.click();}"); pg.wait_for_timeout(200)
    email=pg.evaluate("()=>document.getElementById('email').value")
    print("buttons:", has, "| queue posts:", posts, "| ac:", ac, "| email after pick:", email, "| errs:", errs or "NONE")
    assert has["cancel"] and has["retry"]
    bodies=[json.loads(x) for x in posts]
    assert any(b.get("status")=="cancelled" for b in bodies) and any(b.get("status")=="pending" for b in bodies)
    assert ac["shown"] and ac["items"]>=1 and email=="found@z.com"
    assert not errs, errs
    b.close(); print("OK")
```
Run `python3 /tmp/cl-test/t2.py` → `OK`: queue rows have Cancel + Retry; clicking posts `{status:'cancelled'}` then `{status:'pending'}`; typing in `#email` shows a search dropdown and picking fills `#email=found@z.com`; zero JS errors. (Adapt the `/api/ghl/queue/pending` + `/api/people` mock shapes to the real ones — read them first.) Kill the server.

- [ ] **Step 4: Commit**
```bash
git add static/console-crm.html
git commit -m "feat(console): CRM queue Cancel/Retry + contact search type-ahead"
```

---

### Task 3: Dashboard → Receivables section + key-loss fix

**Files:**
- Modify: `static/dashboard.html` (`ACT_AREA` ~line 974; `actNavigate` ~line 987).
- Verify: headless Playwright (mocked).

**Interfaces:**
- Consumes: the Money board `#receivables` tab (from B1).
- Produces: `actNavigate` builds `/console/money?key=…#receivables` (key before the fragment).

- [ ] **Step 1: Point money-cash at the Receivables tab; drop redirecting targets**

In `static/dashboard.html`, change `ACT_AREA` (~line 974):
```javascript
const ACT_AREA = {
  "money-cash":       "/console/money#receivables",
  "clients-pipeline": "/console",
  "signals-patterns": "/console",
  "shaira-daily":     "/console",
};
```

- [ ] **Step 2: Fix `actNavigate` to place the key BEFORE the `#fragment`**

Replace `actNavigate` (~line 987):
```javascript
function actNavigate(el, e){
  if (e && e.target.closest(".act-menu, .act-panel")) return;  // menu interactions don't navigate
  var href = ACT_AREA[actSlug(el)] || "/console";
  var hash = "";
  var hi = href.indexOf("#");
  if (hi >= 0){ hash = href.slice(hi); href = href.slice(0, hi); }
  var sep = href.indexOf("?") >= 0 ? "&" : "?";
  location.assign(href + sep + "key=" + encodeURIComponent(consoleKey) + hash);
}
```

- [ ] **Step 3: Render-verify (headless, mocked)**

Start the app (PORT=5097). The dashboard gates on a console key + fetches many endpoints; to exercise `actNavigate` deterministically, render a money-cash act element, stub `location.assign`, click it, and assert the captured URL. Save `/tmp/cl-test/t3.py`:
```python
from playwright.sync_api import sync_playwright
import json
def h(route):
    u=route.request.url
    if "/api/intelligence/money-cash" in u:
        return route.fulfill(status=200, content_type="application/json", body=json.dumps({"slug":"money-cash","markdown":"## Money\n[HIGH] Invoice #1234 is 30 days overdue\n","generated_at":"2026-06-26","bytes":40}))
    if "/api/" in u:   # let the rest return empty-ish so the page boots without errors
        return route.fulfill(status=200, content_type="application/json", body=json.dumps({"ok":True,"data":{}}))
    return route.continue_()
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(); errs=[]
    pg.on("pageerror",lambda e:errs.append(str(e)))
    pg.route("**/api/**", h)
    pg.goto("http://127.0.0.1:5097/dashboard?key=test-secret", wait_until="networkidle"); pg.wait_for_timeout(1500)
    # stub navigation capture, then click a money-cash act element
    captured=pg.evaluate("""()=>{
      window.__nav=[]; try{ Object.defineProperty(window.location,'assign',{value:function(u){window.__nav.push(u)},configurable:true}); }catch(e){ window.location.assign=function(u){window.__nav.push(u)}; }
      // find a money-cash card's act element
      var card=document.querySelector('[data-endpoint=\"/api/intelligence/money-cash\"]');
      var act=card? card.querySelector('.act') : document.querySelector('.act');
      if(act){ act.click(); }
      return {found: !!act, nav: window.__nav};
    }""")
    print("captured:", captured, "errs:", errs or "NONE")
    # If the act element isn't rendered (intelligence card collapsed), fall back to asserting the ACT_AREA wiring via a synthetic click is acceptable; the key assertion:
    nav=captured.get("nav") or []
    assert any("/console/money?key=" in u and u.endswith("#receivables") for u in nav), nav
    assert not errs, errs
    b.close(); print("OK")
```
Run `python3 /tmp/cl-test/t3.py` → `OK`: clicking a money-cash alert navigates to `/console/money?key=<enc>#receivables` (key in the query, BEFORE the fragment). If the dashboard's intelligence card doesn't render the act element in the harness (lockout/collapse), instead build a synthetic element with `data-endpoint="/api/intelligence/money-cash"` containing a `.act`, append it, and click it to drive `actNavigate` — the assertion (the captured URL shape) is the same. Kill the server.

- [ ] **Step 4: Commit**
```bash
git add static/dashboard.html
git commit -m "feat(console): dashboard money alert -> Money/Receivables tab + key-before-fragment fix"
```

---

## Verification (whole sub-project)

- Render-verify all three: reveal card Open-Portal anchor + portal `?email=` auto-load; CRM Cancel/Retry post the right status + contact search fills `#email`; dashboard money-cash → `/console/money?key=…#receivables`. Zero JS errors throughout.
- No backend/schema change; the reveal/portal/queue/people endpoints + the Money board are untouched.

## Self-Review Notes

- **Spec coverage:** Reveals→Portal link + `?email=` auto-load (Task 1) ✓; CRM Cancel/Retry via `/api/ghl/queue/result` (Task 2) ✓; CRM contact search via `/api/people?q=` (Task 2) ✓; dashboard Receivables-tab deep-link + key-before-fragment fix (Task 3) ✓; record-level (3b) not built (deferred) ✓; pure-frontend, no backend change (all tasks touch only static HTML) ✓.
- **Type consistency:** `cancelQ(id)`/`retryQ(id)`/`_setQ(id,status)`/`crmSearch`/`crmAcHide` in Task 2; `actNavigate` hash-split in Task 3; numbers-not-objects in onclick.
- **YAGNI:** no new endpoints (reuse `/api/ghl/queue/result`, `/api/people?q=`); no record-level deep-links.
