# Approvals Hub (Sub-project B3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `/console/approvals` triage hub (6 queue cards with live pending counts + Open links) and group the six approval queues under one op-nav **Approvals** entry, de-orphaning the five that have no nav.

**Architecture:** A new `static/console-approvals.html` fetches each queue's existing pending-list endpoint in parallel (fault-isolated per card) and renders a card grid; op-nav collapses the queue entries to one Approvals board and the five bare admin pages get the OPS bar added. No backend/data change, no page merging, no deletions.

**Tech Stack:** Vanilla JS / static HTML, Flask route (`app.py`), headless Playwright render-verify (with mocked endpoint data). No backend change.

## Global Constraints

- **No backend/data change; no queue UI merged; no page deleted.** Each queue's own page, list endpoint, and actions are untouched. This is a hub page + nav grouping only.
- Hub fetches use the **`X-Console-Key` header**; **Open** links carry **`?key=<key>`** (the `/admin/*` queue pages read the key from the URL). Counts are **fault-isolated** — one queue's fetch failing/401 shows "—" on that card and must not blank the others.
- Per-queue pending count sources (exact):
  - reviews `GET /api/console/reviews` → `pending.length`
  - atlas `GET /admin/atlas/pending` → `(data.concepts || concepts).length`
  - clips `GET /admin/clips/pending` → `(data.clips || clips).length`
  - wholesale `GET /admin/wholesale/pending` → `(data.applications || applications).length`
  - cert `GET /api/cert/review/list?status=submitted` → `submissions.length`
  - studio `GET /api/console/studio-credits` → `claims` with a pending status
- Board "Approvals", BOS `data-sub="approvals"`. `/admin/membership` is NOT included.
- Console-key gated (and Rae's OWNER token via sub-project A). Render-verify is the core gate.
- **Test env:** run the local server via `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/<scratch> CONSOLE_SECRET=test-secret PORT=<p> python3 app.py` (mkdir scratch first).

---

### Task 1: Build `console-approvals.html` (the hub) + route

**Files:**
- Create: `static/console-approvals.html`
- Modify: `app.py` — add `@app.route("/console/approvals")` next to another `/console/*` page route (search the file for `console-reviews.html`).
- Verify: headless Playwright with mocked endpoint data.

**Interfaces:**
- Consumes: the six existing pending-list endpoints (unchanged).
- Produces: `GET /console/approvals`; the `QUEUES` config + render.

- [ ] **Step 1: Create the hub page**

Create `static/console-approvals.html`:

```html
<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Approvals · Console</title>
<style>
  body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg,#0a150d);color:var(--cream,#fdf4d8)}
  .wrap{max-width:1000px;margin:0 auto;padding:20px}
  h1{font-size:20px;margin:0 0 4px}
  .sub{color:var(--muted,#a89870);font-size:13px;margin-bottom:16px}
  #gate{display:flex;gap:8px;align-items:center;padding:20px}
  #gate input{padding:8px 10px;border-radius:6px;border:1px solid var(--border,#21472d);background:#0d1117;color:var(--cream,#fdf4d8)}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}
  .qcard{background:var(--surface,#111f16);border:1px solid var(--border,#21472d);border-radius:12px;padding:16px;display:flex;flex-direction:column;gap:8px}
  .qhead{display:flex;align-items:center;justify-content:space-between}
  .qname{font-weight:700;font-size:15px}
  .qbadge{min-width:24px;text-align:center;background:var(--gold,#d4a843);color:#12130f;border-radius:12px;padding:2px 9px;font-weight:700;font-size:13px}
  .qbadge.zero{background:transparent;color:var(--muted,#a89870);border:1px solid var(--border,#21472d)}
  .qdesc{color:var(--muted,#a89870);font-size:12px;flex:1}
  .qcard a.btn{align-self:flex-start;background:transparent;border:1px solid var(--border,#21472d);color:var(--cream,#fdf4d8);text-decoration:none;border-radius:8px;padding:6px 12px;font-size:13px}
  .qcard a.btn:hover{border-color:var(--gold,#d4a843)}
  .toolbar{display:flex;justify-content:flex-end;margin-bottom:10px}
  .toolbar button{background:transparent;border:1px solid var(--border,#21472d);color:var(--muted,#a89870);border-radius:6px;padding:5px 12px;cursor:pointer}
</style>
</head><body>
<script src="/static/op-nav.js" data-active="bos" data-sub="approvals"></script>
<div id="gate"><input id="key" type="password" placeholder="Console key"><button onclick="unlock()">Unlock</button></div>
<div id="app" class="wrap" style="display:none">
  <h1>Approvals</h1>
  <div class="sub">Everything pending a decision, across queues.</div>
  <div class="toolbar"><button onclick="loadAll()">Refresh</button></div>
  <div class="grid" id="grid"></div>
</div>
<script>
  function key(){ return localStorage.getItem('console_key') || ''; }
  (function(){ var u=new URLSearchParams(location.search).get('key'); if(u) localStorage.setItem('console_key', u); })();
  function hdr(){ return { 'X-Console-Key': key() }; }
  function unlock(){ localStorage.setItem('console_key', document.getElementById('key').value); document.getElementById('gate').style.display='none'; document.getElementById('app').style.display=''; loadAll(); }
  function esc(s){ return String(s==null?'':s).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];}); }

  function isPendingClaim(c){
    var s = String((c && c.status) || '').toLowerCase();
    return s !== 'approved' && s !== 'rejected' && s !== 'granted';   // confirm against console-studio-credits.html's own pending filter
  }
  var QUEUES = [
    { id:'reviews',   name:'Reviews',        desc:'Product review moderation + gift approvals', route:'/console/reviews',
      endpoint:'/api/console/reviews',                  count:function(j){ return (j.pending||[]).length; } },
    { id:'atlas',     name:'Atlas',          desc:'Affiliate / Atlas concept applications',     route:'/admin/atlas',
      endpoint:'/admin/atlas/pending',                  count:function(j){ return (((j.data||{}).concepts)||j.concepts||[]).length; } },
    { id:'clips',     name:'Clips',          desc:'Content clip moderation',                    route:'/admin/clips',
      endpoint:'/admin/clips/pending',                  count:function(j){ return (((j.data||{}).clips)||j.clips||[]).length; } },
    { id:'wholesale', name:'Wholesale',      desc:'Wholesale account applications',             route:'/admin/wholesale',
      endpoint:'/admin/wholesale/pending',              count:function(j){ return (((j.data||{}).applications)||j.applications||[]).length; } },
    { id:'cert',      name:'Cert',           desc:'ASH certification submissions',              route:'/console/cert',
      endpoint:'/api/cert/review/list?status=submitted', count:function(j){ return (j.submissions||[]).length; } },
    { id:'studio',    name:'Studio Credits', desc:'ThisStudio free-month claims',               route:'/console/studio-credits',
      endpoint:'/api/console/studio-credits',           count:function(j){ return (j.claims||[]).filter(isPendingClaim).length; } }
  ];

  function card(q){
    var k = key();
    var href = q.route + '?key=' + encodeURIComponent(k);
    return '<div class="qcard" id="card-'+q.id+'">'
      + '<div class="qhead"><span class="qname">'+esc(q.name)+'</span><span class="qbadge" id="badge-'+q.id+'">…</span></div>'
      + '<div class="qdesc">'+esc(q.desc)+'</div>'
      + '<a class="btn" href="'+href+'">Open</a>'
    + '</div>';
  }

  function loadAll(){
    document.getElementById('grid').innerHTML = QUEUES.map(card).join('');
    QUEUES.forEach(function(q){
      fetch(q.endpoint, { headers: hdr() })
        .then(function(r){ if(!r.ok) throw 0; return r.json(); })
        .then(function(j){
          var n = q.count(j) || 0;
          var b = document.getElementById('badge-'+q.id);
          if(b){ b.textContent = String(n); b.className = 'qbadge' + (n===0 ? ' zero' : ''); }
        })
        .catch(function(){ var b = document.getElementById('badge-'+q.id); if(b){ b.textContent = '—'; b.className='qbadge zero'; } });
    });
  }

  if(key()){ document.getElementById('gate').style.display='none'; document.getElementById('app').style.display=''; loadAll(); }
</script>
</body></html>
```
(Before finalizing `isPendingClaim`, read `static/console-studio-credits.html` and match its own notion of a pending/un-actioned claim — adjust the status check if its statuses differ.)

- [ ] **Step 2: Route**

In `app.py`, immediately after the `/console/reviews` route, add:
```python
@app.route("/console/approvals")
def bos_approvals_page():
    resp = send_from_directory(STATIC, "console-approvals.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp
```

- [ ] **Step 3: Render-verify (headless, mocked data)**

`mkdir -p /tmp/appr-test`; start the app on PORT=5097. Save `/tmp/appr-test/av.py`:
```python
from playwright.sync_api import sync_playwright
import json
B="http://127.0.0.1:5097/console/approvals?key=test-secret"
MOCK = {
  "/api/console/reviews": {"pending":[1,2,3]},
  "/admin/atlas/pending": {"data":{"concepts":[1,2]}},
  "/admin/clips/pending": {"data":{"clips":[1]}},
  "/admin/wholesale/pending": {"data":{"applications":[]}},
  "/api/cert/review/list": {"submissions":[1,2,3,4]},
  "/api/console/studio-credits": {"claims":[{"status":"pending"},{"status":"approved"},{"status":"pending"}]},
}
def handle(route):
    u=route.request.url.split("?")[0]
    path=("/"+u.split("//",1)[1].split("/",1)[1]) if "//" in u else u
    for k,v in MOCK.items():
        if path==k: return route.fulfill(status=200, content_type="application/json", body=json.dumps(v))
    if path=="/admin/wholesale/pending2": pass
    return route.continue_()
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(); errs=[]
    pg.on("pageerror",lambda e:errs.append(str(e)))
    pg.on("console", lambda m: errs.append("CJS:"+m.text) if (m.type=="error" and "Failed to load resource" not in m.text) else None)
    # mock everything except the page itself + op-nav assets
    pg.route("**/api/console/reviews", handle); pg.route("**/admin/atlas/pending", handle)
    pg.route("**/admin/clips/pending", handle); pg.route("**/admin/wholesale/pending", handle)
    pg.route("**/api/cert/review/list**", handle); pg.route("**/api/console/studio-credits", handle)
    pg.goto(B, wait_until="networkidle"); pg.wait_for_timeout(1200)
    s=pg.evaluate("""()=>({cards:document.querySelectorAll('.qcard').length,
      badges:Object.fromEntries([...document.querySelectorAll('.qbadge')].map(b=>[b.id, b.textContent])),
      reviewsHref:(document.querySelector('#card-reviews a.btn')||{}).getAttribute?document.querySelector('#card-reviews a.btn').getAttribute('href'):null})""")
    print("HUB:", s); print("errs:", errs or "NONE")
    assert s["cards"]==6
    assert s["badges"]["badge-reviews"]=="3" and s["badges"]["badge-atlas"]=="2" and s["badges"]["badge-clips"]=="1"
    assert s["badges"]["badge-wholesale"]=="0" and s["badges"]["badge-cert"]=="4" and s["badges"]["badge-studio"]=="2"
    assert "?key=test-secret" in s["reviewsHref"] and s["reviewsHref"].startswith("/console/reviews")
    assert not errs, errs
    b.close(); print("OK")
```
Run `python3 /tmp/appr-test/av.py` → `OK`, `errs: NONE`: 6 cards; counts reviews=3, atlas=2, clips=1, wholesale=0, cert=4, studio=2 (pending only); the reviews Open href is `/console/reviews?key=test-secret`. Then a fault-isolation check: leave one endpoint un-mocked (so it hits the real empty/erroring server) and confirm the other cards still get their counts and the failing one shows "—" (re-run with that one route removed). Kill the server.

- [ ] **Step 4: Commit**
```bash
git add static/console-approvals.html app.py
git commit -m "feat(console): Approvals hub — /console/approvals with live per-queue pending counts"
```

---

### Task 2: Nav grouping — Approvals in op-nav + tag the 6 queue pages

**Files:**
- Modify: `static/op-nav.js` (collapse queue entries to one Approvals board).
- Modify: `static/console-reviews.html` (re-tag), and add an `op-nav.js` tag to `static/admin-atlas.html`, `static/admin-clips.html`, `static/admin-wholesale.html`, `static/console-cert.html`, `static/console-studio-credits.html`.
- Verify: headless nav render.

**Interfaces:**
- Consumes: Task 1's `/console/approvals`.
- Produces: op-nav shows one **Approvals** board; the 6 queue pages highlight it.

- [ ] **Step 1: Collapse op-nav**

In `static/op-nav.js` `bosMods`: replace the `reviews` entry with `{ id:"approvals", label:"Approvals", href:"/console/approvals" + qs }`, and remove the entries with ids `atlas`, `clips`, `wholesale`, `cert`, `studio-credits` (5 entries). In `NAV_PROFILES.glen.bos`, replace `"reviews"` with `"approvals"` (the five orphan ids are not in `glen.bos` — they're in the owner-More group, so removing them from `bosMods` suffices). `rae.bos` unchanged. Run `node --check static/op-nav.js`.

- [ ] **Step 2: Re-tag reviews + add op-nav to the 5 bare pages**

In `static/console-reviews.html`, change its existing `op-nav.js` script tag's `data-sub="reviews"` to `data-sub="approvals"`.

In each of `static/admin-atlas.html`, `static/admin-clips.html`, `static/admin-wholesale.html`, `static/console-cert.html`, `static/console-studio-credits.html` — which have **no** `op-nav.js` tag — add this line immediately after the opening `<body>` tag (match the placement other console pages use):
```html
<script src="/static/op-nav.js" data-active="bos" data-sub="approvals"></script>
```
Do not change any other markup or the pages' own scripts.

- [ ] **Step 3: Render-verify nav + the tagged pages**

Start the app (PORT=5098). Verify the BOS row:
```python
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(); errs=[]
    pg.on("pageerror",lambda e:errs.append(str(e)))
    pg.goto("http://127.0.0.1:5098/console/approvals?key=test-secret", wait_until="networkidle"); pg.wait_for_timeout(900)
    ids=pg.evaluate("()=>[...document.querySelectorAll('.op-nav-sub a.op-nav-subtab, #op-nav-more-bos .op-nav-more-menu a')].map(a=>a.dataset.id)")
    print("BOS ids:", ids)
    assert "approvals" in ids
    assert not any(x in ids for x in ["reviews","atlas","clips","wholesale","cert","studio-credits"])
    assert not errs, errs
    # each newly-tagged page renders the OPS bar with Approvals highlighted, zero JS errors
    for route in ["/admin/atlas","/admin/clips","/admin/wholesale","/console/cert","/console/studio-credits","/console/reviews"]:
        e2=[]; pg.on("pageerror", lambda e: e2.append(str(e)))
        pg.goto("http://127.0.0.1:5098"+route+"?key=test-secret", wait_until="networkidle"); pg.wait_for_timeout(700)
        info=pg.evaluate("""()=>({opnav:!!document.querySelector('.op-nav-bar'),
          active:(document.querySelector('.op-nav-sub a.op-nav-subtab.active')||{}).dataset?.id})""")
        print(route, info)
        assert info["opnav"] and info["active"]=="approvals"
    print("errs:", errs or "NONE"); assert not errs, errs
    b.close(); print("OK")
```
Expected: BOS shows `approvals`, the 6 old queue ids gone; each of the 6 queue pages renders the OPS bar with the Approvals sub-tab active; zero JS errors. (The admin pages may also fire their own data fetches that error on a fresh DB — that's a network 'Failed to load resource', not a JS error; the assertion only fails on pageerror.) Kill the server.

- [ ] **Step 4: Commit**
```bash
git add static/op-nav.js static/console-reviews.html static/admin-atlas.html static/admin-clips.html static/admin-wholesale.html static/console-cert.html static/console-studio-credits.html
git commit -m "feat(console): group the 6 approval queues under one Approvals board"
```

---

## Verification (whole sub-project)

- `/console/approvals` 200; hub renders 6 cards with correct per-queue counts (mocked), Open links carry `?key=`, a failing queue shows "—" without breaking the others, zero JS errors.
- `node --check static/op-nav.js`; BOS shows one **Approvals**, the 6 old queue ids gone; each queue page renders the OPS bar with Approvals active.
- No backend/data change; no queue page's own logic altered (only an additive `op-nav.js` tag); no page deleted.

## Self-Review Notes

- **Spec coverage:** hub page + counts + Open-with-key (Task 1) ✓; route (T1 S2) ✓; fault-isolation (T1 loadAll per-queue catch) ✓; op-nav Approvals-replaces-reviews + 5 orphans removed (T2 S1) ✓; tag the 6 pages incl. adding op-nav to the 5 bare (T2 S2) ✓; per-queue count sources match the spec table (T1 QUEUES) ✓; membership excluded (not referenced) ✓.
- **Type consistency:** `QUEUES[].{id,name,desc,route,endpoint,count}` used consistently; `badge-<id>`/`card-<id>` ids; `data-sub="approvals"` everywhere.
- **YAGNI:** no merged UI, no backend, no persistent op-nav sub-row; queues kept as-is.
