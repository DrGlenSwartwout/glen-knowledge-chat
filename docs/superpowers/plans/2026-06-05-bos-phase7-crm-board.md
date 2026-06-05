# BOS Phase 7: CRM board UI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A `/console/crm` board that surfaces the new GHL-write actions (which currently have no UI) and the GHL write-queue status, so the operator can tag a contact, log a note, create an opportunity, or enroll a contact in the onboarding workflow from one place, and see what's queued to sync. The people directory and household/merge review already live in `/console` and are NOT duplicated here.

**Architecture:** A `/console/crm` page route serves a vanilla-JS board (same style/auth-gate as `console-orders.html` / `console-finance.html`). The action panel dispatches the existing `crm.*` enqueue actions via the generic `/api/action/<key>`; the queue panel reads the existing `GET /api/ghl/queue/pending`. No new backend logic.

**Builds on:** the merged Business OS (CRM signal + GHL write-queue + drain). New branch `sess/ec0e1f15` off main, worktree `/tmp/wt-deploy-chat-ec0e1f15`.

---

## File Structure

- `app.py` (modify): add the `/console/crm` page route.
- `static/console-crm.html` (new): the board.
- `tests/test_bos_routes.py` (modify): a page-served test.

---

## Task 1: Page route + stub + test (`app.py`, `static/console-crm.html`)

**Files:**
- Modify: `app.py`
- Create: `static/console-crm.html` (stub)
- Test: `tests/test_bos_routes.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_bos_routes.py`)

```python
def test_crm_page_served(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "c.db"))
    client = app_module.app.test_client()
    r = client.get("/console/crm")
    assert r.status_code == 200
    assert b"CRM" in r.data
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_routes.py -k crm_page -q`
Expected: FAIL if app imports (404), else SKIP.

- [ ] **Step 3: Add the page route** (near `bos_orders_page` / `bos_finance_*`)

```python
@app.route("/console/crm")
def bos_crm_page():
    resp = send_from_directory(STATIC, "console-crm.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp
```

- [ ] **Step 4: Create the stub** `static/console-crm.html`:

```html
<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>CRM - Console</title></head>
<body><h1>CRM</h1></body></html>
```

- [ ] **Step 5: Compile + commit**

Run: `python3 -m py_compile app.py` (OK).

```bash
git add app.py static/console-crm.html tests/test_bos_routes.py
git commit -m "feat(bos): /console/crm page route"
```

---

## Task 2: The board (`static/console-crm.html`)

**Files:**
- Modify: `static/console-crm.html` (replace stub)

Verified by the page-served test + manual review.

- [ ] **Step 1: Replace `static/console-crm.html`** with the board:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>CRM - Business OS</title>
<style>
  :root { --bg:#0a150d; --surface:#111f16; --surface2:#162318; --border:#21472d;
          --cream:#fdf4d8; --muted:#a89870; --gold:#d4a843; --green:#3d8a52; --red:#c0432b; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--cream);
         font-family:"Open Sans",system-ui,sans-serif; padding:24px; max-width:860px; margin:0 auto; }
  h1 { font-family:"Raleway",sans-serif; font-size:22px; margin-bottom:2px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:20px; }
  .panel { background:var(--surface); border:1px solid var(--border); border-radius:12px;
           padding:18px; margin-bottom:16px; }
  .panel h2 { font-family:"Raleway",sans-serif; font-size:15px; margin-bottom:12px; }
  label { display:block; font-size:12px; color:var(--muted); margin:10px 0 4px; }
  input, textarea { width:100%; padding:9px 11px; background:var(--bg); color:var(--cream);
                    border:1px solid var(--border); border-radius:8px; font-size:14px; font-family:inherit; }
  textarea { min-height:54px; resize:vertical; }
  .actions { margin-top:14px; display:flex; flex-wrap:wrap; gap:8px; }
  .actions button { font-size:13px; border:1px solid var(--border); background:transparent;
                    color:var(--cream); border-radius:7px; padding:7px 13px; cursor:pointer; }
  .actions button:hover { border-color:var(--gold); color:var(--gold); }
  #result { margin-top:12px; font-size:13px; color:var(--green); min-height:18px; }
  #result.err { color:var(--red); }
  .row { display:flex; justify-content:space-between; align-items:center;
         border-bottom:1px solid var(--border); padding:8px 0; font-size:13px; }
  .row:last-child { border-bottom:none; }
  .row .op { color:var(--gold); font-size:11px; letter-spacing:.05em; text-transform:uppercase;
             border:1px solid var(--border); border-radius:5px; padding:1px 7px; margin-right:8px; }
  .row .meta { color:var(--muted); }
  .note { color:var(--muted); font-size:12px; margin-top:10px; }
  #gate { position:fixed; inset:0; background:var(--bg); display:flex; align-items:center; justify-content:center; }
  #gate input { max-width:280px; }
</style>
</head>
<body>
  <div id="gate"><div style="text-align:center">
    <p style="margin-bottom:10px;color:var(--muted)">Enter console key</p>
    <input id="key" type="password" placeholder="console key" />
    <button onclick="unlock()" style="padding:10px 14px;margin-left:6px;border-radius:8px;border:1px solid var(--border);background:var(--gold);color:#0a150d;cursor:pointer">Unlock</button>
  </div></div>

  <h1>CRM</h1>
  <div class="sub">Act on a contact in GHL. Writes queue here and push automatically every few minutes. (People directory and household merges live in the main console.)</div>

  <div class="panel">
    <h2>Act on a contact</h2>
    <label for="email">Contact email</label>
    <input id="email" type="email" placeholder="person@example.com" />
    <label for="tag">Tag (for Add tag)</label>
    <input id="tag" type="text" placeholder="e.g. warm-lead" />
    <label for="note">Note (for Log note)</label>
    <textarea id="note" placeholder="What happened / next step"></textarea>
    <div class="actions">
      <button onclick="doAction('crm.add_tag', {tag:val('tag')})">Add tag</button>
      <button onclick="doAction('crm.log_outreach', {note:val('note')})">Log note</button>
      <button onclick="doAction('crm.create_opportunity', {})">Create opportunity</button>
      <button onclick="doAction('crm.enroll_workflow', {})">Enroll in onboarding</button>
    </div>
    <div id="result"></div>
  </div>

  <div class="panel">
    <h2>GHL sync queue <span id="qcount" class="meta" style="font-size:12px"></span></h2>
    <div id="queue"></div>
    <div class="note">Queued writes push to GHL automatically every 5 minutes via the local drain. A failed item stays visible.</div>
  </div>

<script>
  function key(){ return localStorage.getItem('console_key') || ''; }
  function unlock(){ localStorage.setItem('console_key', document.getElementById('key').value); document.getElementById('gate').style.display='none'; loadQueue(); }
  function hdr(){ return { 'X-Console-Key': key(), 'Content-Type':'application/json' }; }
  function esc(s){ return String(s==null?'':s).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];}); }
  function val(id){ return document.getElementById(id).value.trim(); }

  function show(msg, isErr){
    var r = document.getElementById('result');
    r.textContent = msg; r.className = isErr ? 'err' : '';
  }

  async function doAction(actionKey, extra){
    var email = val('email');
    if (!email){ show('Enter a contact email first.', true); return; }
    var params = Object.assign({email: email}, extra || {});
    try {
      var res = await fetch('/api/action/'+actionKey, {method:'POST', headers:hdr(), body:JSON.stringify(params)});
      if (res.status === 401){ document.getElementById('gate').style.display='flex'; return; }
      var body = await res.json();
      if (body.status === 'needs_confirmation'){
        if (!confirm(body.summary || 'Confirm this action?')) return;
        params.confirmed = true;
        body = await (await fetch('/api/action/'+actionKey, {method:'POST', headers:hdr(), body:JSON.stringify(params)})).json();
      }
      if (body.status === 'done'){ show((body.result && body.result.message) || 'Done.', false); }
      else if (body.status === 'denied'){ show('Not permitted.', true); }
      else { show('Error: ' + (body.error || body.status), true); }
      loadQueue();
    } catch(e){ show('Error: ' + e, true); }
  }

  async function loadQueue(){
    try {
      var res = await fetch('/api/ghl/queue/pending', {headers:hdr()});
      if (res.status === 401){ document.getElementById('gate').style.display='flex'; return; }
      var data = await res.json();
      var items = data.queue || [];
      document.getElementById('qcount').textContent = items.length ? '('+items.length+' pending)' : '(clear)';
      var host = document.getElementById('queue');
      if (!items.length){ host.innerHTML = '<div class="row meta">Nothing waiting to sync.</div>'; return; }
      host.innerHTML = items.map(function(q){
        return '<div class="row"><span><span class="op">'+esc(q.op)+'</span>'+esc(q.email)+'</span>'
             + '<span class="meta">'+esc((q.created_at||'').slice(0,16).replace('T',' '))+'</span></div>';
      }).join('');
    } catch(e){ /* leave as-is */ }
  }

  if (key()) { document.getElementById('gate').style.display='none'; loadQueue(); }
</script>
</body>
</html>
```

- [ ] **Step 2: Verify the page serves + parses**

Run: `python3 -c "import html.parser; html.parser.HTMLParser().feed(open('static/console-crm.html').read()); print('parsed OK')"`
Run: `python3 -m pytest tests/test_bos_routes.py -k crm_page -q` (PASS or SKIP locally).
Confirm: no em dashes; every dynamic queue field escaped with `esc()`.

- [ ] **Step 3: Commit**

```bash
git add static/console-crm.html
git commit -m "feat(bos): CRM board UI (contact GHL actions + sync queue)"
```

---

## Self-Review

**Spec coverage:** a CRM board that surfaces the GHL-write actions (tag/note/opportunity/workflow) with a contact-action panel, plus the GHL sync-queue status. Reuses the existing `crm.*` actions + `/api/ghl/queue/pending`; no backend changes beyond the page route.

**Non-duplication:** the people directory + household/merge review stay in `/console` (the board says so). The board adds the action surface those actions previously lacked.

**Out of scope (future):** a leads list (needs a console-keyed leads endpoint), inline dedup actions, queue history (done/failed), per-contact lookup/autocomplete.

**Placeholder scan:** none (the Task 1 stub is explicitly replaced in Task 2).

**Type consistency:** the action keys (`crm.add_tag`/`log_outreach`/`create_opportunity`/`enroll_workflow`), the `{email, ...}` params, the `/api/action/<key>` + `needs_confirmation` contract, and the `/api/ghl/queue/pending` `{queue:[{op,email,created_at}]}` shape match the merged backend.
```
