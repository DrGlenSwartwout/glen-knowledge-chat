# BOS Phase 2b: Orders Board UI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A four-column Orders board (New -> Packed -> Shipped -> Done) where the operator works orders: each card shows customer/items/total/source/age, with buttons that advance the lifecycle through the audited dispatch path. Plus a "set tracking" action so marking shipped can record a tracking number.

**Architecture:** A list endpoint (`GET /api/orders`) returns orders from the Phase 2a `orders` table; a `/console/orders` page serves a vanilla-JS board that buckets orders by status and fires actions via the existing generic `/api/action/<key>` route. One new `orders.set_tracking` action (pure, in `dashboard/orders.py`). No new external dependency. The customer tracking-email send and EasyPost labels come in 2c.

**Builds on:** Phase 2a (`dashboard/orders.py`, the lifecycle actions, `/api/orders` POST) + 1a/1b spine. Same branch `sess/ec0e1f15`, worktree `/tmp/wt-deploy-chat-ec0e1f15`.

---

## File Structure

- `dashboard/orders.py` (modify): add the `orders.set_tracking` lifecycle action.
- `tests/test_bos_orders.py` (modify): add a dispatch test for it.
- `app.py` (modify): extend `/api/orders` to handle `GET` (list), add `/console/orders` page route.
- `static/console-orders.html` (new): the board UI.
- `tests/test_bos_routes.py` (modify): a `/console/orders` page-served test.

---

## Task 1: `orders.set_tracking` action (`dashboard/orders.py`)

**Files:**
- Modify: `dashboard/orders.py`
- Test: `tests/test_bos_orders.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_bos_orders.py`)

```python
def test_set_tracking_action_records_and_ships():
    from dashboard import orders as O, dispatch as D, events as E, rbac as R, actions as A
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    O.init_orders_table(cx)
    oid = O.upsert_order(cx, source="funnel", external_ref="T1")
    assert A.get_action("orders.set_tracking") is not None
    res = D.dispatch_action(cx, "orders.set_tracking",
                            {"order_id": oid, "tracking_number": "9400111899223"},
                            R.Actor(role=R.OWNER))
    assert res["status"] == "done"
    row = O.get_order(cx, oid)
    assert row["status"] == "shipped"
    assert row["tracking_number"] == "9400111899223"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_orders.py -k set_tracking -q`
Expected: FAIL (`assert A.get_action("orders.set_tracking") is not None` -> None).

- [ ] **Step 3: Append the action to `dashboard/orders.py`** (next to the other lifecycle actions)

```python
def _set_tracking_exec(params, ctx):
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection")
    oid = int(params["order_id"])
    tn = str(params.get("tracking_number", "")).strip()
    if not set_order_tracking(cx, oid, tn):
        raise ValueError(f"order #{oid} not found")
    set_order_status(cx, oid, "shipped")
    return {"order_id": oid, "tracking_number": tn, "status": "shipped",
            "message": f"Order #{oid} shipped" + (f" (tracking {tn})." if tn else ".")}


action(key="orders.set_tracking", module="orders", title="Set tracking + ship",
       description="Record a tracking number and mark the order shipped.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_set_tracking_exec)
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_orders.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/orders.py tests/test_bos_orders.py
git commit -m "feat(bos): orders.set_tracking action (record tracking + ship)"
```

---

## Task 2: List endpoint + page route (`app.py`)

**Files:**
- Modify: `app.py`
- Test: `tests/test_bos_routes.py` (append)

- [ ] **Step 1: Write the failing route tests** (append to `tests/test_bos_routes.py`)

```python
def test_orders_list_and_board(monkeypatch, tmp_path):
    app_module = _load_app()
    import sqlite3
    db = str(tmp_path / "o.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    from dashboard import orders as O
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    O.upsert_order(cx, source="funnel", external_ref="L1", email="a@b.com", total_cents=7000)
    cx.commit(); cx.close()

    client = app_module.app.test_client()
    key = app_module.dashboard.CONSOLE_SECRET or ""
    r = client.get("/api/orders", headers={"X-Console-Key": key})
    assert r.status_code == 200
    data = r.get_json()["data"]
    assert any(o["external_ref"] == "L1" for o in data)

    p = client.get("/console/orders")
    assert p.status_code == 200
    assert b"Orders" in p.data
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_routes.py -k "orders_list" -q`
Expected: FAIL if app imports (the GET returns 405 / the page 404), else SKIP.

- [ ] **Step 3: Extend the `/api/orders` route to handle GET**

Find `@app.route("/api/orders", methods=["POST"])` / `def bos_orders_create`. Change the decorator to `methods=["GET", "POST"]` and add a GET branch at the top of the function body, before the existing POST logic:

```python
@app.route("/api/orders", methods=["GET", "POST"])
def bos_orders_create():
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    if request.method == "GET":
        cx = _sqlite3.connect(LOG_DB)
        cx.row_factory = _sqlite3.Row
        try:
            rows = _bos_orders.list_orders(
                cx, status=request.args.get("status"),
                limit=min(int(request.args.get("limit", 200) or 200), 500))
        except (TypeError, ValueError):
            rows = _bos_orders.list_orders(cx)
        finally:
            cx.close()
        return jsonify({"ok": True, "data": rows})
    # --- existing POST body unchanged below ---
    b = request.get_json(silent=True) or {}
    ...
```

(Keep the existing POST body exactly as-is after the GET branch.)

- [ ] **Step 4: Add the `/console/orders` page route** (near `bos_home_page`)

```python
@app.route("/console/orders")
def bos_orders_page():
    resp = send_from_directory(STATIC, "console-orders.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp
```

- [ ] **Step 5: Create a minimal page so the page test passes**

Create `static/console-orders.html`:

```html
<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Orders - Console</title></head>
<body><h1>Orders</h1><div id="board"></div></body></html>
```

- [ ] **Step 6: Compile + verify**

Run: `python3 -m py_compile app.py` (OK).
Run under doppler:
```bash
doppler run -p remedy-match -c prd -- bash -c 'mkdir -p /tmp/bostest && DATA_DIR=/tmp/bostest python3 - <<PY
import app
c = app.app.test_client()
key = app.dashboard.CONSOLE_SECRET or ""
import json
# seed one order through the public model then list it
from dashboard import orders as O, sqlite3 as _; import sqlite3
cx = sqlite3.connect(app.LOG_DB); cx.row_factory=sqlite3.Row
O.upsert_order(cx, source="manual", external_ref="BOARD-1", email="z@z.com", total_cents=5000)
r = c.get("/api/orders", headers={"X-Console-Key": key})
assert r.status_code==200, r.status_code
assert any(o["external_ref"]=="BOARD-1" for o in r.get_json()["data"]), "order not listed"
p = c.get("/console/orders")
assert p.status_code==200
print("ORDERS_2B_OK", len(r.get_json()["data"]), "orders listed")
PY'
rm -rf /tmp/bostest
```
Expected: `ORDERS_2B_OK ...` no assertion error.

```bash
git add app.py static/console-orders.html tests/test_bos_routes.py
git commit -m "feat(bos): GET /api/orders + /console/orders board route"
```

---

## Task 3: The board UI (`static/console-orders.html`)

**Files:**
- Modify: `static/console-orders.html` (replace stub with the full board)

Verified by the page-served test (Task 2) + manual review.

- [ ] **Step 1: Replace `static/console-orders.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Orders - Business OS</title>
<style>
  :root { --bg:#0a150d; --surface:#111f16; --surface2:#162318; --border:#21472d;
          --cream:#fdf4d8; --muted:#a89870; --gold:#d4a843; --green:#3d8a52; --red:#c0432b; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--cream);
         font-family:"Open Sans",system-ui,sans-serif; padding:24px; }
  h1 { font-family:"Raleway",sans-serif; font-size:22px; margin-bottom:2px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:20px; }
  .cols { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; align-items:start; }
  .col { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:12px; }
  .col h2 { font-family:"Raleway",sans-serif; font-size:13px; letter-spacing:.08em;
            text-transform:uppercase; color:var(--muted); margin-bottom:10px; }
  .col h2 .n { color:var(--gold); }
  .card { background:var(--surface2); border:1px solid var(--border); border-radius:9px;
          padding:11px 12px; margin-bottom:10px; }
  .card .who { font-weight:600; font-size:14px; }
  .card .meta { color:var(--muted); font-size:12px; margin-top:3px; line-height:1.5; }
  .card .src { display:inline-block; font-size:10px; letter-spacing:.05em; text-transform:uppercase;
               color:var(--muted); border:1px solid var(--border); border-radius:5px; padding:1px 6px; }
  .card .age.old { color:var(--red); }
  .acts { margin-top:9px; display:flex; flex-wrap:wrap; gap:6px; }
  .acts button { font-size:11px; border:1px solid var(--border); background:transparent;
                 color:var(--cream); border-radius:6px; padding:3px 9px; cursor:pointer; }
  .acts button:hover { border-color:var(--gold); color:var(--gold); }
  .acts button.cancel:hover { border-color:var(--red); color:var(--red); }
  #gate { position:fixed; inset:0; background:var(--bg); display:flex; align-items:center; justify-content:center; }
  #gate input { padding:10px; border-radius:8px; border:1px solid var(--border); background:var(--surface); color:var(--cream); }
</style>
</head>
<body>
  <div id="gate"><div style="text-align:center">
    <p style="margin-bottom:10px;color:var(--muted)">Enter console key</p>
    <input id="key" type="password" placeholder="console key" />
    <button onclick="unlock()" style="padding:10px 14px;margin-left:6px;border-radius:8px;border:1px solid var(--border);background:var(--gold);color:#0a150d;cursor:pointer">Unlock</button>
  </div></div>

  <h1>Orders</h1>
  <div class="sub">Every order, from every channel, through one lifecycle.</div>
  <div class="cols" id="board"></div>

<script>
  var LANES = [["new","New"],["packed","Packed"],["shipped","Shipped"],["done","Done"]];
  function key(){ return localStorage.getItem('console_key') || ''; }
  function unlock(){ localStorage.setItem('console_key', document.getElementById('key').value); document.getElementById('gate').style.display='none'; load(); }
  function hdr(){ return { 'X-Console-Key': key(), 'Content-Type':'application/json' }; }
  function esc(s){ return String(s==null?'':s).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];}); }

  function ageStr(iso){
    try { var h = (Date.now() - new Date(iso).getTime())/3600000;
      if (h < 1) return Math.round(h*60)+'m'; if (h < 48) return Math.round(h)+'h'; return Math.round(h/24)+'d'; }
    catch(e){ return ''; }
  }
  function isOld(iso){ try { return (Date.now()-new Date(iso).getTime())/3600000 > 24; } catch(e){ return false; } }

  function cardHtml(o){
    var total = o.total_cents ? '$'+(o.total_cents/100).toFixed(2) : '';
    var items = (o.items||[]).map(function(i){ return (i.qty?i.qty+'x ':'')+esc(i.name||''); }).join(', ');
    var age = ageStr(o.created_at);
    var acts = '';
    if (o.status==='new')     acts = btn(o.id,'orders.mark_packed','Pack');
    else if (o.status==='packed') acts = shipBtn(o.id);
    else if (o.status==='shipped') acts = btn(o.id,'orders.mark_done','Mark delivered');
    if (o.status!=='done' && o.status!=='cancelled') acts += ' <button class="cancel" onclick="act('+Number(o.id)+',\'orders.cancel\',{})">Cancel</button>';
    return '<div class="card"><div class="who">'+esc(o.name||o.email||'Order #'+o.id)+'</div>'
      + '<div class="meta">'+(items?esc(items)+'<br>':'')
      + (total?total+' · ':'')+'<span class="src">'+esc(o.source)+'</span> · <span class="age'+(isOld(o.created_at)?' old':'')+'">'+age+'</span>'
      + (o.tracking_number?'<br>tracking '+esc(o.tracking_number):'')+'</div>'
      + '<div class="acts">'+acts+'</div></div>';
  }
  function btn(id,key,label){ return '<button onclick="act('+Number(id)+',\''+key+'\',{})">'+label+'</button>'; }
  function shipBtn(id){ return '<button onclick="ship('+Number(id)+')">Ship + tracking</button>'; }

  async function load(){
    var r = await fetch('/api/orders?limit=300', {headers:hdr()});
    if (r.status===401){ document.getElementById('gate').style.display='flex'; return; }
    var orders = (await r.json()).data || [];
    var board = document.getElementById('board'); board.innerHTML='';
    LANES.forEach(function(l){
      var inLane = orders.filter(function(o){ return o.status===l[0]; });
      var col = document.createElement('div'); col.className='col';
      col.innerHTML = '<h2>'+l[1]+' <span class="n">'+inLane.length+'</span></h2>'
        + inLane.map(cardHtml).join('');
      board.appendChild(col);
    });
  }

  async function act(id, key, params){
    params = params || {}; params.order_id = id;
    await fetch('/api/action/'+key, {method:'POST', headers:hdr(), body:JSON.stringify(params)});
    load();
  }
  function ship(id){
    var tn = prompt('Tracking number (optional):','');
    act(id, 'orders.set_tracking', {tracking_number: tn||''});
  }

  if (key()) { document.getElementById('gate').style.display='none'; load(); }
</script>
</body>
</html>
```

- [ ] **Step 2: Verify page serves**

Run: `python3 -m pytest tests/test_bos_routes.py -k "orders_list" -q`
Expected: PASS (or SKIP locally).

- [ ] **Step 3: Commit**

```bash
git add static/console-orders.html
git commit -m "feat(bos): orders board UI (4-lane lifecycle + actions)"
```

---

## Self-Review

**Spec coverage** (the board half of blueprint 5.4):
- Four-column New/Packed/Shipped/Done board -> Task 3.
- Cards show customer/items/total/source/age -> Task 3 `cardHtml`.
- Buttons advance lifecycle via the audited `/api/action/<key>` path -> Task 3 `act()` + the Phase 2a actions.
- Record a tracking number when shipping -> Task 1 `orders.set_tracking` + Task 3 `ship()`.
- List endpoint -> Task 2 `GET /api/orders`.

**Out of scope (2c):** the customer tracking-email send (Gmail), linking + advancing the `shipments` table, and EasyPost label buying + auto-tracking (gated on `EASYPOST_API_KEY`).

**Placeholder scan:** none. The stub page in Task 2 Step 5 is explicitly replaced in Task 3.

**Type consistency:** the order JSON keys (`status`, `items`, `total_cents`, `source`, `created_at`, `tracking_number`, `id`), the action keys (`orders.mark_packed/set_tracking/mark_done/cancel`), and the `/api/action/<key>` + `{order_id}` contract are consistent across Tasks 1-3 and Phase 1a/2a.
