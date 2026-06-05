# BOS Phase 1b: Home Signal Board Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the Home "mission-control" signal board: a 9-cell grid (one per module) showing a color-coded priority signal, a one-line summary, and the top actions needed now, backed by a per-module `signal()` contract and a `/api/home/signals` aggregation.

**Architecture:** A new `dashboard/signals.py` holds the signal contract + a registry of per-module signal functions + the aggregation. It builds on the Phase 1a spine: the `events` table feeds a cross-cutting "pending approvals" overlay so any module with queued actions lights up, even modules whose own `signal()` is not wired yet (they default to gray). Phase 1b ships one real signal (Tasks, from the `todos` table); the other eight modules default to gray and gain real signals in their own phases. A `/console/home` page renders the grid client-side and calls `/api/home/signals` + the existing `/api/events`.

**Tech Stack:** Python 3, Flask, sqlite3, pytest; vanilla JS + HTML for the board (matches `static/console.html` / `static/begin-explore.html` style).

**Builds on:** Phase 1a spine (committed: `dashboard/actions.py`, `rbac.py`, `events.py`, `dispatch.py`, the `events` table, `/api/events`). Same branch `sess/ec0e1f15`, worktree `/tmp/wt-deploy-chat-ec0e1f15`.

---

## File Structure

- `dashboard/signals.py` (new): color constants, `worst_level`, the module list + titles, `@signal` registry, `aggregate_signals`, the Tasks signal, and the pending-approval overlay. One responsibility: computing the Home board state.
- `tests/test_bos_signals.py` (new): unit tests for the signal logic.
- `app.py` (modify): `/api/home/signals` route + `/console/home` page route; import `dashboard.signals` at startup so its registrations load.
- `static/console-home.html` (new): the 9-cell grid UI + pending-approvals + recent-activity, calling `/api/home/signals` and `/api/events`.
- `tests/test_bos_routes.py` (modify): add a `/api/home/signals` route test.

---

## Task 1: Signal contract + registry + aggregation (`dashboard/signals.py`)

**Files:**
- Create: `dashboard/signals.py`
- Test: `tests/test_bos_signals.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bos_signals.py`:

```python
import sqlite3
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _db():
    from dashboard import events as E
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    cx.execute("CREATE TABLE todos (id INTEGER PRIMARY KEY, status TEXT, priority TEXT)")
    cx.commit()
    return cx


def test_worst_level_ordering():
    from dashboard import signals as S
    assert S.worst_level([S.GRAY, S.GREEN, S.AMBER]) == S.AMBER
    assert S.worst_level([S.GREEN, S.RED, S.AMBER]) == S.RED
    assert S.worst_level([]) == S.GRAY
    assert S.worst_level([S.GRAY, S.GREEN]) == S.GREEN


def test_aggregate_returns_nine_cells_in_order():
    from dashboard import signals as S
    cx = _db()
    cells = S.aggregate_signals(cx, actor=None)
    assert [c["module"] for c in cells] == list(S.MODULES)
    for c in cells:
        assert set(("module", "title", "level", "summary", "top_actions", "count")) <= set(c)
        assert c["level"] in (S.RED, S.AMBER, S.GREEN, S.GRAY)


def test_tasks_signal_levels():
    from dashboard import signals as S
    cx = _db()
    # no open todos -> green
    cells = {c["module"]: c for c in S.aggregate_signals(cx, actor=None)}
    assert cells["tasks"]["level"] == S.GREEN
    # an open normal todo -> amber
    cx.execute("INSERT INTO todos (status, priority) VALUES ('open','normal')")
    cx.commit()
    cells = {c["module"]: c for c in S.aggregate_signals(cx, actor=None)}
    assert cells["tasks"]["level"] == S.AMBER
    assert cells["tasks"]["count"] == 1
    # an open high-priority todo -> red
    cx.execute("INSERT INTO todos (status, priority) VALUES ('open','high')")
    cx.commit()
    cells = {c["module"]: c for c in S.aggregate_signals(cx, actor=None)}
    assert cells["tasks"]["level"] == S.RED


def test_unwired_module_defaults_gray():
    from dashboard import signals as S
    cx = _db()
    cells = {c["module"]: c for c in S.aggregate_signals(cx, actor=None)}
    assert cells["money"]["level"] == S.GRAY


def test_pending_approval_overlay_lights_up_module():
    from dashboard import signals as S
    from dashboard import events as E
    cx = _db()
    E.append_event(cx, actor="va", source="justus", action_key="finance.refund_order",
                   module="money", risk_tier="money_send", params={"amount": 5},
                   result=None, status="pending_approval")
    cells = {c["module"]: c for c in S.aggregate_signals(cx, actor=None)}
    money = cells["money"]
    assert money["level"] in (S.AMBER, S.RED)  # bumped up from gray
    assert money["count"] >= 1
    assert any("pending" in (a.get("label", "").lower()) for a in money["top_actions"])
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_signals.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.signals'`.

- [ ] **Step 3: Write the implementation**

Create `dashboard/signals.py`:

```python
"""Business-OS Home signal board: per-module priority signals + aggregation.

Each module may register a signal function:
    signal(cx, actor) -> {level, summary, top_actions, count}
The board aggregates all nine modules (gray default for unwired ones) and
overlays pending-approval events from the Phase 1a event log so queued actions
light up the owning module. Priority rules start as seed heuristics and are
refined from real data over time."""

# Priority colors (worst floats up on the board).
RED = "red"      # urgent, act now
AMBER = "amber"  # needs attention soon
GREEN = "green"  # healthy, nothing required
GRAY = "gray"    # idle / not wired yet

_ORDER = {GRAY: 0, GREEN: 1, AMBER: 2, RED: 3}

# The nine functional modules, in display order. Keys match Action.module.
MODULES = ("money", "crm", "orders", "marketing", "products",
           "content", "comms", "tasks", "b2b")
MODULE_TITLES = {
    "money": "Money & Finance",
    "crm": "Sales & CRM",
    "orders": "Orders & Fulfillment",
    "marketing": "Marketing & Growth",
    "products": "Products & Inventory",
    "content": "Content & Knowledge",
    "comms": "Comms & Calendar",
    "tasks": "Team & Tasks",
    "b2b": "Practitioner & B2B",
}

SIGNAL_REGISTRY = {}


def signal(module_key):
    """Decorator: register a module's signal function."""
    def deco(fn):
        SIGNAL_REGISTRY[module_key] = fn
        return fn
    return deco


def worst_level(levels):
    worst = GRAY
    for lv in levels:
        if _ORDER.get(lv, 0) > _ORDER.get(worst, 0):
            worst = lv
    return worst


def _bump(level, floor):
    """Raise `level` to at least `floor`."""
    return level if _ORDER.get(level, 0) >= _ORDER.get(floor, 0) else floor


def _pending_by_module(cx):
    cur = cx.execute(
        "SELECT module, COUNT(*) AS n FROM events "
        "WHERE status='pending_approval' GROUP BY module")
    return {row[0]: row[1] for row in cur.fetchall()}


def _default_cell():
    return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}


def aggregate_signals(cx, actor=None):
    """Return the ordered list of nine board cells."""
    pending = _pending_by_module(cx)
    cells = []
    for m in MODULES:
        fn = SIGNAL_REGISTRY.get(m)
        sig = fn(cx, actor) if fn else _default_cell()
        sig = {"level": sig.get("level", GRAY),
               "summary": sig.get("summary", ""),
               "top_actions": list(sig.get("top_actions", [])),
               "count": int(sig.get("count", 0) or 0)}
        pc = pending.get(m, 0)
        if pc:
            sig["level"] = _bump(sig["level"], AMBER)
            sig["count"] += pc
            sig["top_actions"] = (
                [{"label": f"Review {pc} pending", "href": "/console/home#pending"}]
                + sig["top_actions"])
        sig["module"] = m
        sig["title"] = MODULE_TITLES[m]
        cells.append(sig)
    return cells


@signal("tasks")
def tasks_signal(cx, actor=None):
    rows = cx.execute(
        "SELECT priority FROM todos "
        "WHERE status NOT IN ('done','dismissed','delegated')").fetchall()
    n = len(rows)
    if n == 0:
        return {"level": GREEN, "summary": "No open tasks", "top_actions": [], "count": 0}
    high = sum(1 for r in rows if (r[0] or "").lower() == "high")
    level = RED if high else AMBER
    summary = f"{n} open" + (f", {high} high priority" if high else "")
    return {"level": level, "summary": summary,
            "top_actions": [{"label": "Open task inbox", "href": "/console"}],
            "count": n}
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_signals.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/signals.py tests/test_bos_signals.py
git commit -m "feat(bos): home signal board contract + aggregation + tasks signal"
```

---

## Task 2: Routes (`app.py`) + route test

**Files:**
- Modify: `app.py` (add `import dashboard.signals` to the BOS startup block; add two routes near the other `bos_*` routes)
- Test: `tests/test_bos_routes.py` (append)

- [ ] **Step 1: Write the failing route test** (append to `tests/test_bos_routes.py`)

```python
def test_home_signals_route(monkeypatch, tmp_path):
    app_module = _load_app()
    import sqlite3
    db = str(tmp_path / "h.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    cx = sqlite3.connect(db)
    from dashboard import events as E
    E.init_event_tables(cx)
    cx.execute("CREATE TABLE IF NOT EXISTS todos (id INTEGER PRIMARY KEY, status TEXT, priority TEXT)")
    cx.execute("INSERT INTO todos (status, priority) VALUES ('open','high')")
    cx.commit(); cx.close()

    client = app_module.app.test_client()
    key = app_module.dashboard.CONSOLE_SECRET or ""
    r = client.get("/api/home/signals", headers={"X-Console-Key": key})
    assert r.status_code == 200
    cells = r.get_json()["data"]
    assert len(cells) == 9
    tasks = [c for c in cells if c["module"] == "tasks"][0]
    assert tasks["level"] == "red"  # an open high-priority todo


def test_home_page_served(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "p.db"))
    client = app_module.app.test_client()
    r = client.get("/console/home")
    assert r.status_code == 200
    assert b"Home" in r.data or b"home" in r.data
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_routes.py -k "home" -q`
Expected: FAIL if `app` imports (404), else SKIP.

- [ ] **Step 3: Add `import dashboard.signals` to the BOS startup block in `app.py`**

Find the BOS startup block (search for `import dashboard.actions_tasks`). Add a line beneath it:

```python
import dashboard.signals as _bos_signals  # noqa: F401 (registers module signals)
```

- [ ] **Step 4: Add the two routes** near the other `bos_*` routes in `app.py`

```python
@app.route("/api/home/signals", methods=["GET"])
def bos_home_signals():
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    cx = _sqlite3.connect(LOG_DB)
    cx.row_factory = _sqlite3.Row
    try:
        cells = _bos_signals.aggregate_signals(cx, actor)
    finally:
        cx.close()
    return jsonify({"ok": True, "data": cells})


@app.route("/console/home")
def bos_home_page():
    resp = send_from_directory(STATIC, "console-home.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp
```

Note: `STATIC` and `send_from_directory` are already used by the existing `/console` route; reuse them.

- [ ] **Step 5: Create a minimal page so the page test passes**

Create `static/console-home.html` with at least a title (full UI is Task 3):

```html
<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Home - Console</title></head>
<body><h1>Home</h1><div id="board"></div></body></html>
```

- [ ] **Step 6: Run + commit**

Run: `python3 -m pytest tests/test_bos_signals.py -q` (still green).
Run: `python3 -m pytest tests/test_bos_routes.py -k "home" -q` (PASS if app imports, else SKIP).
Run: `python3 -m py_compile app.py` (OK).

```bash
git add app.py static/console-home.html tests/test_bos_routes.py
git commit -m "feat(bos): /api/home/signals + /console/home routes"
```

---

## Task 3: The Home board UI (`static/console-home.html`)

**Files:**
- Modify: `static/console-home.html` (replace the stub with the full board)

This task is UI; it is verified by the page-served route test (Task 2) plus manual review. No new unit test.

- [ ] **Step 1: Replace `static/console-home.html` with the full board**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Home - Business OS Console</title>
<style>
  :root { --bg:#0a150d; --surface:#111f16; --border:#21472d; --cream:#fdf4d8;
          --muted:#a89870; --gold:#d4a843;
          --red:#c0432b; --amber:#d4a843; --green:#3d8a52; --gray:#3a4a40; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--cream);
         font-family:"Open Sans",system-ui,sans-serif; padding:28px; }
  h1 { font-family:"Raleway",sans-serif; font-size:24px; margin-bottom:4px; }
  .sub { color:var(--muted); font-size:14px; margin-bottom:24px; }
  .grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
  .cell { background:var(--surface); border:1px solid var(--border);
          border-left:4px solid var(--gray); border-radius:12px; padding:16px 18px;
          cursor:pointer; transition:transform .2s, border-color .2s; }
  .cell:hover { transform:translateY(-2px); }
  .cell.red    { border-left-color:var(--red); }
  .cell.amber  { border-left-color:var(--amber); }
  .cell.green  { border-left-color:var(--green); }
  .cell.gray   { border-left-color:var(--gray); opacity:.75; }
  .cell .dot { display:inline-block; width:9px; height:9px; border-radius:50%;
               margin-right:7px; vertical-align:middle; }
  .red .dot{background:var(--red);} .amber .dot{background:var(--amber);}
  .green .dot{background:var(--green);} .gray .dot{background:var(--gray);}
  .cell h2 { font-family:"Raleway",sans-serif; font-size:15px; display:inline; }
  .cell .summary { color:var(--muted); font-size:13px; margin:8px 0 0; }
  .cell .actions { margin-top:10px; display:flex; flex-wrap:wrap; gap:6px; }
  .cell .actions a { font-size:12px; color:var(--gold); border:1px solid var(--border);
                     border-radius:6px; padding:3px 9px; text-decoration:none; }
  .section { margin-top:32px; }
  .section h3 { font-family:"Raleway",sans-serif; font-size:15px; margin-bottom:10px; }
  .row { display:flex; justify-content:space-between; align-items:center;
         background:var(--surface); border:1px solid var(--border);
         border-radius:8px; padding:10px 14px; margin-bottom:8px; font-size:13px; }
  .row .meta { color:var(--muted); }
  .row button { font-size:12px; border:1px solid var(--border); background:transparent;
                color:var(--cream); border-radius:6px; padding:4px 10px; cursor:pointer; margin-left:6px; }
  .row button.approve { color:var(--green); } .row button.cancel { color:var(--red); }
  #gate { position:fixed; inset:0; background:var(--bg); display:flex;
          align-items:center; justify-content:center; }
  #gate input { padding:10px; border-radius:8px; border:1px solid var(--border);
                background:var(--surface); color:var(--cream); }
</style>
</head>
<body>
  <div id="gate">
    <div style="text-align:center">
      <p style="margin-bottom:10px;color:var(--muted)">Enter console key</p>
      <input id="key" type="password" placeholder="console key" />
      <button onclick="unlock()" style="padding:10px 14px;margin-left:6px;border-radius:8px;border:1px solid var(--border);background:var(--gold);color:#0a150d;cursor:pointer">Unlock</button>
    </div>
  </div>

  <h1>Business OS</h1>
  <div class="sub">Where the business needs you, right now.</div>
  <div class="grid" id="board"></div>

  <div class="section" id="pending"><h3>Pending approvals</h3><div id="pending-list"></div></div>
  <div class="section"><h3>Recent activity</h3><div id="activity-list"></div></div>

<script>
  function key(){ return localStorage.getItem('console_key') || ''; }
  function unlock(){ localStorage.setItem('console_key', document.getElementById('key').value); document.getElementById('gate').style.display='none'; load(); }
  function hdr(){ return { 'X-Console-Key': key() }; }

  async function load(){
    try {
      const s = await fetch('/api/home/signals', {headers:hdr()});
      if (s.status === 401) { document.getElementById('gate').style.display='flex'; return; }
      const cells = (await s.json()).data || [];
      const board = document.getElementById('board');
      board.innerHTML = '';
      cells.forEach(function(c){
        const div = document.createElement('div');
        div.className = 'cell ' + c.level;
        const acts = (c.top_actions||[]).map(function(a){
          return '<a href="'+(a.href||'#')+'">'+a.label+'</a>'; }).join('');
        div.innerHTML = '<span class="dot"></span><h2>'+c.title+'</h2>'
          + '<p class="summary">'+(c.summary||'')+'</p>'
          + '<div class="actions">'+acts+'</div>';
        board.appendChild(div);
      });
      const ev = await fetch('/api/events?limit=40', {headers:hdr()});
      const events = (await ev.json()).data || [];
      renderPending(events.filter(function(e){return e.status==='pending_approval';}));
      renderActivity(events.slice(0,15));
    } catch(e) { console.error(e); }
  }

  function renderPending(list){
    const host = document.getElementById('pending-list');
    if (!list.length){ host.innerHTML = '<div class="row meta">Nothing waiting on you.</div>'; return; }
    host.innerHTML = '';
    list.forEach(function(e){
      const div = document.createElement('div');
      div.className = 'row';
      div.innerHTML = '<span>'+e.action_key+' <span class="meta">('+e.module+', by '+e.actor+')</span></span>'
        + '<span><button class="approve" onclick="act('+e.id+',\'approve\')">Approve</button>'
        + '<button class="cancel" onclick="act('+e.id+',\'cancel\')">Cancel</button></span>';
      host.appendChild(div);
    });
  }

  function renderActivity(list){
    const host = document.getElementById('activity-list');
    host.innerHTML = list.map(function(e){
      return '<div class="row"><span>'+e.action_key+'</span>'
        + '<span class="meta">'+e.status+' · '+(e.actor||'')+'</span></div>';
    }).join('') || '<div class="row meta">No activity yet.</div>';
  }

  async function act(id, what){
    await fetch('/api/events/'+id+'/'+what, {method:'POST', headers:hdr()});
    load();
  }

  if (key()) { document.getElementById('gate').style.display='none'; load(); }
</script>
</body>
</html>
```

- [ ] **Step 2: Verify the page still serves**

Run: `python3 -m pytest tests/test_bos_routes.py -k "home_page" -q`
Expected: PASS (or SKIP locally).

- [ ] **Step 3: Commit**

```bash
git add static/console-home.html
git commit -m "feat(bos): home signal board UI (9-cell grid + pending + activity)"
```

---

## Self-Review

**Spec coverage** (blueprint sections 4.1, 4.2, 5):
- 4.1 Home board: 9-cell grid, color signal, summary, top actions, click-through -> Task 3 UI + Task 1 data.
- 4.2 signal() contract `{level, summary, top_actions, count}` + worst-floats-up + pending-approval feeding signals + seed heuristics -> Task 1.
- The nine modules in order -> `MODULES` in Task 1.
- `/api/home/signals` -> Task 2.
- Real signal for a module whose data exists (Tasks/todos); gray for the rest -> Task 1.
- Pending approvals + recent activity on Home (reusing `/api/events`) -> Task 3.

**Placeholder scan:** none. Eight modules default to gray by design (not a placeholder; each gains a real `signal()` in its own phase).

**Type consistency:** cell dict keys (`module`, `title`, `level`, `summary`, `top_actions`, `count`), color constants (`RED/AMBER/GREEN/GRAY`), and `MODULES` order are used identically across Tasks 1-3 and the route. The route returns `{"ok": True, "data": cells}`, matching the `/api/events` envelope and the UI's `(await res.json()).data`.

**Out of scope (later phases):** real `signal()` for money/crm/orders/marketing/products/content/comms/b2b; briefings-as-actions on Home; the unified shell nav.
