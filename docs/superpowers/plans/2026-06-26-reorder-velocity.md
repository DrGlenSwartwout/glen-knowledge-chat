# Sales Velocity → Reorder Demand — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-generate the reorder report's production plan from sales velocity — per formulation, 3-mo and 12-mo avg units/month from `product_sales`, projected over a horizon — so reorder demand reflects what's actually selling.

**Architecture:** Add `product_velocity` / `velocity_plan` / `velocity_table` to `dashboard/reorder.py`. `velocity_plan` returns the existing `[{formulation_id, qty}]` plan shape, fed unchanged into the existing `bom_demand` → `reorder_report`. The `/api/reorder/report` endpoint gains a `source=velocity` branch; the `/admin/ingredients` Reorder tab gains a "From sales velocity" mode.

**Tech Stack:** Python 3 / Flask, sqlite, pytest, vanilla JS. Reuses `dashboard/reorder.py` (`_connect`, `_num`, `bom_demand`, `reorder_report`), `product_sales` table, `formulations`/`formulation_items` tables.

## Global Constraints

- Velocity = average **units/month**: `vel_N = SUM(units in the N calendar months ending at MAX(period) in product_sales) / N` (missing months count as 0). Window anchored to the **latest period present in `product_sales`** (not wall-clock today). Sums across all `source` values.
- `velocity_plan(basis, horizon_months)`: `basis ∈ {"3mo","12mo","max"}` (default `"3mo"`); `qty = vel(basis) × horizon_months` (default horizon 3); map `product_fmp_id → formulations.fmp_id → formulations.id`; **drop** products with no formulation match or `qty <= 0`. Returns `[{formulation_id, qty}]` — the exact shape `bom_demand`/`reorder_report` accept.
- **Do NOT change** `bom_demand` or `reorder_report` (the shortfall math) — only generate the plan.
- Endpoint + console are **console-key gated** (existing pattern); read-only (no draft POs); additive; no public flag.
- Empty/absent `product_sales` → `velocity_plan` returns `[]`, no error.
- Local test command: pure-module tests run with plain `python3 -m pytest`; app-importing tests via `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest …` (mkdir first).

---

### Task 1: Velocity functions in `dashboard/reorder.py`

**Files:**
- Modify: `dashboard/reorder.py` (add 3 functions; do not touch `bom_demand`/`reorder_report`)
- Test: `tests/test_reorder_velocity.py`

**Interfaces:**
- Consumes: `_connect` (existing in reorder.py); `bom_demand` (existing, for the integration test); the `product_sales` table (from the merged sales feature) and `formulations` table.
- Produces:
  - `product_velocity(db_path=None) -> {product_fmp_id: {"vel_3mo": float, "vel_12mo": float}}`
  - `velocity_plan(basis="3mo", horizon_months=3, db_path=None) -> [{"formulation_id", "qty"}]`
  - `velocity_table(basis="3mo", horizon_months=3, db_path=None) -> [{formulation_id, fmp_id, name, vel_3mo, vel_12mo, projected_qty}]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reorder_velocity.py
import sqlite3
import pytest
from dashboard import reorder as ro


@pytest.fixture
def db(tmp_path):
    p = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(p)
    cx.execute("CREATE TABLE product_sales (product_fmp_id TEXT, period TEXT, units REAL, revenue_cents INTEGER, source TEXT)")
    # product 425: last-3 = 2026-06(12)+2026-05(6) = 18 → vel_3mo 6.0 ; +2025-09(24) within last 12 → 12mo sum 42 → 3.5
    cx.executemany("INSERT INTO product_sales(product_fmp_id,period,units,revenue_cents,source) VALUES (?,?,?,?, 'fmp')", [
        ("425", "2026-06", 12, 0), ("425", "2026-05", 6, 0), ("425", "2025-09", 24, 0),
        ("999", "2026-06", 9, 0),  # has velocity but no formulation → dropped from plan
    ])
    cx.execute("CREATE TABLE formulations (id INTEGER PRIMARY KEY, fmp_id TEXT, name TEXT)")
    cx.execute("INSERT INTO formulations(id,fmp_id,name) VALUES (1,'425','Microbiome')")
    cx.execute("CREATE TABLE formulation_items (formulation_id INTEGER, ingredient_id INTEGER, dose REAL, dose_unit TEXT)")
    cx.execute("INSERT INTO formulation_items(formulation_id,ingredient_id,dose,dose_unit) VALUES (1, 99, 2.0, 'g')")
    cx.commit(); cx.close()
    return p


def test_product_velocity_3_and_12_month(db):
    v = ro.product_velocity(db)
    assert v["425"]["vel_3mo"] == pytest.approx(6.0)    # (12+6)/3
    assert v["425"]["vel_12mo"] == pytest.approx(3.5)   # (12+6+24)/12


def test_velocity_plan_basis_horizon_and_formulation_map(db):
    p3 = ro.velocity_plan(basis="3mo", horizon_months=3, db_path=db)
    assert p3 == [{"formulation_id": 1, "qty": pytest.approx(18.0)}]      # 6 * 3 ; 999 dropped (no formulation)
    p12 = ro.velocity_plan(basis="12mo", horizon_months=2, db_path=db)
    assert p12 == [{"formulation_id": 1, "qty": pytest.approx(7.0)}]      # 3.5 * 2
    pmax = ro.velocity_plan(basis="max", horizon_months=1, db_path=db)
    assert pmax == [{"formulation_id": 1, "qty": pytest.approx(6.0)}]     # max(6, 3.5) * 1


def test_velocity_table_shape(db):
    t = ro.velocity_table(basis="3mo", horizon_months=3, db_path=db)
    assert len(t) == 1 and t[0]["fmp_id"] == "425" and t[0]["name"] == "Microbiome"
    assert t[0]["vel_3mo"] == pytest.approx(6.0) and t[0]["vel_12mo"] == pytest.approx(3.5)
    assert t[0]["projected_qty"] == pytest.approx(18.0)


def test_velocity_plan_feeds_bom_demand(db):
    plan = ro.velocity_plan(basis="3mo", horizon_months=3, db_path=db)
    dem = ro.bom_demand(plan, db_path=db)
    assert dem[99]["demand"] == pytest.approx(36.0)  # dose 2 * qty 18


def test_empty_product_sales_returns_empty(tmp_path):
    p = str(tmp_path / "empty.db")
    cx = sqlite3.connect(p)
    cx.execute("CREATE TABLE product_sales (product_fmp_id TEXT, period TEXT, units REAL, revenue_cents INTEGER, source TEXT)")
    cx.execute("CREATE TABLE formulations (id INTEGER PRIMARY KEY, fmp_id TEXT, name TEXT)")
    cx.commit(); cx.close()
    assert ro.velocity_plan(db_path=p) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && python3 -m pytest tests/test_reorder_velocity.py -q`
Expected: FAIL — `product_velocity`/`velocity_plan`/`velocity_table` undefined.

- [ ] **Step 3: Add the three functions to `dashboard/reorder.py`** (after `reorder_report`; use the existing `_connect`)

```python
def _period_minus(period, n):
    """The 'YYYY-MM' that is n months before `period` (inclusive window helper)."""
    y, m = int(period[:4]), int(period[5:7])
    total = y * 12 + (m - 1) - n
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def product_velocity(db_path=None) -> dict:
    """Per product_fmp_id: avg units/month over the trailing 3 and 12 months,
    anchored to MAX(period) in product_sales (missing months count as 0)."""
    with _connect(db_path) as cx:
        row = cx.execute("SELECT MAX(period) FROM product_sales").fetchone()
        latest = row[0] if row else None
        if not latest:
            return {}
        out = {}
        for months, col in ((3, "vel_3mo"), (12, "vel_12mo")):
            cutoff = _period_minus(latest, months - 1)  # first period in the window
            for pid, units in cx.execute(
                    "SELECT product_fmp_id, SUM(units) FROM product_sales "
                    "WHERE period >= ? AND period <= ? GROUP BY product_fmp_id",
                    (cutoff, latest)).fetchall():
                d = out.setdefault(pid, {"vel_3mo": 0.0, "vel_12mo": 0.0})
                d[col] = (_num(units) or 0.0) / months
        return out


def _pick_velocity(v, basis):
    if basis == "12mo":
        return v["vel_12mo"]
    if basis == "max":
        return max(v["vel_3mo"], v["vel_12mo"])
    return v["vel_3mo"]


def _formulations_by_fmp(cx):
    return {r["fmp_id"]: (r["id"], r["name"]) for r in cx.execute(
        "SELECT id, fmp_id, name FROM formulations WHERE fmp_id IS NOT NULL").fetchall()}


def velocity_plan(basis="3mo", horizon_months=3, db_path=None) -> list:
    """Project sales velocity into a reorder plan [{formulation_id, qty}]. Products
    with no formulation match or zero projected qty are dropped."""
    vel = product_velocity(db_path)
    if not vel:
        return []
    with _connect(db_path) as cx:
        forms = _formulations_by_fmp(cx)
    plan = []
    for pid, v in vel.items():
        f = forms.get(pid)
        if not f:
            continue
        qty = _pick_velocity(v, basis) * float(horizon_months)
        if qty > 0:
            plan.append({"formulation_id": f[0], "qty": qty})
    return plan


def velocity_table(basis="3mo", horizon_months=3, db_path=None) -> list:
    """Per matched formulation: 3-mo & 12-mo velocity + the projected qty (basis × horizon)."""
    vel = product_velocity(db_path)
    with _connect(db_path) as cx:
        forms = _formulations_by_fmp(cx)
    rows = []
    for pid, v in vel.items():
        f = forms.get(pid)
        if not f:
            continue
        rows.append({"formulation_id": f[0], "fmp_id": pid, "name": f[1],
                     "vel_3mo": round(v["vel_3mo"], 2), "vel_12mo": round(v["vel_12mo"], 2),
                     "projected_qty": round(_pick_velocity(v, basis) * float(horizon_months), 2)})
    rows.sort(key=lambda r: -r["projected_qty"])
    return rows
```

> Note: `_connect` returns rows with `sqlite3.Row` access (the module already uses `r["col"]`). The test's raw `sqlite3.connect` must match — if `_connect` sets `row_factory = sqlite3.Row`, the functions work against the test DB regardless (the test seeds plain tables). Confirm `_connect` sets `row_factory`; if it does not, change the SQL reads to positional indices.

- [ ] **Step 4: Run to verify pass**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && python3 -m pytest tests/test_reorder_velocity.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/reorder.py tests/test_reorder_velocity.py
git commit -m "feat(reorder): sales-velocity functions (product_velocity/velocity_plan/velocity_table)"
```

---

### Task 2: `source=velocity` on `/api/reorder/report`

**Files:**
- Modify: `app.py` (the `GET /api/reorder/report` handler — `api_reorder_report_get`, ~line 21814)
- Test: `tests/test_reorder_velocity_api.py`

**Interfaces:**
- Consumes: `_ro.velocity_plan`, `_ro.velocity_table`, `_ro.reorder_report` (`_ro` = `dashboard.reorder`, imported at app.py:21698).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reorder_velocity_api.py
import sqlite3, pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("CREATE TABLE product_sales (product_fmp_id TEXT, period TEXT, units REAL, revenue_cents INTEGER, source TEXT)")
        cx.executemany("INSERT INTO product_sales VALUES (?,?,?,?, 'fmp')",
                       [("425", "2026-06", 12, 0), ("425", "2026-05", 6, 0)])
        cx.execute("CREATE TABLE formulations (id INTEGER PRIMARY KEY, fmp_id TEXT, name TEXT)")
        cx.execute("INSERT INTO formulations(id,fmp_id,name) VALUES (1,'425','Microbiome')")
        cx.execute("CREATE TABLE formulation_items (formulation_id INTEGER, ingredient_id INTEGER, dose REAL, dose_unit TEXT)")
        cx.execute("CREATE TABLE ingredients (id INTEGER PRIMARY KEY, name TEXT, extras TEXT, par_level REAL, par_level_unit TEXT)")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_velocity_source_returns_table(client):
    r = client.get("/api/reorder/report?source=velocity&basis=3mo&horizon=3", headers={"X-Console-Key": "test-secret"})
    assert r.status_code == 200
    body = r.get_json()
    vt = body.get("velocity_table") or (body.get("data") or {}).get("velocity_table")
    assert vt and vt[0]["fmp_id"] == "425" and vt[0]["projected_qty"] == 6.0 * 3


def test_reorder_report_requires_auth(client):
    assert client.get("/api/reorder/report?source=velocity").status_code in (401, 403)
```

> The `body` shape depends on the existing `ok(...)` helper — the assertion checks both a top-level `velocity_table` and a `data`-wrapped form. Keep whichever the existing endpoint uses; the velocity_table must be reachable in the response.

- [ ] **Step 2: Run to verify it fails**

Run: `mkdir -p /tmp/jshell-test && cd /tmp/wt-deploy-chat-6a686b75 && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest tests/test_reorder_velocity_api.py -q`
Expected: FAIL — `source=velocity` not handled (no `velocity_table` in the response).

- [ ] **Step 3: Extend `api_reorder_report_get`** (app.py ~21814). Read the existing handler first; add a `source=velocity` branch BEFORE the default `reorder_report(plan=None)` return, preserving the existing auth + `ok(...)` wrapper and the `include_below_par` arg:

```python
    if request.args.get("source") == "velocity":
        basis = request.args.get("basis", "3mo")
        try:
            horizon = int(request.args.get("horizon", 3))
        except (ValueError, TypeError):
            horizon = 3
        plan = _ro.velocity_plan(basis=basis, horizon_months=horizon)
        rep = _ro.reorder_report(plan=plan, include_below_par=below)
        rep["velocity_table"] = _ro.velocity_table(basis=basis, horizon_months=horizon)
        return ok(rep)
```

(Insert this just after `below` is computed and before the existing `return ok(_ro.reorder_report(plan=None, include_below_par=below))`. Match the exact local var name the handler uses for `include_below_par` — grep the handler.)

- [ ] **Step 4: Run to verify pass**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest tests/test_reorder_velocity_api.py -q`
Expected: PASS (2 tests). If `test_velocity_source_returns_table` fails on the body shape, adjust the assertion to where `ok()` places the payload (top-level vs `data`), not the implementation.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_reorder_velocity_api.py
git commit -m "feat(reorder): source=velocity on /api/reorder/report (velocity plan + table)"
```

---

### Task 3: "From sales velocity" mode in the Reorder console tab

**Files:**
- Modify: `static/admin-ingredients.html` (the Reorder tab — `renderReorderPlan` / the `/api/reorder/report` fetch, ~line 2829–2885)
- Test: manual/visual QA (controller render-smoke).

- [ ] **Step 1: Add the velocity controls + render** to the Reorder tab. Below the existing manual plan builder, add a control row and a velocity table, and wire a fetch:

  - Controls: a basis `<select id="velBasis">` with options `3mo` / `12mo` / `max`, a `<input id="velHorizon" type="number" value="3" min="1" max="24">`, and a `<button onclick="loadVelocity()">From sales velocity</button>`.
  - `loadVelocity()`: `const rep = await api("/api/reorder/report?source=velocity&basis="+document.getElementById("velBasis").value+"&horizon="+(document.getElementById("velHorizon").value||3));` then render `rep.velocity_table` (use the same `api(...)` helper + auth the tab already uses for the manual report) into a table with columns **Formulation · 3-mo/mo · 12-mo/mo · Projected qty**, and render the reorder shopping list from `rep` reusing the tab's existing report-rendering function.
  - Match the existing tab's style/structure (read `renderReorderPlan` and the manual report renderer first and mirror them).

- [ ] **Step 2: Syntax check**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && python3 -c "open('static/admin-ingredients.html').read(); print('read ok')"` (HTML — no JS parser available; rely on the controller's live render-smoke).

- [ ] **Step 3: Commit**

```bash
git add static/admin-ingredients.html
git commit -m "feat(reorder): console — 'From sales velocity' mode (3mo/12mo + projected, reorder list)"
```

---

### Task 4: Integration smoke + real-data check + PR

**Files:** none (verification only).

- [ ] **Step 1: Full new-suite green**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && mkdir -p /tmp/jshell-test && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest tests/test_reorder_velocity.py tests/test_reorder_velocity_api.py -q`
Expected: all pass.

- [ ] **Step 2: Real-data velocity smoke** — import the real invoices into a scratch db, then check velocity:

Run: `cd /tmp/wt-deploy-chat-6a686b75 && python3 scripts/import_invoices_from_fmp.py --items /tmp/fmp-export/newapp/invoice_items.csv --products data/products.json --db /tmp/jshell-test/v.db --write` then `python3 -c "from dashboard import reorder as r; rows=r.velocity_table(db_path='/tmp/jshell-test/v.db'); print(len(rows), rows[:3])"`.
Expected: a non-empty velocity table with the top movers' 3-mo/12-mo numbers (only products that map to a `formulations` row appear — the scratch db has product_sales but no formulations table, so this primarily validates `product_velocity`; seed a formulations row if you want a full plan).

- [ ] **Step 3: Controller render-smoke** of the console velocity mode (the controller does this — start the app with a seeded `product_sales`+`formulations`, open `/admin/ingredients`, Reorder tab, click "From sales velocity", confirm the 3-mo/12-mo table + reorder list render with zero console errors).

- [ ] **Step 4: Report** to the controller — the controller runs the final whole-branch review and opens the PR.

---

## Self-Review

**Spec coverage:**
- `product_velocity` (3/12-mo avg, latest-period-relative, missing months = 0) → Task 1. ✓
- `velocity_plan` (basis 3mo/12mo/max, horizon multiply, fmp→formulation map, drop non-match/zero) → Task 1. ✓
- `velocity_table` (side-by-side 3mo/12mo + projected) → Task 1. ✓
- Feeds existing `bom_demand`/`reorder_report` unchanged → Task 1 (`test_velocity_plan_feeds_bom_demand`). ✓
- `/api/reorder/report?source=velocity&basis=&horizon=` returns velocity_table + report, console-gated → Task 2. ✓
- Console "From sales velocity" mode (basis toggle + horizon + side-by-side table + reorder list) → Task 3. ✓
- Empty product_sales → `[]` no error → Task 1 (`test_empty_product_sales_returns_empty`). ✓
- Read-only, additive, no flag → enforced (no `bom_demand`/`reorder_report` change; no draft POs). ✓

**Placeholder scan:** none — concrete code in Tasks 1–2; Task 3 is HTML (manual-QA) with the exact controls, fetch URL, columns, and the instruction to mirror the existing tab renderers. Two implementer-resolved spots: the `ok()` response shape in Task 2 (assertion tolerant + named), and `_connect`'s `row_factory` (Task 1 note with the positional fallback).

**Type consistency:** `product_velocity`→`{pid:{vel_3mo,vel_12mo}}`, `velocity_plan`→`[{formulation_id,qty}]` (matches `bom_demand`'s input), `velocity_table`→rows with `fmp_id/name/vel_3mo/vel_12mo/projected_qty` consistent across Tasks 1–3; `_pick_velocity`/`_formulations_by_fmp`/`_period_minus` helpers defined and used within Task 1.
