# Phase 3c-3 BOM Demand + Reorder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the migrated data into "what to buy" — explode a planned production list through recipes, net demand (plus par) against on-hand + on-order, and produce a read-only reorder shopping list grouped by preferred supplier.

**Architecture:** A pure-compute module `dashboard/reorder.py` (no new tables) reading recipes (Phase 2), on-hand (3c-1 `inventory_txns`), on-order (3b open POs), and par/preferred-source (Phase 1). Core formula `shortfall = (par or 0) + demand − on_hand − on_order`; reorder when > 0. Two `/api/reorder/*` endpoints + a Reorder console tab.

**Tech Stack:** Python 3 / Flask, SQLite (`chat_log.db`), vanilla-JS static console. Tests: pytest.

## Global Constraints

- No schema changes — pure read/compute. Module fns take optional `db_path`, use `with _connect(db_path)` (reuse the `dashboard/inventory.py` pattern). Reuse `dashboard.inventory.on_hand`.
- Reorder formula: `shortfall = (par_level or 0) + demand − on_hand − on_order`; a line is included only when `shortfall > 0`. No `max()`.
- On-order is computed from **receiving** (`pi.qty − SUM(po_receiving.qty_received)`, floored at 0), over `purchase_orders.status != 'closed'` ingredient lines only — NOT the FMP `qty_left` calc field (same trust rationale as the inventory ledger).
- Ingredient-only throughout (materials excluded). Units are NOT converted — summed numerically, unit strings surfaced; disagreement sets a `unit_warning` (display only).
- Preferred source = `SELECT ... FROM ingredient_sources WHERE ingredient_id=? ORDER BY preferred DESC, price_per_unit LIMIT 1`.
- Endpoints: `@require_console_key`, `ok`/`fail` from `dashboard/__init__.py`, alias `from dashboard import reorder as _ro`. No schema-init wiring (no tables).
- Console: real `api(path, opts={})` (returns `j.data`, throws — no `.data`/`.ok`); reuse the Production tab's search-to-pick formulation picker (index-array pattern) + `escapeHtml`/`showTab`; append to the `labels` array; existing tabs untouched.
- Route tests use the Pinecone `pytest.skip` pattern.
- Module name `dashboard/reorder.py` confirmed collision-free.

---

### Task 1: `dashboard/reorder.py` — bom_demand + on_order + reorder_report

**Files:**
- Create: `dashboard/reorder.py`
- Test: `tests/test_reorder.py`

**Interfaces:**
- Produces: `bom_demand(plan, db_path=None) -> dict`; `on_order_by_ingredient(db_path=None) -> dict`; `reorder_report(plan=None, include_below_par=True, db_path=None) -> dict`; `_round_up_order(shortfall, moq, unit_size) -> float`.
- Consumes: `dashboard.inventory.on_hand`; `formulations`/`formulation_items`; `ingredients`(extras.par_level); `ingredient_sources`; 3b `purchase_orders`/`po_items`/`po_receiving`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reorder.py
import json, sqlite3
import pytest
from dashboard import reorder as ro
from dashboard import inventory as inv


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE ingredients (id INTEGER PRIMARY KEY, name TEXT, extras TEXT)")
        cx.execute("""CREATE TABLE ingredient_sources (id INTEGER PRIMARY KEY, ingredient_id INTEGER,
            supplier_id INTEGER, supplier_name TEXT, price_per_unit REAL, unit_size REAL,
            unit_type TEXT, preferred INTEGER DEFAULT 0, minimum_order REAL, minimum_order_unit TEXT)""")
        cx.execute("CREATE TABLE suppliers (id INTEGER PRIMARY KEY, company TEXT)")
        cx.execute("CREATE TABLE formulations (id INTEGER PRIMARY KEY, name TEXT)")
        cx.execute("""CREATE TABLE formulation_items (id INTEGER PRIMARY KEY, formulation_id INTEGER,
            ingredient_id INTEGER, ingredient_name TEXT, dose REAL, dose_unit TEXT)""")
        cx.execute("CREATE TABLE purchase_orders (id INTEGER PRIMARY KEY, status TEXT)")
        cx.execute("CREATE TABLE po_items (id INTEGER PRIMARY KEY, po_id INTEGER, ingredient_id INTEGER, qty REAL)")
        cx.execute("CREATE TABLE po_receiving (id INTEGER PRIMARY KEY, po_item_id INTEGER, qty_received REAL)")
        inv.init_inventory_schema(cx)
        # ingredient 1: par 3 kg, preferred source MOQ 2 / unit_size 0.5 / $10
        cx.execute("INSERT INTO ingredients VALUES (1,'Mag',?)", (json.dumps({"par_level": "3", "par_level_unit": "kg"}),))
        cx.execute("INSERT INTO ingredients VALUES (2,'Lipoic',?)", (json.dumps({"par_level": "1", "par_level_unit": "kg"}),))
        cx.execute("INSERT INTO suppliers VALUES (7,'NOW Foods')")
        cx.execute("INSERT INTO ingredient_sources (id,ingredient_id,supplier_id,supplier_name,price_per_unit,unit_size,preferred,minimum_order,minimum_order_unit) VALUES (1,1,7,'NOW Foods',10.0,0.5,1,2.0,'kg')")
        cx.execute("INSERT INTO formulations VALUES (1,'Brain Blend')")
        cx.execute("INSERT INTO formulation_items (id,formulation_id,ingredient_id,ingredient_name,dose,dose_unit) VALUES (1,1,1,'Mag',0.5,'kg')")
        # ingredient 1 on-hand 1.0 (baseline); ingredient 2 on-hand 5.0 (well above par)
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'baseline',1.0)")
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (2,'baseline',5.0)")
        cx.commit()
    return db


def test_round_up_order():
    assert ro._round_up_order(1.7, 2.0, 0.5) == 2.0      # MOQ floor
    assert ro._round_up_order(2.1, 2.0, 0.5) == 2.5      # round up to 0.5 multiple
    assert ro._round_up_order(2.1, None, None) == 2.1    # no MOQ/unit_size
    assert ro._round_up_order(0.3, None, 1.0) == 1.0     # ceil to unit_size


def test_on_order_excludes_closed_material_received(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO purchase_orders VALUES (10,'open')")
        cx.execute("INSERT INTO purchase_orders VALUES (11,'closed')")
        cx.execute("INSERT INTO po_items VALUES (100,10,1,4.0)")     # open, ingredient 1, qty 4
        cx.execute("INSERT INTO po_items VALUES (101,10,NULL,9.0)")  # open, material-only (null ingredient)
        cx.execute("INSERT INTO po_items VALUES (102,11,1,8.0)")     # CLOSED po → excluded
        cx.execute("INSERT INTO po_receiving VALUES (1000,100,1.5)") # 1.5 of item 100 already received
        cx.commit()
    oo = ro.on_order_by_ingredient(db)
    assert round(oo[1]["on_order"], 3) == 2.5            # 4.0 − 1.5 received; closed + material excluded
    assert 2 not in oo


def test_bom_demand(tmp_path):
    db = _db(tmp_path)
    d = ro.bom_demand([{"formulation_id": 1, "qty": 4}], db)
    assert d[1]["demand"] == 2.0                          # 0.5 dose × 4
    assert ro.bom_demand([], db) == {}


def test_reorder_report_par_and_plan(tmp_path):
    db = _db(tmp_path)
    # No plan: ingredient 1 par 3 − on_hand 1 − on_order 0 = shortfall 2 → reorder; ingredient 2 (5 ≥ 1) no line
    rep = ro.reorder_report(db_path=db)
    lines = [ln for g in rep["groups"] for ln in g["lines"]]
    by_ing = {ln["ingredient_id"]: ln for ln in lines}
    assert 2 not in by_ing
    assert by_ing[1]["shortfall"] == 2.0
    assert by_ing[1]["suggested_qty"] == 2.0             # MOQ 2 ≥ shortfall 2, on a 0.5 grid
    assert by_ing[1]["est_cost"] == 20.0                 # 2.0 × $10
    assert rep["groups"][0]["supplier"] == "NOW Foods"
    assert rep["groups"][0]["subtotal"] == 20.0
    # With a plan (4 units → demand 2): shortfall = 3 + 2 − 1 = 4 → suggested 4 (MOQ ok, 0.5 grid)
    rep2 = ro.reorder_report(plan=[{"formulation_id": 1, "qty": 4}], db_path=db)
    ln1 = [l for g in rep2["groups"] for l in g["lines"] if l["ingredient_id"] == 1][0]
    assert ln1["demand"] == 2.0 and ln1["shortfall"] == 4.0 and ln1["suggested_qty"] == 4.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_reorder.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `dashboard/reorder.py`**

```python
"""BOM demand + reorder shopping list — pure compute (Phase 3c-3)."""
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Optional

from dashboard.inventory import on_hand


def _default_db_path() -> str:
    base = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))
    return str(Path(base) / "chat_log.db")


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    cx = sqlite3.connect(db_path or _default_db_path())
    cx.row_factory = sqlite3.Row
    cx.execute("PRAGMA foreign_keys=ON")
    return cx


def _num(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _round_up_order(shortfall, moq, unit_size) -> float:
    q = max(float(shortfall), float(moq) if moq else 0.0)
    us = _num(unit_size)
    if us and us > 0:
        q = math.ceil(q / us) * us
    return round(q, 6)


def bom_demand(plan, db_path=None) -> dict:
    out = {}
    if not plan:
        return out
    with _connect(db_path) as cx:
        for line in plan:
            fid = line.get("formulation_id")
            qty = _num(line.get("qty")) or 0.0
            if not fid or qty == 0:
                continue
            rows = cx.execute(
                "SELECT ingredient_id, dose, dose_unit FROM formulation_items WHERE formulation_id=?",
                (fid,)).fetchall()
            for r in rows:
                iid = r["ingredient_id"]
                dose = _num(r["dose"])
                if not iid or dose is None:
                    continue
                d = out.setdefault(iid, {"demand": 0.0, "unit": r["dose_unit"], "units_seen": []})
                d["demand"] += dose * qty
                if r["dose_unit"] and r["dose_unit"] not in d["units_seen"]:
                    d["units_seen"].append(r["dose_unit"])
    return out


def on_order_by_ingredient(db_path=None) -> dict:
    out = {}
    with _connect(db_path) as cx:
        rows = cx.execute("""
            SELECT pi.ingredient_id AS iid, pi.qty AS qty,
                   COALESCE((SELECT SUM(qty_received) FROM po_receiving r WHERE r.po_item_id = pi.id),0) AS received
            FROM po_items pi JOIN purchase_orders po ON po.id = pi.po_id
            WHERE po.status != 'closed' AND pi.ingredient_id IS NOT NULL
        """).fetchall()
    for r in rows:
        remaining = (_num(r["qty"]) or 0.0) - (_num(r["received"]) or 0.0)
        if remaining > 0:
            d = out.setdefault(r["iid"], {"on_order": 0.0})
            d["on_order"] += remaining
    return out


def reorder_report(plan=None, include_below_par=True, db_path=None) -> dict:
    demand = bom_demand(plan or [], db_path)
    onord = on_order_by_ingredient(db_path)
    with _connect(db_path) as cx:
        # candidate set: in demand, in on-order, or (if include_below_par) any ingredient with numeric par
        cand = set(demand) | set(onord)
        ing_rows = {r["id"]: r for r in cx.execute("SELECT id, name, extras FROM ingredients").fetchall()}
        if include_below_par:
            for iid, r in ing_rows.items():
                if _num(_json_get(r["extras"], "par_level")) is not None:
                    cand.add(iid)

        groups = {}
        for iid in cand:
            ing = ing_rows.get(iid)
            if not ing:
                continue
            par = _num(_json_get(ing["extras"], "par_level")) or 0.0
            par_unit = _json_get(ing["extras"], "par_level_unit")
            dem = demand.get(iid, {}).get("demand", 0.0)
            dem_unit = demand.get(iid, {}).get("unit")
            oo = onord.get(iid, {}).get("on_order", 0.0)
            oh = on_hand(iid, db_path)
            shortfall = par + dem - oh - oo
            if shortfall <= 0:
                continue
            src = cx.execute("""
                SELECT s.supplier_id, s.supplier_name, sup.company AS company,
                       s.price_per_unit, s.unit_size, s.unit_type, s.minimum_order, s.minimum_order_unit
                FROM ingredient_sources s LEFT JOIN suppliers sup ON sup.id = s.supplier_id
                WHERE s.ingredient_id=? ORDER BY s.preferred DESC, s.price_per_unit LIMIT 1
            """, (iid,)).fetchone()
            price = _num(src["price_per_unit"]) if src else None
            unit_size = src["unit_size"] if src else None
            moq = src["minimum_order"] if src else None
            sugg = _round_up_order(shortfall, moq, unit_size)
            est_cost = round(sugg * price, 2) if price is not None else None
            units = [u for u in [par_unit if par else None, dem_unit if dem else None,
                                 (src["minimum_order_unit"] if src else None)] if u]
            unit_warning = len(set(units)) > 1
            sup_id = src["supplier_id"] if src else None
            sup_name = (src["company"] or src["supplier_name"]) if src else None
            key = sup_id if sup_id is not None else "—"
            g = groups.setdefault(key, {"supplier_id": sup_id,
                                        "supplier": sup_name or "— no preferred source —",
                                        "lines": [], "subtotal": 0.0})
            g["lines"].append({
                "ingredient_id": iid, "ingredient": ing["name"],
                "on_hand": round(oh, 4), "on_order": round(oo, 4), "par": par,
                "demand": round(dem, 4), "shortfall": round(shortfall, 4),
                "suggested_qty": sugg, "unit": (src["unit_type"] if src else None) or par_unit or dem_unit,
                "price_per_unit": price, "est_cost": est_cost, "unit_warning": unit_warning,
            })
            if est_cost:
                g["subtotal"] = round(g["subtotal"] + est_cost, 2)

    group_list = sorted(groups.values(), key=lambda g: (g["supplier_id"] is None, (g["supplier"] or "").lower()))
    for g in group_list:
        g["lines"].sort(key=lambda l: (l["ingredient"] or "").lower())
    total_cost = round(sum(g["subtotal"] for g in group_list), 2)
    total_lines = sum(len(g["lines"]) for g in group_list)
    return {"groups": group_list, "totals": {"lines": total_lines, "est_cost": total_cost},
            "plan_echo": plan or []}


def _json_get(extras, key):
    if not extras:
        return None
    try:
        return json.loads(extras).get(key)
    except (ValueError, TypeError):
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_reorder.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/reorder.py tests/test_reorder.py
git commit -m "feat(reorder): BOM demand + on-order + netted reorder report"
```

---

### Task 2: `/api/reorder/*` endpoints

**Files:**
- Modify: `app.py`
- Test: `tests/test_admin_reorder_api.py`

**Interfaces:**
- Consumes: Task 1 via `from dashboard import reorder as _ro`.
- Produces: `GET /api/reorder/report`; `POST /api/reorder/report`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_admin_reorder_api.py
import importlib, json, sqlite3, sys
from pathlib import Path
import pytest


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.formulations import init_formulations_schema
    from dashboard.purchase_orders import init_purchase_orders_schema
    from dashboard.inventory import init_inventory_schema
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx); init_formulations_schema(cx)
        init_purchase_orders_schema(cx); init_inventory_schema(cx)
        cx.execute("INSERT INTO ingredients (id,name,extras) VALUES (1,'Mag',?)",
                   (json.dumps({"par_level": "3", "par_level_unit": "kg"}),))
        cx.execute("INSERT INTO formulations (id,name) VALUES (1,'Brain Blend')")
        cx.execute("INSERT INTO formulation_items (formulation_id,ingredient_id,ingredient_name,dose,dose_unit) VALUES (1,1,'Mag',0.5,'kg')")
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'baseline',1.0)")
        cx.commit()
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod.app.test_client()


def test_reorder_get_and_post(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    g = c.get("/api/reorder/report").get_json()["data"]
    line = [l for grp in g["groups"] for l in grp["lines"] if l["ingredient_id"] == 1][0]
    assert line["shortfall"] == 2.0                       # par 3 − on_hand 1
    p = c.post("/api/reorder/report", json={"plan": [{"formulation_id": 1, "qty": 4}], "include_below_par": True}).get_json()["data"]
    line2 = [l for grp in p["groups"] for l in grp["lines"] if l["ingredient_id"] == 1][0]
    assert line2["shortfall"] == 4.0                      # 3 + (0.5×4) − 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_admin_reorder_api.py -q`
Expected: FAIL (404) or SKIP (Pinecone). Proceed to implement.

- [ ] **Step 3: Add endpoints in `app.py`** (beside the `/api/production/*` block)

```python
from dashboard import reorder as _ro


@app.route("/api/reorder/report", methods=["GET"])
@require_console_key
def api_reorder_report_get():
    try:
        below = request.args.get("below_par", "1") != "0"
        return ok(_ro.reorder_report(plan=None, include_below_par=below))
    except Exception as e:
        return fail(e)


@app.route("/api/reorder/report", methods=["POST"])
@require_console_key
def api_reorder_report_post():
    try:
        b = request.get_json(silent=True) or {}
        return ok(_ro.reorder_report(plan=b.get("plan") or [],
                                     include_below_par=bool(b.get("include_below_par", True))))
    except Exception as e:
        return fail(e)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_admin_reorder_api.py -q`
Expected: PASS or SKIP locally on Pinecone. Smoke: `python3 -c "import app"`.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_admin_reorder_api.py
git commit -m "feat(reorder): report endpoints (GET par-only, POST with plan)"
```

---

### Task 3: Reorder console tab + search index

**Files:**
- Modify: `static/admin-ingredients.html`
- Modify: `static/console-search-index.json`

**Interfaces:**
- Consumes: `/api/reorder/report` (GET + POST); `/api/formulations/search` (plan picker).

- [ ] **Step 1: Read the existing Production tab** in `static/admin-ingredients.html` — note the real `api(path, opts={})` (returns `j.data`, throws), the search-to-pick formulation picker (`prodFrmResults` index-array pattern, `/api/formulations/search`), `escapeHtml`/`showTab`/`toast`, the `labels` array, and the `display:none` CSS convention. The Reorder tab reuses these.

- [ ] **Step 2: Append `"reorder"` to the `labels` array** in `showTab`.

- [ ] **Step 3: Add the Reorder tab button + panel.** Panel contents:
  - A plan builder: a formulation search input (`id="ro-frm-search"`) + a results `<ul id="ro-frm-results">` (index-array pattern → clicking sets `roPickedFormId`/`roPickedFormName`), a qty input (`id="ro-frm-qty"`), an "Add to plan" button (→ `roAddPlanLine`). A plan table (`id="ro-plan"`) listing `{name, qty}` rows with a remove button each.
  - An "Include below-par" checkbox (`id="ro-below-par"`, checked by default).
  - A "Compute reorder list" button (→ `roCompute`).
  - A results container (`id="ro-results"`) rendered after compute.

- [ ] **Step 4: Add CSS (mirror the convention) — initial hidden state for the results container:**

```css
#ro-results { display: none; }
#ro-results.active { display: block; }
#ro-frm-results { list-style: none; margin: 0; padding: 0; }
```

- [ ] **Step 5: Add the JS (real `api()` — returns data, throws; reuse `escapeHtml`):**

```javascript
let roFrmResults = [];
let roPickedFormId = null, roPickedFormName = "";
let reorderPlan = [];

async function roFrmSearch() {
  const q = document.getElementById("ro-frm-search").value.trim();
  if (!q) { document.getElementById("ro-frm-results").innerHTML = ""; return; }
  roFrmResults = (await api("/api/formulations/search?q=" + encodeURIComponent(q))).slice(0, 15);
  document.getElementById("ro-frm-results").innerHTML = roFrmResults.map(function (f, i) {
    return '<li style="cursor:pointer;padding:3px 0" onclick="roFrmPick(' + i + ')">' + escapeHtml(f.name) + '</li>';
  }).join("");
}

function roFrmPick(i) {
  const f = roFrmResults[i];
  if (!f) return;
  roPickedFormId = f.id; roPickedFormName = f.name;
  document.getElementById("ro-frm-search").value = f.name;
  document.getElementById("ro-frm-results").innerHTML = "";
}

function roAddPlanLine() {
  const qty = parseFloat(document.getElementById("ro-frm-qty").value);
  if (!roPickedFormId || isNaN(qty) || qty <= 0) { toast("Pick a formulation and a positive qty", "error"); return; }
  reorderPlan.push({ formulation_id: roPickedFormId, name: roPickedFormName, qty: qty });
  roPickedFormId = null; document.getElementById("ro-frm-search").value = ""; document.getElementById("ro-frm-qty").value = "";
  renderReorderPlan();
}

function roRemovePlanLine(i) { reorderPlan.splice(i, 1); renderReorderPlan(); }

function renderReorderPlan() {
  document.getElementById("ro-plan").innerHTML = reorderPlan.map(function (l, i) {
    return '<tr><td>' + escapeHtml(l.name) + '</td><td>' + l.qty +
      '</td><td><button onclick="roRemovePlanLine(' + i + ')">✕</button></td></tr>';
  }).join("");
}

async function roCompute() {
  const body = { plan: reorderPlan.map(function (l) { return { formulation_id: l.formulation_id, qty: l.qty }; }),
                 include_below_par: document.getElementById("ro-below-par").checked };
  let rep;
  try { rep = await api("/api/reorder/report", { method: "POST", body: JSON.stringify(body) }); }
  catch (e) { toast("Compute failed: " + e.message, "error"); return; }
  const el = document.getElementById("ro-results");
  el.classList.add("active");
  if (!rep.groups.length) { el.innerHTML = '<p class="hint">Nothing to reorder — everything is at or above par/demand.</p>'; return; }
  el.innerHTML = rep.groups.map(function (g) {
    var rows = g.lines.map(function (l) {
      var warn = l.unit_warning ? ' <span title="unit mismatch" style="color:#b80">⚠</span>' : '';
      return '<tr><td>' + escapeHtml(l.ingredient) + warn + '</td><td>' + l.on_hand + '</td><td>' + l.on_order +
        '</td><td>' + l.par + '</td><td>' + l.demand + '</td><td><strong>' + l.shortfall + '</strong></td><td>' +
        l.suggested_qty + ' ' + escapeHtml(l.unit || "") + '</td><td>' + (l.est_cost != null ? '$' + l.est_cost : '—') + '</td></tr>';
    }).join("");
    return '<h4>' + escapeHtml(g.supplier) + ' — subtotal $' + g.subtotal + '</h4>' +
      '<table class="data"><thead><tr><th>Ingredient</th><th>On hand</th><th>On order</th><th>Par</th><th>Demand</th><th>Shortfall</th><th>Suggested</th><th>Est. cost</th></tr></thead><tbody>' +
      rows + '</tbody></table>';
  }).join("") + '<p style="margin-top:10px"><strong>Total: ' + rep.totals.lines + ' lines · est. $' + rep.totals.est_cost + '</strong></p>';
}
```

Wire `oninput` on `#ro-frm-search` → `roFrmSearch`; the add/compute buttons; the remove buttons are inline. Verify HTML parses, ids exist, existing tabs untouched. (Match the table CSS class the other tabs use — if they use `class="data"` or similar, reuse it; otherwise plain `<table>` is fine.)

- [ ] **Step 6: Register in `static/console-search-index.json`**

Add: `{ "title": "Reorder / Shopping List", "page": "Products", "url": "/admin/ingredients", "keywords": ["reorder","shopping","buy","purchase","demand","bom","par","shortfall","restock"] }`

- [ ] **Step 7: Commit**

```bash
git add static/admin-ingredients.html static/console-search-index.json
git commit -m "feat(reorder): admin Reorder tab + search index"
```

---

## Self-Review

- **Spec coverage:** bom_demand + on_order + reorder_report + _round_up_order (T1); GET/POST endpoints (T2); console tab + plan builder + grouped output + search index (T3). Formula `par + demand − on_hand − on_order` ✓. On-order from receiving over open POs, ingredient-only ✓. Read-only (no draft POs) ✓. Units summed + warning, not converted ✓. No new tables ✓.
- **Placeholders:** none — full code in every code step.
- **Type consistency:** `bom_demand`/`on_order_by_ingredient`/`reorder_report`/`_round_up_order` used identically across module, endpoints, tests. `_ro` alias. `reorder_report` returns `{groups:[{supplier_id,supplier,lines:[...],subtotal}], totals:{lines,est_cost}, plan_echo}` — the console reads exactly those keys; each line has `ingredient_id,ingredient,on_hand,on_order,par,demand,shortfall,suggested_qty,unit,price_per_unit,est_cost,unit_warning`. The console uses the real `api(path,opts)` shape (returns data, throws) and the index-array picker (no string in onclick).
- **Reviewer note:** the console table CSS class (`class="data"`) is assumed — the implementer should match whatever class the existing tabs' tables use (or plain `<table>`), and confirm in the report.
