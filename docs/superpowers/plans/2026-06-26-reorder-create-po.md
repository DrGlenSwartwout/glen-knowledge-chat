# Reorder → Draft PO (Sub-project C1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Create draft PO" button to the reorder report that turns a supplier group's shortfall lines into a draft purchase order, opened in the PO tab.

**Architecture:** New `purchase_orders.create_draft_po(cx, …)` (forward PO creation — deferred in the FMP migration) + a `reorder.create_po` BOS action (LOW_WRITE, OWNER/OPS) in a new `dashboard/reorder_actions.py`, registered in `app.py`; a per-supplier button in `renderReorderReport` (group-index pattern, never JSON-in-onclick) that posts the action and opens the new draft.

**Tech Stack:** Python (`dashboard/*.py`, sqlite), the BOS action layer, vanilla JS (`static/admin-ingredients.html`, `static/console-products.html`), pytest + headless Playwright (mocked) render-verify.

## Global Constraints

- **C1 creates the draft only** — reviewing/editing is the existing PO tab; ordering/sending a PO (vendor #, email) is a future step. No schema change.
- Draft PO mapping (verbatim): `purchase_orders` row `status='draft'`, `po_date=today`, `vendor_po_no='DRAFT-'+YYYYMMDD+'-'+supplier_id`; per line → `po_items` `item_kind='ingredient'`, `item_label=<ingredient name>`, `ingredient_id`, `qty=<suggested_qty>`, `qty_unit=<unit>`, `cost=<price_per_unit>` (**NULL when the line has no price — include, don't drop**), `extras=json({unit_size,packs,est_cost})`.
- Action `reorder.create_po`: `risk_tier=LOW_WRITE`, `permission=(OWNER, OPS)`. (Rae's token resolves to OWNER via sub-project A, so she's covered.)
- **UI must NOT `JSON.stringify` line data into an `onclick`** (that has broken on quotes/apostrophes here repeatedly). Stash the report and pass a group **index**; the render-verify tests an apostrophe supplier name.
- Console-key / OWNER-token gated.
- **Test env:** the Python tests here are pure-module (`import dashboard.purchase_orders` / `reorder_actions` — no network import), so run with **plain `python3 -m pytest`** (no Doppler needed). The render-verify runs the app via `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/<scratch> CONSOLE_SECRET=test-secret PORT=<p> python3 app.py` (mkdir scratch first).

---

### Task 1: Backend — `create_draft_po` + the `reorder.create_po` action

**Files:**
- Modify: `dashboard/purchase_orders.py` (add `create_draft_po`).
- Create: `dashboard/reorder_actions.py`.
- Modify: `app.py` (register the action module, in the block near `from dashboard import reviews_actions as _ra` ~line 24125).
- Test: `tests/test_create_draft_po.py` (create).

**Interfaces:**
- Consumes: the `purchase_orders`/`po_items` tables (`init_purchase_orders_schema`); `dashboard.actions` (`Action`, `register_action`, `get_action`, `LOW_WRITE`); `dashboard.rbac` (`OWNER`, `OPS`).
- Produces: `purchase_orders.create_draft_po(cx, supplier_id, supplier_name, lines) -> {"po_id": int, "line_count": int}`; the registered `reorder.create_po` action with executor `_exec_create_po(params, ctx) -> {"ok": bool, "po_id"?, "line_count"?, "error"?}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_create_draft_po.py`:

```python
"""C1: create_draft_po inserts a draft PO + items; reorder.create_po action wraps it."""
import sqlite3
import pytest


@pytest.fixture
def cx():
    cx = sqlite3.connect(":memory:")
    from dashboard import purchase_orders as po
    po.init_purchase_orders_schema(cx)
    return cx


def test_create_draft_po_inserts_po_and_items(cx):
    from dashboard import purchase_orders as po
    lines = [
        {"ingredient_id": 5, "ingredient": "Ashwagandha", "suggested_qty": 100.0, "unit": "g",
         "price_per_unit": 12.5, "unit_size": 50.0, "packs": 2, "est_cost": 25.0},
        {"ingredient_id": 6, "ingredient": "NoPrice", "suggested_qty": 30.0, "unit": "g",
         "price_per_unit": None, "unit_size": None, "packs": None, "est_cost": None},
    ]
    res = po.create_draft_po(cx, 9, "Acme Botanicals", lines)
    assert res["line_count"] == 2 and res["po_id"]
    hdr = cx.execute("SELECT supplier_id, supplier_name, status, vendor_po_no FROM purchase_orders WHERE id=?",
                     (res["po_id"],)).fetchone()
    assert hdr[0] == 9 and hdr[1] == "Acme Botanicals" and hdr[2] == "draft" and hdr[3].startswith("DRAFT-")
    items = cx.execute("SELECT ingredient_id, qty, qty_unit, cost, item_kind FROM po_items WHERE po_id=? ORDER BY ingredient_id",
                       (res["po_id"],)).fetchall()
    assert items[0] == (5, 100.0, "g", 12.5, "ingredient")
    assert items[1][0] == 6 and items[1][3] is None          # no-price line: cost NULL, still inserted


def test_create_draft_po_skips_lines_missing_id_or_qty(cx):
    from dashboard import purchase_orders as po
    res = po.create_draft_po(cx, 1, "X", [{"ingredient": "bad", "suggested_qty": 5}, {"ingredient_id": 7, "suggested_qty": None}])
    assert res["line_count"] == 0


def test_exec_create_po_ok(cx):
    from dashboard import reorder_actions as ra
    res = ra._exec_create_po({"supplier_id": 9, "supplier_name": "Acme",
                              "lines": [{"ingredient_id": 5, "suggested_qty": 100.0, "unit": "g", "price_per_unit": 12.5}]},
                             {"cx": cx, "actor": None})
    assert res["ok"] is True and res["po_id"]


def test_exec_create_po_no_supplier(cx):
    from dashboard import reorder_actions as ra
    res = ra._exec_create_po({"supplier_id": None, "lines": []}, {"cx": cx})
    assert res["ok"] is False


def test_action_registered_metadata():
    from dashboard import reorder_actions as ra
    from dashboard.actions import get_action, LOW_WRITE
    from dashboard.rbac import OWNER, OPS
    ra.register()
    a = get_action("reorder.create_po")
    assert a is not None and a.risk_tier == LOW_WRITE and a.permission == (OWNER, OPS)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_create_draft_po.py -q`
Expected: FAIL — `module 'dashboard.purchase_orders' has no attribute 'create_draft_po'` / no `dashboard.reorder_actions`.

- [ ] **Step 3: Add `create_draft_po` to `dashboard/purchase_orders.py`**

At the top of `dashboard/purchase_orders.py`, ensure `import json` and `from datetime import date` are present (add if missing). Add:

```python
def create_draft_po(cx, supplier_id, supplier_name, lines):
    """Create a draft purchase order + its line items from reorder-report lines.
    `cx` is an open sqlite3 connection. Lines missing ingredient_id or suggested_qty are
    skipped; price_per_unit may be None (cost stored NULL). Returns {po_id, line_count}."""
    today = date.today().isoformat()
    vendor_po_no = "DRAFT-" + today.replace("-", "") + "-" + str(supplier_id)
    cur = cx.execute(
        "INSERT INTO purchase_orders (supplier_id, supplier_name, vendor_po_no, po_date, status) "
        "VALUES (?,?,?,?,'draft')",
        (supplier_id, supplier_name or "", vendor_po_no, today))
    po_id = cur.lastrowid
    n = 0
    for ln in (lines or []):
        ing_id = ln.get("ingredient_id")
        qty = ln.get("suggested_qty")
        if ing_id is None or qty is None:
            continue
        c = ln.get("price_per_unit")
        cost = float(c) if c not in (None, "") else None
        extras = json.dumps({k: ln.get(k) for k in ("unit_size", "packs", "est_cost")})
        cx.execute(
            "INSERT INTO po_items (po_id, item_kind, item_label, ingredient_id, qty, qty_unit, cost, extras) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (po_id, "ingredient", ln.get("ingredient") or "", int(ing_id),
             float(qty), ln.get("unit"), cost, extras))
        n += 1
    cx.commit()
    return {"po_id": po_id, "line_count": n}
```

- [ ] **Step 4: Create `dashboard/reorder_actions.py`**

```python
"""C1 console action: create a draft purchase order from a reorder-report supplier group."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import purchase_orders as _po

_LINE_KEYS = ("ingredient_id", "ingredient", "suggested_qty", "unit",
              "price_per_unit", "unit_size", "packs", "est_cost")


def _sanitize(lines):
    out = []
    for ln in (lines or []):
        if not isinstance(ln, dict):
            continue
        if ln.get("ingredient_id") is None or ln.get("suggested_qty") is None:
            continue
        out.append({k: ln.get(k) for k in _LINE_KEYS})
    return out


def _exec_create_po(params, ctx):
    sid = params.get("supplier_id")
    if sid is None:
        return {"ok": False, "error": "no supplier — assign a preferred source first"}
    lines = _sanitize(params.get("lines"))
    if not lines:
        return {"ok": False, "error": "no orderable lines"}
    res = _po.create_draft_po(ctx["cx"], int(sid), params.get("supplier_name") or "", lines)
    return {"ok": True, **res}


def register():
    if get_action("reorder.create_po"):
        return
    register_action(Action(
        key="reorder.create_po", module="reorder", title="Create draft PO",
        description="Create a draft purchase order from a reorder-report supplier group.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_create_po))
```

- [ ] **Step 5: Register the action in `app.py`**

In `app.py`, in the action-registration block (next to `from dashboard import reviews_actions as _ra` ~line 24125), add:
```python
from dashboard import reorder_actions as _roa
_roa.register()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_create_draft_po.py -q`
Expected: PASS — 5 passed.

- [ ] **Step 7: Commit**
```bash
git add dashboard/purchase_orders.py dashboard/reorder_actions.py app.py tests/test_create_draft_po.py
git commit -m "feat(po): create_draft_po + reorder.create_po action (draft PO from reorder report)"
```

---

### Task 2: UI — "Create draft PO" button + Backorders link

**Files:**
- Modify: `static/admin-ingredients.html` (`renderReorderReport`, add `createDraftPO`, `?tab=` init).
- Modify: `static/console-products.html` (Backorders "Open reorder report →" link).
- Verify: headless Playwright (mocked report + action).

**Interfaces:**
- Consumes: `POST /api/action/reorder.create_po` (Task 1); the page's `api(path, opts)` (~line 1148), `showTab(name)` (~1165), `openPo(id)` (~2411), `escapeHtml` (~1159).
- Produces: a per-supplier-group "Create draft PO" button + `createDraftPO(idx)`; `?tab=` deep-link support.

- [ ] **Step 1: Stash the report + render the button/note in `renderReorderReport`**

In `static/admin-ingredients.html`, in `renderReorderReport(el, rep)` (~line 2902): immediately on entry, stash the report so the button can reference a group by index (no JSON-in-onclick):
```javascript
  window._lastReorderRep = rep;
```
In the per-supplier group header string (currently `'<h4 …>' + escapeHtml(g.supplier) + ' — subtotal $' + g.subtotal + '</h4>'`), append, using the group's index `i` from the `rep.groups.map(function(g, i){…})` callback (add the `, i` index param to the map callback if absent):
```javascript
  + (g.supplier_id != null
     ? ' <button class="btn" style="margin-left:10px" onclick="createDraftPO(' + i + ')">Create draft PO</button>'
     : ' <span style="margin-left:10px;color:var(--muted);font-size:12px">Assign a preferred source to order these</span>')
```

- [ ] **Step 2: Add `createDraftPO(idx)`**

Add near `renderReorderReport` (or with the other Reorder-tab functions):
```javascript
async function createDraftPO(idx) {
  var rep = window._lastReorderRep || {};
  var g = (rep.groups || [])[idx];
  if (!g || g.supplier_id == null) return;
  if (!confirm('Create draft PO for ' + g.supplier + ' — ' + (g.lines || []).length + ' lines, ~$' + g.subtotal + '?')) return;
  var res = await api('/api/action/reorder.create_po', {
    method: 'POST',
    body: JSON.stringify({ supplier_id: g.supplier_id, supplier_name: g.supplier, lines: g.lines })
  });
  if (res && res.ok && res.po_id) {
    showTab('po');
    openPo(res.po_id);
  } else {
    alert('Could not create PO: ' + ((res && res.error) || 'unknown error'));
  }
}
```
(Match the exact shape `api()` returns — if `api()` already parses JSON and returns the body, use it directly as above; if it returns a Response, adapt to `await res.json()`. Read `api()` at ~line 1148 first and conform.)

- [ ] **Step 3: Add `?tab=` deep-link support**

Find where the page selects its initial tab on load (the bottom-of-script init, near where `showTab('ingredients')` would run or the default tab is set). Add, before/at init:
```javascript
  (function(){ var t = new URLSearchParams(location.search).get('tab');
    if (t && document.getElementById('tab-' + t)) showTab(t); })();
```
(`tab-reorder` exists at line ~1059; `showTab('reorder')` is a valid call.)

- [ ] **Step 4: Backorders link in `static/console-products.html`**

In the Backorders tab section, add a shortcut link (carry the console key the page already resolves — match how the page references its key variable; if it uses a `KEY`/`key()` like other console pages, use it):
```html
<a href="/admin/ingredients?tab=reorder" id="bo-reorder-link">Open reorder report →</a>
```
If the page carries a key in JS, set the href to include it on load (e.g. `document.getElementById('bo-reorder-link').href = '/admin/ingredients?tab=reorder&key=' + encodeURIComponent(theKey)`); otherwise the static link is fine (op-nav/localStorage will carry the key on the destination).

- [ ] **Step 5: Render-verify (headless, mocked)**

`mkdir -p /tmp/createpo-test`; start the app on PORT=5097. Save `/tmp/createpo-test/cv.py`:
```python
from playwright.sync_api import sync_playwright
import json
B = "http://127.0.0.1:5097/admin/ingredients?key=test-secret&tab=reorder"
REPORT = {"ok": True, "data": {  # match the shape /api/reorder/report returns (ok-wrapped); adapt if it returns the report directly
  "groups": [
    {"supplier_id": 9, "supplier": "O'Brien Botanicals", "subtotal": 25.0,
     "lines": [{"ingredient_id": 5, "ingredient": "Ashwagandha", "suggested_qty": 100.0, "unit": "g",
                "price_per_unit": 12.5, "unit_size": 50.0, "packs": 2, "est_cost": 25.0,
                "on_hand": 0, "on_order": 0, "par": 100, "demand": 0, "shortfall": 100}]},
    {"supplier_id": None, "supplier": "— no preferred source —", "subtotal": 0.0,
     "lines": [{"ingredient_id": 6, "ingredient": "Orphan", "suggested_qty": 30.0, "unit": "g",
                "price_per_unit": None, "on_hand": 0, "on_order": 0, "par": 30, "demand": 0, "shortfall": 30}]}
  ], "totals": {"lines": 2, "est_cost": 25.0}, "plan_echo": []}}
created = {"n": 0}
def handle(route):
    u = route.request.url
    if "/api/reorder/report" in u:
        return route.fulfill(status=200, content_type="application/json", body=json.dumps(REPORT))
    if "/api/action/reorder.create_po" in u:
        created["n"] += 1
        return route.fulfill(status=200, content_type="application/json", body=json.dumps({"ok": True, "po_id": 7, "line_count": 1}))
    if "/api/po/7" in u:
        return route.fulfill(status=200, content_type="application/json", body=json.dumps({"ok": True, "data": {"id": 7, "status": "draft", "supplier_name": "O'Brien Botanicals", "items": [], "receiving": []}}))
    return route.continue_()
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width":1280,"height":900}); errs=[]
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append("CJS:"+m.text) if (m.type=="error" and "Failed to load resource" not in m.text) else None)
    pg.route("**/api/**", handle)
    pg.on("dialog", lambda d: d.accept())   # auto-accept the confirm()
    pg.goto(B, wait_until="networkidle"); pg.wait_for_timeout(800)
    # Trigger the reorder report render (click the reorder compute/run control if the tab doesn't auto-load).
    # The Reorder tab must be active via ?tab=reorder; run its report if a button is needed:
    pg.evaluate("() => { if (typeof roCompute === 'function') roCompute(); }")
    pg.wait_for_timeout(800)
    s1 = pg.evaluate("""()=>({
      reorderTabActive: getComputedStyle(document.getElementById('tab-reorder')).display !== 'none',
      createBtns: document.querySelectorAll('#tab-reorder button').length,
      hasCreate: [...document.querySelectorAll('#tab-reorder button')].some(b=>/Create draft PO/.test(b.textContent)),
      hasNote: /Assign a preferred source/.test(document.getElementById('tab-reorder').innerText)
    })""")
    print("RENDER:", s1)
    # click the Create draft PO button → confirm auto-accepted → action posts → PO tab opens
    pg.evaluate("""()=>{ var b=[...document.querySelectorAll('#tab-reorder button')].find(x=>/Create draft PO/.test(x.textContent)); if(b) b.click(); }""")
    pg.wait_for_timeout(1000)
    s2 = pg.evaluate("()=>({poTabActive: getComputedStyle(document.getElementById('tab-po')).display !== 'none'})")
    print("AFTER CLICK:", s2, "action posts:", created["n"], "errs:", errs or "NONE")
    assert s1["hasCreate"] and s1["hasNote"], s1
    assert created["n"] == 1 and s2["poTabActive"], (created, s2)
    assert not errs, errs
    b.close(); print("OK")
```
Run `python3 /tmp/createpo-test/cv.py` → `OK`, `errs: NONE`: the priced group shows **Create draft PO**, the no-source group shows the **note**, clicking the button (apostrophe supplier name "O'Brien" — proves no JSON-in-onclick breakage) posts the action **once** and switches to the **PO tab**. Then load `/admin/ingredients?key=test-secret&tab=reorder` and assert the Reorder tab is active (the `?tab=` support). Check the Backorders link separately: `/console/products?key=test-secret` shows the "Open reorder report →" link. Kill the server.

(Note: read the real `api()` return shape and the exact `/api/reorder/report` response envelope before finalizing the mock — adjust `REPORT`/`res` handling so the assertions hold against the real code.)

- [ ] **Step 6: Commit**
```bash
git add static/admin-ingredients.html static/console-products.html
git commit -m "feat(console): Create draft PO button on the reorder report + Backorders link"
```

---

## Verification (whole sub-project)

- `python3 -m pytest tests/test_create_draft_po.py -q` → 5 pass (create_draft_po inserts draft + items incl. NULL-cost no-price line; `_exec_create_po` ok + no-supplier error; action metadata LOW_WRITE/(OWNER,OPS)).
- Render-verify: Create-draft-PO button on priced supplier groups, note on the no-source group, the button posts `reorder.create_po` once and opens the PO tab; apostrophe supplier name works (no JSON-in-onclick break); `?tab=reorder` deep-links; Backorders shows the reorder link; zero JS errors.
- No schema change; `/api/reorder/report` and the PO read endpoints untouched.

## Self-Review Notes

- **Spec coverage:** `create_draft_po` (Task 1) ✓; `reorder.create_po` action + register (Task 1) ✓; no-price line → cost NULL (Task 1 + test) ✓; LOW_WRITE/(OWNER,OPS) (Task 1 + test) ✓; button index-pattern not JSON-in-onclick (Task 2 + apostrophe test) ✓; no-source-group note (Task 2) ✓; open new draft in PO tab (Task 2) ✓; `?tab=` support (Task 2) ✓; Backorders link (Task 2) ✓; draft-only/no-send (no ordering code anywhere) ✓.
- **Type consistency:** `create_draft_po(cx, supplier_id, supplier_name, lines) -> {po_id, line_count}`; `_exec_create_po(params, ctx) -> {ok, po_id?, error?}`; JS `createDraftPO(idx)` reads `window._lastReorderRep.groups[idx]` — names consistent across tasks.
- **YAGNI:** no PO send/email, no Backorders data bridge, client-sent lines (sanitized) rather than server re-derivation.
