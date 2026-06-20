# Product Page Images — Phase C1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log per-product distinct-session image exposures, and build a console leaderboard ranking prompt variations and models by Wilson-lower-bound pick-rate (votes ÷ exposures-of-products-containing-the-tag) across all products.

**Architecture:** A new `sales_image_exposures` table records one row per (product, session), written from the data endpoint when the vote feature is on and the grid is shown (dedup via UNIQUE, so polling/reloads don't inflate). A pure aggregation module joins `sales_page_votes` × `sales_page_images` × exposures into ranked variation/model tables and renders HTML. A console-auth route exposes it. Read-only; nothing on the public page changes.

**Tech Stack:** Python 3 / Flask, SQLite (`LOG_DB`), pytest.

## Global Constraints

- **No new public flag.** Exposure logging is gated only by the existing `_SALES_IMAGE_VOTE_ENABLED` (same lifecycle as votes). The leaderboard route is gated by `_sales_console_ok()` (the Phase-5 console gate: `dashboard.CONSOLE_SECRET` via `X-Console-Key` header or `?key=`). Nothing on the public page changes.
- **Impressions = distinct sessions per product** (`UNIQUE(product_slug, session_id)`); empty `session_id` is ignored.
- **A tag's impressions = sum of exposures over the distinct products whose ready images contain that tag** (applies to BOTH `prompt_variant_id` and `model_id`). Votes per tag = count of `sales_page_votes` rows with that tag.
- **Ranking = Wilson lower bound** (z = 1.96) on (votes, impressions), descending; `low_volume = impressions < min_volume` with `min_volume` default **30**.
- Tests: pytest, `sqlite3.connect(":memory:")` per test, import `dashboard.*` directly, no live network. New file `tests/test_sales_pages_phase_c.py`. Follow `tests/test_sales_pages_phase4b.py` style.
- Sandbox: use `python3` (no `python`). `import app` CANNOT run here (Pinecone client at import → network auth) — verify app.py edits with `python3 -m py_compile app.py` + the unit-tested helpers they call.
- Work in worktree `/tmp/wt-deploy-chat-db16e904` (branch `sess/db16e904`, at Phase B tip `d80dae4` which is in `main`). Commit per task. No edits to `main`.

---

### Task 1: Exposure logging table

**Files:**
- Create: `dashboard/sales_image_exposures.py`
- Test: `tests/test_sales_pages_phase_c.py`

**Interfaces:**
- Produces: `record(cx, slug, session_id)` (insert-or-ignore, skips empty session); `per_product_counts(cx) -> {slug: count}` (count = distinct sessions); `init_table(cx)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sales_pages_phase_c.py
import sqlite3
from dashboard import sales_image_exposures as ex

def _cx(): return sqlite3.connect(":memory:")

def test_record_dedups_per_session():
    cx = _cx()
    ex.record(cx, "a", "s1")
    ex.record(cx, "a", "s1")     # same session -> no new row
    ex.record(cx, "a", "s2")     # different session
    ex.record(cx, "b", "s1")     # different product
    assert ex.per_product_counts(cx) == {"a": 2, "b": 1}

def test_record_ignores_empty_session():
    cx = _cx()
    ex.record(cx, "a", "")
    ex.record(cx, "a", None)
    assert ex.per_product_counts(cx) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sales_pages_phase_c.py -q`
Expected: FAIL (module `sales_image_exposures` does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/sales_image_exposures.py
import datetime

def _now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def init_table(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS sales_image_exposures ("
               "product_slug TEXT, session_id TEXT, created_at TEXT DEFAULT '', "
               "UNIQUE(product_slug, session_id))")
    cx.commit()

def record(cx, slug, session_id):
    session_id = (session_id or "").strip()
    if not session_id:
        return
    init_table(cx)
    cx.execute("INSERT INTO sales_image_exposures (product_slug, session_id, created_at) "
               "VALUES (?,?,?) ON CONFLICT(product_slug, session_id) DO NOTHING",
               (slug, session_id, _now()))
    cx.commit()

def per_product_counts(cx):
    init_table(cx)
    rows = cx.execute("SELECT product_slug, COUNT(*) FROM sales_image_exposures "
                      "GROUP BY product_slug").fetchall()
    return {r[0]: r[1] for r in rows}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_sales_pages_phase_c.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_image_exposures.py tests/test_sales_pages_phase_c.py
git commit -m "feat(sales-img): per-product distinct-session exposure logging (Phase C1 task 1)"
```

---

### Task 2: Leaderboard aggregation + HTML

**Files:**
- Create: `dashboard/sales_image_leaderboard.py`
- Test: `tests/test_sales_pages_phase_c.py` (append)

**Interfaces:**
- Consumes: `sales_image_exposures.per_product_counts` (Task 1); tables `sales_page_votes` (cols `prompt_variant_id`, `model_id`), `sales_page_images` (cols `product_slug`, `prompt_variant_id`, `model_id`, `state`), and label tables `sales_prompt_variations(id,label)` / `sales_image_models(id,label)`.
- Produces: `wilson_lower(pos, n, z=1.96) -> float`; `leaderboard(cx, min_volume=30) -> {"variations": [row...], "models": [row...]}` where `row = {key, label, votes, impressions, rate, wilson, low_volume, rank}` sorted by `wilson` desc; `render_html(data) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sales_pages_phase_c.py
from dashboard import sales_image_leaderboard as lb
from dashboard import sales_images as si
from dashboard import sales_votes as sv

def test_wilson_lower_rewards_volume_at_equal_rate():
    assert lb.wilson_lower(0, 0) == 0.0
    assert lb.wilson_lower(80, 100) > lb.wilson_lower(8, 10)   # same 0.8 rate, more data -> higher lower bound
    assert 0.0 < lb.wilson_lower(8, 10) < 0.8

def test_leaderboard_model_impressions_use_containing_products():
    cx = _cx()
    # product a contains models flux + imagen; product b contains flux + recraft
    si.record_image(cx, "a", "botanical", 1, "a-b1.png", prompt_variant_id=1, model_id="flux-1.1-pro")
    si.record_image(cx, "a", "mechanism", 1, "a-m1.png", prompt_variant_id=5, model_id="imagen-4")
    si.record_image(cx, "b", "botanical", 1, "b-b1.png", prompt_variant_id=1, model_id="flux-1.1-pro")
    si.record_image(cx, "b", "mechanism", 1, "b-m1.png", prompt_variant_id=5, model_id="recraft-v3")
    for i in range(10): ex.record(cx, "a", f"a{i}")   # a: 10 sessions
    for i in range(5):  ex.record(cx, "b", f"b{i}")   # b: 5 sessions
    # votes (tagged with model_id, Phase-B style)
    for i in range(6): sv.record_pick(cx, "a", "botanical", 1, f"va{i}", model_id="flux-1.1-pro", prompt_variant_id=1)
    for i in range(2): sv.record_pick(cx, "a", "mechanism", 1, f"vm{i}", model_id="imagen-4", prompt_variant_id=5)
    sv.record_pick(cx, "b", "mechanism", 1, "vb0", model_id="recraft-v3", prompt_variant_id=5)
    data = lb.leaderboard(cx, min_volume=8)
    models = {r["key"]: r for r in data["models"]}
    assert models["flux-1.1-pro"]["impressions"] == 15   # a(10) + b(5)
    assert models["imagen-4"]["impressions"] == 10        # a only
    assert models["recraft-v3"]["impressions"] == 5       # b only
    assert models["flux-1.1-pro"]["votes"] == 6
    assert abs(models["flux-1.1-pro"]["rate"] - 6/15) < 1e-9
    assert models["recraft-v3"]["low_volume"] is True     # 5 < 8
    assert data["models"][0]["rank"] == 1                 # ranked, wilson desc
    # variations present too
    keys = {r["key"] for r in data["variations"]}
    assert 1 in keys and 5 in keys
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sales_pages_phase_c.py -q`
Expected: FAIL (`sales_image_leaderboard` does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/sales_image_leaderboard.py
def wilson_lower(pos, n, z=1.96):
    if n <= 0:
        return 0.0
    phat = pos / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * ((phat * (1 - phat) + z * z / (4 * n)) / n) ** 0.5
    return (centre - margin) / denom

def _labels(cx, table):
    try:
        return {r[0]: r[1] for r in cx.execute(f"SELECT id, label FROM {table}").fetchall()}
    except Exception:
        return {}

def _agg(cx, tag_col, label_map, min_volume):
    from dashboard import sales_image_exposures as _ex
    exp = _ex.per_product_counts(cx)
    votes = {tag: n for tag, n in cx.execute(
        f"SELECT {tag_col}, COUNT(*) FROM sales_page_votes "
        f"WHERE {tag_col} IS NOT NULL GROUP BY {tag_col}").fetchall()}
    prods = {}
    for slug, tag in cx.execute(
        f"SELECT DISTINCT product_slug, {tag_col} FROM sales_page_images "
        f"WHERE {tag_col} IS NOT NULL AND state='ready'").fetchall():
        prods.setdefault(tag, set()).add(slug)
    rows = []
    for tag in (set(votes) | set(prods)):
        impr = sum(exp.get(p, 0) for p in prods.get(tag, ()))
        v = votes.get(tag, 0)
        rows.append({"key": tag, "label": label_map.get(tag, str(tag)),
                     "votes": v, "impressions": impr,
                     "rate": (v / impr) if impr else 0.0,
                     "wilson": wilson_lower(v, impr),
                     "low_volume": impr < min_volume})
    rows.sort(key=lambda r: r["wilson"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows

def leaderboard(cx, min_volume=30):
    # tag_col values are fixed literals (not user input) -> safe to interpolate
    var_labels = _labels(cx, "sales_prompt_variations")
    model_labels = _labels(cx, "sales_image_models")
    return {"variations": _agg(cx, "prompt_variant_id", var_labels, min_volume),
            "models": _agg(cx, "model_id", model_labels, min_volume)}

def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

def render_html(data):
    def _rows(items):
        out = []
        for r in items:
            badge = ' <span style="color:#b00">low volume</span>' if r["low_volume"] else ""
            out.append(f"<tr><td>{r['rank']}</td><td>{_esc(r['label'])}</td>"
                       f"<td>{r['rate']*100:.1f}%</td><td>{r['votes']}</td>"
                       f"<td>{r['impressions']}</td><td>{badge}</td></tr>")
        return "".join(out)
    head = ("<tr><th>Rank</th><th>Label</th><th>Pick-rate</th>"
            "<th>Votes</th><th>Exposures</th><th></th></tr>")
    return ("<!doctype html><meta charset=utf-8><title>Image Leaderboard</title>"
            "<style>body{font-family:system-ui;margin:2rem}"
            "table{border-collapse:collapse;margin-bottom:2rem}"
            "td,th{border:1px solid #ccc;padding:4px 10px;text-align:left}</style>"
            "<h1>Image Split-Test Leaderboard</h1>"
            f"<h2>Prompt Variations</h2><table>{head}{_rows(data['variations'])}</table>"
            f"<h2>Models</h2><table>{head}{_rows(data['models'])}</table>")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_sales_pages_phase_c.py -q`
Expected: PASS (4 tests total).

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_image_leaderboard.py tests/test_sales_pages_phase_c.py
git commit -m "feat(sales-img): leaderboard aggregation + Wilson ranking + HTML (Phase C1 task 2)"
```

---

### Task 3: Wire exposure logging into the data endpoint

**Files:**
- Modify: `app.py` — the grouped+vote branch (inside `if _SALES_IMAGE_VOTE_ENABLED:`, ~line 3271-3276)

**Interfaces:**
- Consumes: `sales_image_exposures.record` (Task 1); `_SALES_IMAGE_VOTE_ENABLED`, `_vsess`, `_grouped`, `_cx2` (already in scope in that branch).

- [ ] **Step 1: Add the exposure record**

In `app.py`, the grouped+vote branch currently ends with the picks line:

```python
                        if _SALES_IMAGE_VOTE_ENABLED:
                            from dashboard import sales_votes as _sv2
                            _vsess = request.cookies.get("amg_session", "")
                            _vau = get_authenticated_user(request)
                            _vem = ((_vau or {}).get("email") or "").strip().lower() if _vau else ""
                            _img_sec["body"]["picks"] = _sv2.get_picks(_cx2, slug, session_id=_vsess, email=_vem)
```

Append, immediately after the `_img_sec["body"]["picks"] = ...` line (same indentation, still inside `if _SALES_IMAGE_VOTE_ENABLED:`):

```python
                            if any(_grouped.values()):
                                from dashboard import sales_image_exposures as _ex2
                                _ex2.record(_cx2, slug, _vsess)
```

- [ ] **Step 2: Verify it compiles**

Run: `python3 -m py_compile app.py`
Expected: succeeds. (Do NOT run `import app` — Pinecone-at-import fails in this sandbox.)

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(sales-img): log image exposures from the data endpoint (Phase C1 task 3)"
```

---

### Task 4: Console leaderboard route

**Files:**
- Modify: `app.py` — add `GET /console/image-leaderboard` (near the other `/console/*` routes, e.g. after `/console/sales-pages`)

**Interfaces:**
- Consumes: `_sales_console_ok()` (existing), `sales_image_leaderboard.leaderboard` + `render_html` (Task 2).

- [ ] **Step 1: Add the route**

Add near the other `@app.route("/console/...")` handlers in `app.py`:

```python
@app.route("/console/image-leaderboard")
def console_image_leaderboard():
    _gate = _sales_console_ok()
    if _gate is not None:
        return _gate
    from dashboard import sales_image_leaderboard as _lb
    with sqlite3.connect(LOG_DB) as cx:
        data = _lb.leaderboard(cx)
    if request.args.get("format") == "json":
        return jsonify(data)
    return Response(_lb.render_html(data), mimetype="text/html")
```

(`Response` is from Flask; confirm it's imported at the top of `app.py` — search `from flask import`. If `Response` is not already in that import list, add it.)

- [ ] **Step 2: Verify it compiles**

Run: `python3 -m py_compile app.py`
Expected: succeeds.

Also confirm `Response` is importable in app's flask import line:
Run: `grep -n "from flask import" app.py | head -1`
Expected: the import line includes `Response` (add it if missing, then re-run py_compile).

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(sales-img): /console/image-leaderboard route (Phase C1 task 4)"
```

---

### Task 5: Regression + flag-off parity

**Files:** none (verification)

- [ ] **Step 1: Run the Phase C + B + A unit suites**

Run: `python3 -m pytest tests/test_sales_pages_phase_c.py tests/test_sales_pages_phase_b.py tests/test_sales_pages_phase_a.py -q`
Expected: all PASS (4 Phase C + 5 Phase B + 18 Phase A = 27).

- [ ] **Step 2: Confirm flag-off parity + compile**

With `SALES_PAGES_IMAGE_VOTE` unset: the data endpoint's `if _SALES_IMAGE_VOTE_ENABLED:` block is skipped, so NO exposure is recorded (and no `picks`). The `/console/image-leaderboard` route still works (console-auth) and simply shows whatever data exists (empty tables if none). Confirm:
Run: `python3 -m py_compile app.py` → clean.
Run: `python3 -c "from dashboard import sales_image_leaderboard as lb; import sqlite3; cx=sqlite3.connect(':memory:'); print(lb.leaderboard(cx))"`
Expected: `{'variations': [], 'models': []}` (empty DB → empty leaderboard, no crash).

- [ ] **Step 3: Commit (if any fixups)**

```bash
git add -A && git commit -m "test(sales-img): Phase C1 regression pass" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:** exposure table + record + counts (T1), wilson + leaderboard aggregation + render_html (T2), data-endpoint exposure wiring (T3), console route + JSON/HTML (T4), regression + flag-off (T5). Spec sections 1-4, data flow, testing, and the "no new flag / console-auth / app-import" notes all map to tasks. Out-of-scope (C2/C3, per-product drill-down, rolling window) correctly have no task. ✔

**Placeholder scan:** all code blocks complete; verifications are concrete commands. The console HTML render check is manual (console-auth, app can't boot) but the underlying `render_html`/`leaderboard` are unit-tested. No "TODO/handle edge cases" in code.

**Type consistency:** `record(cx, slug, session_id)` / `per_product_counts(cx) -> {slug:int}` (T1) match `_agg`'s `exp = per_product_counts(cx)` and the data-endpoint call `_ex2.record(_cx2, slug, _vsess)` (T3); `leaderboard(cx, min_volume=30) -> {"variations":[...],"models":[...]}` with row keys `{key,label,votes,impressions,rate,wilson,low_volume,rank}` (T2) match the route's `jsonify(data)` + `render_html(data)` consumption (T4) and the T2 test asserts; `wilson_lower(pos,n,z=1.96)` consistent. The vote rows carry `prompt_variant_id`/`model_id` (Phase B) which `_agg` reads.
