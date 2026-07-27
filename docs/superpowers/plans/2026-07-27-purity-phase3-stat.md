# Purity Phase 3 (slice 2) — public aggregate stat + embeddable widget — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the "X% of professional-quality products we've screened fail our purity standard" authority number as a public, live-computed endpoint plus a self-contained embeddable widget page.

**Architecture:** A pure `aggregate_confirmed(cx)` counts confirmed `product_ratings` by color; a public unauthenticated `GET /api/purity/stats` turns that into counts + percentages + a headline (aggregate ONLY — never a product name); a self-contained `static/purity-stat.html` widget (served at `/purity-stat`) fetches the endpoint and renders the blunt headline for iframe/embed anywhere.

**Tech Stack:** Python 3.9 / Flask (sqlite dev, Postgres adapter prod), vanilla JS static widget.

## Global Constraints

- **Public + aggregate-only.** `GET /api/purity/stats` is unauthenticated (no `@require_console_key`, no `_portal_console_ok`). It returns ONLY counts/percentages/headline — NEVER a product name or brand (the spec's primary-use boundary: only the aggregate rate is public, never a per-competitor badge). A test asserts no product identity leaks.
- **Confirmed-only, live.** Counts come from `status='confirmed'` rows (unrated is never confirmed). Computed on each request so the denominator grows as more are rated.
- **"fail" = red / confirmed.** Headline verbatim: `"<fail_pct>% of professional-quality products we've screened fail our purity standard"`. `fail_pct = round(100*red/screened)`. Honest denominator surfaced as "of the N we've screened".
- **Best-effort / never 500.** A DB error yields a zeroed aggregate and a graceful headline, not an error.
- **Postgres-safe:** `SELECT color, COUNT(*) … GROUP BY color`; no PRAGMA/lastrowid/ON CONFLICT.
- **Not flag-gated:** the endpoint + widget are inert until Glen embeds them; publishing the aggregate is the intent.
- **Brand colors** in the widget: green `#2f6f5e`, gold `#c9a227`, plus a neutral red-grey for the fail figure; self-contained inline styles (embeddable, no external assets).

---

### Task 1: `aggregate_confirmed` + public `GET /api/purity/stats`

**Files:**
- Modify: `dashboard/product_ratings.py` (append `aggregate_confirmed`)
- Modify: `app.py` (add the public route; place it near the other `/api/purity/*` console routes ~line 27927+, but WITHOUT the console gate)
- Test: `tests/test_purity_stats.py` (create)

**Interfaces:**
- Produces:
  - `product_ratings.aggregate_confirmed(cx) -> {"screened": int, "red": int, "yellow": int, "green": int}` — counts of confirmed rows by color; `screened` = red+yellow+green.
  - `GET /api/purity/stats` → JSON `{"screened","red","yellow","green","fail_pct","clean_pct","filler_pct","headline"}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_purity_stats.py
import json, sqlite3, pytest, app as app_mod
from dashboard import product_ratings as pr


def _confirm_color(cx, key, color):
    pr.record_screen(cx, key, brand="B", product_name="P"+key,
                     other_ingredients_raw="x", other_ingredients_parsed=["x"],
                     screen={"color": color, "red_hits": [], "yellow_hits": [], "avoidlist_version": "v1"})
    if color == "red":
        pr.confirm(cx, key)                     # red: screened -> confirmed
    else:
        pr.set_tier2(cx, key, None, "{}"); pr.confirm(cx, key)   # green/yellow via ai_draft


@pytest.fixture
def client(monkeypatch, tmp_path):
    db = str(tmp_path / "s.db")
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    pr.init_tables(cx)
    _confirm_color(cx, "fullscript::r1", "red")
    _confirm_color(cx, "fullscript::g1", "green")
    _confirm_color(cx, "fullscript::g2", "green")
    _confirm_color(cx, "fullscript::y1", "yellow")
    # a screened-but-unconfirmed row must NOT count
    pr.record_screen(cx, "fullscript::r2", brand="B", product_name="R2",
                     other_ingredients_raw="magnesium stearate", other_ingredients_parsed=["magnesium stearate"],
                     screen={"color": "red", "red_hits": ["stearate"], "yellow_hits": [], "avoidlist_version": "v1"})
    cx.commit(); cx.close()
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    app_mod.app.config["TESTING"] = True
    return app_mod.app.test_client()


def test_aggregate_confirmed_counts_only_confirmed():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    pr.init_tables(cx)
    _confirm_color(cx, "fullscript::a", "green")
    pr.record_screen(cx, "fullscript::b", brand="B", product_name="B",   # screened, not confirmed
                     other_ingredients_raw="x", other_ingredients_parsed=["x"],
                     screen={"color": "red", "red_hits": [], "yellow_hits": [], "avoidlist_version": "v1"})
    agg = pr.aggregate_confirmed(cx)
    assert agg == {"screened": 1, "red": 0, "yellow": 0, "green": 1}


def test_stats_endpoint_public_aggregate(client):
    r = client.get("/api/purity/stats")
    assert r.status_code == 200
    b = r.get_json()
    assert b["screened"] == 4                       # 1 red + 2 green + 1 yellow (r2 unconfirmed excluded)
    assert b["red"] == 1 and b["green"] == 2 and b["yellow"] == 1
    assert b["fail_pct"] == 25                       # 1/4
    assert b["headline"] == "25% of professional-quality products we've screened fail our purity standard"
    # aggregate-only: no product identity leaks into the public payload
    blob = json.dumps(b).lower()
    assert "product_key" not in blob and "fullscript::" not in blob and "brand" not in blob


def test_stats_endpoint_no_auth_required(client):
    # no console key header -> still 200 (public)
    assert client.get("/api/purity/stats").status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_purity_stats.py -v`
Expected: FAIL — `aggregate_confirmed` missing / route 404.

- [ ] **Step 3: Implement**

Append to `dashboard/product_ratings.py`:

```python
def aggregate_confirmed(cx):
    """Counts of CONFIRMED product ratings by color, for the public aggregate
    stat. unrated is never confirmed, so only red/yellow/green appear. Returns
    NO product identities. `screened` is the confirmed total."""
    counts = {"red": 0, "yellow": 0, "green": 0}
    for row in cx.execute("SELECT color, COUNT(*) FROM product_ratings "
                          "WHERE status='confirmed' GROUP BY color").fetchall():
        c = row[0]
        if c in counts:
            counts[c] = row[1]
    counts["screened"] = counts["red"] + counts["yellow"] + counts["green"]
    return counts
```

Add the public route to `app.py` (immediately after `api_console_purity_ratings_list`, ~line 28051 — but note: NO console gate):

```python
@app.route("/api/purity/stats", methods=["GET"])
def api_purity_stats():
    """PUBLIC aggregate purity stat: counts + percentages + a headline. Returns
    NEVER a product name/brand -- only the aggregate rate is public (spec
    primary-use boundary). Live group-by-color over CONFIRMED product_ratings;
    best-effort (a DB error yields a zeroed aggregate, never a 500)."""
    from dashboard import product_ratings as _pr
    try:
        with db.connect(LOG_DB) as cx:
            _pr.init_tables(cx)
            agg = _pr.aggregate_confirmed(cx)
    except Exception:
        agg = {"screened": 0, "red": 0, "yellow": 0, "green": 0}
    n = agg["screened"]
    def pct(k):
        return round(100 * agg[k] / n) if n else 0
    fail_pct = pct("red")
    headline = (f"{fail_pct}% of professional-quality products we've screened "
                "fail our purity standard") if n else "No products screened yet."
    return jsonify({"screened": n, "red": agg["red"], "yellow": agg["yellow"],
                    "green": agg["green"], "fail_pct": fail_pct,
                    "clean_pct": pct("green"), "filler_pct": pct("yellow"),
                    "headline": headline})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e501a0cc && OPENAI_API_KEY=x PINECONE_API_KEY=x python3 -m pytest tests/test_purity_stats.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e501a0cc
git add dashboard/product_ratings.py app.py tests/test_purity_stats.py
git commit -m "feat(purity): public GET /api/purity/stats aggregate authority endpoint"
```

---

### Task 2: `/purity-stat` embeddable widget page

**Files:**
- Create: `static/purity-stat.html`
- Modify: `app.py` (add a `/purity-stat` route serving it, mirroring the `send_from_directory(STATIC, …)` pattern used for `/begin` etc.)
- Verify: headless fetch+render (controller; see Step 4)

**Interfaces:**
- Consumes: `GET /api/purity/stats` (Task 1).
- Produces: a self-contained HTML widget at `/purity-stat`, iframe/embed-friendly, rendering the blunt headline + the honest denominator + the clean/filler/fail split.

- [ ] **Step 1: Create the widget**

Create `static/purity-stat.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Purity screen — the numbers</title>
<style>
  :root { --green:#2f6f5e; --gold:#c9a227; --fail:#8a4b4b; --ink:#1c2a26; --muted:#6b7a75; }
  html,body { margin:0; background:transparent; }
  .wrap { font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; color:var(--ink);
          max-width:520px; margin:0 auto; padding:1.25rem 1.4rem; box-sizing:border-box; }
  .big { font-size:2.6rem; font-weight:800; line-height:1; color:var(--fail); }
  .headline { font-size:1.05rem; font-weight:600; margin:.5rem 0 .1rem; }
  .denom { color:var(--muted); font-size:.85rem; margin-bottom:.9rem; }
  .split { display:flex; gap:.5rem; font-size:.8rem; }
  .split div { flex:1; text-align:center; padding:.4rem .2rem; border-radius:.5rem; background:#f3f6f4; }
  .split b { display:block; font-size:1.1rem; }
  .g { color:var(--green); } .y { color:var(--gold); } .r { color:var(--fail); }
  .err { color:var(--muted); font-size:.9rem; }
</style>
</head>
<body>
<div class="wrap" id="root"><span class="err">Loading purity screen…</span></div>
<script>
(function(){
  var root = document.getElementById("root");
  fetch("/api/purity/stats").then(function(r){ return r.json(); }).then(function(d){
    if (!d || !d.screened) { root.innerHTML = '<span class="err">No products screened yet.</span>'; return; }
    root.innerHTML =
      '<div class="big">' + d.fail_pct + '%</div>' +
      '<div class="headline">of professional-quality products we’ve screened fail our purity standard</div>' +
      '<div class="denom">Of ' + d.screened + ' professional products we’ve screened.</div>' +
      '<div class="split">' +
        '<div><b class="g">' + d.clean_pct + '%</b>fully clean</div>' +
        '<div><b class="y">' + d.filler_pct + '%</b>filler only</div>' +
        '<div><b class="r">' + d.fail_pct + '%</b>excipients we avoid</div>' +
      '</div>';
  }).catch(function(){ root.innerHTML = '<span class="err">Purity screen unavailable.</span>'; });
})();
</script>
</body>
</html>
```

- [ ] **Step 2: Add the route**

In `app.py`, add near the other `send_from_directory(STATIC, …)` page routes (e.g. after the `/begin` route ~line 2637):

```python
@app.route("/purity-stat")
def purity_stat_page():
    """Public embeddable widget rendering GET /api/purity/stats. iframe-friendly;
    self-contained. No auth."""
    return send_from_directory(STATIC, "purity-stat.html")
```

- [ ] **Step 3: Commit**

```bash
cd /tmp/wt-deploy-chat-e501a0cc
git add static/purity-stat.html app.py
git commit -m "feat(purity): /purity-stat embeddable authority-stat widget"
```

- [ ] **Step 4: Verify (controller, headless)**

After the branch is served locally (or on the deployed dark build), load `/purity-stat` headless with a stubbed `/api/purity/stats` (or against real data), and assert the rendered DOM shows: the big `fail_pct%`, the headline text "fail our purity standard", the "Of N … we've screened" denominator, and the three-way clean/filler/fail split. The widget must degrade to a plain "No products screened yet."/"unavailable" message on empty or failed fetch (verify by pointing it at an empty stats response).

---

## Self-Review

**Spec coverage (Section 3 — aggregate stat slice):**
- Public live endpoint, aggregate-only (no product names) → Task 1 (route + the no-leak test). ✅
- `fail = red / confirmed`, blunt headline verbatim, honest denominator → Task 1 headline + Task 2 denom line. ✅
- Embeddable self-contained widget → Task 2 (`static/purity-stat.html`, inline styles, `/purity-stat`). ✅
- Confirmed-only, grows live → `aggregate_confirmed` filters `status='confirmed'`, computed per request. ✅
- Not flag-gated, best-effort → route has no gate and a zeroed-aggregate fallback. ✅
- Postgres-safe → `GROUP BY color`, no PRAGMA/lastrowid. ✅

**Placeholder scan:** none — full code in every step; Step 4 names concrete DOM assertions.

**Type consistency:** `aggregate_confirmed(cx) -> {"screened","red","yellow","green"}` (Task 1) is consumed by the route's `pct()`/`agg[...]`; the JSON keys the route emits (`fail_pct/clean_pct/filler_pct/headline/screened`) are exactly the keys `static/purity-stat.html` reads (`d.fail_pct`, `d.clean_pct`, `d.filler_pct`, `d.screened`). Consistent.
