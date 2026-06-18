# New-Format In-Funnel Sales Pages — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the funnel's remedymatch.com sales-page links with new-format pages rendered from our own data inside `/begin/match` (route `/begin/product/<slug>`), with on-demand generation, a draft/approve workflow, an interactive comparison table, community image feedback, and a bottle-science visual.

**Architecture:** Render-from-data — one shared static template (`static/begin-product.html`) + per-product structured content stored in SQLite (`chat_log.db`). Copy is generated in-process on Render and streamed live (SSE); the 2 images are generated off Render by a local launchd watcher (the image-studio infra lives on Glen's Mac). A console editor owns content + an approve action; the global format is evolved in code. External/off-catalog remedies keep today's link-out behavior.

**Tech Stack:** Python 3.11 / Flask, SQLite (`chat_log.db` under `DATA_DIR`), Anthropic SDK (`_cl`, `claude-haiku-4-5-20251001`) for copy, Replicate/Flux (local) for images, server-sent events, vanilla static HTML+JS (no Jinja for `/begin/*`), the `dispatch_action` RBAC/event spine, `dashboard/points.py` ledger, `dashboard/inbox.py` Gmail send.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-18-funnel-sales-pages-design.md` (authoritative).
- **Copy conventions (Glen):** NO emojis in client-facing copy — use SVG/text icons (the 👍/👎/✓/✗ in this plan denote UI affordances rendered as SVG/text glyphs, never emoji characters); "Order" not "Reorder"; keep patient/UGC videos short. Ingredient labels follow `Compound Class: Common Name [% if <95] (Latin, italic) dose [%DV]`, elemental minerals × FMP %, A/D/E in IU, current 2016 FDA DVs.
- **No new excipients / claims discipline:** comparison rows use only facts on a comparator's own published Supplement Facts panel; hidden-excipient line stays category-level ("industry can legally include up to ~3% stearates undisclosed"); never name a competitor.
- **Platform:** code that runs in a Render web request must be fast — NO slow image generation in web workers (a cold-start burst previously wedged gunicorn → site-wide 502). Slow work goes through the queue → local watcher.
- **Env separation:** Render env vars live in the Render dashboard for service `glen-knowledge-chat`, NOT Doppler. New env (e.g. flags) are set there.
- **DB:** all tables live in `chat_log.db` resolved via `DATA_DIR`; reuse the connection pattern in `dashboard/client_portal.py`. Never mint/persist with `DATA_DIR` unset.
- **Test invocation:** `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest`. In the worktree, set `DATA_DIR` to a tmp dir after `--`. Mock live Supabase; `pytest.importorskip` for playwright. Pre-existing failures to ignore: `test_pf_playwright_fetch`, `test_bos_routes::test_home_page_served`.
- **Feature flag:** the whole funnel-facing surface is gated by env `SALES_PAGES_ENABLED` (default off) until content exists and Glen approves go-live.

---

## Phasing (each phase is independently shippable, flag-gated)

This spec is too large for one plan. It is split into five sequenced plans; **this document fully details Phase 1** and scopes Phases 2–5 (each gets its own plan when we reach it). Phase 1 delivers a real improvement on its own: a flag-gated new-format page the funnel links to, rendering existing product data in the new section model with the comparison table, Miron rotator, and in-funnel CTA — no AI generation yet.

- **Phase 1 — Page shell + section model + routing (this plan).** New page renders existing `product-data` in the 8-section model; comparison table (generic archetype, static); Miron rotator; CTA → `/begin/buy`; match card links to it. Flag-gated. No generation, no credit, no console.
- **Phase 2 — Content store + copy generation (in-process, streamed).** `sales_pages` table + state machine; the in-process copy generator + SSE reveal; caveat banner; draft persistence.
- **Phase 3 — Image generation via local watcher.** Queue table + `sales_page_watcher.py`; 2 Flux images written back; page backfills.
- **Phase 4 — Image feedback + credit + per-viewer memory.** `sales_page_viewers` table; thumbs ratings; `points.credit` + `_settle_order_points` gating; section-open memory.
- **Phase 5 — Console editor + approve/regenerate + approval email + reviews interface.** `dashboard/sales_pages.py` actions; `console-sales-pages.html`; approval email; read-only reviews-store contract (testimonials + UGC video entries).

---

## Phase 1 — File Structure

- **Create** `static/begin-product.html` — the new-format page template (static HTML + vanilla JS; fetches data, renders the 8 sections, comparison table, Miron rotator, CTA). One file, one responsibility: render the page.
- **Create** `data/sales-page-archetypes.json` — the static generic comparison archetypes ("a leading professional-channel formula", "a top-selling mass-market formula") and the category excipient callout text. Data, not code.
- **Create** `data/miron-assets.json` — the self-hosted Miron/bottle visual asset list (URLs under `/static/media/...`), seeded with the one backgrounder; rotator renders however many exist.
- **Modify** `app.py` — add `SALES_PAGES_ENABLED` flag read; add `product_url` to the match event; add `/begin/product/<slug>` route; add `/begin/product-page-data/<slug>` data endpoint (section model + comparison + assets); reuse `_get_product`, `_product_card`, `_product_how`, `_read_open_sections`.
- **Modify** `static/begin-match.html` — `renderMatch()` prefers `product_url` over `buy_url`/`url`/`search_url`.
- **Test** `tests/test_sales_pages_phase1.py` — Flask test-client coverage of the flag, route, and data endpoint.

---

## Phase 1 — Tasks

### Task 1: Feature flag + `product_url` on the match event

**Files:**
- Modify: `app.py` (match-event construction ~L2048-2053; add a module-level flag read near other `os.environ` flags)
- Test: `tests/test_sales_pages_phase1.py`

**Interfaces:**
- Consumes: existing `_resolve_buy_slug(name) -> str|None`, `_resolve_remedy_url`, `_store_search_url`.
- Produces: match event dict now carries `product_url: str` (`"/begin/product/<slug>"` when the flag is on and a slug resolves, else `""`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sales_pages_phase1.py
import importlib, os
import pytest

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SALES_PAGES_ENABLED", "true")
    import app as appmod
    importlib.reload(appmod)
    appmod.app.config["TESTING"] = True
    return appmod.app, appmod

def test_product_url_built_when_flag_on(client):
    _, appmod = client
    # pick any real slug from the loaded catalog
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    name = appmod._PRODUCTS["products"][slug]["name"]
    assert appmod._sales_page_url(name) == f"/begin/product/{slug}"

def test_product_url_empty_when_flag_off(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SALES_PAGES_ENABLED", "false")
    import importlib, app as appmod
    importlib.reload(appmod)
    name = next(iter(appmod._PRODUCTS["products"].values()))["name"]
    assert appmod._sales_page_url(name) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_sales_pages_phase1.py -k product_url -v`
Expected: FAIL — `AttributeError: module 'app' has no attribute '_sales_page_url'`

- [ ] **Step 3: Write minimal implementation**

```python
# app.py — near other flag reads (e.g. _STRIPE_ACTIVE)
_SALES_PAGES_ENABLED = os.environ.get("SALES_PAGES_ENABLED", "").lower() in ("1", "true", "yes")

def _sales_page_url(name):
    """New-format in-funnel sales page URL for an in-catalog remedy, or '' if
    disabled / off-catalog. Flag-gated so we can ship dark."""
    if not _SALES_PAGES_ENABLED:
        return ""
    slug = _resolve_buy_slug(name)
    return f"/begin/product/{slug}" if slug else ""
```

```python
# app.py — match-event construction (~L2052), add one field:
            "product_url": _sales_page_url(obj["name"]),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... -m pytest tests/test_sales_pages_phase1.py -k product_url -v`
Expected: PASS (both)

- [ ] **Step 5: Commit**

```bash
git add tests/test_sales_pages_phase1.py app.py
git commit -m "feat: flag-gated product_url on funnel match event"
```

---

### Task 2: `/begin/product/<slug>` route

**Files:**
- Modify: `app.py` (add route after `begin_buy_page` ~L2787)
- Create: `static/begin-product.html` (minimal stub this task; fleshed out in Task 4)
- Test: `tests/test_sales_pages_phase1.py`

**Interfaces:**
- Consumes: `_get_product(slug) -> dict|None`, `send_from_directory(STATIC, ...)`.
- Produces: route serving `begin-product.html`; sets `amg_session` cookie if absent (same as `begin_buy_page`).

- [ ] **Step 1: Write the failing test**

```python
def test_product_page_200_known_slug(client):
    appmod = client[1]
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    c = appmod.app.test_client()
    r = c.get(f"/begin/product/{slug}")
    assert r.status_code == 200
    assert b"begin-product" in r.data or b"<!DOCTYPE html" in r.data

def test_product_page_404_unknown_slug(client):
    appmod = client[1]
    c = appmod.app.test_client()
    assert c.get("/begin/product/nope-not-real").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_sales_pages_phase1.py -k product_page -v`
Expected: FAIL — 404 for the known slug (route not defined) / template missing.

- [ ] **Step 3: Write minimal implementation**

```html
<!-- static/begin-product.html (stub; Task 4 fills it in) -->
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>begin-product</title></head>
<body><div id="sp-root" data-slug=""></div></body></html>
```

```python
# app.py — after begin_buy_page
@app.route("/begin/product/<slug>")
def begin_product_page(slug):
    if not _get_product(slug):
        return ("", 404)
    resp = send_from_directory(STATIC, "begin-product.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    if not request.cookies.get("amg_session"):
        resp.set_cookie("amg_session", uuid.uuid4().hex, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... -m pytest tests/test_sales_pages_phase1.py -k product_page -v`
Expected: PASS (both)

- [ ] **Step 5: Commit**

```bash
git add app.py static/begin-product.html
git commit -m "feat: /begin/product/<slug> route + page stub"
```

---

### Task 3: `/begin/product-page-data/<slug>` — section model + comparison + assets

**Files:**
- Create: `data/sales-page-archetypes.json`, `data/miron-assets.json`
- Modify: `app.py` (new endpoint; module-level loaders for the two JSON files, mirroring `_PRODUCT_ALIASES` loading)
- Test: `tests/test_sales_pages_phase1.py`

**Interfaces:**
- Consumes: `_get_product`, `_product_card(p) -> {description, ingredients, benefits}`, `_product_how(p) -> str`, `_read_open_sections(session, email) -> list`.
- Produces: JSON with keys `slug, name, price, price_cents, sections[], comparison{}, miron_assets[], cta_url, open_sections[]`. `sections` is an ordered list of `{id, title, default_open: bool, body}` where `id ∈ {intro, description, video, ingredients, comparison, research, images, cta}`.

- [ ] **Step 1: Write the failing test**

```python
def test_product_page_data_shape(client):
    appmod = client[1]
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    c = appmod.app.test_client()
    data = c.get(f"/begin/product-page-data/{slug}").get_json()
    ids = [s["id"] for s in data["sections"]]
    assert ids == ["intro", "description", "video", "ingredients",
                   "comparison", "research", "images", "cta"]
    assert next(s for s in data["sections"] if s["id"] == "intro")["default_open"] is True
    assert all(s["default_open"] is False for s in data["sections"] if s["id"] != "intro")
    assert data["cta_url"] == f"/begin/buy/{slug}"
    # comparison carries packaging + microplastics rows and the category excipient callout
    rows = {r["label"] for r in data["comparison"]["rows"]}
    assert "Packaging" in rows and "Microplastic exposure" in rows
    assert "stearates" in data["comparison"]["excipient_callout"].lower()
    assert len(data["comparison"]["columns"]) == 3  # ours + 2 anonymized archetypes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_sales_pages_phase1.py -k page_data -v`
Expected: FAIL — 404 (endpoint not defined).

- [ ] **Step 3: Write the data files + implementation**

```json
// data/sales-page-archetypes.json
{
  "columns": ["This formula", "A leading professional-channel formula", "A top-selling mass-market formula"],
  "excipient_callout": "Industry note: a product can legally contain up to ~3% stearates (and other flow agents) with no label disclosure.",
  "rows": [
    {"label": "Ingredient form quality", "ours": "ok", "pro": "warn", "mass": "bad"},
    {"label": "Excipient-free", "ours": "ok", "pro": "warn", "mass": "bad"},
    {"label": "Completeness (no key actives missing)", "ours": "ok", "pro": "warn", "mass": "bad"},
    {"label": "Packaging", "ours": "ok", "pro": "bad", "mass": "bad",
     "ours_note": "Miron biophotonic violet glass", "pro_note": "plastic / clear glass", "mass_note": "plastic"},
    {"label": "Microplastic exposure", "ours": "ok", "pro": "bad", "mass": "bad",
     "ours_note": "none", "pro_note": "likely", "mass_note": "likely"}
  ]
}
```

```json
// data/miron-assets.json
{ "assets": [ {"src": "/static/media/miron-violetglass-backgrounder.png",
               "alt": "Miron biophotonic violet glass — why it protects actives"} ] }
```

```python
# app.py — module-level loaders (near _PRODUCT_ALIASES load)
def _load_json_data(fname, default):
    try:
        with open(os.path.join(os.path.dirname(__file__), "data", fname)) as f:
            return json.load(f)
    except Exception:
        return default

_SALES_ARCHETYPES = _load_json_data("sales-page-archetypes.json", {"columns": [], "rows": [], "excipient_callout": ""})
_MIRON_ASSETS = _load_json_data("miron-assets.json", {"assets": []})

# app.py — endpoint
@app.route("/begin/product-page-data/<slug>")
def begin_product_page_data(slug):
    p = _get_product(slug)
    if not p:
        return jsonify({"error": "not found"}), 404
    card = _product_card(p) if not p.get("info_only") else {}
    how = "" if p.get("info_only") else _product_how(p)
    ingredients = p.get("ingredients") or card.get("ingredients", [])
    intro = p.get("intro") or (card.get("description", "") or "").split(". ")[0]
    sections = [
        {"id": "intro", "title": "What this does", "default_open": True, "body": intro},
        {"id": "description", "title": "Overview", "default_open": False,
         "body": p.get("description") or card.get("description", "")},
        {"id": "video", "title": "Watch", "default_open": False, "body": {"videos": p.get("videos", [])}},
        {"id": "ingredients", "title": "What's inside", "default_open": False, "body": {"ingredients": ingredients}},
        {"id": "comparison", "title": "How it compares", "default_open": False, "body": {}},
        {"id": "research", "title": "The research", "default_open": False,
         "body": {"how_it_works": how, "learn_url": f"/begin/learn/{slug}"}},
        {"id": "images", "title": "Help shape this", "default_open": False, "body": {"images": p.get("page_images", [])}},
        {"id": "cta", "title": "Order", "default_open": False, "body": {}},
    ]
    return jsonify({
        "slug": slug, "name": p["name"], "price_cents": p["price_cents"],
        "price": f"${p['price_cents']/100:.2f}", "cta_url": f"/begin/buy/{slug}",
        "sections": sections, "comparison": _SALES_ARCHETYPES, "miron_assets": _MIRON_ASSETS["assets"],
        "open_sections": _read_open_sections(request.cookies.get("amg_session", ""),
                                             (get_authenticated_user(request) or {}).get("email", "")),
    })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... -m pytest tests/test_sales_pages_phase1.py -k page_data -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data/sales-page-archetypes.json data/miron-assets.json app.py tests/test_sales_pages_phase1.py
git commit -m "feat: product-page-data endpoint (section model + comparison + assets)"
```

---

### Task 4: `begin-product.html` — render the 8-section model

**Files:**
- Modify: `static/begin-product.html` (full implementation)
- Manual verification (this repo does not unit-test static JS; playwright is `importorskip`)

**Interfaces:**
- Consumes: `GET /begin/product-page-data/<slug>` (Task 3 shape); `POST /begin/open-section` (existing `_read_open_sections` write path — verify the existing endpoint name in `app.py`; reuse it).
- Produces: rendered page; intro open by default; other sections render their body on first click; open/closed state persists per viewer.

- [ ] **Step 1: Implement the page**

Build `static/begin-product.html` modeled on `static/begin-buy.html`:
1. Read `slug` from the path (`location.pathname.split('/').pop()`).
2. `fetch('/begin/product-page-data/' + slug, {credentials:'same-origin'})` → render.
3. Render each `section` as a `<details>`/accordion. `intro` starts open; for others, lazy-render `body` on first `toggle`→open and POST the open event to the existing open-sections endpoint so it persists (restore from `open_sections` on load).
4. `comparison` section → render via Task 5 component. `images` → Task 6 handles the Miron rotator; the rated images block is Phase 4 (render a placeholder note "Community image feedback coming soon" guarded by a flag).
5. CTA renders an `<a class="mc-open" href=cta_url>Order →</a>` (text "Order", never "Reorder", no emoji).
6. Match the funnel's existing CSS variables/classes from `begin-buy.html` for visual consistency.

- [ ] **Step 2: Manual verification**

```bash
# from worktree, with the venv + DATA_DIR set and SALES_PAGES_ENABLED=true
doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" SALES_PAGES_ENABLED=true \
  ~/.venvs/deploy-chat311/bin/python -m app  # or the app's run entrypoint
# then open http://localhost:<port>/begin/product/<a-real-slug>
```
Confirm: intro open by default; clicking each other section reveals its content; reload preserves which sections were open; CTA links to `/begin/buy/<slug>`; no emoji; "Order" label.

- [ ] **Step 3: Commit**

```bash
git add static/begin-product.html
git commit -m "feat: begin-product page renders 8-section model with per-viewer open memory"
```

---

### Task 5: Comparison table component (green ✓ / red ✗ scoring)

**Files:**
- Modify: `static/begin-product.html` (the `comparison` renderer)
- Manual verification

**Interfaces:**
- Consumes: `data.comparison = {columns[3], rows[], excipient_callout}` from Task 3.
- Produces: a 3-column table; each cell shows an SVG check (ok), warn, or cross (bad) plus optional note; the excipient callout renders below as a category-level note.

- [ ] **Step 1: Implement** — map `ok→green check SVG`, `warn→amber dash SVG`, `bad→red X SVG`; render `*_note` under the glyph; render `excipient_callout` in a muted caption. Anonymized column headers only (no competitor names). Icons are inline SVG, not emoji.

- [ ] **Step 2: Manual verification** — open a product page, expand "How it compares": three columns ("This formula" + two anonymized), Packaging + Microplastic rows present, checks/crosses render, excipient note visible, no competitor named.

- [ ] **Step 3: Commit**

```bash
git add static/begin-product.html
git commit -m "feat: comparison table with packaging + microplastics rows and excipient callout"
```

---

### Task 6: Miron bottle-science rotator (≤10s, reduced-motion aware)

**Files:**
- Modify: `static/begin-product.html` (rotator component, rendered near the comparison/bottle area)
- Add asset: `static/media/miron-violetglass-backgrounder.png` (self-hosted copy of the seed asset)
- Manual verification

**Interfaces:**
- Consumes: `data.miron_assets = [{src, alt}, ...]`.
- Produces: 0 assets → nothing; 1 → static `<img>`; 2+ → auto-rotating slider on a ≤10s interval. Honors `window.matchMedia('(prefers-reduced-motion: reduce)')` → no auto-rotate (show first, allow manual next/prev). Links to `https://skepticalreviews.com/bottles`.

- [ ] **Step 1: Implement** the rotator with a `setInterval` (≤10000ms) cleared when `prefers-reduced-motion` matches; render manual prev/next controls; "Learn the science ↗" link to `skepticalreviews.com/bottles` (`rel="noopener"`).

- [ ] **Step 2: Manual verification** — with 1 asset: static image, no timer. Add a 2nd temp entry to `data/miron-assets.json`: slider alternates ≤10s. Toggle OS reduced-motion: auto-rotate stops, manual controls work.

- [ ] **Step 3: Commit**

```bash
git add static/begin-product.html static/media/miron-violetglass-backgrounder.png
git commit -m "feat: Miron bottle-science rotator (reduced-motion aware, 1=static/2+=slider)"
```

---

### Task 7: Match card prefers `product_url`

**Files:**
- Modify: `static/begin-match.html` (`renderMatch()` ~L271-307)
- Manual verification

**Interfaces:**
- Consumes: `match.product_url` (Task 1).
- Produces: when `product_url` present → primary "View [name] →" links to it (internal page); `buy_url`/`url`/`search_url` fall to secondary/unchanged. When absent → today's behavior exactly.

- [ ] **Step 1: Implement** — at the top of the link block, add:

```javascript
const productUrl = (match.product_url || '').trim();
if (productUrl) {
  html += '<a class="mc-open" href="' + escAttr(productUrl) + '">View ' + escHtml(match.name) + ' →</a>';
} else if (buyUrl) {
  // ... existing buyUrl branch unchanged ...
}
```
Keep all existing branches as the `else` chain so off-catalog/external behavior is untouched.

- [ ] **Step 2: Manual verification** — flag ON: an in-catalog match card links to `/begin/product/<slug>`. Flag OFF: identical to current (Buy/Open/Search). An off-catalog match (e.g. an Amazon remedy): unchanged, with the Associate disclosure.

- [ ] **Step 3: Commit**

```bash
git add static/begin-match.html
git commit -m "feat: match card links to new-format product page when enabled"
```

---

### Task 8: Phase 1 integration check + flag default

**Files:**
- Modify: `tests/test_sales_pages_phase1.py` (one end-to-end route test)

- [ ] **Step 1: Write the test**

```python
def test_match_to_product_page_roundtrip(client):
    appmod = client[1]
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    name = appmod._PRODUCTS["products"][slug]["name"]
    assert appmod._sales_page_url(name) == f"/begin/product/{slug}"
    c = appmod.app.test_client()
    assert c.get(f"/begin/product/{slug}").status_code == 200
    data = c.get(f"/begin/product-page-data/{slug}").get_json()
    assert data["cta_url"] == f"/begin/buy/{slug}"
```

- [ ] **Step 2: Run the full Phase-1 file**

Run: `... -m pytest tests/test_sales_pages_phase1.py -v`
Expected: PASS (all)

- [ ] **Step 3: Confirm `SALES_PAGES_ENABLED` defaults OFF** in Render (do not set it yet). Ship dark.

- [ ] **Step 4: Commit + open PR**

```bash
git add tests/test_sales_pages_phase1.py
git commit -m "test: phase-1 match→product-page roundtrip"
```

---

## Phases 2–5 — scope (each gets its own plan)

**Phase 2 — content store + copy generation.** Add `sales_pages` table (`dashboard/sales_pages.py` data layer, `client_portal.py` pattern). State machine `none→generating→draft→approved`. In-process copy generator: build per-section prompts in Python from `product_content` + Glen's copy rules; stream via `sse()` + `_cl.messages.stream("claude-haiku-4-5-20251001")`; persist draft. Caveat banner on non-approved pages. The data endpoint reads draft copy when present, else falls back to Phase-1 product-data.

**Phase 3 — image generation via local watcher.** `sales_page_gen_queue` table + `process_queue.enqueue` pattern; `sales_page_watcher.py` (launchd) renders 2 Flux images via the local image-studio infra (Replicate, `deploy-chat311` venv), writes to `static/media/sales/<slug>/` + updates the draft record; page polls/backfills. Keep all Replicate calls off Render web workers.

**Phase 4 — image feedback + credit + memory.** `sales_page_viewers` table (`person_id, product_slug, sections_viewed, image_ratings_json, credit_earned`). Endpoints to record 👍/👎 and section-open state (extend the existing open-sections write). On both images rated → `points.credit(cx, email, value_cents=100, reason="page_rate_images", order_ref=f"page_{slug}")`; realize inside `_settle_order_points` gated on the order containing that slug; idempotent via `points.has_entry`.

**Phase 5 — console + approval + reviews interface.** `dashboard/sales_pages.py` actions (`sales_pages.generate|regenerate|approve`) on the `dispatch_action` spine, RBAC `(OWNER, OPS)`, `LOW_WRITE`. `static/console-sales-pages.html` (biofield-editor pattern) + `op-nav.js` BOS sub-tab. Approve → state `approved`, drop banner, swap reviewed comparators, `inbox.send_email(...)` to `triggering_email`. Define the **read-only reviews-store contract** the page consumes for the testimonial section + UGC video entries (Spec 2 populates it); empty store → those entries don't render.

---

## Phase 1 — Verification (end to end)

1. **Tests:** `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_sales_pages_phase1.py -v` → all pass; full suite shows no new failures (only the two known pre-existing ones).
2. **Flag off (default):** funnel match cards behave exactly as today (remedymatch/buy/search); `/begin/product/<slug>` still serves (route exists) but nothing links to it.
3. **Flag on locally:** an in-catalog match links to `/begin/product/<slug>`; the page renders all 8 sections (intro open, others lazy on click), comparison table with packaging + microplastics + excipient callout, Miron rotator (static with 1 asset), CTA → `/begin/buy/<slug>`; reopening preserves open sections; no emoji; "Order" label.
4. **Off-catalog match:** unchanged link-out + Amazon disclosure.
