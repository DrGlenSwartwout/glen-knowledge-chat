# Console Biofield Portal Editor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A console form at `/console/biofield-portal` to compose/edit and publish a client's biofield causal-chain content (with a paste-portal-seed.json bridge) straight to their portal.

**Architecture:** New console-key-gated API (`GET` load, `GET` catalog, `POST` publish) over the existing `client_portal.upsert_portal`; a static vanilla-JS form; a nav entry. No new storage; mirrors `/admin/portal/upsert` semantics with validation.

**Tech Stack:** Flask, sqlite (DATA_DIR/chat_log.db), vanilla JS. Tests: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-06-16-biofield-portal-editor-design.md`

---

## File Structure

- **Modify** `app.py` — 3 API routes + 1 page route (place near `/admin/portal/upsert`, ~app.py:7463). Reuse `_portal_console_ok()`, `_cp.upsert_portal`, `_cp.get_portal_content_by_email`, `_send_full_report_email`, `_bos_products.catalog`.
- **Create** `static/console-biofield-portal.html` — the form (vanilla JS).
- **Modify** `static/op-nav.js` — add a "Biofield" BOS sub-tab.
- **Create** `tests/test_console_biofield_portal.py` — API + page route tests.

---

## Task 1: Publish API — `POST /api/console/biofield-portal`

**Files:**
- Modify: `app.py` (after `admin_client_portal_upsert`, ~line 7495)
- Test: `tests/test_console_biofield_portal.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_console_biofield_portal.py
import sqlite3
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


_LAYER = {"n": 1, "title": "Calm", "meaning": "settle", "remedy": "Terrain Restore",
          "dosing": "10 drops 3x/day"}


def test_post_creates_portal_and_returns_url(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"email": "x@y.com", "name": "X",
                     "content": {"greeting": "Aloha", "layers": [_LAYER]}})
    assert r.status_code == 200
    j = r.get_json()
    assert j["token"] and j["url"].endswith(j["token"])
    # content round-trips through the public portal API
    r2 = c.get(f"/api/portal/{j['token']}")
    assert r2.get_json()["layers"][0]["title"] == "Calm"


def test_post_requires_console_key(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal",
               json={"email": "x@y.com", "content": {"layers": [_LAYER]}})
    assert r.status_code == 401


def test_post_requires_email(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"content": {"layers": [_LAYER]}})
    assert r.status_code == 400


def test_post_requires_some_content(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"email": "x@y.com", "name": "X", "content": {}})
    assert r.status_code == 400


def test_post_send_emails_link(client, monkeypatch):
    c, appmod = client
    sent = {}
    monkeypatch.setattr(appmod, "_send_full_report_email",
                        lambda to, name, subj, body, **k: sent.update(to=to, body=body))
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"email": "e@y.com", "name": "E", "send": True,
                     "content": {"greeting": "hi", "layers": [_LAYER]}})
    tok = r.get_json()["token"]
    assert sent["to"] == "e@y.com" and tok in sent["body"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `... -m pytest tests/test_console_biofield_portal.py -q`
Expected: FAIL — routes return 404 (not yet defined).

- [ ] **Step 3: Implement the POST route**

In `app.py`, after the `admin_client_portal_upsert` function:

```python
def _biofield_content_clean(content):
    """Drop blank layers (no title); return (clean_content, has_content)."""
    content = dict(content or {})
    layers = [L for L in (content.get("layers") or []) if (L.get("title") or "").strip()]
    content["layers"] = layers
    has = bool(layers or (content.get("video") or {}).get("url") or (content.get("greeting") or "").strip())
    return content, has


@app.route("/api/console/biofield-portal", methods=["POST"])
def api_console_biofield_publish():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    name = (body.get("name") or "").strip()
    if not email:
        return jsonify({"error": "email required"}), 400
    content, has = _biofield_content_clean(body.get("content") or {})
    if not has:
        return jsonify({"error": "Add some content — at least one layer, a video, or a greeting."}), 400
    from dashboard import client_portal as _cp
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        token, pid = _cp.upsert_portal(cx, email, name, content)
    url = f"{PUBLIC_BASE_URL}/portal/{token}" if token else None
    emailed = False
    if token and body.get("send"):
        try:
            _send_full_report_email(
                email, name, "Your personal healing home is ready 🌺",
                f"Aloha {name or ''},\n\nYour personal healing home is ready:\n\n{url}\n\n"
                f"With aloha,\nDr. Glen & Rae")
            emailed = True
        except Exception as e:
            print(f"[biofield-publish] send failed: {e!r}", flush=True)
    return jsonify({"ok": True, "token": token, "url": url, "portal_id": pid,
                    "updated": token is None, "emailed": emailed})
```

- [ ] **Step 4: Run to verify they pass**

Run: `... -m pytest tests/test_console_biofield_portal.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_console_biofield_portal.py
git commit -m "Biofield portal editor: publish API"
```

---

## Task 2: Load API — `GET /api/console/biofield-portal` + catalog

**Files:**
- Modify: `app.py`
- Test: `tests/test_console_biofield_portal.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_console_biofield_portal.py`:

```python
def _seed(appmod, email="seed@y.com", name="Seed"):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    cp.upsert_portal(cx, email, name, {"greeting": "hello", "layers": [_LAYER]})
    cx.close()


def test_get_loads_existing_portal(client):
    c, appmod = client
    _seed(appmod, "seed@y.com", "Seed")
    j = c.get("/api/console/biofield-portal?key=test-secret&email=seed@y.com").get_json()
    assert j["found"] is True
    assert j["name"] == "Seed"
    assert j["content"]["layers"][0]["title"] == "Calm"


def test_get_unknown_returns_scaffold(client):
    c, _ = client
    j = c.get("/api/console/biofield-portal?key=test-secret&email=nobody@y.com").get_json()
    assert j["found"] is False
    assert j["content"] == {}


def test_get_requires_key(client):
    c, _ = client
    assert c.get("/api/console/biofield-portal?email=x@y.com").status_code == 401


def test_catalog_returns_products(client):
    c, _ = client
    j = c.get("/api/console/biofield-portal/catalog?key=test-secret").get_json()
    assert isinstance(j["products"], list) and j["products"]
    assert "slug" in j["products"][0] and "name" in j["products"][0]
```

- [ ] **Step 2: Run to verify they fail**

Run: `... -m pytest tests/test_console_biofield_portal.py -k "loads_existing or scaffold or get_requires or catalog" -q`
Expected: FAIL — 404 / route missing.

- [ ] **Step 3: Implement the GET routes**

In `app.py`, next to the POST route:

```python
@app.route("/api/console/biofield-portal", methods=["GET"])
def api_console_biofield_load():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    email = (request.args.get("email") or "").strip().lower()
    if not email:
        return jsonify({"found": False, "content": {}})
    from dashboard import client_portal as _cp
    with sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        rec = _cp.get_portal_content_by_email(cx, email)
    if not rec:
        return jsonify({"found": False, "name": "", "content": {}, "has_token": False})
    return jsonify({"found": True, "name": rec.get("name") or "",
                    "content": rec.get("content") or {}, "has_token": True})


@app.route("/api/console/biofield-portal/catalog", methods=["GET"])
def api_console_biofield_catalog():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    items = []
    for p in _bos_products.catalog(with_ingredients_only=False):
        items.append({"slug": p.get("slug"), "name": p.get("name"),
                      "price_cents": p.get("price_cents")})
    return jsonify({"products": items})
```

- [ ] **Step 4: Run to verify they pass**

Run: `... -m pytest tests/test_console_biofield_portal.py -q`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_console_biofield_portal.py
git commit -m "Biofield portal editor: load + catalog API"
```

---

## Task 3: Page route

**Files:**
- Modify: `app.py`
- Create: `static/console-biofield-portal.html` (full form — see Task 4)
- Test: `tests/test_console_biofield_portal.py`

- [ ] **Step 1: Write the failing test**

Add:

```python
def test_page_served(client):
    c, _ = client
    assert c.get("/console/biofield-portal").status_code == 200
```

- [ ] **Step 2: Run to verify it fails**

Run: `... -m pytest tests/test_console_biofield_portal.py::test_page_served -q`
Expected: FAIL — 404 (route + file missing).

- [ ] **Step 3: Implement the page route + a stub file**

In `app.py` (next to `console_pricing_settings_page`):

```python
@app.route("/console/biofield-portal")
def console_biofield_portal_page():
    resp = send_from_directory(STATIC, "console-biofield-portal.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp
```

Create `static/console-biofield-portal.html` with a minimal stub (replaced fully in Task 4):

```html
<!DOCTYPE html><html><head><meta charset="utf-8"><title>Biofield Portal</title></head>
<body><div id="app">Biofield Portal Editor</div></body></html>
```

- [ ] **Step 4: Run to verify it passes**

Run: `... -m pytest tests/test_console_biofield_portal.py::test_page_served -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py static/console-biofield-portal.html tests/test_console_biofield_portal.py
git commit -m "Biofield portal editor: page route + stub"
```

---

## Task 4: The form UI

**Files:**
- Modify: `static/console-biofield-portal.html` (replace the stub with the full form)

No unit test (vanilla JS, no harness) — covered by the API tests + manual smoke.

- [ ] **Step 1: Write the full form**

Replace `static/console-biofield-portal.html` with a form that:
- reads the console key from `?key=` or `localStorage.getItem("console_key")` (match other console pages — check `console-pricing-settings.html` for the exact key-resolution snippet and reuse it verbatim);
- **Client:** an `email` input + `name` input + a "Load existing" button → `GET /api/console/biofield-portal?key=<k>&email=<e>`, populating the form on `found`;
- **Paste box:** a `<textarea>` + "Fill form" button → `JSON.parse`; on success populate greeting/video/layers/reorder_items/pricing_note; on error show an inline message and leave the form untouched;
- **Greeting** textarea; **Video** `url` + `label` inputs;
- **Layers** list: each row has number `n`, `title`, `meaning`, `remedy`, `dosing`, and a delete button; an "Add layer" button appends a row;
- **Reorder items** list: each row has `slug` (an `<input list="catalog">` backed by a `<datalist id="catalog">` filled from `GET /api/console/biofield-portal/catalog`), `qty`, optional `price_cents`; "Add item" appends;
- **pricing_note** textarea;
- a `buildContent()` JS function assembling `{greeting, video:{url,label}, layers:[...], reorder_items:[...], pricing_note}` (omit empty video/reorder);
- **Publish** and **Publish & email** buttons → `POST /api/console/biofield-portal?key=<k>` with `{email,name,content,send}`; on success show the returned `url`, a **Preview** link (`<a target="_blank" href=url>`), and "updated" vs "created";
- console-styled CSS consistent with other `console-*.html` pages.

Use `static/console-pricing-settings.html` as the structural reference (key resolution, fetch wrapper, status banner).

- [ ] **Step 2: Manual smoke**

Run the app (or rely on Task 6's suite). Open `/console/biofield-portal?key=<CONSOLE_SECRET>`, paste `~/AI-Training/05 Clients/Brooke Webb portal-seed.json`, confirm the form fills, Publish to a test email, open the Preview link, confirm the biofield block renders.

- [ ] **Step 3: Commit**

```bash
git add static/console-biofield-portal.html
git commit -m "Biofield portal editor: full form UI"
```

---

## Task 5: Nav entry

**Files:**
- Modify: `static/op-nav.js` (BOS sub-tab list, ~lines 73-78)

- [ ] **Step 1: Add the sub-tab**

In `static/op-nav.js`, in the Business OS module sub-tab array (after the `products` entry, ~line 76), add:

```javascript
    { id: "biofield", label: "Biofield", href: "/console/biofield-portal" + qs },
```

And in `static/console-biofield-portal.html`'s `<body>` tag (or wherever op-nav reads active state), set `data-active="bos" data-sub="biofield"` so the tab highlights, matching `console-products.html`.

- [ ] **Step 2: Manual check**

Open `/console/biofield-portal?key=<k>` and confirm the "Biofield" sub-tab appears and is active.

- [ ] **Step 3: Commit**

```bash
git add static/op-nav.js static/console-biofield-portal.html
git commit -m "Biofield portal editor: nav sub-tab"
```

---

## Task 6: Full suite, push, PR

- [ ] **Step 1: Run the whole suite**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest -q`
Expected: all pass, 0 failed.

- [ ] **Step 2: Push + PR**

```bash
git push -u origin sess/5326cc61
gh pr create --base main --title "Feature #3 slice 1: console biofield portal editor" --body "..."
```

---

## Self-Review notes

- **Spec coverage:** page route (Task 3), load API (Task 2), publish API + validation + send (Task 1), catalog autocomplete (Task 2 + form Task 4), paste-JSON bridge (Task 4), nav (Task 5). Out-of-scope items (FMP/E4L import, remedy autocomplete, live preview, drag-reorder) correctly absent.
- **Placeholder scan:** all code steps show real code; Task 4 (vanilla-JS form) describes exact fields/endpoints/behaviors and points at `console-pricing-settings.html` for the key-resolution/fetch snippet rather than guessing it — acceptable since it's UI glue with no unit test, and every endpoint/shape it calls is concretely defined in Tasks 1-2.
- **Type consistency:** content shape `{greeting, video:{url,label}, layers:[{n,title,meaning,remedy,dosing}], reorder_items:[{slug,qty,price_cents?}], pricing_note}` identical across publish/load/form; API paths `/api/console/biofield-portal` (GET/POST) + `/api/console/biofield-portal/catalog` consistent; auth via `_portal_console_ok()` everywhere.
- **Open verification (do during impl):** confirm `_bos_products.catalog(with_ingredients_only=False)` items carry `slug`/`name`/`price_cents` keys (adjust the mapping in Task 2 if names differ); copy the exact console-key resolution snippet from `console-pricing-settings.html` in Task 4.
