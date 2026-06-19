# Phase 3 — Sales Page AI Product Image Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate 2 image variants per type (botanical + mechanism = 4) per product via Flux on Replicate, run in the existing background scheduler, save to Render's persistent disk, serve via a route, and display one per type in the images section — behind a new `SALES_PAGES_AI_IMAGES` flag.

**Architecture:** Render-side end-to-end. A SQLite queue is filled on first images-section open; a job in the existing APScheduler `_start_scheduler` drains it, calls Replicate's REST API (Flux 1.1 Pro) per prompt, writes PNGs to `DATA_DIR/sales-images/<slug>/`, records rows, and the page backfills via a serving route. No Mac dependency.

**Tech Stack:** Python 3.11 / Flask, SQLite (`chat_log.db` via `LOG_DB`), `requests` (existing dep) for the Replicate REST API, APScheduler (existing `_start_scheduler`), persistent disk under `DATA_DIR`, vanilla static JS.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-18-phase3-sales-page-images-design.md` (authoritative).
- **Flag:** `SALES_PAGES_AI_IMAGES`, default OFF, `.strip().lower() in ("1","true","yes")` (existing flag idiom). With it off: no queue, no gen route, no images field — Phase-1/2 behavior identical.
- **Generation is off web workers:** runs only in the scheduler job; web requests only enqueue or serve files. NO Replicate call in a web request.
- **2 variants per type**, kinds exactly `botanical` and `mechanism`; display one per type.
- **Model:** Flux 1.1 Pro via Replicate REST (`black-forest-labs/flux-1.1-pro`); needs `REPLICATE_API_TOKEN` env. No new pip dependency — use `requests`.
- **Images are enhancement:** any failure logs + degrades to showing nothing; the page never breaks.
- **No emoji** in client-facing copy.
- **DB:** `chat_log.db` via `LOG_DB`; data layer takes an open `cx`. Disk dir `DATA_DIR/sales-images` (mirror `_CLIPS_DIR = DATA_DIR/clips` at app.py:11827).
- **Test invocation:** `doppler run -p remedy-match -c prd -- env DATA_DIR="$(mktemp -d)" ~/.venvs/deploy-chat311/bin/python -m pytest <file> -v`. Pure-function/data-layer tests need only the venv python. Mock Supabase; `importorskip` playwright.

---

## File Structure

- **Create** `dashboard/sales_images.py` — data layer: image queue + `sales_page_images` rows. One responsibility: persistence.
- **Create** `dashboard/sales_image_prompts.py` — pure prompt builder for the 2 modes × 2 variants.
- **Create** `dashboard/replicate_client.py` — `generate_image(prompt) -> bytes` via Replicate REST.
- **Modify** `app.py` — flag `_SALES_AI_IMAGES_ENABLED`; disk dir `_SALES_IMG_DIR`; worker `_drain_sales_image_queue` + scheduler registration; routes `/begin/product-image/<slug>/<filename>` (serve) and `/begin/product-image-gen/<slug>` (enqueue); images-section marker in `begin_product_page_data`.
- **Modify** `static/begin-product.html` — `renderImagesBody`: show images / generating placeholder + enqueue + poll.
- **Test** `tests/test_sales_pages_phase3.py`.

---

## Task 1: data layer `dashboard/sales_images.py`

**Files:** Create `dashboard/sales_images.py`; Test `tests/test_sales_pages_phase3.py`

**Interfaces (produces):** `init_tables(cx)`; `enqueue(cx, slug)`; `list_pending(cx) -> list[str]`; `mark_done(cx, slug)`; `mark_failed(cx, slug)`; `queue_state(cx, slug) -> str|None`; `record_image(cx, slug, kind, variant, filename)`; `get_images(cx, slug) -> list[dict]`; `display_images(cx, slug) -> dict` (`{"botanical": filename|None, "mechanism": filename|None}` — first ready variant per kind).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sales_pages_phase3.py
import sqlite3
from dashboard import sales_images as si

def _cx(): return sqlite3.connect(":memory:")

def test_queue_enqueue_pending_done():
    cx = _cx()
    si.enqueue(cx, "longevity")
    assert si.list_pending(cx) == ["longevity"]
    assert si.queue_state(cx, "longevity") == "pending"
    si.mark_done(cx, "longevity")
    assert si.list_pending(cx) == []
    assert si.queue_state(cx, "longevity") == "done"

def test_enqueue_idempotent_resets_to_pending():
    cx = _cx()
    si.enqueue(cx, "energy"); si.mark_failed(cx, "energy")
    si.enqueue(cx, "energy")
    assert si.queue_state(cx, "energy") == "pending"

def test_record_and_display_first_ready_per_kind():
    cx = _cx()
    si.record_image(cx, "longevity", "botanical", 1, "botanical-1.png")
    si.record_image(cx, "longevity", "botanical", 2, "botanical-2.png")
    si.record_image(cx, "longevity", "mechanism", 1, "mechanism-1.png")
    disp = si.display_images(cx, "longevity")
    assert disp == {"botanical": "botanical-1.png", "mechanism": "mechanism-1.png"}
    assert len(si.get_images(cx, "longevity")) == 3
```

- [ ] **Step 2: Run to verify it fails** — `... -m pytest tests/test_sales_pages_phase3.py -k "queue or record" -v` → FAIL (no module).

- [ ] **Step 3: Implement**

```python
# dashboard/sales_images.py
import datetime

def _now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def init_tables(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS sales_image_queue ("
               "product_slug TEXT PRIMARY KEY, state TEXT DEFAULT 'pending', "
               "requested_at TEXT DEFAULT '', updated_at TEXT DEFAULT '')")
    cx.execute("CREATE TABLE IF NOT EXISTS sales_page_images ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, product_slug TEXT, kind TEXT, "
               "variant INTEGER, filename TEXT, state TEXT DEFAULT 'ready', created_at TEXT DEFAULT '')")
    cx.commit()

def enqueue(cx, slug):
    init_tables(cx); now = _now()
    cx.execute("INSERT INTO sales_image_queue (product_slug, state, requested_at, updated_at) "
               "VALUES (?, 'pending', ?, ?) ON CONFLICT(product_slug) DO UPDATE SET "
               "state='pending', requested_at=?, updated_at=?", (slug, now, now, now, now))
    cx.commit()

def list_pending(cx):
    init_tables(cx)
    return [r[0] for r in cx.execute(
        "SELECT product_slug FROM sales_image_queue WHERE state='pending' ORDER BY requested_at").fetchall()]

def _set_state(cx, slug, state):
    init_tables(cx)
    cx.execute("UPDATE sales_image_queue SET state=?, updated_at=? WHERE product_slug=?", (state, _now(), slug))
    cx.commit()

def mark_done(cx, slug):   _set_state(cx, slug, "done")
def mark_failed(cx, slug): _set_state(cx, slug, "failed")

def queue_state(cx, slug):
    init_tables(cx)
    row = cx.execute("SELECT state FROM sales_image_queue WHERE product_slug=?", (slug,)).fetchone()
    return row[0] if row else None

def record_image(cx, slug, kind, variant, filename):
    init_tables(cx)
    cx.execute("INSERT INTO sales_page_images (product_slug, kind, variant, filename, state, created_at) "
               "VALUES (?,?,?,?, 'ready', ?)", (slug, kind, int(variant), filename, _now()))
    cx.commit()

def get_images(cx, slug):
    init_tables(cx)
    rows = cx.execute("SELECT kind, variant, filename FROM sales_page_images "
                      "WHERE product_slug=? AND state='ready' ORDER BY kind, variant", (slug,)).fetchall()
    return [{"kind": r[0], "variant": r[1], "filename": r[2]} for r in rows]

def display_images(cx, slug):
    out = {"botanical": None, "mechanism": None}
    for img in get_images(cx, slug):
        if img["kind"] in out and out[img["kind"]] is None:
            out[img["kind"]] = img["filename"]
    return out
```

- [ ] **Step 4: Run** — `... -k "queue or record" -v` → PASS (3).
- [ ] **Step 5: Commit** — `git add dashboard/sales_images.py tests/test_sales_pages_phase3.py && git commit -m "feat: sales image queue + image-record data layer"`

---

## Task 2: prompt builder `dashboard/sales_image_prompts.py`

**Files:** Create `dashboard/sales_image_prompts.py`; Test append to `tests/test_sales_pages_phase3.py`

**Interfaces (produces):** `IMAGE_KINDS = ("botanical","mechanism")`; `build_image_prompts(product) -> {"botanical": [str, str], "mechanism": [str, str]}`. `product` has `name` and `ingredients` (list of `{name,dose}` or str).

- [ ] **Step 1: Write the failing test**

```python
from dashboard import sales_image_prompts as sip

def test_prompts_two_modes_two_variants_each():
    p = sip.build_image_prompts({"name": "Longevity", "ingredients": [{"name": "Resveratrol"}]})
    assert set(p.keys()) == {"botanical", "mechanism"}
    assert len(p["botanical"]) == 2 and len(p["mechanism"]) == 2
    # variants within a kind are distinct
    assert p["botanical"][0] != p["botanical"][1]

def test_prompts_ground_in_ingredients_and_name():
    p = sip.build_image_prompts({"name": "Longevity", "ingredients": [{"name": "Resveratrol"}, "Quercetin"]})
    joined = " ".join(p["botanical"] + p["mechanism"])
    assert "Resveratrol" in joined and "Quercetin" in joined
    # botanical references the lifestyle scene; mechanism references the protective-field concept
    assert "kitchen" in p["botanical"][0].lower()
    assert "cell" in p["mechanism"][0].lower() or "field" in p["mechanism"][0].lower()
```

- [ ] **Step 2: Run** — `... -k prompts -v` → FAIL (no module).

- [ ] **Step 3: Implement**

```python
# dashboard/sales_image_prompts.py
IMAGE_KINDS = ("botanical", "mechanism")

def _ingredient_names(product):
    out = []
    for ing in (product.get("ingredients") or []):
        if isinstance(ing, dict) and ing.get("name"): out.append(ing["name"])
        elif isinstance(ing, str) and ing.strip(): out.append(ing.strip())
    return out

# Two style directives per kind so the variants are genuinely distinct (Phase-4 A/B).
_BOTANICAL_VARIANTS = [
    "warm natural daylight, eye-level composition",
    "soft golden-hour light, slightly elevated three-quarter angle",
]
_MECHANISM_VARIANTS = [
    "clean studio render, deep teal background",
    "luminous dark background with volumetric light, dramatic angle",
]

def build_image_prompts(product):
    name = product.get("name", "")
    ings = _ingredient_names(product)
    ing_phrase = ", ".join(ings[:6]) if ings else "fresh botanicals"
    botanical = [
        (f"Photo-quality botanical lifestyle scene for the supplement '{name}': the formula's fresh and "
         f"powdered botanical ingredients ({ing_phrase}) arranged on a natural wooden kitchen counter, an "
         f"attractive mature woman preparing them, a lush herb garden visible behind her; {style}.")
        for style in _BOTANICAL_VARIANTS
    ]
    mechanism = [
        (f"Photo-quality conceptual mechanism render for the supplement '{name}': a living human cell "
         f"surrounded by a radiant protective energy field, nourished by the formula's key compounds "
         f"({ing_phrase}), conveying cellular resilience and protection; {style}.")
        for style in _MECHANISM_VARIANTS
    ]
    return {"botanical": botanical, "mechanism": mechanism}
```

- [ ] **Step 4: Run** — `... -k prompts -v` → PASS (2).
- [ ] **Step 5: Commit** — `git add dashboard/sales_image_prompts.py tests/test_sales_pages_phase3.py && git commit -m "feat: grounded Flux prompt builder (botanical + mechanism, 2 variants each)"`

---

## Task 3: Replicate REST client `dashboard/replicate_client.py`

**Files:** Create `dashboard/replicate_client.py`; Test append.

**Interfaces (produces):** `generate_image(prompt, *, token=None, aspect_ratio="1:1", timeout=120) -> bytes`. Raises on missing token / failure / timeout.

- [ ] **Step 1: Write the failing test**

```python
from dashboard import replicate_client as rc

class _Resp:
    def __init__(self, js=None, content=b"", status=200): self._js=js; self.content=content; self.status_code=status
    def json(self): return self._js
    def raise_for_status(self):
        if self.status_code >= 400: raise RuntimeError("http %d" % self.status_code)

def test_generate_image_returns_bytes(monkeypatch):
    calls = {"post": 0, "get": 0}
    def fake_post(url, **kw):
        calls["post"] += 1
        return _Resp(js={"status": "succeeded", "output": "https://img/x.png", "urls": {"get": "https://api/get"}})
    def fake_get(url, **kw):
        calls["get"] += 1
        return _Resp(content=b"PNGBYTES")
    monkeypatch.setattr(rc.requests, "post", fake_post)
    monkeypatch.setattr(rc.requests, "get", fake_get)
    out = rc.generate_image("a prompt", token="tok")
    assert out == b"PNGBYTES" and calls["post"] == 1

def test_generate_image_raises_on_failed_status(monkeypatch):
    monkeypatch.setattr(rc.requests, "post", lambda url, **kw: _Resp(js={"status": "failed", "urls": {"get": "g"}}))
    import pytest
    with pytest.raises(Exception):
        rc.generate_image("p", token="tok")

def test_generate_image_requires_token(monkeypatch):
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    import pytest
    with pytest.raises(Exception):
        rc.generate_image("p")
```

- [ ] **Step 2: Run** — `... -k generate_image -v` → FAIL (no module).

- [ ] **Step 3: Implement**

```python
# dashboard/replicate_client.py
import os, time, requests

_MODEL_URL = "https://api.replicate.com/v1/models/black-forest-labs/flux-1.1-pro/predictions"

def generate_image(prompt, *, token=None, aspect_ratio="1:1", timeout=120):
    token = token or os.environ.get("REPLICATE_API_TOKEN", "")
    if not token:
        raise RuntimeError("REPLICATE_API_TOKEN not set")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Prefer": "wait"}
    body = {"input": {"prompt": prompt, "aspect_ratio": aspect_ratio, "output_format": "png"}}
    r = requests.post(_MODEL_URL, headers=headers, json=body, timeout=90)
    r.raise_for_status()
    pred = r.json()
    get_url = (pred.get("urls") or {}).get("get")
    deadline = time.time() + timeout
    while pred.get("status") not in ("succeeded", "failed", "canceled"):
        if time.time() > deadline:
            raise TimeoutError("replicate prediction timed out")
        time.sleep(2)
        pred = requests.get(get_url, headers=headers, timeout=30).json()
    if pred.get("status") != "succeeded":
        raise RuntimeError(f"replicate prediction {pred.get('status')}")
    out = pred.get("output")
    url = out[0] if isinstance(out, list) else out
    if not url:
        raise RuntimeError("replicate returned no output")
    img = requests.get(url, timeout=60)
    img.raise_for_status()
    return img.content
```

- [ ] **Step 4: Run** — `... -k generate_image -v` → PASS (3).
- [ ] **Step 5: Commit** — `git add dashboard/replicate_client.py tests/test_sales_pages_phase3.py && git commit -m "feat: Replicate REST client for Flux 1.1 Pro (requests, no new dep)"`

---

## Task 4: flag + disk dir + worker + scheduler registration

**Files:** Modify `app.py`; Test append.

**Interfaces:** `_SALES_AI_IMAGES_ENABLED` (bool); `_SALES_IMG_DIR` (Path); `_drain_sales_image_queue()` worker (flag-gated; processes pending slugs); registered in `_start_scheduler`.

**Consumes:** `dashboard.sales_images`, `dashboard.sales_image_prompts`, `dashboard.replicate_client`, `_get_product`, `_product_card`, `LOG_DB`.

- [ ] **Step 1: Write the failing test**

```python
import importlib, sqlite3, pathlib

def _reload(monkeypatch, tmp_path, imgs="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path)); monkeypatch.setenv("SALES_PAGES_ENABLED", "true")
    monkeypatch.setenv("SALES_PAGES_AI_IMAGES", imgs)
    import app as appmod; importlib.reload(appmod); return appmod

def test_worker_generates_and_records(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    monkeypatch.setattr(appmod, "_product_card", lambda p: {"ingredients": [{"name": "Resveratrol"}]})
    from dashboard import replicate_client as rc
    monkeypatch.setattr(rc, "generate_image", lambda prompt, **kw: b"PNG")
    from dashboard import sales_images as si
    with sqlite3.connect(appmod.LOG_DB) as cx: si.enqueue(cx, slug)
    appmod._drain_sales_image_queue()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert si.queue_state(cx, slug) == "done"
        assert len(si.get_images(cx, slug)) == 4
    files = list((appmod._SALES_IMG_DIR / slug).glob("*.png"))
    assert len(files) == 4

def test_worker_flag_off_noop(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path, imgs="false")
    assert appmod._SALES_AI_IMAGES_ENABLED is False
    from dashboard import sales_images as si
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    with sqlite3.connect(appmod.LOG_DB) as cx: si.enqueue(cx, slug)
    appmod._drain_sales_image_queue()  # flag off → no-op
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert si.queue_state(cx, slug) == "pending"

def test_worker_marks_failed_on_error(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    monkeypatch.setattr(appmod, "_product_card", lambda p: {"ingredients": []})
    from dashboard import replicate_client as rc
    def boom(prompt, **kw): raise RuntimeError("replicate down")
    monkeypatch.setattr(rc, "generate_image", boom)
    from dashboard import sales_images as si
    with sqlite3.connect(appmod.LOG_DB) as cx: si.enqueue(cx, slug)
    appmod._drain_sales_image_queue()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert si.queue_state(cx, slug) == "failed"
```

- [ ] **Step 2: Run** — `... -k worker -v` → FAIL (no `_drain_sales_image_queue`).

- [ ] **Step 3: Implement**

```python
# app.py — near _SALES_AI_COPY_ENABLED (~L2301)
_SALES_AI_IMAGES_ENABLED = os.environ.get("SALES_PAGES_AI_IMAGES", "").strip().lower() in ("1", "true", "yes")

# app.py — near _CLIPS_DIR (~L11827)
_SALES_IMG_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent))) / "sales-images"
_SALES_IMG_DIR.mkdir(parents=True, exist_ok=True)

# app.py — define before _start_scheduler (e.g. just above _run_cron)
def _drain_sales_image_queue():
    """Scheduler job: render queued product images via Replicate (off web workers)."""
    if not _SALES_AI_IMAGES_ENABLED:
        return
    from dashboard import sales_images as _si, sales_image_prompts as _sip, replicate_client as _rc
    try:
        with sqlite3.connect(LOG_DB) as cx:
            pending = _si.list_pending(cx)[:2]   # cap per tick (Replicate spend)
    except Exception as e:
        print(f"[sales-img] queue read failed: {e}", flush=True); return
    for slug in pending:
        p = _get_product(slug)
        if not p:
            with sqlite3.connect(LOG_DB) as cx: _si.mark_failed(cx, slug)
            continue
        prod = dict(p)
        if not prod.get("ingredients"):
            prod["ingredients"] = (_product_card(p) or {}).get("ingredients", [])
        prompts = _sip.build_image_prompts(prod)
        dest = _SALES_IMG_DIR / slug
        dest.mkdir(parents=True, exist_ok=True)
        ok = 0
        for kind in _sip.IMAGE_KINDS:
            for variant, prompt in enumerate(prompts[kind], start=1):
                try:
                    data = _rc.generate_image(prompt)
                    fname = f"{kind}-{variant}.png"
                    (dest / fname).write_bytes(data)
                    with sqlite3.connect(LOG_DB) as cx:
                        _si.record_image(cx, slug, kind, variant, fname)
                    ok += 1
                except Exception as e:
                    print(f"[sales-img] {slug} {kind}-{variant} failed: {e}", flush=True)
        with sqlite3.connect(LOG_DB) as cx:
            (_si.mark_done if ok else _si.mark_failed)(cx, slug)

# app.py — inside _start_scheduler, after the existing add_job calls:
        scheduler.add_job(_drain_sales_image_queue, "interval", minutes=1, id="sales_image_gen")
```

- [ ] **Step 4: Run** — `... -k worker -v` → PASS (3).
- [ ] **Step 5: Commit** — `git add app.py tests/test_sales_pages_phase3.py && git commit -m "feat: image-gen worker + flag + scheduler registration"`

---

## Task 5: serving + enqueue routes

**Files:** Modify `app.py`; Test append.

**Interfaces (produces):** `GET /begin/product-image/<slug>/<filename>` → serves PNG from `_SALES_IMG_DIR/<slug>`, 404 if absent/invalid; `POST /begin/product-image-gen/<slug>` → enqueues (flag-gated; 404 if off / unknown slug), idempotent, returns `{"ok": true, "state": "<queue state>"}`.

- [ ] **Step 1: Write the failing test**

```python
def test_enqueue_route_and_404_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    c = appmod.app.test_client()
    r = c.post(f"/begin/product-image-gen/{slug}")
    assert r.status_code == 200 and r.get_json().get("ok") is True
    from dashboard import sales_images as si
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx: assert si.queue_state(cx, slug) == "pending"
    # flag off
    off = _reload(monkeypatch, tmp_path, imgs="false")
    assert off.app.test_client().post(f"/begin/product-image-gen/{slug}").status_code == 404

def test_serve_image_route(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    d = appmod._SALES_IMG_DIR / slug; d.mkdir(parents=True, exist_ok=True)
    (d / "botanical-1.png").write_bytes(b"\x89PNG\r\n")
    c = appmod.app.test_client()
    assert c.get(f"/begin/product-image/{slug}/botanical-1.png").status_code == 200
    assert c.get(f"/begin/product-image/{slug}/missing.png").status_code == 404
    assert c.get(f"/begin/product-image/{slug}/../evil.png").status_code in (400, 404)
```

- [ ] **Step 2: Run** — `... -k "enqueue_route or serve_image" -v` → FAIL (routes undefined).

- [ ] **Step 3: Implement**

```python
# app.py — after begin_product_page_gen (or near the other /begin routes)
@app.route("/begin/product-image/<slug>/<filename>")
def begin_product_image(slug, filename):
    if not re.match(r'^[\w\-]+\.png$', filename):
        return ("", 404)
    d = _SALES_IMG_DIR / slug
    if not (d / filename).exists():
        return ("", 404)
    return send_from_directory(str(d), filename, mimetype="image/png")

@app.route("/begin/product-image-gen/<slug>", methods=["POST"])
def begin_product_image_gen(slug):
    if not _SALES_AI_IMAGES_ENABLED or not _get_product(slug):
        return ("", 404)
    from dashboard import sales_images as _si
    with sqlite3.connect(LOG_DB) as cx:
        _si.enqueue(cx, slug)
        state = _si.queue_state(cx, slug)
    return jsonify({"ok": True, "state": state})
```

- [ ] **Step 4: Run** — `... -k "enqueue_route or serve_image" -v` → PASS.
- [ ] **Step 5: Commit** — `git add app.py tests/test_sales_pages_phase3.py && git commit -m "feat: image serve + enqueue routes"`

---

## Task 6: page-data images marker

**Files:** Modify `app.py` (`begin_product_page_data`); Test append.

**Interfaces (produces):** when `SALES_PAGES_AI_IMAGES` on, the `images` section body becomes `{"images": [{"kind","url"}...], "state": "ready"|"generating"|"none"}`. `ready` when display images exist (urls = `/begin/product-image/<slug>/<file>`); `generating` when queue pending and no ready images; else `none`. Flag off → images body unchanged (no new fields).

- [ ] **Step 1: Write the failing test**

```python
def test_page_data_images_states(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    c = appmod.app.test_client()
    # none
    d = c.get(f"/begin/product-page-data/{slug}").get_json()
    img = next(s for s in d["sections"] if s["id"] == "images")["body"]
    assert img.get("state") == "none"
    # generating
    import sqlite3
    from dashboard import sales_images as si
    with sqlite3.connect(appmod.LOG_DB) as cx: si.enqueue(cx, slug)
    img = next(s for s in c.get(f"/begin/product-page-data/{slug}").get_json()["sections"] if s["id"]=="images")["body"]
    assert img.get("state") == "generating"
    # ready
    with sqlite3.connect(appmod.LOG_DB) as cx:
        si.record_image(cx, slug, "botanical", 1, "botanical-1.png")
        si.record_image(cx, slug, "mechanism", 1, "mechanism-1.png")
    img = next(s for s in c.get(f"/begin/product-page-data/{slug}").get_json()["sections"] if s["id"]=="images")["body"]
    assert img.get("state") == "ready"
    urls = [i["url"] for i in img["images"]]
    assert f"/begin/product-image/{slug}/botanical-1.png" in urls

def test_page_data_images_flag_off(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path, imgs="false")
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    img = next(s for s in appmod.app.test_client().get(f"/begin/product-page-data/{slug}").get_json()["sections"] if s["id"]=="images")["body"]
    assert "state" not in img
```

- [ ] **Step 2: Run** — `... -k "page_data_images" -v` → FAIL.

- [ ] **Step 3: Implement** — in `begin_product_page_data`, after the `_SALES_AI_COPY_ENABLED` block, add:

```python
    if _SALES_AI_IMAGES_ENABLED:
        import sqlite3 as _sq2
        from dashboard import sales_images as _si2
        try:
            with _sq2.connect(LOG_DB) as _cx2:
                _disp = _si2.display_images(_cx2, slug)
                _qstate = _si2.queue_state(_cx2, slug)
            _imgs = [{"kind": k, "url": f"/begin/product-image/{slug}/{fn}"}
                     for k, fn in _disp.items() if fn]
            _img_sec = next((s for s in sections if s["id"] == "images"), None)
            if _img_sec is not None:
                if _imgs:
                    _img_sec["body"] = {"images": _imgs, "state": "ready"}
                elif _qstate == "pending":
                    _img_sec["body"] = {"images": [], "state": "generating"}
                else:
                    _img_sec["body"] = {"images": [], "state": "none"}
        except Exception as _e:
            print(f"[sales-img] page-data marker skipped: {_e}", flush=True)
```

- [ ] **Step 4: Run** — `... -k "page_data_images" -v` → PASS.
- [ ] **Step 5: Commit** — `git add app.py tests/test_sales_pages_phase3.py && git commit -m "feat: page-data images section state (ready/generating/none)"`

---

## Task 7: frontend — render images / generating placeholder + enqueue + poll

**Files:** Modify `static/begin-product.html` (`renderImagesBody`); Manual verification.

**Interfaces:** consumes the `images` section body `{images:[{kind,url}], state}`; uses existing `BASE`, `slug`.

- [ ] **Step 1: Implement** `renderImagesBody(body)`:
  - `state === 'ready'` → render each `images[i]` as `<img src=url alt="...">` (one botanical, one mechanism), styled to fit (reuse `.miron-rotator img` sizing or a new `.sp-product-img` rule — full width, rounded).
  - `state === 'generating'` → a muted "Generating product imagery…" line; start a poll: every 4s `fetch('/begin/product-page-data/'+slug)`, read the images section; when `state === 'ready'`, re-render the block; stop after ~10 tries.
  - `state === 'none'` → on first render of an open images section, `fetch(BASE + '/begin/product-image-gen/' + slug, {method:'POST', credentials:'same-origin'})` once (guard with a `dataset` flag), show the "Generating…" line, and start the same poll.
  - No `state`/undefined (flag off) → keep current empty/placeholder behavior.
  - NO emoji. Match existing styling.

- [ ] **Step 2: Manual verification** — locally with `SALES_PAGES_ENABLED=true SALES_PAGES_AI_IMAGES=true` and `REPLICATE_API_TOKEN` set: open a product, open the images section → "Generating…" → enqueue → (scheduler tick / or trigger `_drain_sales_image_queue()` once) → poll backfills the botanical + mechanism images. Flag off → images section unchanged. Full visual is a manual product-owner pass.

- [ ] **Step 3: Commit** — `git add static/begin-product.html && git commit -m "feat: images section renders generated imagery + generating placeholder/poll"`

---

## Task 8: integration + flag default

**Files:** Modify `tests/test_sales_pages_phase3.py`

- [ ] **Step 1: Write the test**

```python
def test_flag_defaults_off(monkeypatch, tmp_path):
    monkeypatch.delenv("SALES_PAGES_AI_IMAGES", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path)); monkeypatch.setenv("SALES_PAGES_ENABLED", "true")
    import importlib, app as appmod; importlib.reload(appmod)
    assert appmod._SALES_AI_IMAGES_ENABLED is False
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    assert appmod.app.test_client().post(f"/begin/product-image-gen/{slug}").status_code == 404
```

- [ ] **Step 2: Run the full Phase-3 file + Phase 1/2** — `... -m pytest tests/test_sales_pages_phase3.py tests/test_sales_pages_phase2.py tests/test_sales_pages_phase1.py -v` → all pass.
- [ ] **Step 3: Confirm flag + token default OFF/unset in Render** — do NOT set `SALES_PAGES_AI_IMAGES` yet; `REPLICATE_API_TOKEN` is the other go-live prerequisite. Ship dark.
- [ ] **Step 4: Commit** — `git add tests/test_sales_pages_phase3.py && git commit -m "test: phase-3 integration + flag-default-off"`

---

## Verification (end to end)

1. `... -m pytest tests/test_sales_pages_phase3.py tests/test_sales_pages_phase2.py tests/test_sales_pages_phase1.py -v` → all pass; full suite no new failures.
2. **Flag off (default):** pages render exactly as live Phase 1/2 — no images field, gen/serve routes behave (serve 404s on missing; enqueue 404s), no scheduler image work.
3. **Flag on + `REPLICATE_API_TOKEN` set (local):** opening the images section enqueues; the scheduler worker renders 4 Flux images to `DATA_DIR/sales-images/<slug>/`, records rows; the page polls and shows one botanical + one mechanism; cached thereafter; Replicate failure degrades to no images without breaking the page.
