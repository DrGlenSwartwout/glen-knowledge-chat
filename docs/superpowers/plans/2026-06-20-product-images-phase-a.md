# Product Page Images — Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show 4 images per type (botanical + mechanism = 8/product) on `/begin/product/<slug>` pages, each generated from a distinct prompt variation × rotating model, every image tagged with both for later split-testing, with a per-type grid, model-source labels, and visitor-triggered lazy top-up.

**Architecture:** Two new SQLite-backed registries (prompt variations, image models) seed defaults on first use. A balanced assignment builds up to 8 generation jobs/product (all 4 variations per kind × models rotated by a per-product offset), each tagged. The existing on-open generate+poll path gains a top-up gate; the worker generates only missing slots via a model dispatcher and records both tags. The data endpoint emits a grouped, stateful payload; the template renders two responsive grids with model captions and generating placeholders. Everything new is gated by `SALES_PAGES_IMAGE_VARIATIONS` and ships dark.

**Tech Stack:** Python 3 / Flask, SQLite (`LOG_DB`), Replicate HTTP API (Flux 1.1 Pro / Imagen 4 / Recraft V3), vanilla JS template (`static/begin-product.html`), pytest.

## Global Constraints

- New behavior gated by env flag `SALES_PAGES_IMAGE_VARIATIONS` (truthy = `1`/`true`/`yes`); OFF = exact current behavior. Read as module global `_SALES_IMAGE_VARIATIONS_ENABLED` in `app.py`.
- Image prompts carry **NO product names and NO text** — always append `sales_image_prompts._NO_TEXT`; never inject product name/ingredients into the prompt string.
- Target is **4 images per kind** (kinds = `("botanical","mechanism")`), **8 per product**. Image filenames: `"{kind}-{variant}.png"`, `variant` = slot index `1..4`.
- Tests: pytest, `sqlite3.connect(":memory:")` per test, import `dashboard.*` modules directly, no live network (monkeypatch Replicate). Follow `tests/test_sales_pages_phase4b.py` style. Honor deploy-chat test isolation (no shared DB, no live Supabase).
- Work in worktree `/tmp/wt-deploy-chat-db16e904` (branch `sess/db16e904`). Commit per task. No edits to `main`. Confirm live Replicate pricing/refs for Imagen 4 & Recraft V3 before enabling in prod (does not block building/testing with mocks).

---

### Task 1: Prompt-variation registry

**Files:**
- Create: `dashboard/sales_prompt_variations.py`
- Test: `tests/test_sales_pages_phase_a.py`

**Interfaces:**
- Produces: `init_table(cx)`, `seed(cx)` (idempotent), `active_variations(cx, kind) -> list[dict]` with keys `id, kind, label, prompt_template`, ordered by `id`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sales_pages_phase_a.py
import sqlite3
from dashboard import sales_prompt_variations as pv

def _cx(): return sqlite3.connect(":memory:")

def test_seed_creates_four_active_variations_per_kind():
    cx = _cx(); pv.seed(cx)
    for kind in ("botanical", "mechanism"):
        rows = pv.active_variations(cx, kind)
        assert len(rows) == 4
        assert all(r["kind"] == kind for r in rows)
        assert all(r["prompt_template"] and r["label"] for r in rows)
        # distinct scenes, not duplicates
        assert len({r["prompt_template"] for r in rows}) == 4

def test_seed_is_idempotent():
    cx = _cx(); pv.seed(cx); pv.seed(cx)
    assert len(pv.active_variations(cx, "botanical")) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sales_pages_phase_a.py -q`
Expected: FAIL (module `sales_prompt_variations` does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/sales_prompt_variations.py
import datetime

def _now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

_SEED = {
    "botanical": [
        ("flat-lay", "Warm overhead flat-lay of an abundance of fresh whole herbs, green leaves, "
                     "roots, and colorful botanical ingredients arranged on a rustic wooden kitchen "
                     "counter, soft natural daylight."),
        ("kitchen-woman", "An attractive mature woman in a sunlit farmhouse kitchen gently preparing "
                          "fresh herbs at a wooden counter, a lush green herb garden visible through the "
                          "window behind her, golden-hour light."),
        ("still-life", "A close intimate still-life of fresh botanical ingredients — sprigs, flowers, and "
                       "sliced roots — on a weathered cutting board, shallow depth of field, soft morning light."),
        ("market-basket", "An abundant woven market basket overflowing with fresh colorful botanicals and "
                          "leafy greens on a garden table outdoors, dappled natural sunlight, lush plants behind."),
    ],
    "mechanism": [
        ("shielded-cell", "A single glowing living human cell surrounded by a radiant spherical protective "
                          "energy field, luminous particles flowing inward, deep teal studio background, "
                          "clean conceptual render."),
        ("dramatic-shield", "A luminous human cell with a shimmering protective shield, dramatic volumetric "
                            "light on a dark background, iridescent particles drifting toward it."),
        ("cross-section", "A cross-section of a vibrant human cell with a glowing luminous membrane and "
                          "energized interior, symmetrical centered composition, iridescent blue-violet palette."),
        ("repelling-field", "A radiant cellular energy field repelling dark chaotic stressor particles away "
                            "from a healthy glowing cell, warm amber glow on a black background, shallow depth of field."),
    ],
}

def init_table(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS sales_prompt_variations ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, label TEXT, "
               "prompt_template TEXT, state TEXT DEFAULT 'active', "
               "created_at TEXT DEFAULT '', retired_at TEXT DEFAULT '')")
    cx.commit()

def seed(cx):
    init_table(cx)
    n = cx.execute("SELECT COUNT(*) FROM sales_prompt_variations").fetchone()[0]
    if n:
        return
    now = _now()
    for kind, items in _SEED.items():
        for label, template in items:
            cx.execute("INSERT INTO sales_prompt_variations (kind, label, prompt_template, state, created_at) "
                       "VALUES (?,?,?, 'active', ?)", (kind, label, template, now))
    cx.commit()

def active_variations(cx, kind):
    seed(cx)
    rows = cx.execute("SELECT id, kind, label, prompt_template FROM sales_prompt_variations "
                      "WHERE kind=? AND state='active' ORDER BY id", (kind,)).fetchall()
    return [{"id": r[0], "kind": r[1], "label": r[2], "prompt_template": r[3]} for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sales_pages_phase_a.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_prompt_variations.py tests/test_sales_pages_phase_a.py
git commit -m "feat(sales-img): prompt-variation registry (Phase A task 1)"
```

---

### Task 2: Image-model registry

**Files:**
- Create: `dashboard/sales_image_models.py`
- Test: `tests/test_sales_pages_phase_a.py` (append)

**Interfaces:**
- Produces: `init_table(cx)`, `seed(cx)`, `active_models(cx) -> list[dict]` with keys `id, label, engine, engine_ref`, ordered by `id`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sales_pages_phase_a.py
from dashboard import sales_image_models as mods

def test_seed_creates_three_active_models():
    cx = _cx(); mods.seed(cx)
    rows = mods.active_models(cx)
    ids = [m["id"] for m in rows]
    assert ids == ["flux-1.1-pro", "imagen-4", "recraft-v3"]
    assert all(m["engine"] == "replicate" and m["engine_ref"] for m in rows)
    assert all(m["label"] for m in rows)

def test_models_seed_idempotent():
    cx = _cx(); mods.seed(cx); mods.seed(cx)
    assert len(mods.active_models(cx)) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sales_pages_phase_a.py -q`
Expected: FAIL (module `sales_image_models` does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/sales_image_models.py
import datetime

def _now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

# id, label, engine_ref (Replicate model path) — ordered intentionally; baseline first.
_SEED = [
    ("flux-1.1-pro", "Flux 1.1 Pro", "black-forest-labs/flux-1.1-pro"),
    ("imagen-4",     "Imagen 4",     "google/imagen-4"),
    ("recraft-v3",   "Recraft V3",   "recraft-ai/recraft-v3"),
]

def init_table(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS sales_image_models ("
               "id TEXT PRIMARY KEY, label TEXT, engine TEXT DEFAULT 'replicate', "
               "engine_ref TEXT, state TEXT DEFAULT 'active', created_at TEXT DEFAULT '')")
    cx.commit()

def seed(cx):
    init_table(cx)
    n = cx.execute("SELECT COUNT(*) FROM sales_image_models").fetchone()[0]
    if n:
        return
    now = _now()
    for mid, label, ref in _SEED:
        cx.execute("INSERT INTO sales_image_models (id, label, engine, engine_ref, state, created_at) "
                   "VALUES (?,?, 'replicate', ?, 'active', ?)", (mid, label, ref, now))
    cx.commit()

def active_models(cx):
    seed(cx)
    rows = cx.execute("SELECT id, label, engine, engine_ref FROM sales_image_models "
                      "WHERE state='active' ORDER BY rowid").fetchall()
    return [{"id": r[0], "label": r[1], "engine": r[2], "engine_ref": r[3]} for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sales_pages_phase_a.py -q`
Expected: PASS (4 tests total).

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_image_models.py tests/test_sales_pages_phase_a.py
git commit -m "feat(sales-img): image-model registry (Phase A task 2)"
```

---

### Task 3: Multi-model Replicate client

**Files:**
- Modify: `dashboard/replicate_client.py`
- Test: `tests/test_sales_pages_phase_a.py` (append)

**Interfaces:**
- Produces: `generate_image(prompt, *, token=None, aspect_ratio="1:1", timeout=120, model_ref="black-forest-labs/flux-1.1-pro") -> bytes`. URL derived from `model_ref`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sales_pages_phase_a.py
from dashboard import replicate_client as rc

class _Resp:
    def __init__(self, j=None, content=b""): self._j = j or {}; self.content = content
    def json(self): return self._j
    def raise_for_status(self): pass

def test_generate_image_uses_model_ref_url(monkeypatch):
    calls = {}
    def fake_post(url, **kw):
        calls["url"] = url
        return _Resp({"status": "succeeded", "output": ["http://img/x.png"], "urls": {"get": "http://g"}})
    def fake_get(url, **kw):
        return _Resp(content=b"PNGDATA")
    monkeypatch.setattr(rc.requests, "post", fake_post)
    monkeypatch.setattr(rc.requests, "get", fake_get)
    out = rc.generate_image("hello", token="t", model_ref="google/imagen-4")
    assert out == b"PNGDATA"
    assert "google/imagen-4" in calls["url"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sales_pages_phase_a.py::test_generate_image_uses_model_ref_url -q`
Expected: FAIL (`generate_image` has no `model_ref` kwarg).

- [ ] **Step 3: Write minimal implementation**

Replace the top of `dashboard/replicate_client.py` (the `_MODEL_URL` constant and `generate_image` signature/first line) with:

```python
import os, time, requests

_DEFAULT_REF = "black-forest-labs/flux-1.1-pro"

def _model_url(model_ref):
    return f"https://api.replicate.com/v1/models/{model_ref}/predictions"

def generate_image(prompt, *, token=None, aspect_ratio="1:1", timeout=120, model_ref=_DEFAULT_REF):
    token = token or os.environ.get("REPLICATE_API_TOKEN", "")
    if not token:
        raise RuntimeError("REPLICATE_API_TOKEN not set")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Prefer": "wait"}
    body = {"input": {"prompt": prompt, "aspect_ratio": aspect_ratio, "output_format": "png"}}
    r = requests.post(_model_url(model_ref), headers=headers, json=body, timeout=90)
```

Leave the rest of the function body (poll loop, output extraction, image download) unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sales_pages_phase_a.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/replicate_client.py tests/test_sales_pages_phase_a.py
git commit -m "feat(sales-img): replicate_client accepts model_ref (Phase A task 3)"
```

---

### Task 4: Model dispatcher with fallback

**Files:**
- Modify: `dashboard/sales_image_models.py` (add `generate`)
- Test: `tests/test_sales_pages_phase_a.py` (append)

**Interfaces:**
- Consumes: `active_models(cx)` (Task 2); `replicate_client.generate_image(prompt, *, aspect_ratio, model_ref)` (Task 3).
- Produces: `generate(cx, model_id, prompt, *, aspect="1:1") -> (bytes, used_model_id)`. On engine error, retries with the baseline Flux ref and returns `used_model_id="flux-1.1-pro"`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sales_pages_phase_a.py
def test_dispatch_uses_requested_model_ref(monkeypatch):
    cx = _cx(); mods.seed(cx)
    seen = {}
    def fake_gen(prompt, *, aspect_ratio="1:1", model_ref=None, **kw):
        seen["ref"] = model_ref; return b"IMG"
    monkeypatch.setattr(mods, "_rc_generate", fake_gen, raising=False)
    monkeypatch.setattr("dashboard.replicate_client.generate_image", fake_gen)
    data, used = mods.generate(cx, "recraft-v3", "p")
    assert data == b"IMG" and used == "recraft-v3"
    assert seen["ref"] == "recraft-ai/recraft-v3"

def test_dispatch_falls_back_to_flux_on_error(monkeypatch):
    cx = _cx(); mods.seed(cx)
    calls = {"n": 0}
    def flaky(prompt, *, aspect_ratio="1:1", model_ref=None, **kw):
        calls["n"] += 1
        if model_ref != "black-forest-labs/flux-1.1-pro":
            raise RuntimeError("engine down")
        return b"FALLBACK"
    monkeypatch.setattr("dashboard.replicate_client.generate_image", flaky)
    data, used = mods.generate(cx, "imagen-4", "p")
    assert data == b"FALLBACK" and used == "flux-1.1-pro"
    assert calls["n"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sales_pages_phase_a.py -q`
Expected: FAIL (`mods.generate` undefined).

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/sales_image_models.py`:

```python
_BASELINE_REF = "black-forest-labs/flux-1.1-pro"

def generate(cx, model_id, prompt, *, aspect="1:1"):
    """Return (image_bytes, used_model_id). Falls back to baseline Flux on engine error."""
    from dashboard import replicate_client as _rc
    by_id = {m["id"]: m for m in active_models(cx)}
    m = by_id.get(model_id)
    ref = m["engine_ref"] if m else _BASELINE_REF
    try:
        return _rc.generate_image(prompt, aspect_ratio=aspect, model_ref=ref), model_id
    except Exception as e:
        if ref == _BASELINE_REF:
            raise
        data = _rc.generate_image(prompt, aspect_ratio=aspect, model_ref=_BASELINE_REF)
        return data, "flux-1.1-pro"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sales_pages_phase_a.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_image_models.py tests/test_sales_pages_phase_a.py
git commit -m "feat(sales-img): model dispatcher with Flux fallback (Phase A task 4)"
```

---

### Task 5: Image tags (schema + record + counts)

**Files:**
- Modify: `dashboard/sales_images.py`
- Modify: `dashboard/sales_image_prompts.py` (expose `NO_TEXT`)
- Test: `tests/test_sales_pages_phase_a.py` (append)

**Interfaces:**
- Produces: `record_image(cx, slug, kind, variant, filename, prompt_variant_id=None, model_id=None)`; `get_images(cx, slug)` rows now include `prompt_variant_id, model_id`; `tagged_count(cx, slug) -> int`; `needs_topup(cx, slug, target=8) -> bool`; `sales_image_prompts.NO_TEXT` constant.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sales_pages_phase_a.py
from dashboard import sales_images as si
from dashboard import sales_image_prompts as sip

def test_record_image_persists_tags_and_counts():
    cx = _cx()
    si.record_image(cx, "p", "botanical", 1, "botanical-1.png", prompt_variant_id=3, model_id="imagen-4")
    si.record_image(cx, "p", "botanical", 2, "botanical-2.png")   # legacy, untagged
    rows = {r["variant"]: r for r in si.get_images(cx, "p")}
    assert rows[1]["prompt_variant_id"] == 3 and rows[1]["model_id"] == "imagen-4"
    assert rows[2]["prompt_variant_id"] is None
    assert si.tagged_count(cx, "p") == 1
    assert si.needs_topup(cx, "p") is True

def test_no_text_constant_exposed():
    assert "No text" in sip.NO_TEXT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sales_pages_phase_a.py -q`
Expected: FAIL (`record_image` has no tag kwargs / `NO_TEXT` missing).

- [ ] **Step 3: Write minimal implementation**

In `dashboard/sales_image_prompts.py`, add right after the `_NO_TEXT = (...)` definition:

```python
NO_TEXT = _NO_TEXT   # public alias
```

In `dashboard/sales_images.py`, extend `init_tables` (after the `sales_page_images` CREATE, before `cx.commit()`):

```python
    for _col, _decl in (("prompt_variant_id", "INTEGER"), ("model_id", "TEXT")):
        try:
            cx.execute(f"ALTER TABLE sales_page_images ADD COLUMN {_col} {_decl}")
        except Exception:
            pass
```

Replace `record_image` and `get_images` and add the helpers:

```python
def record_image(cx, slug, kind, variant, filename, prompt_variant_id=None, model_id=None):
    init_tables(cx)
    cx.execute("INSERT INTO sales_page_images "
               "(product_slug, kind, variant, filename, state, created_at, prompt_variant_id, model_id) "
               "VALUES (?,?,?,?, 'ready', ?, ?, ?)",
               (slug, kind, int(variant), filename, _now(), prompt_variant_id, model_id))
    cx.commit()

def get_images(cx, slug):
    init_tables(cx)
    rows = cx.execute("SELECT kind, variant, filename, prompt_variant_id, model_id "
                      "FROM sales_page_images WHERE product_slug=? AND state='ready' "
                      "ORDER BY kind, variant", (slug,)).fetchall()
    return [{"kind": r[0], "variant": r[1], "filename": r[2],
             "prompt_variant_id": r[3], "model_id": r[4]} for r in rows]

def tagged_count(cx, slug):
    init_tables(cx)
    r = cx.execute("SELECT COUNT(*) FROM sales_page_images WHERE product_slug=? AND state='ready' "
                   "AND prompt_variant_id IS NOT NULL", (slug,)).fetchone()
    return r[0] if r else 0

def needs_topup(cx, slug, target=8):
    return tagged_count(cx, slug) < target
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sales_pages_phase_a.py -q`
Expected: PASS. Also run the existing image tests to confirm no regression: `python -m pytest tests/test_sales_pages_phase4b.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_images.py dashboard/sales_image_prompts.py tests/test_sales_pages_phase_a.py
git commit -m "feat(sales-img): tag images with prompt_variant_id + model_id (Phase A task 5)"
```

---

### Task 6: Balanced generation-job builder

**Files:**
- Modify: `dashboard/sales_images.py` (add `build_generation_jobs`)
- Test: `tests/test_sales_pages_phase_a.py` (append)

**Interfaces:**
- Consumes: `sales_prompt_variations.active_variations`, `sales_image_models.active_models`, `sales_image_prompts.IMAGE_KINDS` + `NO_TEXT`, `get_images` (Task 5).
- Produces: `build_generation_jobs(cx, slug) -> list[dict]` with keys `kind, variant, prompt_variant_id, model_id, prompt_text`; returns only **missing** (kind, slot) cells; deterministic per slug.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sales_pages_phase_a.py
def test_build_jobs_full_set_covers_all_variations_and_8_slots():
    cx = _cx()
    jobs = si.build_generation_jobs(cx, "alpha")
    assert len(jobs) == 8
    for kind in ("botanical", "mechanism"):
        kjobs = [j for j in jobs if j["kind"] == kind]
        assert sorted(j["variant"] for j in kjobs) == [1, 2, 3, 4]
        assert len({j["prompt_variant_id"] for j in kjobs}) == 4     # all 4 variations
        assert all("No text" in j["prompt_text"] for j in kjobs)     # NO_TEXT appended
        assert all(j["model_id"] in ("flux-1.1-pro", "imagen-4", "recraft-v3") for j in kjobs)

def test_build_jobs_skips_present_slots():
    cx = _cx()
    si.record_image(cx, "beta", "botanical", 1, "botanical-1.png", prompt_variant_id=1, model_id="flux-1.1-pro")
    jobs = si.build_generation_jobs(cx, "beta")
    assert ("botanical", 1) not in {(j["kind"], j["variant"]) for j in jobs}
    assert len(jobs) == 7

def test_build_jobs_deterministic_and_model_offset_varies_by_slug():
    cx = _cx()
    j1 = si.build_generation_jobs(cx, "slug-one")
    j1b = si.build_generation_jobs(cx, "slug-one")
    assert [j["model_id"] for j in j1] == [j["model_id"] for j in j1b]   # deterministic
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sales_pages_phase_a.py -q`
Expected: FAIL (`build_generation_jobs` undefined).

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/sales_images.py`:

```python
def build_generation_jobs(cx, slug):
    """Missing (kind, slot) generation jobs for `slug`, up to 4/kind. Each job is tagged
    with its prompt variation and an assigned model. All 4 variations are covered per kind;
    models rotate by a per-product offset for balanced marginal coverage across products."""
    import zlib
    from dashboard import sales_prompt_variations as _pv
    from dashboard import sales_image_models as _mods
    from dashboard import sales_image_prompts as _sip
    init_tables(cx)
    present = {(im["kind"], im["variant"]) for im in get_images(cx, slug)}
    models = _mods.active_models(cx)
    if not models:
        return []
    offset = zlib.crc32(slug.encode("utf-8")) % len(models)
    jobs = []
    for kind in _sip.IMAGE_KINDS:
        variations = _pv.active_variations(cx, kind)[:4]
        for i, var in enumerate(variations):
            slot = i + 1
            if (kind, slot) in present:
                continue
            model = models[(i + offset) % len(models)]
            prompt_text = f"{var['prompt_template']} {_sip.NO_TEXT}"
            jobs.append({"kind": kind, "variant": slot,
                         "prompt_variant_id": var["id"], "model_id": model["id"],
                         "prompt_text": prompt_text})
    return jobs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sales_pages_phase_a.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_images.py tests/test_sales_pages_phase_a.py
git commit -m "feat(sales-img): balanced variation x model job builder (Phase A task 6)"
```

---

### Task 7: Grouped display + state

**Files:**
- Modify: `dashboard/sales_images.py` (add `display_images_grouped`, `images_grouped_state`)
- Test: `tests/test_sales_pages_phase_a.py` (append)

**Interfaces:**
- Consumes: `sales_image_models.active_models`, `sales_image_prompts.IMAGE_KINDS`.
- Produces: `display_images_grouped(cx, slug, per_kind=4) -> {kind: [ {url, variant, prompt_variant_id, model_id, model_label} ]}` (≤`per_kind` tagged per kind ordered by variant; legacy untagged shown only when a kind has no tagged images, with `model_label=None`); `images_grouped_state(cx, slug, target=8) -> "ready"|"generating"`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sales_pages_phase_a.py
def test_grouped_returns_tagged_with_labels_and_state():
    cx = _cx()
    for v in (1, 2, 3, 4):
        si.record_image(cx, "p", "botanical", v, f"botanical-{v}.png", prompt_variant_id=v, model_id="imagen-4")
    g = si.display_images_grouped(cx, "p")
    assert [e["variant"] for e in g["botanical"]] == [1, 2, 3, 4]
    assert g["botanical"][0]["model_label"] == "Imagen 4"
    assert g["botanical"][0]["url"] == "/begin/product-image/p/botanical-1.png"
    assert g["mechanism"] == []
    assert si.images_grouped_state(cx, "p") == "generating"   # only 4 tagged of 8

def test_grouped_legacy_fallback_no_label():
    cx = _cx()
    si.record_image(cx, "leg", "botanical", 1, "botanical-1.png")   # untagged legacy
    g = si.display_images_grouped(cx, "leg")
    assert len(g["botanical"]) == 1 and g["botanical"][0]["model_label"] is None

def test_grouped_state_ready_at_8():
    cx = _cx()
    n = 0
    for kind in ("botanical", "mechanism"):
        for v in (1, 2, 3, 4):
            n += 1
            si.record_image(cx, "full", kind, v, f"{kind}-{v}.png", prompt_variant_id=n, model_id="flux-1.1-pro")
    assert si.images_grouped_state(cx, "full") == "ready"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sales_pages_phase_a.py -q`
Expected: FAIL (`display_images_grouped` undefined).

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/sales_images.py`:

```python
def display_images_grouped(cx, slug, per_kind=4):
    from dashboard import sales_image_models as _mods
    from dashboard import sales_image_prompts as _sip
    init_tables(cx)
    labels = {m["id"]: m["label"] for m in _mods.active_models(cx)}
    out = {k: [] for k in _sip.IMAGE_KINDS}
    legacy = {k: [] for k in _sip.IMAGE_KINDS}
    for im in get_images(cx, slug):   # ordered by kind, variant
        k = im["kind"]
        if k not in out:
            continue
        entry = {"url": f"/begin/product-image/{slug}/{im['filename']}", "variant": im["variant"],
                 "prompt_variant_id": im["prompt_variant_id"], "model_id": im["model_id"],
                 "model_label": labels.get(im["model_id"])}
        if im["prompt_variant_id"] is not None:
            if len(out[k]) < per_kind:
                out[k].append(entry)
        else:
            legacy[k].append(entry)
    for k in _sip.IMAGE_KINDS:
        if not out[k] and legacy[k]:
            out[k] = [dict(e, model_label=None) for e in legacy[k][:per_kind]]
    return out

def images_grouped_state(cx, slug, target=8):
    return "ready" if tagged_count(cx, slug) >= target else "generating"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sales_pages_phase_a.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_images.py tests/test_sales_pages_phase_a.py
git commit -m "feat(sales-img): grouped display + state helper (Phase A task 7)"
```

---

### Task 8: Generate-missing orchestrator

**Files:**
- Modify: `dashboard/sales_images.py` (add `generate_missing`)
- Test: `tests/test_sales_pages_phase_a.py` (append)

**Interfaces:**
- Consumes: `build_generation_jobs` (Task 6), `record_image` (Task 5).
- Produces: `generate_missing(cx, slug, dest_dir, *, generate_fn) -> int` where `generate_fn(model_id, prompt_text) -> (bytes, used_model_id)`; writes `{kind}-{variant}.png` into `dest_dir`, records each with tags (using the **used** model id), returns count generated.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sales_pages_phase_a.py
def test_generate_missing_writes_files_and_tags(tmp_path):
    cx = _cx()
    def fake_gen(model_id, prompt):
        return (b"PNG", model_id)
    n = si.generate_missing(cx, "p", tmp_path, generate_fn=fake_gen)
    assert n == 8
    assert (tmp_path / "botanical-1.png").read_bytes() == b"PNG"
    assert si.tagged_count(cx, "p") == 8
    rows = {(r["kind"], r["variant"]): r for r in si.get_images(cx, "p")}
    assert rows[("botanical", 1)]["model_id"] is not None

def test_generate_missing_records_used_model_on_fallback(tmp_path):
    cx = _cx()
    def fallback_gen(model_id, prompt):
        return (b"PNG", "flux-1.1-pro")   # dispatcher fell back
    si.generate_missing(cx, "q", tmp_path, generate_fn=fallback_gen)
    assert all(r["model_id"] == "flux-1.1-pro" for r in si.get_images(cx, "q"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sales_pages_phase_a.py -q`
Expected: FAIL (`generate_missing` undefined).

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/sales_images.py`:

```python
def generate_missing(cx, slug, dest_dir, *, generate_fn):
    """Generate only the missing slots for `slug`. `generate_fn(model_id, prompt)` returns
    (image_bytes, used_model_id). Writes files into dest_dir and records each tagged image."""
    from pathlib import Path
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ok = 0
    for job in build_generation_jobs(cx, slug):
        data, used_model = generate_fn(job["model_id"], job["prompt_text"])
        fname = f"{job['kind']}-{job['variant']}.png"
        (dest / fname).write_bytes(data)
        record_image(cx, slug, job["kind"], job["variant"], fname,
                     prompt_variant_id=job["prompt_variant_id"], model_id=used_model)
        ok += 1
    return ok
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sales_pages_phase_a.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_images.py tests/test_sales_pages_phase_a.py
git commit -m "feat(sales-img): generate_missing orchestrator (Phase A task 8)"
```

---

### Task 9: Feature flag + admin pre-warm helper

**Files:**
- Modify: `app.py` (flag def near line 2542; backfill-slugs helper)
- Test: `tests/test_sales_pages_phase_a.py` (append — helper only)

**Interfaces:**
- Produces: `app._SALES_IMAGE_VARIATIONS_ENABLED` (bool); `sales_images.backfill_slugs(cx, arg, all_slugs) -> list[str]` (pure helper: `arg` is a slug or `"all"`; returns slugs to enqueue).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sales_pages_phase_a.py
def test_backfill_slugs_single_and_all():
    cx = _cx()
    assert si.backfill_slugs(cx, "abc", ["x", "y"]) == ["abc"]
    assert si.backfill_slugs(cx, "all", ["x", "y"]) == ["x", "y"]
    assert si.backfill_slugs(cx, "", ["x"]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sales_pages_phase_a.py::test_backfill_slugs_single_and_all -q`
Expected: FAIL (`backfill_slugs` undefined).

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/sales_images.py`:

```python
def backfill_slugs(cx, arg, all_slugs):
    """Resolve the admin pre-warm target: a single slug, or every product slug for 'all'."""
    arg = (arg or "").strip()
    if not arg:
        return []
    if arg == "all":
        return list(all_slugs)
    return [arg]
```

In `app.py`, add the flag right after line 2542 (`_SALES_IMAGE_TOURNAMENT_ENABLED = ...`):

```python
_SALES_IMAGE_VARIATIONS_ENABLED = os.environ.get("SALES_PAGES_IMAGE_VARIATIONS", "").strip().lower() in ("1", "true", "yes")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sales_pages_phase_a.py -q`
Expected: PASS. Confirm app imports: `python -c "import app"` → no error.

- [ ] **Step 5: Commit**

```bash
git add app.py dashboard/sales_images.py tests/test_sales_pages_phase_a.py
git commit -m "feat(sales-img): SALES_PAGES_IMAGE_VARIATIONS flag + backfill helper (Phase A task 9)"
```

---

### Task 10: Worker top-up generation

**Files:**
- Modify: `app.py` — `_drain_sales_image_queue` (~16456-16490)

**Interfaces:**
- Consumes: `_SALES_IMAGE_VARIATIONS_ENABLED` (Task 9), `sales_images.generate_missing` (Task 8), `sales_image_models.generate` (Task 4).

- [ ] **Step 1: Modify the worker loop body**

Inside `_drain_sales_image_queue`, replace the per-slug body (from `prod = dict(p)` through the `mark_done/mark_failed` line) with a branch on the flag. New code:

```python
        prod = dict(p)
        dest = _SALES_IMG_DIR / slug
        dest.mkdir(parents=True, exist_ok=True)
        if _SALES_IMAGE_VARIATIONS_ENABLED:
            from dashboard import sales_image_models as _mods
            try:
                with sqlite3.connect(LOG_DB) as cx:
                    n = _si.generate_missing(
                        cx, slug, dest,
                        generate_fn=lambda mid, prompt: _mods.generate(cx, mid, prompt))
                with sqlite3.connect(LOG_DB) as cx:
                    _si.mark_done(cx, slug)
            except Exception as e:
                print(f"[sales-img] {slug} variation gen failed: {e}", flush=True)
                with sqlite3.connect(LOG_DB) as cx:
                    _si.mark_failed(cx, slug)
            continue
        if not prod.get("ingredients"):
            prod["ingredients"] = (_product_card(p) or {}).get("ingredients", [])
        prompts = _sip.build_image_prompts(prod)
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
```

(The `dest = ...; dest.mkdir(...)` lines move above the branch; remove the old duplicate `dest`/`mkdir` lines further down.)

- [ ] **Step 2: Verify import + smoke**

Run: `python -c "import app"`
Expected: no error.

- [ ] **Step 3: Manual worker check (mocked)**

Run this scratch check (then delete it):

```bash
python - <<'PY'
import os, sqlite3, app
from unittest import mock
os.environ  # flag read at import; force-set the module global for this check:
app._SALES_IMAGE_VARIATIONS_ENABLED = True
with mock.patch("dashboard.replicate_client.generate_image", return_value=b"PNG"):
    # enqueue a known product slug, then drain once
    from dashboard import sales_images as si
    with sqlite3.connect(app.LOG_DB) as cx:
        si.enqueue(cx, "test-slug-doesnotexist")  # will mark_failed (no product) — proves no crash
    app._drain_sales_image_queue()
print("drain ran without raising")
PY
```
Expected: prints "drain ran without raising".

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(sales-img): worker tops up missing slots via dispatcher (Phase A task 10)"
```

---

### Task 11: Gen endpoint top-up gate + admin pre-warm route

**Files:**
- Modify: `app.py` — `begin_product_image_gen` (~3790-3801); add `/admin/sales-images/backfill`

**Interfaces:**
- Consumes: `_SALES_IMAGE_VARIATIONS_ENABLED`, `sales_images.needs_topup` (Task 5), `sales_images.backfill_slugs` (Task 9), `sales_images.enqueue`.

- [ ] **Step 1: Replace the gate in `begin_product_image_gen`**

Replace the `with sqlite3.connect(LOG_DB) as cx:` block body with:

```python
    with sqlite3.connect(LOG_DB) as cx:
        if _SALES_IMAGE_VARIATIONS_ENABLED:
            if not _si.needs_topup(cx, slug):
                return jsonify({"ok": True, "state": "done"})
            _si.enqueue(cx, slug)
            return jsonify({"ok": True, "state": "generating"})
        disp = _si.display_images(cx, slug)
        if any(disp.values()):
            return jsonify({"ok": True, "state": "done"})
        _si.enqueue(cx, slug)
        state = _si.queue_state(cx, slug)
    return jsonify({"ok": True, "state": state})
```

- [ ] **Step 2: Add the admin pre-warm route**

Add near the other `/admin/...` routes (follow the existing auth/guard pattern used by neighbors such as `/admin/sync-people-to-ghl`; match its `require_*`/secret check):

```python
@app.route("/admin/sales-images/backfill", methods=["POST"])
def admin_sales_images_backfill():
    # AUTH: mirror the guard used by the neighboring /admin/* routes in this file.
    if not _SALES_IMAGE_VARIATIONS_ENABLED:
        return jsonify({"ok": False, "error": "variations disabled"}), 400
    arg = (request.values.get("slug") or "").strip()
    from dashboard import sales_images as _si
    with sqlite3.connect(LOG_DB) as cx:
        targets = _si.backfill_slugs(cx, arg, _si.list_image_slugs(cx) if arg == "all"
                                     else [arg] if arg else [])
        enq = []
        for s in targets:
            if _si.needs_topup(cx, s):
                _si.enqueue(cx, s); enq.append(s)
    return jsonify({"ok": True, "enqueued": enq, "count": len(enq)})
```

Note: for `all`, `list_image_slugs` only returns products that already have at least one image. To pre-warm products with **zero** images, pass their slugs explicitly (one POST per slug) — acceptable for Phase A; a full product catalog sweep is a Phase C nicety.

- [ ] **Step 3: Verify import**

Run: `python -c "import app"`
Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(sales-img): top-up gate + admin pre-warm route (Phase A task 11)"
```

---

### Task 12: Data endpoint grouped payload

**Files:**
- Modify: `app.py` — Phase-3 image branch (~3258-3276) and pick-branch guard (~3277)

**Interfaces:**
- Consumes: `_SALES_IMAGE_VARIATIONS_ENABLED`, `sales_images.display_images_grouped` + `images_grouped_state` (Task 7).

- [ ] **Step 1: Replace the Phase-3 branch body**

Replace lines ~3258-3276 (the `if _SALES_AI_IMAGES_ENABLED:` block) with:

```python
    if _SALES_AI_IMAGES_ENABLED:
        import sqlite3 as _sq2
        from dashboard import sales_images as _si2
        try:
            _img_sec = next((s for s in sections if s["id"] == "images"), None)
            if _img_sec is not None:
                with _sq2.connect(LOG_DB) as _cx2:
                    if _SALES_IMAGE_VARIATIONS_ENABLED:
                        _grouped = _si2.display_images_grouped(_cx2, slug)
                        _state = _si2.images_grouped_state(_cx2, slug)
                        _img_sec["body"] = {"grouped": _grouped, "state": _state, "target": 8}
                    else:
                        _disp = _si2.display_images(_cx2, slug)
                        _qstate = _si2.queue_state(_cx2, slug)
                        _imgs = [{"kind": k, "url": f"/begin/product-image/{slug}/{fn}"}
                                 for k, fn in _disp.items() if fn]
                        if _imgs:
                            _img_sec["body"] = {"images": _imgs, "state": "ready"}
                        elif _qstate == "pending":
                            _img_sec["body"] = {"images": [], "state": "generating"}
                        else:
                            _img_sec["body"] = {"images": [], "state": "none"}
        except Exception as _e:
            print(f"[sales-img] page-data marker skipped: {_e}", flush=True)
```

- [ ] **Step 2: Guard the pick branch (avoid grid+pick collision)**

Change the pick-branch condition (~3277) from:

```python
    if _SALES_IMAGE_PICK_ENABLED:
```
to:
```python
    if _SALES_IMAGE_PICK_ENABLED and not _SALES_IMAGE_VARIATIONS_ENABLED:
```

(Variations grid and the pairwise pick UI are mutually exclusive; the grid wins when its flag is on.)

- [ ] **Step 3: Verify import**

Run: `python -c "import app"`
Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(sales-img): data endpoint emits grouped image payload (Phase A task 12)"
```

---

### Task 13: Template — per-type grid, model labels, generating placeholders

**Files:**
- Modify: `static/begin-product.html` — `renderImagesBody` (~700-768) + a small scoped CSS block

**Interfaces:**
- Consumes: the grouped payload `{grouped:{botanical:[{url,model_label,...}], mechanism:[...]}, state, target}` (Task 12).

- [ ] **Step 1: Replace `renderImagesBody` with grouped-aware rendering**

Replace the whole `renderImagesBody(body)` function (~700-768) with:

```javascript
    function renderImagesBody(body){
      var wrap = document.createElement('div');

      // Legacy (flag off): flat list of 1/kind.
      function renderFlat(images){
        wrap.className = ''; wrap.innerHTML = '';
        (images || []).forEach(function(img){
          var el = document.createElement('img');
          el.className = 'sp-product-img';
          el.src = img.url; el.alt = (img.kind || 'product') + ' image of the product';
          wrap.appendChild(el);
        });
      }

      var KIND_LABEL = { botanical: 'Botanical', mechanism: 'Mechanism' };

      // Grouped (flag on): two labeled grids of up to 4, plus generating placeholders.
      function renderGrouped(grouped, target){
        wrap.className = 'sp-img-groups'; wrap.innerHTML = '';
        ['botanical', 'mechanism'].forEach(function(kind){
          var tiles = (grouped && grouped[kind]) || [];
          var sec = document.createElement('div'); sec.className = 'sp-img-group';
          var h = document.createElement('div'); h.className = 'sp-img-group-title';
          h.textContent = KIND_LABEL[kind] || kind; sec.appendChild(h);
          var grid = document.createElement('div'); grid.className = 'sp-img-grid';
          tiles.forEach(function(t){
            var fig = document.createElement('figure'); fig.className = 'sp-img-tile';
            var el = document.createElement('img'); el.className = 'sp-product-img';
            el.src = t.url; el.alt = kind + ' image of the product';
            fig.appendChild(el);
            if (t.model_label){
              var cap = document.createElement('figcaption'); cap.className = 'sp-img-cap';
              cap.textContent = 'made with ' + t.model_label; fig.appendChild(cap);
            }
            grid.appendChild(fig);
          });
          for (var i = tiles.length; i < 4; i++){       // generating placeholders
            var ph = document.createElement('div'); ph.className = 'sp-img-tile sp-img-ph';
            ph.textContent = '…'; grid.appendChild(ph);
          }
          sec.appendChild(grid); wrap.appendChild(sec);
        });
      }

      if (!body || !body.state){
        var ph0 = document.createElement('div'); ph0.id = 'sp-images';
        wrap.appendChild(ph0); return wrap;
      }
      if (body.pick){ renderPick(wrap, body.pick); return wrap; }

      var grouped = body.grouped;          // present only when variations flag on
      var _enqueueGuard = false;

      function startPoll(){
        var polls = 0;
        var timer = setInterval(function(){
          polls++;
          if (polls > 30){ clearInterval(timer); return; }   // 30 x 4s = 2min for 8 imgs
          fetch(BASE + '/begin/product-page-data/' + slug, {credentials: 'same-origin'})
            .then(function(r){ return r.json(); })
            .then(function(data){
              var secs = data.sections || [];
              for (var i = 0; i < secs.length; i++){
                if (secs[i].id === 'images' && secs[i].body){
                  var b = secs[i].body;
                  if (b.grouped){ renderGrouped(b.grouped, b.target); }
                  if (b.state === 'ready'){ clearInterval(timer); if (!b.grouped){ renderFlat(b.images); } return; }
                  return;
                }
              }
            }).catch(function(){});
        }, 4000);
      }

      function triggerGen(){
        if (_enqueueGuard) return; _enqueueGuard = true;
        fetch(BASE + '/begin/product-image-gen/' + slug, {method: 'POST', credentials: 'same-origin'}).catch(function(){});
      }

      if (grouped){                                  // variations flag ON
        renderGrouped(grouped, body.target);
        if (body.state !== 'ready'){ triggerGen(); startPoll(); }
      } else if (body.state === 'ready'){
        renderFlat(body.images);
      } else if (body.state === 'generating'){
        wrap.className = 'loading'; wrap.textContent = 'Generating product imagery…'; startPoll();
      } else if (body.state === 'none'){
        wrap.className = 'loading'; wrap.textContent = 'Generating product imagery…';
        triggerGen(); startPoll();
      }
      return wrap;
    }
```

- [ ] **Step 2: Add scoped CSS**

Add to the page's `<style>` block (near the existing `.sp-product-img` rule):

```css
.sp-img-group-title{font-weight:600;margin:14px 0 6px;font-size:0.95rem;letter-spacing:.02em}
.sp-img-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
@media(min-width:768px){.sp-img-grid{grid-template-columns:repeat(4,1fr)}}
.sp-img-tile{margin:0}
.sp-img-tile .sp-product-img{width:100%;height:auto;display:block;border-radius:8px}
.sp-img-cap{font-size:.72rem;opacity:.6;margin-top:4px;text-align:center}
.sp-img-ph{display:flex;align-items:center;justify-content:center;min-height:120px;
  border-radius:8px;background:rgba(127,127,127,.12);color:rgba(127,127,127,.6);font-size:1.4rem}
```

- [ ] **Step 3: Manual verification**

Run the app locally with the flag on and a known product slug, then confirm the grouped JSON and the rendered grid:

```bash
SALES_PAGES_AI_IMAGES=1 SALES_PAGES_IMAGE_VARIATIONS=1 python app.py &   # or the project's run command
# replace <slug> with a real product slug:
curl -s localhost:5000/begin/product-page-data/<slug> | python -m json.tool | grep -A3 '"grouped"'
```
Expected: the `images` section body contains a `grouped` object with `botanical`/`mechanism` arrays and a `state`. Load `/begin/product/<slug>` in a browser: two labeled grids (4-across desktop / 2×2 mobile), captions "made with …", placeholders fill in as generation completes.

- [ ] **Step 4: Commit**

```bash
git add static/begin-product.html
git commit -m "feat(sales-img): per-type image grid + model labels + placeholders (Phase A task 13)"
```

---

### Task 14: Full-suite regression + flag-off safety

**Files:** none (verification)

- [ ] **Step 1: Run the new + existing image/sales tests**

Run: `python -m pytest tests/test_sales_pages_phase_a.py tests/test_sales_pages_phase3.py tests/test_sales_pages_phase4.py tests/test_sales_pages_phase4b.py -q`
Expected: all PASS (no regression in existing phases).

- [ ] **Step 2: Confirm flag-off parity**

With `SALES_PAGES_IMAGE_VARIATIONS` unset, confirm `display_images`, the data-endpoint `images` body, the gen-endpoint `any()` gate, and the worker all behave exactly as before (covered by the existing phase3/4/4b tests passing in Step 1).

- [ ] **Step 3: Commit (if any fixups)**

```bash
git add -A && git commit -m "test(sales-img): Phase A regression pass" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:** registry (T1), models (T2), multi-model client (T3), dispatcher+fallback (T4), schema tags (T5), balanced assignment (T6), grouped display+state (T7), generate_missing (T8), flag+backfill helper (T9), worker top-up (T10), gate+admin route (T11), data endpoint (T12), template grid+labels+placeholders+poll-window (T13), regression+flag-off (T14). Spec sections 1–8, data flow, testing, and the poll-window risk all map to tasks. ✔

**Placeholder scan:** all code blocks are complete; the only deferred items are (a) the admin route's exact auth guard — flagged to mirror neighboring `/admin/*` routes, since that pattern is file-specific — and (b) live Replicate pricing/refs confirmation (a global constraint, not a code gap). No "TODO/handle edge cases" left in code.

**Type consistency:** `generate(cx, model_id, prompt)->(bytes, used_model_id)` matches `generate_fn` in `generate_missing`; `build_generation_jobs` job keys (`kind, variant, prompt_variant_id, model_id, prompt_text`) match their consumers; `record_image(..., prompt_variant_id, model_id)` matches `get_images` keys and `display_images_grouped`/`tagged_count` reads; `NO_TEXT` exposed in T5 and used in T6.
