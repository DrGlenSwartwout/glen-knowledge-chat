# Phase 2 — Sales Page AI Narrative Copy Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate rich, grounded narrative copy (intro / overview / research framing) per product on the in-funnel sales page, streamed token-by-token into each section on first open, cached in a `sales_pages` draft store — behind a new `SALES_PAGES_AI_COPY` flag.

**Architecture:** A `sales_pages` SQLite table holds per-section draft copy. A new `GET /begin/product-page-gen/<slug>/<section>` SSE endpoint generates one section via streamed Claude (same in-request streaming pattern as the live chat) and caches it; cached sections stream back instantly. The page-data endpoint tags narrative sections `ai: cached|pending`; the frontend opens an EventSource per pending section on first open and streams the copy in live. Factual panels are untouched.

**Tech Stack:** Python 3.11 / Flask, SQLite (`chat_log.db` via `LOG_DB`), Anthropic SDK (`_cl`, `claude-haiku-4-5-20251001`), Server-Sent Events (`sse()` + `stream_with_context`), vanilla static JS (EventSource).

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-18-phase2-sales-page-copy-gen-design.md` (authoritative).
- **Flag:** `SALES_PAGES_AI_COPY`, default OFF; truthy ∈ {1,true,yes} case-insensitive (use `.strip().lower()` to match existing flag idiom). With it off, every endpoint/path behaves byte-identically to live Phase 1.
- **Narrative sections only:** exactly `intro`, `description`, `research`. Never generate ingredient panel, comparison, video, images, or CTA.
- **Compliance (every prompt):** structure/function language only ("supports / promotes / helps maintain"); NO claim to diagnose, treat, cure, or prevent disease; no invented studies; Glen's voice — no fluff, no AI-pleasantry filler, no clichés.
- **Model:** `claude-haiku-4-5-20251001` (matches existing calls). Streaming via `_cl.messages.stream(...)` + `for tok in stream.text_stream`.
- **DB:** `chat_log.db` via the module-level `LOG_DB` path; connect with `sqlite3.connect(LOG_DB)`. Data layer takes an open connection so tests use a tmp DB.
- **Test invocation:** `doppler run -p remedy-match -c prd -- env DATA_DIR="$(mktemp -d)" ~/.venvs/deploy-chat311/bin/python -m pytest <file> -v`. Mock live Supabase; `pytest.importorskip` playwright. Pre-existing ignorable failures: `test_pf_playwright_fetch`, `test_bos_routes::test_home_page_served`.
- **No web-worker risk:** copy streaming runs in-request (like the live chat endpoint) — acceptable. No Replicate / slow image work here.

---

## File Structure

- **Create** `dashboard/sales_pages.py` — data layer: `init_table`, `get_page`, `get_section`, `upsert_section`. One responsibility: persistence of per-section draft copy.
- **Create** `dashboard/sales_copy.py` — generation inputs: `NARRATIVE_SECTIONS`, `COMPLIANCE`, `SECTION_BRIEFS`, `build_section_prompt(section, product) -> (system, user)`. Pure, no I/O — unit-testable.
- **Modify** `app.py` — add `_SALES_AI_COPY_ENABLED` flag; add `GET /begin/product-page-gen/<slug>/<section>` SSE endpoint; tag narrative sections in `begin_product_page_data` with `ai` markers.
- **Modify** `static/begin-product.html` — EventSource-stream pending narrative sections on first open; caveat banner.
- **Test** `tests/test_sales_pages_phase2.py` — data layer, prompt builder, page-data markers, gen endpoint (mocked `_cl`).

---

## Task 1: `sales_pages` data layer

**Files:**
- Create: `dashboard/sales_pages.py`
- Test: `tests/test_sales_pages_phase2.py`

**Interfaces:**
- Produces: `init_table(cx)`; `get_page(cx, slug) -> dict|None` (`{"product_slug","state","content":dict,"model","generated_at"}`); `get_section(cx, slug, section) -> str|None`; `upsert_section(cx, slug, section, text, model="") -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sales_pages_phase2.py
import sqlite3
from dashboard import sales_pages as sp

def _cx():
    return sqlite3.connect(":memory:")

def test_upsert_then_get_section_roundtrip():
    cx = _cx()
    assert sp.get_section(cx, "longevity", "intro") is None
    sp.upsert_section(cx, "longevity", "intro", "Hello world.", model="m1")
    assert sp.get_section(cx, "longevity", "intro") == "Hello world."

def test_upsert_accretes_sections_in_one_row():
    cx = _cx()
    sp.upsert_section(cx, "energy", "intro", "A.")
    sp.upsert_section(cx, "energy", "description", "B.")
    page = sp.get_page(cx, "energy")
    assert page["content"] == {"intro": "A.", "description": "B."}
    assert page["state"] == "draft"

def test_get_page_missing_returns_none():
    assert sp.get_page(_cx(), "nope") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_sales_pages_phase2.py -k "roundtrip or accretes or missing" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.sales_pages'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/sales_pages.py
import json, datetime

def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS sales_pages ("
        "product_slug TEXT PRIMARY KEY, state TEXT DEFAULT 'draft', "
        "content_json TEXT DEFAULT '{}', model TEXT DEFAULT '', "
        "generated_at TEXT DEFAULT '', created_at TEXT DEFAULT '', updated_at TEXT DEFAULT '')")
    cx.commit()

def get_page(cx, slug):
    init_table(cx)
    row = cx.execute(
        "SELECT product_slug, state, content_json, model, generated_at "
        "FROM sales_pages WHERE product_slug=?", (slug,)).fetchone()
    if not row:
        return None
    return {"product_slug": row[0], "state": row[1],
            "content": json.loads(row[2] or "{}"), "model": row[3], "generated_at": row[4]}

def get_section(cx, slug, section):
    page = get_page(cx, slug)
    if not page:
        return None
    return page["content"].get(section) or None

def upsert_section(cx, slug, section, text, model=""):
    init_table(cx)
    now = _now()
    row = cx.execute("SELECT content_json FROM sales_pages WHERE product_slug=?", (slug,)).fetchone()
    content = json.loads(row[0]) if row and row[0] else {}
    content[section] = text
    cx.execute(
        "INSERT INTO sales_pages (product_slug, state, content_json, model, generated_at, created_at, updated_at) "
        "VALUES (?, 'draft', ?, ?, ?, ?, ?) "
        "ON CONFLICT(product_slug) DO UPDATE SET content_json=excluded.content_json, "
        "model=excluded.model, generated_at=excluded.generated_at, updated_at=excluded.updated_at",
        (slug, json.dumps(content), model, now, now, now))
    cx.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... -m pytest tests/test_sales_pages_phase2.py -k "roundtrip or accretes or missing" -v`
Expected: PASS (3)

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_pages.py tests/test_sales_pages_phase2.py
git commit -m "feat: sales_pages draft-copy data layer"
```

---

## Task 2: section prompt builder (`dashboard/sales_copy.py`)

**Files:**
- Create: `dashboard/sales_copy.py`
- Test: `tests/test_sales_pages_phase2.py`

**Interfaces:**
- Produces: `NARRATIVE_SECTIONS = ("intro","description","research")`; `COMPLIANCE` (str); `SECTION_BRIEFS` (dict); `build_section_prompt(section, product) -> (system_str, user_str)`. `product` is a dict with `name` and `ingredients` (list of `{name,dose}` or str).

- [ ] **Step 1: Write the failing test**

```python
from dashboard import sales_copy as sc

def test_prompt_includes_compliance_and_no_disease_claim():
    system, user = sc.build_section_prompt("intro", {"name": "Longevity", "ingredients": []})
    assert "treat" in system.lower() and "cure" in system.lower() and "prevent" in system.lower()
    assert "supports" in system.lower() or "structure/function" in system.lower()

def test_prompt_grounds_in_product_name_and_ingredients():
    prod = {"name": "Longevity", "ingredients": [{"name": "Resveratrol", "dose": "200 mg"}, "Quercetin"]}
    system, user = sc.build_section_prompt("research", prod)
    assert "Longevity" in user
    assert "Resveratrol" in user and "200 mg" in user and "Quercetin" in user

def test_narrative_sections_are_exactly_three():
    assert sc.NARRATIVE_SECTIONS == ("intro", "description", "research")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_sales_pages_phase2.py -k "prompt or narrative_sections" -v`
Expected: FAIL — `No module named 'dashboard.sales_copy'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/sales_copy.py
NARRATIVE_SECTIONS = ("intro", "description", "research")

COMPLIANCE = (
    "Use structure/function language only (supports, promotes, helps maintain). "
    "Do NOT claim to diagnose, treat, cure, or prevent any disease. Make no medical "
    "claims and cite no invented studies. This is educational and not a substitute for "
    "medical advice."
)

SECTION_BRIEFS = {
    "intro": ("Write ONE warm, concrete paragraph (about 2-4 sentences): what this formula "
              "does for the person and why it matters, grounded in its ingredients."),
    "description": ("Write a fuller plain-language overview in 2-3 short paragraphs: what the "
                    "formula is, what it's built from, and who it's for."),
    "research": ("Explain how it works in lay language, 1-2 short paragraphs, grounded in the "
                 "mechanisms of the listed ingredients."),
}

def _ingredient_lines(product):
    out = []
    for ing in (product.get("ingredients") or []):
        if isinstance(ing, dict):
            out.append((f"- {ing.get('name','')} {ing.get('dose','')}").rstrip())
        elif isinstance(ing, str) and ing.strip():
            out.append(f"- {ing.strip()}")
    return "\n".join(out)

def build_section_prompt(section, product):
    name = product.get("name", "")
    ings = _ingredient_lines(product)
    brief = SECTION_BRIEFS[section]
    system = ("You are writing sales-page copy for Dr. Glen Swartwout's nutritional formulas. "
              "Voice: warm, clinically grounded, and specific — no fluff, no AI-pleasantry filler, "
              "no clichés. " + COMPLIANCE)
    user = (f"Product: {name}\n\nIngredient stack:\n{ings or '(not specified)'}\n\n"
            f"Task: {brief}\n\nReturn only the copy itself — no headings, labels, or preamble.")
    return system, user
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... -m pytest tests/test_sales_pages_phase2.py -k "prompt or narrative_sections" -v`
Expected: PASS (3)

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_copy.py tests/test_sales_pages_phase2.py
git commit -m "feat: grounded, compliance-constrained section prompt builder"
```

---

## Task 3: `SALES_PAGES_AI_COPY` flag + page-data `ai` markers

**Files:**
- Modify: `app.py` (flag read near `_SALES_PAGES_ENABLED` ~L2296; `begin_product_page_data` ~L2852-2876)
- Test: `tests/test_sales_pages_phase2.py`

**Interfaces:**
- Consumes: `dashboard.sales_pages.get_section`, `_get_product`, `LOG_DB`.
- Produces: when `SALES_PAGES_AI_COPY` is ON, each narrative section in the page-data response carries `"ai": "cached"` (body replaced with the stored draft) or `"ai": "pending"` (fallback body kept). For the `research` section, the draft replaces `body["how_it_works"]`. Flag OFF → no `ai` key anywhere (Phase-1 identical).

- [ ] **Step 1: Write the failing test**

```python
import importlib, os, sqlite3, pytest

def _reload_app(monkeypatch, tmp_path, ai="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SALES_PAGES_ENABLED", "true")
    monkeypatch.setenv("SALES_PAGES_AI_COPY", ai)
    import app as appmod
    importlib.reload(appmod)
    return appmod

def test_page_data_marks_narrative_pending_when_no_draft(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    data = appmod.app.test_client().get(f"/begin/product-page-data/{slug}").get_json()
    nar = {s["id"]: s for s in data["sections"] if s["id"] in ("intro","description","research")}
    assert all(s.get("ai") == "pending" for s in nar.values())

def test_page_data_serves_cached_draft(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    from dashboard import sales_pages as sp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sp.upsert_section(cx, slug, "intro", "Cached intro copy.")
    data = appmod.app.test_client().get(f"/begin/product-page-data/{slug}").get_json()
    intro = next(s for s in data["sections"] if s["id"] == "intro")
    assert intro["ai"] == "cached" and intro["body"] == "Cached intro copy."

def test_page_data_no_ai_field_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, ai="false")
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    data = appmod.app.test_client().get(f"/begin/product-page-data/{slug}").get_json()
    assert all("ai" not in s for s in data["sections"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_sales_pages_phase2.py -k page_data -v`
Expected: FAIL — narrative sections have no `ai` key.

- [ ] **Step 3: Write minimal implementation**

```python
# app.py — near _SALES_PAGES_ENABLED (~L2296)
_SALES_AI_COPY_ENABLED = os.environ.get("SALES_PAGES_AI_COPY", "").strip().lower() in ("1", "true", "yes")
```

```python
# app.py — in begin_product_page_data, AFTER the `sections = [...]` list is built and
# BEFORE the `return jsonify(...)`:
    if _SALES_AI_COPY_ENABLED:
        import sqlite3 as _sq
        from dashboard import sales_pages as _sp
        try:
            with _sq.connect(LOG_DB) as _cx:
                for _s in sections:
                    if _s["id"] not in ("intro", "description", "research"):
                        continue
                    _draft = _sp.get_section(_cx, slug, _s["id"])
                    if _draft:
                        _s["ai"] = "cached"
                        if _s["id"] == "research" and isinstance(_s["body"], dict):
                            _s["body"]["how_it_works"] = _draft
                        else:
                            _s["body"] = _draft
                    else:
                        _s["ai"] = "pending"
        except Exception as _e:
            print(f"[sales-ai] page-data marker skipped: {_e}", flush=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... -m pytest tests/test_sales_pages_phase2.py -k page_data -v`
Expected: PASS (3)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_sales_pages_phase2.py
git commit -m "feat: SALES_PAGES_AI_COPY flag + page-data ai markers"
```

---

## Task 4: generation SSE endpoint

**Files:**
- Modify: `app.py` (add route after `begin_product_page_data`)
- Test: `tests/test_sales_pages_phase2.py`

**Interfaces:**
- Consumes: `_SALES_AI_COPY_ENABLED`, `_get_product`, `_product_card`, `_cl`, `sse`, `LOG_DB`, `dashboard.sales_pages`, `dashboard.sales_copy`.
- Produces: route `GET /begin/product-page-gen/<slug>/<section>` → `text/event-stream`. Cached → one `{"token": <draft>}` + `{"done": true, "cached": true}`. Uncached → streamed `{"token": …}` frames, persists full text via `upsert_section`, `{"done": true}`. On exception → `{"error": true}`. Flag off OR bad section OR unknown slug → 404.

- [ ] **Step 1: Write the failing test**

```python
class _FakeStream:
    def __init__(self, toks): self._toks = toks
    def __enter__(self): return self
    def __exit__(self, *a): return False
    @property
    def text_stream(self):
        for t in self._toks: yield t

class _FakeMessages:
    def __init__(self, toks, boom=False): self._toks=toks; self.boom=boom; self.calls=0
    def stream(self, **kw):
        self.calls += 1
        if self.boom: raise RuntimeError("claude down")
        return _FakeStream(self._toks)

class _FakeCl:
    def __init__(self, toks, boom=False): self.messages=_FakeMessages(toks, boom)

def _frames(resp):
    return resp.get_data(as_text=True)

def test_gen_streams_and_persists(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    monkeypatch.setattr(appmod, "_product_card", lambda p: {"ingredients": [{"name": "Resveratrol", "dose": "200 mg"}]})
    monkeypatch.setattr(appmod, "_cl", _FakeCl(["Live ", "intro ", "copy."]))
    body = _frames(appmod.app.test_client().get(f"/begin/product-page-gen/{slug}/intro"))
    assert "Live " in body and '"done": true' in body
    import sqlite3
    from dashboard import sales_pages as sp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert sp.get_section(cx, slug, "intro") == "Live intro copy."

def test_gen_returns_cached_without_calling_claude(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    import sqlite3
    from dashboard import sales_pages as sp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sp.upsert_section(cx, slug, "intro", "Cached copy.")
    fake = _FakeCl([], boom=True)
    monkeypatch.setattr(appmod, "_cl", fake)
    body = _frames(appmod.app.test_client().get(f"/begin/product-page-gen/{slug}/intro"))
    assert "Cached copy." in body and fake.messages.calls == 0

def test_gen_404_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, ai="false")
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    assert appmod.app.test_client().get(f"/begin/product-page-gen/{slug}/intro").status_code == 404

def test_gen_error_frame_on_claude_failure(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    monkeypatch.setattr(appmod, "_product_card", lambda p: {"ingredients": []})
    monkeypatch.setattr(appmod, "_cl", _FakeCl([], boom=True))
    body = _frames(appmod.app.test_client().get(f"/begin/product-page-gen/{slug}/intro"))
    assert '"error": true' in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... -m pytest tests/test_sales_pages_phase2.py -k gen -v`
Expected: FAIL — route not defined (404 for the streaming cases).

- [ ] **Step 3: Write minimal implementation**

```python
# app.py — after begin_product_page_data
@app.route("/begin/product-page-gen/<slug>/<section>")
def begin_product_page_gen(slug, section):
    from dashboard import sales_copy as _sc
    if not _SALES_AI_COPY_ENABLED or section not in _sc.NARRATIVE_SECTIONS:
        return ("", 404)
    p = _get_product(slug)
    if not p:
        return ("", 404)

    def generate():
        import sqlite3 as _sq
        from dashboard import sales_pages as _sp
        try:
            with _sq.connect(LOG_DB) as cx:
                cached = _sp.get_section(cx, slug, section)
            if cached:
                yield sse({"token": cached}); yield sse({"done": True, "cached": True}); return
            prod = dict(p)
            if not prod.get("ingredients"):
                prod["ingredients"] = (_product_card(p) or {}).get("ingredients", [])
            system, user = _sc.build_section_prompt(section, prod)
            acc = []
            with _cl.messages.stream(model="claude-haiku-4-5-20251001", max_tokens=600,
                                     system=system,
                                     messages=[{"role": "user", "content": user}]) as stream:
                for tok in stream.text_stream:
                    acc.append(tok); yield sse({"token": tok})
            text = "".join(acc).strip()
            if text:
                try:
                    with _sq.connect(LOG_DB) as cx:
                        _sp.upsert_section(cx, slug, section, text, model="claude-haiku-4-5-20251001")
                except Exception as e:
                    print(f"[sales-gen] cache write failed: {e}", flush=True)
            yield sse({"done": True})
        except Exception as e:
            print(f"[sales-gen] {e}", flush=True)
            yield sse({"error": True})

    resp = Response(stream_with_context(generate()), content_type="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... -m pytest tests/test_sales_pages_phase2.py -k gen -v`
Expected: PASS (4)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_sales_pages_phase2.py
git commit -m "feat: per-section sales-copy generation SSE endpoint (cache-or-stream)"
```

---

## Task 5: frontend — stream pending sections on open + caveat banner

**Files:**
- Modify: `static/begin-product.html`
- Manual verification (repo does not unit-test static JS).

**Interfaces:**
- Consumes: page-data sections with optional `ai: "cached"|"pending"`; the SSE endpoint `/begin/product-page-gen/<slug>/<section>` (frames: `{token}`, `{done}`, `{error}`).
- Produces: pending narrative sections stream their copy in live on first open (intro on load); cached → render stored draft; caveat banner when any section is `ai`-tagged.

- [ ] **Step 1: Implement**

In `renderProduct`, capture the slug. When building each narrative section's body (`intro`, `description`, `research`):
1. If `sec.ai === 'cached'` → render `sec.body` (for `research`, `sec.body.how_it_works`) as today.
2. If `sec.ai === 'pending'` → render the fallback body text now, but mark the section's content element so that **on first open** (intro: immediately on load) it calls `streamSection(sec.id, targetEl)`:
   ```js
   function streamSection(section, el){
     if (el.dataset.streamed) return; el.dataset.streamed = '1';
     var es = new EventSource(BASE + '/begin/product-page-gen/' + encodeURIComponent(slug) + '/' + section);
     var first = true, buf = '';
     es.onmessage = function(e){
       var d = JSON.parse(e.data);
       if (d.error){ es.close(); return; }            // keep fallback text already shown
       if (d.token){ if (first){ el.textContent=''; first=false; } buf += d.token; el.textContent = buf; }
       if (d.done){ es.close(); }
     };
     es.onerror = function(){ es.close(); };           // keep whatever is shown
   }
   ```
   Wire `streamSection` into the existing lazy-open handler (same place section bodies are first rendered). For `research`, the target element is the how-it-works paragraph, not the whole section.
3. If `sec.ai` is undefined (flag off) → existing Phase-1 rendering untouched.
4. **Caveat banner:** if any section has an `ai` field, render a banner element at the top of the page: "Generated from Dr. Glen's knowledge base — pending his personal review for final approval." (muted, no emoji). One banner per page.

- [ ] **Step 2: Manual verification**

```
# locally with both flags on (reuse the Phase-1 preview technique, but point at the real app):
doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" SALES_PAGES_ENABLED=true SALES_PAGES_AI_COPY=true ~/.venvs/deploy-chat311/bin/python -m app
# open /begin/product/<slug>: intro streams in live on load; open Overview/Research → each streams once;
# reload → now cached, renders instantly; caveat banner shows. Flag AI off → Phase-1 page, no banner.
```

- [ ] **Step 3: Commit**

```bash
git add static/begin-product.html
git commit -m "feat: stream pending narrative sections via EventSource + AI caveat banner"
```

---

## Task 6: integration sanity + flag default

**Files:**
- Modify: `tests/test_sales_pages_phase2.py`

- [ ] **Step 1: Write the test**

```python
def test_full_phase2_file_and_flag_default(monkeypatch, tmp_path):
    # flag unset → disabled
    monkeypatch.delenv("SALES_PAGES_AI_COPY", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path)); monkeypatch.setenv("SALES_PAGES_ENABLED", "true")
    import importlib, app as appmod; importlib.reload(appmod)
    assert appmod._SALES_AI_COPY_ENABLED is False
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    assert appmod.app.test_client().get(f"/begin/product-page-gen/{slug}/intro").status_code == 404
```

- [ ] **Step 2: Run the full Phase-2 file**

Run: `... -m pytest tests/test_sales_pages_phase2.py -v`
Expected: PASS (all). Then run `tests/test_sales_pages_phase1.py` too — must stay green (Phase 1 untouched when flag off).

- [ ] **Step 3: Confirm both flags default OFF in Render** — do NOT set `SALES_PAGES_AI_COPY` yet. Ship dark.

- [ ] **Step 4: Commit**

```bash
git add tests/test_sales_pages_phase2.py
git commit -m "test: phase-2 integration + flag-default-off"
```

---

## Verification (end to end)

1. `... -m pytest tests/test_sales_pages_phase2.py tests/test_sales_pages_phase1.py -v` → all pass; full suite no new failures (only the two known pre-existing).
2. **Flag AI off (default):** product pages render exactly as live Phase 1 — no `ai` markers, no gen endpoint (404), no banner.
3. **Flag AI on locally:** intro streams live on load; Overview/Research stream on first open; reload serves cached copy instantly; caveat banner present; generated copy is structure/function only (no disease claims), in Glen's voice, grounded in the product's ingredients; Claude failure falls back to the Phase-1 copy without breaking the page.
