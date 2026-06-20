# Ingredient Page — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A paid-gated, built-on-request per-ingredient page (`/begin/ingredient/<slug>`) mirroring the formulation page, with a heavy research section, two 1-10 gauges, traditional-use detail, related-forms (superior/inferior), and links both ways; AI proposes, Glen verifies in a console, requesters are emailed when ready.

**Architecture:** New sibling modules to the sales-page subsystem: `dashboard/ingredients.py` (resolver), `dashboard/ingredient_pages.py` (store + request/notify, mirrors `sales_pages.py` + the Phase-5b viewer pattern), `dashboard/ingredient_copy.py` (mirrors `sales_copy.py`), `dashboard/ingredient_page_actions.py` (mirrors `sales_pages_actions.py`). New routes + `static/begin-ingredient.html` + `static/console-ingredient-pages.html`. A paid gate (`_active_membership_for_email`) governs access; the public page only ever emits APPROVED content.

**Tech Stack:** Flask, SQLite, Pinecone `ingredients` namespace, haiku `claude-haiku-4-5-20251001`, the dispatch spine, pytest.

## Global Constraints

- No emoji, no em dashes. Live, no feature flag (the paid gate + `INGREDIENT_PAGES_PAID_ONLY` govern access). `main` auto-deploys.
- **The public page-data NEVER emits draft content** - it returns `state:"locked"` (not paid), `state:"preparing"` (paid, no approved row), or `state:"approved"` (paid + approved) with the content only in the approved case.
- **Paid gate fail-safe:** `_active_membership_for_email` error -> treat as not-paid (locked), never bypass. `_ingredient_paid_ok(email)` = `not INGREDIENT_PAGES_PAID_ONLY or bool(_active_membership_for_email(email))`.
- **AI proposals are provisional; Glen verifies.** `propose_curation`'s prompt instructs the model to OMIT any classical formula it is not confident is real (never invent). Same compliance system prompt as `sales_copy` (structure/function, no disease claims, no em dashes).
- `notify_on_approve` is best-effort + at-most-once (`emailed_at`); never fails the approve.
- Store getters use a per-cursor Row factory (no connection-state mutation). Console actions on the dispatch spine, RBAC `(OWNER, OPS)`.
- XSS-safe front-end: all dynamic text via `textContent`; links set `.href` to server `/begin/...` or study `url` with `target=_blank rel=noopener`.
- Test harness: `importlib` app load (skip if not importable); tmp `LOG_DB`; mock the Anthropic client + Pinecone + `_active_membership_for_email` + the send fn; mock GHL on any free-tier transition. Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest <target> -v`. Spec: `docs/superpowers/specs/2026-06-20-ingredient-page-design.md`.

## Critical files / anchors
- Mirror targets: `dashboard/sales_pages.py` (store), `dashboard/sales_copy.py` (`NARRATIVE_SECTIONS`/`build_section_prompt`/`_ingredient_lines`/compliance system), `dashboard/sales_pages_actions.py` (`configure`/`register`/`_exec_*`), `dashboard/sales_page_viewers.py` (`record_viewer`/`notify_on_approve`), `static/console-sales-pages.html`, `static/begin-product.html` (`renderIngredientsBody` ~341).
- `_active_membership_for_email` (app.py:5688, returns a dict or None); `_product_card`/`product_content._research_sources` (Pinecone ingredients); `dashboard/ingredient_content.get(name)`; `data/fmp-ingredient-content.json`; `data/products.json`; `_slugify_product` (app.py:491); product page-data `ingredients` section (app.py:3171/3185).

---

### Task 1: `dashboard/ingredients.py` resolver

**Files:** Create `dashboard/ingredients.py`; Create `tests/test_ingredients_resolver.py`.

**Interfaces produced:** `slugify(name)->str`; `resolve(slug)->{slug,name,fmp}|None`; `formulations_with(name)->[{slug,name}]`; `research_studies(name, k=12)->[{study_title,publication,year,url,text}]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ingredients_resolver.py
import sys
from pathlib import Path
import pytest


def _mod():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dashboard import ingredients
        return ingredients
    except Exception as e:
        pytest.skip(f"module not importable: {e}")


def test_slugify():
    m = _mod()
    assert m.slugify("HMC Hesperidin") == "hmc-hesperidin"
    assert m.slugify("Acai 20:1 Freeze-Dried") == "acai-20-1-freeze-dried"


def test_resolve_known_and_unknown():
    m = _mod()
    # a name that exists in fmp-ingredient-content.json - resolve its slug back
    name = next(iter(m._name_index().values()))  # any known canonical name
    slug = m.slugify(name)
    r = m.resolve(slug)
    assert r is not None and r["slug"] == slug and r["name"]
    assert isinstance(r.get("fmp"), dict)
    assert m.resolve("totally-bogus-ingredient-xyz") is None


def test_formulations_with_returns_list():
    m = _mod()
    # any ingredient that appears in products.json; assert a list of {slug,name}
    out = m.formulations_with(next(iter(m._name_index().values())))
    assert isinstance(out, list)
    for f in out:
        assert "slug" in f and "name" in f
```

- [ ] **Step 2: Run to verify fail.** `... -m pytest tests/test_ingredients_resolver.py -v` -> FAIL.

- [ ] **Step 3: Implement `dashboard/ingredients.py`**

```python
"""Ingredient resolver for the ingredient page. Maps a URL slug to an ingredient
name + its FMP record, the formulations that use it, and its research studies."""
import json
import re
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_FMP = _ROOT / "data" / "fmp-ingredient-content.json"
_PRODUCTS = _ROOT / "data" / "products.json"


def slugify(name):
    s = (name or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-")[:40]


@lru_cache(maxsize=1)
def _fmp_records():
    try:
        return json.loads(_FMP.read_text())
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _name_index():
    """{slug: canonical_name} over all known ingredient names (FMP + products)."""
    idx = {}
    for rec in _fmp_records().values():
        nm = (rec.get("name") or "").strip()
        if nm:
            idx.setdefault(slugify(nm), nm)
    try:
        prods = json.loads(_PRODUCTS.read_text()).get("products", {})
    except Exception:
        prods = {}
    for p in prods.values():
        for ing in (p.get("ingredients") or []):
            nm = (ing.get("name") if isinstance(ing, dict) else ing) or ""
            nm = nm.strip()
            if nm:
                idx.setdefault(slugify(nm), nm)
    return idx


def _fmp_for(name):
    try:
        from dashboard import ingredient_content
        return ingredient_content.get(name) or {}
    except Exception:
        return {}


def resolve(slug):
    name = _name_index().get(slug)
    if not name:
        return None
    return {"slug": slug, "name": name, "fmp": _fmp_for(name)}


def formulations_with(name):
    target = slugify(name)
    out = []
    try:
        prods = json.loads(_PRODUCTS.read_text()).get("products", {})
    except Exception:
        return out
    for pslug, p in prods.items():
        for ing in (p.get("ingredients") or []):
            nm = (ing.get("name") if isinstance(ing, dict) else ing) or ""
            if slugify(nm) == target:
                out.append({"slug": pslug, "name": p.get("name", pslug)})
                break
    return out


def research_studies(name, k=12):
    try:
        from dashboard import product_content
        return product_content._research_sources(name, k=k) or []
    except Exception:
        return []
```

(If `product_content._research_sources` has a different signature, the implementer adapts the call to return the documented study-dict shape; degrade to `[]` on any error.)

- [ ] **Step 4: Run to verify pass.** `... -v` -> PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/ingredients.py tests/test_ingredients_resolver.py
git commit -m "feat: ingredient resolver (slug, fmp, formulations, research studies)"
```

---

### Task 2: `dashboard/ingredient_pages.py` store + requests

**Files:** Create `dashboard/ingredient_pages.py`; Create `tests/test_ingredient_pages_store.py`.

**Interfaces produced:** `init_table(cx)`; `get_page(cx, slug)`; `get_section(cx, slug, section)`/`upsert_section(cx, slug, section, text, model="")`; `set_state(cx, slug, state, by="")`; `set_scores(cx, slug, research, traditional)`; `set_related_forms(cx, slug, forms)`; `set_traditional_use(cx, slug, entries)`; `set_name(cx, slug, name)`; `record_request(cx, slug, email)`; `requesters_to_email(cx, slug)`; `mark_emailed(cx, slug, email)`; `notify_on_approve(cx, slug, name, base_url, *, send, strip=None)`.

- [ ] **Step 1: Write the failing tests** (cover: init; upsert/get section; set_scores clamps to 1-10; set_traditional_use + set_related_forms round-trip; set_state draft->approved; record_request once per (slug,email); requesters_to_email excludes emailed; notify_on_approve emails each once + marks emailed; per-cursor Row factory does not leak).

```python
# tests/test_ingredient_pages_store.py
import sqlite3, sys
from pathlib import Path
import pytest


def _m():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dashboard import ingredient_pages
        return ingredient_pages
    except Exception as e:
        pytest.skip(f"module not importable: {e}")


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    _m().init_table(cx)
    return cx


def test_section_scores_state(tmp_path):
    m = _m(); cx = _cx(tmp_path)
    m.upsert_section(cx, "zinc", "what_it_is", "An essential mineral.")
    assert m.get_section(cx, "zinc", "what_it_is") == "An essential mineral."
    m.set_scores(cx, "zinc", 9, 12)          # 12 clamps to 10
    p = m.get_page(cx, "zinc")
    assert p["research_score"] == 9 and p["traditional_score"] == 10
    m.set_traditional_use(cx, "zinc", [{"system": "TCM", "formula": "X", "uses": "y", "forms": "powder"}])
    m.set_related_forms(cx, "zinc", [{"name": "Zinc Oxide", "slug": "zinc-oxide", "verdict": "inferior", "note": "n"}])
    p = m.get_page(cx, "zinc")
    assert p["traditional_use"][0]["system"] == "TCM" and p["related_forms"][0]["verdict"] == "inferior"
    m.set_state(cx, "zinc", "approved", by="glen")
    assert m.get_page(cx, "zinc")["state"] == "approved"
    # bare query after a getter must still return tuples (no row_factory leak)
    assert isinstance(cx.execute("SELECT 1").fetchone(), tuple)


def test_requests_and_notify(tmp_path):
    m = _m(); cx = _cx(tmp_path)
    m.record_request(cx, "zinc", "a@x.com")
    m.record_request(cx, "zinc", "a@x.com")   # idempotent
    m.record_request(cx, "zinc", "b@x.com")
    assert {r["email"] for r in m.requesters_to_email(cx, "zinc")} == {"a@x.com", "b@x.com"}
    sent = []
    m.notify_on_approve(cx, "zinc", "Zinc", "https://x.test",
                        send=lambda to, subject, body: sent.append((to, body)) or True)
    assert len(sent) == 2 and all("/begin/ingredient/zinc" in b for _, b in sent)
    assert m.requesters_to_email(cx, "zinc") == []   # all marked emailed
    m.notify_on_approve(cx, "zinc", "Zinc", "https://x.test", send=lambda *a: sent.append(a) or True)
    assert len(sent) == 2   # at-most-once
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Implement `dashboard/ingredient_pages.py`** - model `dashboard/sales_pages.py` for the page table/getters (per-cursor Row factory) and `dashboard/sales_page_viewers.py` for the requests/notify. Table columns exactly as the spec's Storage section (`ingredient_slug PK, name, state, content_json, research_score, traditional_score, traditional_use_json, related_forms_json, model, generated_at, approved_at, approved_by, created_at, updated_at`) + the `ingredient_page_requests` table. `set_scores` clamps each score to `max(1, min(10, int(v)))` or stores None when v is None. `notify_on_approve`: for each `requesters_to_email`, call `send(email, "Your <name> deep-dive is ready", body)` where body contains `f"{base_url}/begin/ingredient/{slug}"`, then `mark_emailed`; wrap each send so one failure does not stop the rest and never raises. `get_page` returns a dict with parsed `traditional_use`/`related_forms`/`content` and the scores.

- [ ] **Step 4: Run -> PASS.**

- [ ] **Step 5: Commit**

```bash
git add dashboard/ingredient_pages.py tests/test_ingredient_pages_store.py
git commit -m "feat: ingredient_pages store + request/notify (mirrors sales_pages + viewer notify)"
```

---

### Task 3: Paid gate + route + page-data state machine + the page

**Files:** Modify `app.py` (the gate helper + `/begin/ingredient/<slug>` + `/begin/ingredient-page-data/<slug>`); Create `static/begin-ingredient.html`; add tests to a new `tests/test_ingredient_routes.py`.

**Interfaces:** Consumes `dashboard.ingredients`, `dashboard.ingredient_pages`, `_active_membership_for_email`.

- [ ] **Step 1: Write the failing tests** (paid gate + 3 states). Add `tests/test_ingredient_routes.py`:

```python
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _fresh(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    from dashboard import ingredient_pages
    with sqlite3.connect(db) as cx:
        ingredient_pages.init_table(cx)
    return db


def _known_slug():
    from dashboard import ingredients
    return next(iter(ingredients._name_index().keys()))


def test_pagedata_locked_for_non_member(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: None)
    monkeypatch.setattr(app_module, "INGREDIENT_PAGES_PAID_ONLY", True, raising=False)
    monkeypatch.setattr(app_module, "get_authenticated_user", lambda req: {"email": "non@x.com"})
    body = app_module.app.test_client().get(f"/begin/ingredient-page-data/{_known_slug()}").get_json()
    assert body["state"] == "locked"
    assert "research_score" not in body and "sections" not in body


def test_pagedata_preparing_for_paid_no_approved(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    monkeypatch.setattr(app_module, "INGREDIENT_PAGES_PAID_ONLY", True, raising=False)
    monkeypatch.setattr(app_module, "get_authenticated_user", lambda req: {"email": "paid@x.com"})
    # neutralize the background build so the test does not call the model
    monkeypatch.setattr(app_module, "_ingredient_kickoff_build", lambda slug, name: None, raising=False)
    slug = _known_slug()
    body = app_module.app.test_client().get(f"/begin/ingredient-page-data/{slug}").get_json()
    assert body["state"] == "preparing"
    from dashboard import ingredient_pages
    with sqlite3.connect(db) as cx:
        assert any(r["email"] == "paid@x.com" for r in ingredient_pages.requesters_to_email(cx, slug))


def test_pagedata_approved_returns_full(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    monkeypatch.setattr(app_module, "INGREDIENT_PAGES_PAID_ONLY", True, raising=False)
    monkeypatch.setattr(app_module, "get_authenticated_user", lambda req: {"email": "paid@x.com"})
    slug = _known_slug()
    from dashboard import ingredient_pages as ip
    with sqlite3.connect(db) as cx:
        ip.upsert_section(cx, slug, "what_it_is", "Hello.")
        ip.set_scores(cx, slug, 8, 7)
        ip.set_state(cx, slug, "approved", by="glen")
    body = app_module.app.test_client().get(f"/begin/ingredient-page-data/{slug}").get_json()
    assert body["state"] == "approved" and body["research_score"] == 8
    assert "formulations" in body and any(s["id"] == "what_it_is" for s in body["sections"])


def test_unknown_slug_pagedata(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    monkeypatch.setattr(app_module, "get_authenticated_user", lambda req: {"email": "paid@x.com"})
    r = app_module.app.test_client().get("/begin/ingredient-page-data/bogus-xyz")
    assert r.status_code == 404 or r.get_json().get("state") == "unknown"
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Implement the gate + routes** in `app.py`:

```python
INGREDIENT_PAGES_PAID_ONLY = os.environ.get("INGREDIENT_PAGES_PAID_ONLY", "true").strip().lower() in ("1", "true", "yes", "on")


def _ingredient_viewer_email():
    au = get_authenticated_user(request) or {}
    return (au.get("email") or request.cookies.get("rm_reorder_email", "") or "").strip().lower()


def _ingredient_paid_ok(email):
    if not INGREDIENT_PAGES_PAID_ONLY:
        return True
    try:
        return bool(_active_membership_for_email(email))
    except Exception:
        return False


def _ingredient_kickoff_build(slug, name):
    """Best-effort, non-blocking AI draft build (Task 5 fills this in)."""
    return None


@app.route("/begin/ingredient/<slug>")
def begin_ingredient_page(slug):
    return send_from_directory(STATIC, "begin-ingredient.html")


@app.route("/begin/ingredient-page-data/<slug>")
def begin_ingredient_page_data(slug):
    from dashboard import ingredients as _ing, ingredient_pages as _ip
    info = _ing.resolve(slug)
    if not info:
        return jsonify({"state": "unknown"}), 404
    name = info["name"]
    email = _ingredient_viewer_email()
    if not _ingredient_paid_ok(email):
        return jsonify({"slug": slug, "name": name, "state": "locked"})
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _ip.init_table(cx)
        page = _ip.get_page(cx, slug)
        if page and page.get("state") == "approved":
            sections = [{"id": s, "text": (page.get("content") or {}).get(s, "")}
                        for s in ("what_it_is", "research")]
            return jsonify({
                "slug": slug, "name": name, "state": "approved",
                "sections": sections,
                "research_score": page.get("research_score"),
                "traditional_score": page.get("traditional_score"),
                "traditional_use": page.get("traditional_use") or [],
                "related_forms": page.get("related_forms") or [],
                "research_studies": _ing.research_studies(name),
                "fmp": info.get("fmp") or {},
                "formulations": _ing.formulations_with(name),
            })
        # paid, not approved -> record request + kick off build, show preparing
        if email:
            _ip.record_request(cx, slug, email)
            _ip.set_name(cx, slug, name)
    if email:
        _ingredient_kickoff_build(slug, name)
    return jsonify({"slug": slug, "name": name, "state": "preparing"})
```

(`get_authenticated_user`, `send_from_directory`, `STATIC`, `_db_lock` exist. The `research_studies` call in the approved branch is fine - it is paid-gated. If you prefer, cache studies in the row at build time; rendering them live is acceptable.)

- [ ] **Step 4: Create `static/begin-ingredient.html`** - reads `/begin/ingredient-page-data/<slug>` (slug from `location.pathname`), renders by `state`: `locked` -> upgrade prompt + a membership link; `preparing` -> the "we are preparing your deep-dive on <name>, we will email you" message; `approved` -> the accordion: the two gauges (green->gold fill = `score/10`, hidden when null), What it is, Details (`fmp` fields), The research (the `sections` research text + the `research_studies` list, each title a `_blank` link to its `url`), Traditional use (`traditional_use` entries), Related forms (`related_forms`, each name a `_blank` link to `/begin/ingredient/<slug>`), In these formulations (`formulations`, links to `/begin/product/<slug>`). `unknown` -> a friendly not-found. textContent only; model styling on `begin-product.html`. No emoji/em dash.

- [ ] **Step 5: Run + commit**

Run: `... -m pytest tests/test_ingredient_routes.py -v` -> PASS.

```bash
git add app.py static/begin-ingredient.html tests/test_ingredient_routes.py
git commit -m "feat: ingredient page - paid gate + locked/preparing/approved state machine"
```

---

### Task 4: Link ingredients on the formulation page

**Files:** Modify `app.py` (`begin_product_page_data` - add a per-ingredient `slug`); Modify `static/begin-product.html` (`renderIngredientsBody`); add a serve test.

- [ ] **Step 1: Write the failing test** (assert `begin-product.html` renders each ingredient name as a link to `/begin/ingredient/<slug>` with `_blank`).

```python
def test_product_page_links_ingredients(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    html = app_module.app.test_client().get("/begin/product/" + "any").get_data(as_text=True) if False else (app_module.STATIC / "begin-product.html").read_text()
    assert "/begin/ingredient/" in html and "_blank" in html
```

(The link is built client-side in `renderIngredientsBody`, so the assertion is on the static file's JS. If `begin_product_page_data` is changed to include per-ingredient slugs, also assert the page-data shape in a separate test.)

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Add per-ingredient slug to product page-data.** In `begin_product_page_data` (app.py ~3171), normalize the ingredients list so each item is `{name, dose, slug}` using `dashboard.ingredients.slugify(name)` (keep strings working: a bare-string ingredient becomes `{name: s, slug: slugify(s)}`). Put this normalized list in the `"ingredients"` section body.

- [ ] **Step 4: Update `renderIngredientsBody`** in `static/begin-product.html` (~341): wrap the ingredient `name` span in an `<a href="/begin/ingredient/<slug>" target="_blank" rel="noopener">` (slug from `ing.slug`, else compute it client-side with the same `[^a-z0-9]+ -> -` rule). Keep the dose. All via `textContent`/`setAttribute`.

- [ ] **Step 5: Run + commit**

```bash
git add app.py static/begin-product.html tests/test_ingredient_routes.py
git commit -m "feat: link each ingredient name on the formulation page to its ingredient page"
```

---

### Task 5: AI draft build + request kickoff (`ingredient_copy.py`, `propose_curation`, gen)

**Files:** Create `dashboard/ingredient_copy.py`; Modify `app.py` (`_ingredient_kickoff_build` + a gen SSE endpoint + the Anthropic client injection); add tests.

- [ ] **Step 1: Write the failing tests** (mock the Anthropic client: `propose_curation` returns clamped scores + lists; `build_section_prompt` returns a (system,user) grounded in the ingredient; `_ingredient_kickoff_build` writes a draft row paid-gated).

- [ ] **Step 2-4: Implement** `dashboard/ingredient_copy.py` mirroring `dashboard/sales_copy.py`: `NARRATIVE_SECTIONS=("what_it_is","research")`; `build_section_prompt(section, ingredient)` grounded in `ingredient["fmp"]` + `ingredient["studies"]` with the same compliance system prompt; `propose_curation(ingredient, client)` -> a single haiku JSON call returning `{research_score, traditional_score, related_forms, traditional_use}` (prompt: omit any classical formula not confidently real; clamp scores 1-10; slug each related form via `ingredients.slugify`); safe defaults on failure. In `app.py`, implement `_ingredient_kickoff_build(slug, name)` to run in a background thread (mirror the funnel `_onboard` threading pattern): build `ingredient` (fmp + studies), call `propose_curation` + generate the two sections via the haiku client, and write them to `ingredient_pages` as a draft (state stays `draft`). Add a gen SSE endpoint `GET /begin/ingredient-page-gen/<slug>/<section>` (mirror `/begin/product-page-gen`, paid-gated) for the console preview. Inject the Anthropic client the same way the sales-page gen does.

- [ ] **Step 5: Run + commit** (`... -m pytest tests/test_ingredient_copy.py tests/test_ingredient_routes.py -v`).

```bash
git add dashboard/ingredient_copy.py app.py tests/test_ingredient_copy.py
git commit -m "feat: ingredient AI draft build - propose_curation + section gen + background kickoff"
```

---

### Task 6: Console review + email-when-ready

**Files:** Create `dashboard/ingredient_page_actions.py`; Modify `app.py` (register/configure + the console list/serve routes + nav); Create `static/console-ingredient-pages.html`; add tests.

- [ ] **Step 1: Write the failing tests** (`ingredient_page.edit` updates sections/scores/traditional-use/forms and stays draft; `ingredient_page.approve` -> approved AND calls the injected send fn once per requester; the console list route returns draft rows; RBAC OWNER/OPS).

- [ ] **Step 2-4: Implement** `dashboard/ingredient_page_actions.py` mirroring `dashboard/sales_pages_actions.py`: `configure(**kw)` (base_url, send, strip); `_exec_edit` (set_section/set_scores/set_traditional_use/set_related_forms per params, stays draft); `_exec_approve` (`set_state approved`, then `ingredient_pages.notify_on_approve(cx, slug, name, base_url, send=..., strip=...)` - wrapped, never fails approve); `_exec_regenerate` (re-run propose_curation + sections); `register()` idempotent, RBAC `(OWNER, OPS)`. Register + configure at app startup beside `sales_pages_actions`. Add `GET /api/console/ingredient-pages` (list draft rows, console-auth) + `GET /console/ingredient-pages` (serve page). Create `static/console-ingredient-pages.html` modelled on `console-sales-pages.html` (edit the two narrative sections + the two scores + the traditional-use list + the related-forms list; approve). Add an "Ingredient Pages" nav sub-tab in `op-nav.js`.

- [ ] **Step 5: Run the focused tests + the sweep** (`... -m pytest tests/test_ingredient_*.py -v`; then `... -m pytest tests/ -k "ingredient or begin or sales" -v` - no regressions).

- [ ] **Step 6: Commit**

```bash
git add dashboard/ingredient_page_actions.py app.py static/console-ingredient-pages.html static/op-nav.js tests/
git commit -m "feat: ingredient-page console review + approve + email-when-ready"
```

---

## Self-Review

**1. Spec coverage:** resolver (T1); store + request/notify (T2); paid gate + locked/preparing/approved state machine + page (T3); formulation links (T4); AI build + propose_curation + kickoff (T5); console + approve + notify (T6). Paid gate + `INGREDIENT_PAGES_PAID_ONLY` -> T3. Public never emits draft -> T3 (content only in the approved branch; the preparing/locked branches carry no content; tests assert it). Email-when-ready -> T2 (`notify_on_approve`) + T6 (approve fires it). Two gauges / traditional-use / related-forms / heavy research / both-way links -> T3 page + T4. AI proposes-Glen-verifies / no invented formulas -> T5 prompt.

**2. Placeholder scan:** No TBD. T2/T5/T6 lean on "mirror `sales_pages.py`/`sales_copy.py`/`sales_pages_actions.py`" with the explicit deltas (columns, the two extra lists, the requests table, the paid gate) named; the mirror targets are concrete existing files. The HTML tasks pin their contract via serve-test assertions.

**3. Type consistency:** `slugify`/`resolve`/`formulations_with`/`research_studies` (T1) consumed by T3/T4. Store fns (T2) used by T3 (`get_page`/`record_request`), T5 (`upsert_section`/`set_scores`/`set_traditional_use`/`set_related_forms`), T6 (`set_state`/`notify_on_approve`). `_ingredient_paid_ok`/`_ingredient_viewer_email`/`_ingredient_kickoff_build` defined in T3, the kickoff filled in T5. `page-data` state values `locked|preparing|approved|unknown` consistent between the route (T3) and the page (T3 Step 4). `propose_curation -> {research_score, traditional_score, related_forms, traditional_use}` (T5) matches the store setters (T2) and the console edit (T6).
