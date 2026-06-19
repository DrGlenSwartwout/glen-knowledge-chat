# Phase 5 — Console Sales-Page Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Glen/Rae a console page to review draft AI sales-page copy — edit it, regenerate it in-process, and approve it (which drops the live page's "pending review" banner).

**Architecture:** Extend the Phase-2 `sales_pages` store with `approved_at`/`approved_by` + `set_state`/`list_draft_pages`. Three new dispatch-spine actions (`sales_pages.approve`/`edit`/`regenerate`, RBAC OWNER+OPS, LOW_WRITE) live in a new `dashboard/sales_pages_actions.py` with app-injected deps so the regenerate executor can call the Anthropic client. A new console page (`static/console-sales-pages.html`, BOS sub-tab) lists draft pages and drives those actions via the existing `/api/action/<key>` route. Page-data gains an `ai_state` field; `begin-product.html` only shows the caveat banner while `ai_state !== 'approved'`.

**Tech Stack:** Python 3.11, Flask, SQLite (`chat_log.db` via `LOG_DB`), Anthropic SDK (`_cl`, model `claude-haiku-4-5-20251001`), vanilla-JS static pages, APScheduler (unused here), pytest.

## Global Constraints

- Console-only feature — no new public flag. Console endpoints gated by `_check_console_auth()`; actions gated by RBAC `(OWNER, OPS)` via `dispatch_action`.
- Copy compliance is owned by `dashboard/sales_copy.build_section_prompt` (structure/function language, no disease claims). Regenerated copy MUST be run through `_strip_dash` (no em dashes) before storing — same rule as Phase-2 gen.
- NO emoji in any client- or console-facing copy/markup (SVG/text glyphs only).
- Reuse the Phase-2 `sales_pages` table and `dashboard/sales_pages.py` functions; do not create a parallel store.
- `register()` MUST be idempotent (app.py is reloaded in the test suite via `importlib.reload`; re-registering a duplicate action key raises).
- Narrative sections are exactly `dashboard.sales_copy.NARRATIVE_SECTIONS = ("intro", "description", "research")`.
- Run tests with: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_sales_pages_phase5.py -v`

---

### Task 1: Data layer — approved columns, `set_state`, `list_draft_pages`

**Files:**
- Modify: `dashboard/sales_pages.py`
- Test: `tests/test_sales_pages_phase5.py` (create)

**Interfaces:**
- Consumes: existing `init_table(cx)`, `get_page(cx, slug)`, `upsert_section(cx, slug, section, text, model="")`, `_now()`.
- Produces:
  - `set_state(cx, slug, state, by="") -> None` — sets `state`; when `state=="approved"` also stamps `approved_at`/`approved_by`.
  - `list_draft_pages(cx) -> list[dict]` — `[{"slug": str, "state": str, "sections": [str,...]}]` for every page whose `content_json` is non-empty, newest-updated first.
  - `init_table` now guarantees `approved_at`/`approved_by` columns exist.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sales_pages_phase5.py`:

```python
import sqlite3
from dashboard import sales_pages as sp


def _cx():
    return sqlite3.connect(":memory:")


def test_set_state_approved_stamps_by_and_time():
    cx = _cx()
    sp.upsert_section(cx, "x", "intro", "hello")
    sp.set_state(cx, "x", "approved", by="Glen")
    page = sp.get_page(cx, "x")
    assert page["state"] == "approved"
    row = cx.execute(
        "SELECT approved_at, approved_by FROM sales_pages WHERE product_slug='x'").fetchone()
    assert row[1] == "Glen" and row[0]  # approved_by set, approved_at non-empty


def test_set_state_draft_does_not_stamp_approver():
    cx = _cx()
    sp.upsert_section(cx, "x", "intro", "hello")
    sp.set_state(cx, "x", "approved", by="Glen")
    sp.set_state(cx, "x", "draft")
    page = sp.get_page(cx, "x")
    assert page["state"] == "draft"


def test_list_draft_pages_includes_content_excludes_empty():
    cx = _cx()
    sp.upsert_section(cx, "with-copy", "intro", "hello")
    sp.init_table(cx)
    # a row with empty content_json should be excluded
    cx.execute("INSERT INTO sales_pages (product_slug, content_json) VALUES ('empty','{}')")
    cx.commit()
    rows = sp.list_draft_pages(cx)
    slugs = [r["slug"] for r in rows]
    assert "with-copy" in slugs and "empty" not in slugs
    row = next(r for r in rows if r["slug"] == "with-copy")
    assert row["state"] == "draft" and row["sections"] == ["intro"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_sales_pages_phase5.py -v`
Expected: FAIL with `AttributeError: module 'dashboard.sales_pages' has no attribute 'set_state'`.

- [ ] **Step 3: Implement the data-layer changes**

In `dashboard/sales_pages.py`, add the column guard and call it from `init_table`, then add the two functions. Replace the existing `init_table` body and append the new functions:

```python
def _ensure_columns(cx):
    cols = {r[1] for r in cx.execute("PRAGMA table_info(sales_pages)").fetchall()}
    if "approved_at" not in cols:
        cx.execute("ALTER TABLE sales_pages ADD COLUMN approved_at TEXT DEFAULT ''")
    if "approved_by" not in cols:
        cx.execute("ALTER TABLE sales_pages ADD COLUMN approved_by TEXT DEFAULT ''")


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS sales_pages ("
        "product_slug TEXT PRIMARY KEY, state TEXT DEFAULT 'draft', "
        "content_json TEXT DEFAULT '{}', model TEXT DEFAULT '', "
        "generated_at TEXT DEFAULT '', created_at TEXT DEFAULT '', updated_at TEXT DEFAULT '')")
    _ensure_columns(cx)
    cx.commit()


def set_state(cx, slug, state, by=""):
    init_table(cx)
    now = _now()
    if state == "approved":
        cx.execute(
            "UPDATE sales_pages SET state=?, approved_at=?, approved_by=?, updated_at=? "
            "WHERE product_slug=?", (state, now, by, now, slug))
    else:
        cx.execute("UPDATE sales_pages SET state=?, updated_at=? WHERE product_slug=?",
                   (state, now, slug))
    cx.commit()


def list_draft_pages(cx):
    init_table(cx)
    rows = cx.execute(
        "SELECT product_slug, state, content_json FROM sales_pages "
        "ORDER BY updated_at DESC").fetchall()
    out = []
    for slug, state, cj in rows:
        content = json.loads(cj or "{}")
        if not content:
            continue
        out.append({"slug": slug, "state": state or "draft",
                    "sections": sorted(content.keys())})
    return out
```

Note: the existing `init_table` (lines 9-15) is REPLACED by the version above; do not leave two `init_table` definitions.

- [ ] **Step 4: Run tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_sales_pages_phase5.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_pages.py tests/test_sales_pages_phase5.py
git commit -m "feat(sales-phase5): sales_pages set_state + list_draft_pages + approved columns"
```

---

### Task 2: Console actions module (`sales_pages_actions.py`)

**Files:**
- Create: `dashboard/sales_pages_actions.py`
- Test: `tests/test_sales_pages_phase5.py` (append)

**Interfaces:**
- Consumes: Task 1's `sales_pages.set_state`, `sales_pages.upsert_section`; `sales_copy.NARRATIVE_SECTIONS`, `sales_copy.build_section_prompt`; `dashboard.actions.{register_action, Action, LOW_WRITE, get_action}`; `dashboard.rbac.{OWNER, OPS}`.
- Produces:
  - `configure(**kw)` — app injects `client`, `get_product`, `product_card`, `strip_dash`.
  - `regenerate_copy(slug) -> dict|None` — `{section: text}` for all narrative sections, em dashes stripped; `None` if deps/product missing.
  - `register()` — idempotently registers actions `sales_pages.approve`, `sales_pages.edit`, `sales_pages.regenerate`.
  - Executors are referenced by Task 3 through `dispatch_action`, not imported directly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sales_pages_phase5.py`:

```python
from dashboard import sales_pages_actions as spa
from dashboard.rbac import Actor, OWNER


class _Blk:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Msg:
    def __init__(self, text):
        self.content = [_Blk(text)]


class _FakeMessages:
    def create(self, **kw):
        # echo the section brief marker so each call is distinct-enough; em dash on purpose
        return _Msg("Supports vitality — grounded in the stack.")


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def _configure_fake():
    spa.configure(client=_FakeClient(),
                  get_product=lambda s: {"name": "Test", "ingredients": [{"name": "Magnesium"}]},
                  product_card=lambda p: {"ingredients": p.get("ingredients", [])},
                  strip_dash=lambda s: s.replace("—", ","))


def test_regenerate_copy_strips_dashes_all_sections():
    _configure_fake()
    out = spa.regenerate_copy("x")
    assert set(out.keys()) == {"intro", "description", "research"}
    assert all("—" not in v and v for v in out.values())


def test_regenerate_copy_none_without_product():
    spa.configure(client=_FakeClient(), get_product=lambda s: None,
                  product_card=lambda p: {}, strip_dash=lambda s: s)
    assert spa.regenerate_copy("nope") is None


def test_exec_edit_forces_draft():
    _configure_fake()
    spa.register()
    cx = _cx()
    sp.upsert_section(cx, "x", "intro", "old")
    sp.set_state(cx, "x", "approved", by="Glen")
    from dashboard.actions import get_action
    act = get_action("sales_pages.edit")
    act.executor({"slug": "x", "section": "intro", "text": "new copy"},
                 {"cx": cx, "actor": Actor(role=OWNER, name="Glen")})
    page = sp.get_page(cx, "x")
    assert page["content"]["intro"] == "new copy"
    # any edit returns the page to draft: edited copy must be re-approved before the
    # banner drops again (Approve is the single deliberate publish step).
    assert page["state"] == "draft"


def test_exec_approve_sets_approved():
    _configure_fake()
    spa.register()
    cx = _cx()
    sp.upsert_section(cx, "x", "intro", "hi")
    from dashboard.actions import get_action
    get_action("sales_pages.approve").executor(
        {"slug": "x"}, {"cx": cx, "actor": Actor(role=OWNER, name="Glen")})
    assert sp.get_page(cx, "x")["state"] == "approved"


def test_exec_regenerate_sets_draft_and_writes_copy():
    _configure_fake()
    spa.register()
    cx = _cx()
    sp.upsert_section(cx, "x", "intro", "old")
    sp.set_state(cx, "x", "approved", by="Glen")
    from dashboard.actions import get_action
    res = get_action("sales_pages.regenerate").executor(
        {"slug": "x"}, {"cx": cx, "actor": Actor(role=OWNER, name="Glen")})
    page = sp.get_page(cx, "x")
    assert page["state"] == "draft"
    assert page["content"]["intro"] == res["content"]["intro"]


def test_register_idempotent():
    spa.register()
    spa.register()  # second call must not raise duplicate-key
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_sales_pages_phase5.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.sales_pages_actions'`.

- [ ] **Step 3: Implement the actions module**

Create `dashboard/sales_pages_actions.py`:

```python
"""Phase-5 console actions for sales-page copy review: approve / edit / regenerate.
Registered on the Business-OS dispatch spine. The regenerate executor needs the
Anthropic client + product lookups, which app.py injects via configure()."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import sales_pages as _sp
from dashboard import sales_copy as _sc

_MODEL = "claude-haiku-4-5-20251001"
_DEPS = {}  # client, get_product, product_card, strip_dash — set by app.py at startup


def configure(**kw):
    _DEPS.update(kw)


def regenerate_copy(slug):
    """Generate all narrative sections synchronously; return {section: text} or None."""
    client = _DEPS.get("client")
    get_product = _DEPS.get("get_product")
    product_card = _DEPS.get("product_card")
    strip = _DEPS.get("strip_dash") or (lambda s: s)
    if client is None or get_product is None:
        return None
    p = get_product(slug)
    if not p:
        return None
    prod = dict(p)
    if not prod.get("ingredients") and product_card is not None:
        prod["ingredients"] = (product_card(p) or {}).get("ingredients", [])
    out = {}
    for section in _sc.NARRATIVE_SECTIONS:
        system, user = _sc.build_section_prompt(section, prod)
        msg = client.messages.create(
            model=_MODEL, max_tokens=600, system=system,
            messages=[{"role": "user", "content": user}])
        text = "".join(getattr(b, "text", "") for b in msg.content
                       if getattr(b, "type", "") == "text")
        out[section] = strip(text).strip()
    return out


def _actor_name(actor):
    return (getattr(actor, "name", "") or getattr(actor, "role", "") or "console")


def _exec_approve(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    _sp.set_state(ctx["cx"], slug, "approved", by=_actor_name(ctx.get("actor")))
    return {"slug": slug, "state": "approved"}


def _exec_edit(params, ctx):
    slug = (params.get("slug") or "").strip()
    section = (params.get("section") or "").strip()
    if not slug or section not in _sc.NARRATIVE_SECTIONS:
        raise ValueError("slug and valid section required")
    cx = ctx["cx"]
    _sp.upsert_section(cx, slug, section, params.get("text") or "")
    # any edit returns the page to draft; edited copy must be re-approved (never auto-approves)
    _sp.set_state(cx, slug, "draft")
    return {"slug": slug, "section": section, "saved": True}


def _exec_regenerate(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    content = regenerate_copy(slug)
    if content is None:
        raise RuntimeError("regeneration unavailable")
    cx = ctx["cx"]
    for section, text in content.items():
        if text:
            _sp.upsert_section(cx, slug, section, text, model=_MODEL)
    _sp.set_state(cx, slug, "draft")
    return {"slug": slug, "state": "draft", "content": content}


def register():
    if get_action("sales_pages.approve"):
        return
    register_action(Action(
        key="sales_pages.approve", module="sales_pages", title="Approve sales page",
        description="Mark a product's AI sales copy approved (drops the draft banner on the live page).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_approve))
    register_action(Action(
        key="sales_pages.edit", module="sales_pages", title="Edit sales-page section",
        description="Save edited copy for one narrative section (stays draft).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_edit))
    register_action(Action(
        key="sales_pages.regenerate", module="sales_pages", title="Regenerate sales copy",
        description="Regenerate all narrative sections in-process for review (stays draft).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_regenerate))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_sales_pages_phase5.py -v`
Expected: all pass (9 total in this file so far).

- [ ] **Step 5: Commit**

```bash
git add dashboard/sales_pages_actions.py tests/test_sales_pages_phase5.py
git commit -m "feat(sales-phase5): sales_pages approve/edit/regenerate dispatch actions"
```

---

### Task 3: App wiring — register actions, page-data `ai_state`, banner drop

**Files:**
- Modify: `app.py` (register/configure block; `begin_product_page_data`)
- Modify: `static/begin-product.html:848-854` (banner condition)
- Test: `tests/test_sales_pages_phase5.py` (append)

**Interfaces:**
- Consumes: Task 2's `sales_pages_actions.{register, configure}`; existing app globals `_cl`, `_get_product`, `_product_card`, `_strip_dash`; `dispatch_action`; `dashboard.rbac`.
- Produces: page-data JSON now carries top-level `"ai_state"` (`"none"` | `"draft"` | `"approved"`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sales_pages_phase5.py`:

```python
import importlib


def _reload_app(monkeypatch, tmp_path, copy="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SALES_PAGES_ENABLED", "true")
    monkeypatch.setenv("SALES_PAGES_AI_COPY", copy)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_page_data_ai_state_none_when_no_page(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    data = appmod.app.test_client().get(f"/begin/product-page-data/{slug}").get_json()
    assert data["ai_state"] == "none"


def test_page_data_ai_state_reflects_state(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    import sqlite3
    from dashboard import sales_pages as sp2
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sp2.upsert_section(cx, slug, "intro", "draft copy")
    data = appmod.app.test_client().get(f"/begin/product-page-data/{slug}").get_json()
    assert data["ai_state"] == "draft"
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sp2.set_state(cx, slug, "approved", by="Glen")
    data = appmod.app.test_client().get(f"/begin/product-page-data/{slug}").get_json()
    assert data["ai_state"] == "approved"


def test_dispatch_approve_flips_state(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    import sqlite3
    from dashboard import sales_pages as sp2
    from dashboard import dispatch as d
    from dashboard.rbac import Actor, OWNER
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sp2.upsert_section(cx, slug, "intro", "draft copy")
        res = d.dispatch_action(cx, "sales_pages.approve", {"slug": slug},
                                Actor(role=OWNER, name="Glen"), source="panel")
        assert res["status"] == "done"
        assert sp2.get_page(cx, slug)["state"] == "approved"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_sales_pages_phase5.py -k "ai_state or dispatch_approve" -v`
Expected: FAIL — `ai_state` missing from page-data (`KeyError`) and `sales_pages.approve` unknown action.

- [ ] **Step 3a: Register + configure the actions at app startup**

In `app.py`, find the `bos_action` route (search for `@app.route("/api/action/<path:key>", methods=["POST"])`). Immediately AFTER that function's body (after its `return jsonify(res)` line), add a module-level registration block:

```python
# ── Phase 5: sales-page review actions (approve/edit/regenerate) ──────────────
from dashboard import sales_pages_actions as _spa
_spa.register()
_spa.configure(client=_cl, get_product=_get_product,
               product_card=_product_card, strip_dash=_strip_dash)
```

(This runs at import; `register()` is idempotent across the test suite's reloads. `_cl`, `_get_product`, `_product_card`, `_strip_dash` are all defined earlier in the module.)

- [ ] **Step 3b: Add `ai_state` to page-data**

In `begin_product_page_data` (app.py ~2873), introduce a default before the `_SALES_AI_COPY_ENABLED` block and set it inside, then include it in the response. Change the block at lines 2898-2916 to capture state:

```python
    _ai_state = "none"
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
                _pg = _sp.get_page(_cx, slug)
                _ai_state = _pg["state"] if _pg else "none"
        except Exception as _e:
            print(f"[sales-ai] page-data marker skipped: {_e}", flush=True)
```

Then add `"ai_state": _ai_state,` to the final `jsonify({...})` dict (alongside `"slug"`, `"name"`, etc.):

```python
    return jsonify({
        "slug": slug, "name": p["name"], "price_cents": p["price_cents"],
        "price": f"${p['price_cents']/100:.2f}", "cta_url": f"/begin/buy/{slug}",
        "ai_state": _ai_state,
        "sections": sections, "miron_assets": _MIRON_ASSETS["assets"],
        "miron_story": _MIRON_ASSETS.get("story", []),
        "open_sections": _read_open_sections(request.cookies.get("amg_session", ""),
                                             (get_authenticated_user(request) or {}).get("email", "")),
    })
```

- [ ] **Step 3c: Drop the banner on approval**

In `static/begin-product.html`, change the banner condition (lines 848-849) from:

```javascript
      var hasAi = (data.sections || []).some(function(s){ return s.ai !== undefined; });
      if (hasAi){
```

to:

```javascript
      var hasAi = (data.sections || []).some(function(s){ return s.ai !== undefined; });
      if (hasAi && data.ai_state !== 'approved'){
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_sales_pages_phase5.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app.py static/begin-product.html tests/test_sales_pages_phase5.py
git commit -m "feat(sales-phase5): register review actions, page-data ai_state, banner drops on approval"
```

---

### Task 4: Console API + UI page + nav sub-tab

**Files:**
- Modify: `app.py` (3 new routes)
- Create: `static/console-sales-pages.html`
- Modify: `static/op-nav.js` (add `sales` sub-tab + update data-sub comment)
- Test: `tests/test_sales_pages_phase5.py` (append)

**Interfaces:**
- Consumes: Task 1 `sales_pages.{list_draft_pages, get_page}`; `sales_copy.NARRATIVE_SECTIONS`; `_get_product`; `_check_console_auth`; `STATIC`; existing `/api/action/<key>` route (Task 3 registration).
- Produces:
  - `GET /api/console/sales-pages` → `{"ok": True, "pages": [{slug, name, state, sections}]}`.
  - `GET /api/console/sales-page/<slug>` → `{"ok": True, slug, name, state, sections:[{id,text}], live_url}`.
  - `GET /console/sales-pages` → serves the console HTML.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sales_pages_phase5.py`:

```python
def test_console_list_and_load(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    import dashboard as _d
    _d.CONSOLE_SECRET = ""  # auth passes through when unset
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    import sqlite3
    from dashboard import sales_pages as sp2
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sp2.upsert_section(cx, slug, "intro", "draft copy")
    c = appmod.app.test_client()
    lst = c.get("/api/console/sales-pages").get_json()
    assert lst["ok"] and any(p["slug"] == slug for p in lst["pages"])
    one = c.get(f"/api/console/sales-page/{slug}").get_json()
    assert one["ok"] and one["state"] == "draft"
    assert [s["id"] for s in one["sections"]] == ["intro", "description", "research"]
    assert next(s for s in one["sections"] if s["id"] == "intro")["text"] == "draft copy"
    assert one["live_url"] == f"/begin/product/{slug}"


def test_console_page_served(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    r = appmod.app.test_client().get("/console/sales-pages")
    assert r.status_code == 200 and b"Sales Pages" in r.data


def test_console_list_gated_when_secret_set(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    import dashboard as _d
    _d.CONSOLE_SECRET = "topsecret"
    appmod.CONSOLE_SECRET = "topsecret"
    r = appmod.app.test_client().get("/api/console/sales-pages")  # no key
    assert r.status_code == 401
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_sales_pages_phase5.py -k console -v`
Expected: FAIL — routes return 404.

- [ ] **Step 3a: Add the three routes**

In `app.py`, after the `console_biofield_portal_page` route (search `@app.route("/console/biofield-portal")`), add:

```python
@app.route("/console/sales-pages")
def console_sales_pages_page():
    resp = send_from_directory(STATIC, "console-sales-pages.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/console/sales-pages", methods=["GET"])
def api_console_sales_pages_list():
    bad = _check_console_auth()
    if bad:
        return bad
    from dashboard import sales_pages as _sp
    with sqlite3.connect(LOG_DB) as cx:
        pages = _sp.list_draft_pages(cx)
    for pg in pages:
        pg["name"] = (_get_product(pg["slug"]) or {}).get("name", pg["slug"])
    return jsonify({"ok": True, "pages": pages})


@app.route("/api/console/sales-page/<slug>", methods=["GET"])
def api_console_sales_page_load(slug):
    bad = _check_console_auth()
    if bad:
        return bad
    from dashboard import sales_pages as _sp
    from dashboard import sales_copy as _sc
    p = _get_product(slug)
    with sqlite3.connect(LOG_DB) as cx:
        page = _sp.get_page(cx, slug)
    content = (page or {}).get("content", {})
    sections = [{"id": s, "text": content.get(s, "")} for s in _sc.NARRATIVE_SECTIONS]
    return jsonify({"ok": True, "slug": slug, "name": (p or {}).get("name", slug),
                    "state": (page or {}).get("state", "none"),
                    "sections": sections, "live_url": f"/begin/product/{slug}"})
```

- [ ] **Step 3b: Create the console page**

Create `static/console-sales-pages.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sales Pages &middot; Console</title>
<style>
  :root{ --bg:#0f1115; --card:#171a21; --line:#262b36; --fg:#e8ebf0; --muted:#9aa3b2;
         --accent:#5b8cff; --ok:#3fb968; --err:#e0556a; }
  *{ box-sizing:border-box; }
  body{ margin:0; background:var(--bg); color:var(--fg);
        font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }
  .wrap{ max-width:1100px; margin:0 auto; padding:18px; display:flex; gap:18px; }
  .col-list{ flex:0 0 280px; } .col-edit{ flex:1; min-width:0; }
  .card{ background:var(--card); border:1px solid var(--line); border-radius:10px;
         padding:14px; margin-bottom:14px; }
  h1{ font-size:20px; margin:0 0 4px; } h2{ font-size:15px; margin:0 0 8px; }
  .sub{ color:var(--muted); margin:0 0 14px; }
  .item{ padding:9px 11px; border:1px solid var(--line); border-radius:8px;
         margin-bottom:7px; cursor:pointer; display:flex; justify-content:space-between; }
  .item:hover{ border-color:var(--accent); } .item.sel{ border-color:var(--accent); }
  .pill{ font-size:11px; padding:1px 7px; border-radius:20px; border:1px solid var(--line);
         color:var(--muted); } .pill.approved{ color:var(--ok); border-color:var(--ok); }
  label{ display:block; font-size:12px; color:var(--muted); margin:10px 0 4px; }
  textarea{ width:100%; min-height:90px; background:#0c0e12; color:var(--fg);
            border:1px solid var(--line); border-radius:8px; padding:9px; font:inherit; resize:vertical; }
  .btn{ background:var(--accent); color:#fff; border:0; border-radius:8px;
        padding:8px 14px; font:inherit; cursor:pointer; }
  .btn.ghost{ background:transparent; color:var(--fg); border:1px solid var(--line); }
  .btn.ok{ background:var(--ok); }
  .row{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top:12px; }
  .preview{ color:var(--accent); text-decoration:none; font-size:13px; }
  .status{ min-height:20px; color:var(--muted); font-size:13px; margin-top:10px; }
  body > :not(.op-nav-bar):not(.op-nav-sub):not(#gate):not(script):not(style){ }
  #gate{ position:fixed; inset:0; display:flex; align-items:center; justify-content:center;
         background:var(--bg); z-index:50; }
</style>
</head>
<body>
<script src="/static/op-nav.js" data-active="bos" data-sub="sales"></script>

<div id="gate" style="display:none"><div class="card" style="min-width:300px">
  <h2>Console key</h2>
  <input id="key" type="password" placeholder="CONSOLE_SECRET" style="width:100%;padding:8px">
  <p style="margin-top:10px"><button class="btn" onclick="unlock()">Unlock</button></p>
</div></div>

<div id="app" class="wrap" style="display:none">
  <div class="col-list">
    <h1>Sales Pages</h1>
    <p class="sub">Review AI-generated copy, edit it, regenerate, and approve.</p>
    <div id="list"></div>
  </div>
  <div class="col-edit">
    <div id="editor" class="card" style="display:none">
      <h2 id="ed-title">&nbsp;</h2>
      <a id="ed-live" class="preview" target="_blank" rel="noopener">View live page &#8599;</a>
      <span id="ed-state" class="pill" style="margin-left:8px"></span>
      <div id="ed-sections"></div>
      <div class="row">
        <button class="btn ghost" id="btn-regen" onclick="regen()">Regenerate &amp; review</button>
        <button class="btn ok" id="btn-approve" onclick="approve()">Approve</button>
      </div>
      <div id="status" class="status"></div>
    </div>
    <div id="empty" class="card"><p class="sub" style="margin:0">Select a product on the left to review its copy.</p></div>
  </div>
</div>

<script>
function key(){ return localStorage.getItem('console_key') || ''; }
(function(){ var u=new URLSearchParams(location.search).get('key'); if(u) localStorage.setItem('console_key', u); })();
function unlock(){ localStorage.setItem('console_key', document.getElementById('key').value);
  document.getElementById('gate').style.display='none'; boot(); }
const $ = id => document.getElementById(id);
const esc = s => (s==null?'':String(s)).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const SECTION_LABELS = { intro:'Intro (what this does)', description:'Overview', research:'The research' };
let CURRENT = null;

async function api(method, path, body){
  const opt = { method, headers:{'X-Console-Key':key()} };
  if(body){ opt.headers['Content-Type']='application/json'; opt.body=JSON.stringify(body); }
  const r = await fetch(path, opt);
  return { ok:r.ok, status:r.status, json: await r.json().catch(()=>({})) };
}

function boot(){
  if(!key()){ $('gate').style.display='flex'; return; }
  $('app').style.display='flex';
  loadList();
}

async function loadList(){
  const r = await api('GET', '/api/console/sales-pages');
  if(!r.ok){ $('list').innerHTML = '<p class="sub">Could not load (check console key).</p>'; return; }
  const pages = r.json.pages || [];
  if(!pages.length){ $('list').innerHTML = '<p class="sub">No draft pages yet.</p>'; return; }
  $('list').innerHTML = pages.map(p =>
    `<div class="item" data-slug="${esc(p.slug)}" onclick="select('${esc(p.slug)}')">`
    + `<span>${esc(p.name||p.slug)}</span>`
    + `<span class="pill ${p.state==='approved'?'approved':''}">${esc(p.state)}</span></div>`).join('');
}

async function select(slug){
  CURRENT = slug;
  document.querySelectorAll('.item').forEach(el =>
    el.classList.toggle('sel', el.dataset.slug===slug));
  const r = await api('GET', `/api/console/sales-page/${encodeURIComponent(slug)}`);
  if(!r.ok) return;
  const d = r.json;
  $('empty').style.display='none';
  $('editor').style.display='block';
  $('ed-title').textContent = d.name || slug;
  $('ed-live').href = d.live_url || ('/begin/product/'+slug);
  setStatePill(d.state);
  $('ed-sections').innerHTML = (d.sections||[]).map(s =>
    `<label>${esc(SECTION_LABELS[s.id]||s.id)}</label>`
    + `<textarea id="sec-${esc(s.id)}">${esc(s.text)}</textarea>`
    + `<div class="row"><button class="btn ghost" onclick="saveSection('${esc(s.id)}')">Save ${esc(s.id)}</button></div>`).join('');
  $('status').textContent = '';
}

function setStatePill(state){
  const el = $('ed-state');
  el.textContent = state || 'none';
  el.className = 'pill' + (state==='approved' ? ' approved' : '');
}

async function act(key, params){
  const r = await api('POST', '/api/action/'+key, params);
  return r;
}

async function saveSection(id){
  const text = $('sec-'+id).value;
  $('status').textContent = 'Saving ' + id + '...';
  const r = await act('sales_pages.edit', { slug: CURRENT, section: id, text });
  $('status').textContent = (r.ok && r.json.status==='done') ? 'Saved ' + id + ' (still draft).' : 'Save failed.';
  setStatePill('draft');
}

async function regen(){
  $('status').textContent = 'Regenerating all sections...';
  $('btn-regen').disabled = true;
  const r = await act('sales_pages.regenerate', { slug: CURRENT });
  $('btn-regen').disabled = false;
  if(r.ok && r.json.status==='done'){
    const content = (r.json.result||{}).content || {};
    Object.keys(content).forEach(id => { const t=$('sec-'+id); if(t) t.value = content[id]; });
    setStatePill('draft');
    $('status').textContent = 'Regenerated. Review the new copy, then Approve.';
  } else {
    $('status').textContent = 'Regenerate failed.';
  }
}

async function approve(){
  $('status').textContent = 'Approving...';
  const r = await act('sales_pages.approve', { slug: CURRENT });
  if(r.ok && r.json.status==='done'){
    setStatePill('approved');
    $('status').textContent = 'Approved. The live page no longer shows the draft banner.';
    loadList();
  } else {
    $('status').textContent = 'Approve failed.';
  }
}

boot();
</script>
</body>
</html>
```

- [ ] **Step 3c: Add the nav sub-tab**

In `static/op-nav.js`, add the `sales` entry to the BOS sub-tabs array right after the `biofield` entry (line 77):

```javascript
    { id: "biofield", label: "Biofield",  href: "/console/biofield-portal" + qs },
    { id: "sales",    label: "Sales Pages", href: "/console/sales-pages" + qs },
```

And update the valid-`data-sub` comment (line 11) to include `sales`:

```javascript
 * Valid data-sub values: "orders" | "finance" | "crm" | "products" | "biofield" | "sales" | "shipping" | "neworder"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_sales_pages_phase5.py -v`
Expected: all pass.

- [ ] **Step 5: Run the full phase-1..5 sales suite for regressions**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/ -k sales -v`
Expected: all sales-pages tests pass (phases 1-5), no new failures.

- [ ] **Step 6: Commit**

```bash
git add app.py static/console-sales-pages.html static/op-nav.js tests/test_sales_pages_phase5.py
git commit -m "feat(sales-phase5): console sales-page review UI + list/load API + nav sub-tab"
```

---

## Manual Verification (after Task 4)

- With `SALES_PAGES_ENABLED=true` + `SALES_PAGES_AI_COPY=true` locally: open a product page → caveat banner shows. Open `/console/sales-pages?key=<CONSOLE_SECRET>` → product appears in the list → select it → three textareas load. Edit intro + Save → state stays draft. Click "Regenerate & review" → textareas refill with new copy, state draft. Click Approve → pill turns green; reload the public product page → no banner. NO emoji anywhere. The nav shows a "Sales Pages" sub-tab under Business OS.

## Self-Review Notes (plan author)

- **Spec coverage:** data columns + set_state/list_draft_pages (Task 1) → spec "Data"; actions approve/edit/regenerate (Task 2) → spec "Console actions"; regenerate helper in-process (Task 2 `regenerate_copy`) → spec "Regeneration helper"; ai_state + banner (Task 3) → spec "Banner drop"; list/load API + console UI + nav (Task 4) → spec "Console API + UI". All spec sections covered.
- **Confirmed choices:** (a) edit keeps draft — `_exec_edit` forces `state=draft` after every edit, so editing an approved page returns it to draft and the edited copy must be re-approved before the banner drops (test `test_exec_edit_forces_draft`). Approve is the single deliberate publish step. (b) Regenerate is immediate in-process — `_exec_regenerate` calls `regenerate_copy` synchronously and returns content for review (test `test_exec_regenerate_sets_draft_and_writes_copy`).
- **Type consistency:** `set_state(cx, slug, state, by="")`, `list_draft_pages(cx) -> [{slug,state,sections}]`, `regenerate_copy(slug) -> {section:text}|None`, action keys `sales_pages.approve|edit|regenerate`, page-data field `ai_state` — used identically across tasks.
