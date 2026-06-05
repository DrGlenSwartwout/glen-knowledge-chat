# BOS Phase 11c: Products module (board + stale-page work queue)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** A Products module over the now-enriched `products.json`: upgrade the Products Home signal to surface the stale-GrooveKart-page work queue, add a `/console/products` board (browse the catalog + ingredients, work the 48 stale pages), and an audited `products.mark_page_fixed` action that shrinks the queue as pages are updated. Persists the "fixed" set on the /data disk so it survives redeploys.

**Architecture:** `dashboard/products.py` owns the catalog reads, the upgraded `@signal("products")`, the stale-page helpers, and the `products.mark_page_fixed` action. It REPLACES the informational products signal currently in `dashboard/module_signals.py`. `app.py` adds read routes + the page route. The board is vanilla JS like the other console boards.

**Builds on:** the enriched `products.json` (145 products carry `ingredients`/`description`/`gk_stale`/`gk_stale_reason`). New branch `sess/ec0e1f15`, worktree `/tmp/wt-deploy-chat-ec0e1f15`.

**Persistence note:** `products.json` is a repo file (read-only at runtime). The "fixed" set persists to `DATA_DIR/products-page-fixed.json` (the /data disk), so marking a page fixed survives redeploys. Price-editing is deferred (needs a /data override overlay) — out of scope here.

---

## File Structure
- `dashboard/products.py` (new): catalog loader, `stale_pages`, `products_signal` (`@signal("products")`), `products.mark_page_fixed` action, fixed-set persistence.
- `dashboard/module_signals.py` (modify): REMOVE its `products_signal` (replaced by products.py).
- `tests/test_bos_products.py` (new): signal levels + stale helper + mark_page_fixed.
- `app.py` (modify): import the module; add `GET /api/products`, `GET /api/products/stale`, `GET /console/products`.
- `static/console-products.html` (new): the board.

---

## Task 1: `dashboard/products.py` + tests

**Files:** Create `dashboard/products.py`, `tests/test_bos_products.py`; Modify `dashboard/module_signals.py`.

- [ ] **Step 1: Write failing tests** `tests/test_bos_products.py`:

```python
import json, os, sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _products(tmp_path, products):
    d = tmp_path / "data"; d.mkdir()
    (d / "products.json").write_text(json.dumps({"products": products}))
    os.environ["DATA_DIR"] = str(d)
    from dashboard import products as P
    P._FIXED_CACHE = None
    return P


def test_stale_pages_excludes_fixed(tmp_path, monkeypatch):
    P = _products(tmp_path, {
        "a": {"name": "A", "ingredients": [{"name": "X"}], "gk_stale": True, "gk_stale_reason": "missing X"},
        "b": {"name": "B", "ingredients": [{"name": "Y"}]},
    })
    monkeypatch.setattr(P, "_products_path", lambda: str(tmp_path / "data" / "products.json"))
    sp = P.stale_pages()
    assert len(sp) == 1 and sp[0]["slug"] == "a"


def test_products_signal_amber_on_stale(tmp_path, monkeypatch):
    from dashboard import signals as S
    P = _products(tmp_path, {"a": {"name": "A", "ingredients": [{"name": "X"}], "gk_stale": True}})
    monkeypatch.setattr(P, "_products_path", lambda: str(tmp_path / "data" / "products.json"))
    sig = P.products_signal(None, None)
    assert sig["level"] == S.AMBER and sig["count"] == 1


def test_products_signal_green_when_clear(tmp_path, monkeypatch):
    from dashboard import signals as S
    P = _products(tmp_path, {"a": {"name": "A", "ingredients": [{"name": "X"}]}})
    monkeypatch.setattr(P, "_products_path", lambda: str(tmp_path / "data" / "products.json"))
    assert P.products_signal(None, None)["level"] == S.GREEN


def test_mark_page_fixed_action(tmp_path, monkeypatch):
    import sqlite3
    from dashboard import dispatch as D, events as E, rbac as R, actions as A
    P = _products(tmp_path, {"a": {"name": "A", "ingredients": [{"name": "X"}], "gk_stale": True}})
    monkeypatch.setattr(P, "_products_path", lambda: str(tmp_path / "data" / "products.json"))
    assert A.get_action("products.mark_page_fixed") is not None
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row; E.init_event_tables(cx)
    res = D.dispatch_action(cx, "products.mark_page_fixed", {"slug": "a"}, R.Actor(role=R.OWNER))
    assert res["status"] == "done"
    assert P.stale_pages() == []  # 'a' now fixed -> queue empty


def test_products_signal_registered():
    from dashboard import products as P, signals as S  # noqa
    assert S.SIGNAL_REGISTRY.get("products") is not None
```

- [ ] **Step 2: Run, verify fail.** `python3 -m pytest tests/test_bos_products.py -q` → fail.

- [ ] **Step 3: Implement `dashboard/products.py`:**

```python
"""Business-OS Products module over the enriched products.json. Surfaces the
catalog + ingredients and the stale-GrooveKart-page work queue, and persists the
'fixed' set on the /data disk (products.json itself is a read-only repo file)."""
import json
import os
from dashboard.signals import signal as _signal, AMBER, GREEN, GRAY
from dashboard.actions import action, LOW_WRITE
from dashboard.rbac import OWNER, OPS, VA

_REPO_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _products_path():
    d = os.environ.get("DATA_DIR")
    if d and os.path.exists(os.path.join(d, "products.json")):
        return os.path.join(d, "products.json")
    p = os.path.join(_REPO_DATA, "products.json")
    return p if os.path.exists(p) else None


def _fixed_path():
    d = os.environ.get("DATA_DIR") or _REPO_DATA
    return os.path.join(d, "products-page-fixed.json")


def load_products():
    p = _products_path()
    if not p:
        return {}
    try:
        return (json.load(open(p)) or {}).get("products", {})
    except Exception:
        return {}


def _fixed_set():
    try:
        return set(json.load(open(_fixed_path())))
    except Exception:
        return set()


def stale_pages(products=None, fixed=None):
    products = load_products() if products is None else products
    fixed = _fixed_set() if fixed is None else fixed
    return [{"slug": s, "name": p.get("name"), "reason": p.get("gk_stale_reason", "")}
            for s, p in products.items()
            if p.get("gk_stale") and s not in fixed]


def catalog(with_ingredients_only=True):
    out = []
    for s, p in load_products().items():
        if with_ingredients_only and not p.get("ingredients"):
            continue
        out.append({"slug": s, "name": p.get("name"), "price_cents": p.get("price_cents"),
                    "ingredients": p.get("ingredients", []), "description": p.get("description", ""),
                    "ingredients_source": p.get("ingredients_source"), "gk_stale": bool(p.get("gk_stale"))})
    out.sort(key=lambda x: (x["name"] or "").lower())
    return out


def products_signal(cx=None, actor=None):
    try:
        products = load_products()
        total = len(products)
        with_ing = sum(1 for p in products.values() if p.get("ingredients"))
        stale = len(stale_pages(products))
    except Exception:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    if total == 0:
        return {"level": GRAY, "summary": "No catalog", "top_actions": [], "count": 0}
    if stale:
        return {"level": AMBER, "summary": f"{stale} sales page{'s' if stale != 1 else ''} to update",
                "top_actions": [{"label": "Open products", "href": "/console/products"}], "count": stale}
    return {"level": GREEN, "summary": f"{with_ing}/{total} products enriched",
            "top_actions": [{"label": "Open products", "href": "/console/products"}], "count": 0}


products_signal = _signal("products")(products_signal)


def _mark_page_fixed_exec(params, ctx):
    slug = (params.get("slug") or "").strip()
    if not slug:
        raise ValueError("slug required")
    fixed = _fixed_set()
    fixed.add(slug)
    try:
        json.dump(sorted(fixed), open(_fixed_path(), "w"))
    except Exception as e:
        raise RuntimeError(f"could not persist fixed set: {e}")
    return {"slug": slug, "remaining": len(stale_pages()),
            "message": f"Marked {slug}'s GrooveKart page as updated."}


action(key="products.mark_page_fixed", module="products", title="Mark GK page updated",
       description="Record that a product's GrooveKart sales page now matches the current formula.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS, VA))(_mark_page_fixed_exec)
```

- [ ] **Step 4: Remove the old products signal** from `dashboard/module_signals.py` — delete its `products_signal` def + its `@signal("products")` registration (it is replaced by products.py; leave the other module_signals intact). If `app.py` imports `module_signals` BEFORE `products`, the products.py registration must win — ensure `import dashboard.products` comes AFTER `import dashboard.module_signals` in app.py (Task 2).

- [ ] **Step 5: Run tests.** `python3 -m pytest tests/test_bos_products.py tests/test_bos_signals.py -q` → green.

- [ ] **Step 6: Commit.** `git add dashboard/products.py dashboard/module_signals.py tests/test_bos_products.py && git commit -m "feat(bos): Products module signal + stale-page queue + mark_page_fixed action"`

---

## Task 2: app.py routes + board

**Files:** Modify `app.py`; Create `static/console-products.html`.

- [ ] **Step 1: Import + routes in app.py** (in the BOS startup block, AFTER `import dashboard.module_signals`):

```python
import dashboard.products as _bos_products  # noqa: F401 (registers products signal + action; replaces module_signals' products signal)


@app.route("/api/products")
def bos_products_list():
    if not _ghl_queue_auth() and not (request.headers.get("X-Console-Key", "") == (os.environ.get("CONSOLE_SECRET", ""))):
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"products": _bos_products.catalog()})


@app.route("/api/products/stale")
def bos_products_stale():
    if request.headers.get("X-Console-Key", "") != (os.environ.get("CONSOLE_SECRET", "")):
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"stale": _bos_products.stale_pages()})


@app.route("/console/products")
def bos_products_page():
    resp = send_from_directory(STATIC, "console-products.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp
```

(Use the console-key auth pattern already used by the other console APIs; reuse `_ghl_queue_auth` if it fits, else the `X-Console-Key == CONSOLE_SECRET` check shown.)

- [ ] **Step 2: Create `static/console-products.html`** — a board (same style/auth-gate as `console-crm.html`) with two tabs/sections:
  1. **Stale pages** (default): `GET /api/products/stale` → a list of `{slug, name, reason}`; each row has a "Mark updated" button → `POST /api/action/products.mark_page_fixed {slug}` (handle the standard dispatch envelope), then refresh.
  2. **Catalog**: `GET /api/products` → searchable list; clicking a product shows its `ingredients` ({name, dose}) + `description` + a "page outdated" badge when `gk_stale`.
  Escape every dynamic field with an `esc()` helper. Console-key localStorage gate. No em dashes.

- [ ] **Step 3: Compile + verify under doppler:**
```bash
python3 -m py_compile app.py
python3 -c "import html.parser; html.parser.HTMLParser().feed(open('static/console-products.html').read()); print('parsed OK')"
doppler run -p remedy-match -c prd -- bash -c 'mkdir -p /tmp/bostest && DATA_DIR=/tmp/bostest python3 - <<PY
import app, sqlite3
from dashboard import signals as S, products as P
assert S.SIGNAL_REGISTRY.get("products") is not None
# products signal resolves over the real repo products.json (DATA_DIR has none -> falls back to repo)
import dashboard.products as DP
cells = {c["module"]: c for c in S.aggregate_signals(sqlite3.connect(":memory:"), None)}
print("products cell:", cells["products"]["level"], "-", cells["products"]["summary"])
c = app.app.test_client(); key = app.dashboard.CONSOLE_SECRET or ""
r = c.get("/api/products/stale", headers={"X-Console-Key": key}); assert r.status_code == 200
print("stale count:", len(r.get_json()["stale"]))
print("PRODUCTS_MODULE_OK")
PY'
rm -rf /tmp/bostest
```
Expected: the products cell shows amber with the stale count (the repo products.json has gk_stale flags) + `PRODUCTS_MODULE_OK`.

Run: `python3 -m pytest tests/test_bos_products.py tests/test_bos_spine.py tests/test_bos_signals.py -q` → green.

- [ ] **Step 4: Commit.** `git add app.py static/console-products.html && git commit -m "feat(bos): /console/products board + products API routes"`

---

## Self-Review
**Spec:** Products module surfaces the enriched catalog + the stale-GK work queue; the Products Home cell goes amber with the count of pages to update; `products.mark_page_fixed` (audited) shrinks the queue and persists on /data. Price-editing deferred (needs a /data override overlay).
**Replaces** the informational products signal from Phase 5 (module_signals).
**Placeholder scan:** none.
**Type consistency:** `stale_pages`/`catalog`/`products_signal`/`mark_page_fixed`, the `{slug,name,reason}` + `{slug,name,ingredients,description,gk_stale}` shapes, and the signal cell shape match the spine/signals contracts.
