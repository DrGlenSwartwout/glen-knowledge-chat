# Biofield Intake → Client Portal Publish Connector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One "Publish to portal" action in the local Biofield Intake tool that turns an authored intake report into the illtowell.com client-portal `content` shape and POSTs it to the prod `/admin/portal/upsert`, returning the `/portal/<token>` URL for Glen to send.

**Architecture:** A new pure builder module `dashboard/biofield_portal_publish.py` (none-raising, cx-based, offline-testable) maps `authored_report` + narrative → portal content (layers with narrative-segmented meaning, remedies deduped to slugs at a flat courtesy price, status `confirmed`). A new local route `POST /test/<id>/publish-portal` builds the payload and POSTs it to prod via an injectable `publish_to_portal`, returning the URL; a "Publish to portal" button on the report view calls it. PHI stays local; only the finished payload crosses to prod (same trust path as reveal-push).

**Tech Stack:** Python 3.11, Flask (local app), sqlite3, `requests`, pytest. Reuses `dashboard.practitioner_portal.name_to_slug`, `dashboard.wholesale_pricing._load_catalog`, `dashboard.biofield_authoring.authored_report`, `dashboard.biofield_narrative.get_narrative`.

## Global Constraints

- New module is **pure / none-raising** where the spec says so; cx-based; offline-testable (tmp sqlite, no network — the prod POST is injected/mocked).
- **No changes** to the prod app, `client-portal.html`, or `/admin/portal/upsert`. The connector only *calls* the existing endpoint.
- The upsert call always sends `send: false` (Glen emails the link himself).
- `biofield_status` is always `"confirmed"` (authored → un-blurred).
- Courtesy price is a flat per-bottle `special_price_cents` passed at publish, applied to every reorder line.
- Unresolved remedy slugs are surfaced, never silently dropped (route returns 409 with the list; no partial publish).
- Alias overrides (normalized, alphanumeric-only key) are applied BEFORE fuzzy resolution:
  `focusneuromagnesium → neuro-magnesium`, `communityspiritformulainterrainrestore → terrain-restore`.
- Catalog is the slug-keyed dict from `wholesale_pricing._load_catalog()` (i.e. `data/products.json`'s `products` map); `name_to_slug(name, catalog)` takes that dict.
- Tests run with: `~/.venvs/deploy-chat311/bin/python -m pytest tests/<file> -v` (pure-module tests). Route tests build the app via `biofield_local_app.create_app(db_path=<tmp>)` and run offline.

---

### Task 1: Slug resolution (aliases + fuzzy)

**Files:**
- Create: `dashboard/biofield_portal_publish.py`
- Test: `tests/test_biofield_portal_publish_resolve.py`

**Interfaces:**
- Consumes: `dashboard.practitioner_portal.name_to_slug(name, catalog) -> str|None`; `dashboard.wholesale_pricing._load_catalog() -> dict[str,dict]`.
- Produces: `ALIAS_SLUGS: dict[str,str]`; `_norm_key(s) -> str`; `resolve_remedy_slug(name, catalog) -> str|None`; `load_catalog() -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_portal_publish_resolve.py
from dashboard import biofield_portal_publish as bpp

CATALOG = {
    "vitality":       {"name": "Vitality"},
    "chelation":      {"name": "Chelation"},
    "nous-energy":    {"name": "Nous Energy"},
    "neuro-magnesium":{"name": "Neuro Magnesium"},
    "terrain-restore":{"name": "Terrain Restore"},
}

def test_alias_overrides_take_precedence():
    assert bpp.resolve_remedy_slug("Focus, Neuromagnesium", CATALOG) == "neuro-magnesium"
    assert bpp.resolve_remedy_slug("Focus Neuro-Magnesium", CATALOG) == "neuro-magnesium"
    assert bpp.resolve_remedy_slug(
        "Community Spirit Formula in Terrain Restore", CATALOG) == "terrain-restore"

def test_exact_names_resolve_via_name_to_slug():
    assert bpp.resolve_remedy_slug("Vitality", CATALOG) == "vitality"
    assert bpp.resolve_remedy_slug("Chelation", CATALOG) == "chelation"
    assert bpp.resolve_remedy_slug("Nous Energy", CATALOG) == "nous-energy"

def test_unresolvable_returns_none():
    assert bpp.resolve_remedy_slug("Totally Invented Remedy XYZ", CATALOG) is None
    assert bpp.resolve_remedy_slug("", CATALOG) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_portal_publish_resolve.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.biofield_portal_publish'`.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/biofield_portal_publish.py
"""Publish an authored Biofield Intake report to the illtowell.com client portal.

Pure / none-raising builder + an injectable prod POST. PHI stays local; only the
finished portal payload crosses to prod via the existing /admin/portal/upsert.
"""
import re

from dashboard.practitioner_portal import name_to_slug
from dashboard import wholesale_pricing as _pricing

# Protocol wordings that differ from the catalog. Keyed by alphanumeric-only,
# lowercased remedy text so "Focus, Neuromagnesium" and "Focus Neuro-Magnesium"
# collapse to the same key.
ALIAS_SLUGS = {
    "focusneuromagnesium": "neuro-magnesium",
    "communityspiritformulainterrainrestore": "terrain-restore",
}


def _norm_key(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def load_catalog():
    """The slug-keyed products map (data/products.json 'products')."""
    return _pricing._load_catalog()


def resolve_remedy_slug(name, catalog):
    """Resolve a protocol remedy name to a catalog slug: alias override first,
    then the in-repo fuzzy resolver. None when genuinely unresolvable."""
    if not (name or "").strip():
        return None
    alias = ALIAS_SLUGS.get(_norm_key(name))
    if alias:
        return alias
    return name_to_slug(name, catalog)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_portal_publish_resolve.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_portal_publish.py tests/test_biofield_portal_publish_resolve.py
git commit -m "feat(portal-publish): remedy slug resolution with alias overrides"
```

---

### Task 2: Dosing string + narrative segmentation

**Files:**
- Modify: `dashboard/biofield_portal_publish.py`
- Test: `tests/test_biofield_portal_publish_text.py`

**Interfaces:**
- Produces: `_dosing(layer: dict) -> str`; `segment_narrative(narrative: str, layers: list[dict]) -> list[str]` (returns a list aligned to `layers`, or `[]` when it cannot align 1:1).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_portal_publish_text.py
from dashboard import biofield_portal_publish as bpp

def test_dosing_joins_present_fields_and_skips_blanks():
    assert bpp._dosing({"dosage": "1 capsule", "frequency": "daily",
                        "timing": "with food"}) == "1 capsule daily with food"
    assert bpp._dosing({"dosage": "10 drops", "frequency": "", "timing": ""}) == "10 drops"
    assert bpp._dosing({"dosage": "", "frequency": "", "timing": ""}) == ""

def test_segment_narrative_splits_layer_by_layer():
    layers = [{"remedy": "Vitality", "head": "ED3 Cell Driver"},
              {"remedy": "Chelation", "head": "Kidney"},
              {"remedy": "Nous Energy", "head": "Kidney"}]
    narr = ("Aloha Karin. The surface layer needs Vitality to restore energy. "
            "Next, Chelation clears the burden. Finally, Nous Energy steadies the mind.")
    segs = bpp.segment_narrative(narr, layers)
    assert len(segs) == 3
    assert "Vitality" in segs[0]
    assert "Chelation" in segs[1]
    assert "Nous Energy" in segs[2]

def test_segment_narrative_returns_empty_when_not_alignable():
    layers = [{"remedy": "Vitality", "head": "ED3"},
              {"remedy": "Chelation", "head": "Kidney"}]
    # Narrative never mentions the second remedy/head -> cannot align 1:1.
    assert bpp.segment_narrative("A generic message with no cues at all.", layers) == []
    assert bpp.segment_narrative("", layers) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_portal_publish_text.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_dosing'`.

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/biofield_portal_publish.py`:

```python
def _dosing(layer):
    parts = [(layer.get("dosage") or "").strip(),
             (layer.get("frequency") or "").strip(),
             (layer.get("timing") or "").strip()]
    return " ".join(p for p in parts if p)


def _cue_candidates(layer):
    """Ordered phrases to locate this layer in the narrative blob."""
    rem = (layer.get("remedy") or "").strip()
    out = []
    if rem:
        out.append(rem)
        first = rem.split(",")[0].strip()      # "Focus, Neuromagnesium" -> "Focus"
        if first and first != rem:
            out.append(first)
    head = (layer.get("head") or "").strip()
    if head:
        out.append(head)
    return out


def segment_narrative(narrative, layers):
    """Split the single narrative blob into one segment per layer, by locating
    each layer's cue (remedy, else its first word, else head) in increasing
    order. Returns a list aligned to ``layers``; ``[]`` when it cannot align."""
    text = narrative or ""
    if not text or not layers:
        return []
    low = text.lower()
    positions = []
    cursor = 0
    for layer in layers:
        found = -1
        for cue in _cue_candidates(layer):
            idx = low.find(cue.lower(), cursor)
            if idx != -1:
                found = idx
                break
        if found == -1:
            return []                          # a layer has no cue -> fall back
        positions.append(found)
        cursor = found + 1
    # positions are strictly increasing by construction (each search starts past
    # the previous hit). Slice between consecutive cue starts.
    segs = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        segs.append(text[start:end].strip())
    return segs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_portal_publish_text.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_portal_publish.py tests/test_biofield_portal_publish_text.py
git commit -m "feat(portal-publish): dosing string + narrative segmentation with fallback"
```

---

### Task 3: build_portal_content

**Files:**
- Modify: `dashboard/biofield_portal_publish.py`
- Test: `tests/test_biofield_portal_publish_build.py`

**Interfaces:**
- Consumes: `dashboard.biofield_authoring.authored_report(cx, test_id)`, `add_chain_row`, `create_test`; `dashboard.biofield_narrative.save_narrative`/`get_narrative`; Task 1/2 helpers.
- Produces: `build_portal_content(cx, test_id, *, special_price_cents, catalog=None) -> dict` with keys `email, name, scan_date, scan_id, content, unresolved`.

**Note on `authored_report` layer fields:** each layer dict has `layer` (int), `head`, `most_affected`, `remedy`, `dosage`, `frequency`, `timing`, `rid`. `authored_report(cx, tid)` returns `{test_id, client:{name,email}, date, layers:[...], schedule}`. Test IDs passed to authoring functions use the `"a<id>"` form (e.g. `"a1"`); `_num()` strips the prefix internally.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_portal_publish_build.py
import sqlite3
from dashboard import biofield_portal_publish as bpp
from dashboard.biofield_authoring import create_test, add_chain_row
from dashboard.biofield_narrative import save_narrative

CATALOG = {
    "vitality":       {"name": "Vitality"},
    "chelation":      {"name": "Chelation"},
    "nous-energy":    {"name": "Nous Energy"},
    "neuro-magnesium":{"name": "Neuro Magnesium"},
    "terrain-restore":{"name": "Terrain Restore"},
}

def _seed_karin(cx):
    tid = create_test(cx, "Karin Takahashi", "permanentlyyours777@hawaiiantel.net",
                      "2026-06-25")
    aid = f"a{tid}"
    add_chain_row(cx, aid, layer=1, head="ED3 Cell Driver", most_affected="Circulation",
                  remedy="Vitality", dosage="1 capsule", frequency="daily", timing="with food")
    add_chain_row(cx, aid, layer=2, head="EI6 Kidney pH", most_affected="Kidney",
                  remedy="Chelation", dosage="1 capsule", frequency="daily", timing="")
    add_chain_row(cx, aid, layer=2, head="EI6 Kidney pH", most_affected="Kidney",
                  remedy="Nous Energy", dosage="one a day", frequency="", timing="")
    add_chain_row(cx, aid, layer=3, head="EI10 Circulation", most_affected="Heart",
                  remedy="Focus, Neuromagnesium", dosage="two scoops", frequency="a day", timing="")
    add_chain_row(cx, aid, layer=4, head="Psychoemotional", most_affected="Psychoemotional",
                  remedy="Community Spirit Formula in Terrain Restore",
                  dosage="10 drops", frequency="3 times a day", timing="before meals")
    return aid

def test_build_maps_layers_dedups_and_prices(tmp_path):
    cx = sqlite3.connect(":memory:")
    aid = _seed_karin(cx)
    out = bpp.build_portal_content(cx, aid, special_price_cents=5000, catalog=CATALOG)

    assert out["email"] == "permanentlyyours777@hawaiiantel.net"
    assert out["name"] == "Karin Takahashi"
    assert out["scan_date"] == "2026-06-25"
    assert out["unresolved"] == []
    c = out["content"]
    assert c["biofield_status"] == "confirmed"
    # 5 chain rows -> 5 walkthrough layers
    assert len(c["layers"]) == 5
    l0 = c["layers"][0]
    assert l0["n"] == 1 and l0["title"] == "ED3 Cell Driver" and l0["remedy"] == "Vitality"
    assert l0["dosing"] == "1 capsule daily with food"
    # reorder deduped to 5 unique slugs (Focus,Neuromagnesium -> one neuro-magnesium line)
    slugs = [it["slug"] for it in c["reorder_items"]]
    assert sorted(slugs) == ["chelation", "neuro-magnesium", "nous-energy",
                             "terrain-restore", "vitality"]
    assert all(it["price_cents"] == 5000 and it["qty"] == 1 for it in c["reorder_items"])

def test_build_meaning_from_narrative_segments(tmp_path):
    cx = sqlite3.connect(":memory:")
    aid = _seed_karin(cx)
    save_narrative(cx, aid,
        "Aloha Karin. Vitality restores your surface energy. Chelation clears the burden. "
        "Nous Energy steadies you. Focus, Neuromagnesium sharpens you. "
        "Community Spirit Formula in Terrain Restore holds your heart.")
    out = bpp.build_portal_content(cx, aid, special_price_cents=5000, catalog=CATALOG)
    assert "Vitality" in out["content"]["layers"][0]["meaning"]
    assert out["content"]["greeting"].startswith("Aloha")

def test_build_unresolved_remedy_is_reported_not_published(tmp_path):
    cx = sqlite3.connect(":memory:")
    tid = create_test(cx, "Test One", "t@example.com", "2026-06-25")
    aid = f"a{tid}"
    add_chain_row(cx, aid, layer=1, head="X", most_affected="X",
                  remedy="Invented Remedy ZZZ", dosage="1", frequency="daily", timing="")
    out = bpp.build_portal_content(cx, aid, special_price_cents=5000, catalog=CATALOG)
    assert out["unresolved"] == ["Invented Remedy ZZZ"]
    assert out["content"]["reorder_items"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_portal_publish_build.py -v`
Expected: FAIL — `AttributeError: ... 'build_portal_content'`.

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/biofield_portal_publish.py` (add `from dashboard.biofield_authoring import authored_report` and `from dashboard.biofield_narrative import get_narrative` at the top with the other imports):

```python
def build_portal_content(cx, test_id, *, special_price_cents, catalog=None):
    """Map an authored intake report to the portal content payload.

    Returns {email, name, scan_date, scan_id, content, unresolved}. Never raises
    on missing narrative (falls back to greeting=full narrative, blank meanings)."""
    cat = catalog if catalog is not None else load_catalog()
    rep = authored_report(cx, test_id)
    raw_layers = rep.get("layers") or []
    client = rep.get("client") or {}
    name = (client.get("name") or "").strip()
    first = name.split()[0] if name else ""

    narrative = get_narrative(cx, test_id) or ""
    segs = segment_narrative(narrative, raw_layers)
    if segs:
        greeting = f"Aloha {first}," if first else "Aloha,"
        meanings = segs
    else:
        greeting = narrative or (f"Aloha {first}," if first else "Aloha,")
        meanings = [""] * len(raw_layers)

    layers, reorder, seen, unresolved = [], [], set(), []
    for i, L in enumerate(raw_layers):
        remedy = (L.get("remedy") or "").strip()
        layers.append({
            "n": L.get("layer"),
            "title": (L.get("head") or "").strip(),
            "meaning": meanings[i] if i < len(meanings) else "",
            "remedy": remedy,
            "dosing": _dosing(L),
        })
        if not remedy:
            continue
        slug = resolve_remedy_slug(remedy, cat)
        if slug is None:
            if remedy not in unresolved:
                unresolved.append(remedy)
            continue
        if slug in seen:
            continue
        seen.add(slug)
        reorder.append({"slug": slug, "qty": 1, "price_cents": int(special_price_cents)})

    content = {
        "greeting": greeting,
        "video": {"url": "", "label": "Watch your message from Dr. Glen"},
        "layers": layers,
        "reorder_items": reorder,
        "pricing_note": "",
        "findings": [],
        "biofield_status": "confirmed",
    }
    return {
        "email": (client.get("email") or "").strip().lower(),
        "name": name,
        "scan_date": rep.get("date") or "",
        "scan_id": "",
        "content": content,
        "unresolved": unresolved,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_portal_publish_build.py -v`
Expected: PASS (3 tests). If `add_chain_row`'s signature differs, read `dashboard/biofield_authoring.py` and adjust the seed calls — do not change the asserted behavior.

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_portal_publish.py tests/test_biofield_portal_publish_build.py
git commit -m "feat(portal-publish): build_portal_content from authored intake report"
```

---

### Task 4: publish_to_portal (prod POST, injectable)

**Files:**
- Modify: `dashboard/biofield_portal_publish.py`
- Test: `tests/test_biofield_portal_publish_post.py`

**Interfaces:**
- Produces: `publish_to_portal(payload, *, base_url, console_key, http_post=None) -> dict`. POSTs `{**payload, "send": False}` to `{base_url}/admin/portal/upsert` with header `X-Console-Key: console_key`; returns parsed JSON; raises `RuntimeError` on non-2xx.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_portal_publish_post.py
import json
import pytest
from dashboard import biofield_portal_publish as bpp

class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)
    def json(self):
        return self._body

def test_publish_posts_with_key_and_send_false_and_returns_json():
    captured = {}
    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _Resp(200, {"ok": True, "url": "https://illtowell.com/portal/abc",
                           "token": "abc"})
    out = bpp.publish_to_portal(
        {"email": "k@example.com", "name": "K", "content": {}, "scan_date": "2026-06-25"},
        base_url="https://illtowell.com", console_key="secret", http_post=fake_post)
    assert out["url"] == "https://illtowell.com/portal/abc"
    assert captured["url"] == "https://illtowell.com/admin/portal/upsert"
    assert captured["headers"]["X-Console-Key"] == "secret"
    assert captured["json"]["send"] is False
    assert captured["json"]["email"] == "k@example.com"

def test_publish_raises_on_non_2xx():
    def fake_post(url, json=None, headers=None, timeout=None):
        return _Resp(401, {"error": "unauthorized"})
    with pytest.raises(RuntimeError):
        bpp.publish_to_portal({"email": "k@example.com"}, base_url="https://x",
                              console_key="bad", http_post=fake_post)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_portal_publish_post.py -v`
Expected: FAIL — `AttributeError: ... 'publish_to_portal'`.

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/biofield_portal_publish.py` (add `import requests` at the top):

```python
def publish_to_portal(payload, *, base_url, console_key, http_post=None):
    """POST the portal payload to the prod /admin/portal/upsert (send disabled).
    Returns the parsed JSON (contains url/token). Raises RuntimeError on non-2xx."""
    post = http_post or requests.post
    url = f"{base_url.rstrip('/')}/admin/portal/upsert"
    body = {**payload, "send": False}
    r = post(url, json=body, headers={"X-Console-Key": console_key}, timeout=30)
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"portal upsert failed {r.status_code}: {r.text[:300]}")
    return r.json()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_portal_publish_post.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_portal_publish.py tests/test_biofield_portal_publish_post.py
git commit -m "feat(portal-publish): injectable prod upsert POST"
```

---

### Task 5: Local route + "Publish to portal" button

**Files:**
- Modify: `biofield_local_app.py` (add a route inside `create_app`, before the final `return app` at line ~707; add the button to the report/author view template)
- Test: `tests/test_biofield_portal_publish_route.py`

**Interfaces:**
- Consumes: `dashboard.biofield_portal_publish.build_portal_content`, `publish_to_portal`; env `PORTAL_PUBLISH_BASE_URL`, `CONSOLE_SECRET`.
- Produces: `POST /test/<test_id>/publish-portal` → JSON. On unresolved remedies: `409 {"ok": False, "unresolved": [...]}`. On success: `200 {"ok": True, "url": ..., "unresolved": []}`.

**Read first:** `tests/test_biofield_local_app.py` for the existing offline app-test pattern (how it builds `create_app(db_path=...)` and a Flask test client, and how it seeds a test). Match it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_portal_publish_route.py
import sqlite3
import biofield_local_app
from dashboard import biofield_portal_publish as bpp
from dashboard.biofield_authoring import create_test, add_chain_row

def _client(tmp_path):
    db = str(tmp_path / "t.db")
    cx = sqlite3.connect(db)
    tid = create_test(cx, "Karin", "k@example.com", "2026-06-25")
    aid = f"a{tid}"
    add_chain_row(cx, aid, layer=1, head="ED3", most_affected="Circ",
                  remedy="Vitality", dosage="1 cap", frequency="daily", timing="")
    cx.commit(); cx.close()
    app = biofield_local_app.create_app(db_path=db)
    return app.test_client(), aid

def test_publish_route_success(tmp_path, monkeypatch):
    monkeypatch.setattr(bpp, "load_catalog", lambda: {"vitality": {"name": "Vitality"}})
    monkeypatch.setattr(bpp, "publish_to_portal",
                        lambda payload, **kw: {"ok": True,
                                               "url": "https://illtowell.com/portal/xyz"})
    monkeypatch.setenv("PORTAL_PUBLISH_BASE_URL", "https://illtowell.com")
    monkeypatch.setenv("CONSOLE_SECRET", "")     # gate open in tests
    client, aid = _client(tmp_path)
    r = client.post(f"/test/{aid}/publish-portal", json={"special_price_cents": 5000})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["url"] == "https://illtowell.com/portal/xyz"

def test_publish_route_409_on_unresolved(tmp_path, monkeypatch):
    monkeypatch.setattr(bpp, "load_catalog", lambda: {})   # nothing resolves
    called = {"n": 0}
    monkeypatch.setattr(bpp, "publish_to_portal",
                        lambda payload, **kw: called.__setitem__("n", called["n"] + 1))
    monkeypatch.setenv("CONSOLE_SECRET", "")
    client, aid = _client(tmp_path)
    r = client.post(f"/test/{aid}/publish-portal", json={"special_price_cents": 5000})
    assert r.status_code == 409
    assert r.get_json()["unresolved"] == ["Vitality"]
    assert called["n"] == 0      # no publish attempted
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_portal_publish_route.py -v`
Expected: FAIL — 404 (route not registered).

- [ ] **Step 3: Write minimal implementation**

Add this route inside `create_app` in `biofield_local_app.py` (place it next to the other `/test/<test_id>/...` routes, before `return app`):

```python
    @app.route("/test/<test_id>/publish-portal", methods=["POST"])
    def publish_portal(test_id):
        from dashboard import biofield_portal_publish as _bpp
        body = request.get_json(silent=True) or {}
        try:
            special = int(body.get("special_price_cents") or 0)
        except (TypeError, ValueError):
            return {"ok": False, "error": "special_price_cents must be an integer"}, 400
        with sqlite3.connect(db_path) as cx:
            payload = _bpp.build_portal_content(cx, test_id, special_price_cents=special)
        if payload["unresolved"]:
            return {"ok": False, "unresolved": payload["unresolved"]}, 409
        base = os.environ.get("PORTAL_PUBLISH_BASE_URL", "")
        key = os.environ.get("CONSOLE_SECRET", "")
        if not base:
            return {"ok": False, "error": "PORTAL_PUBLISH_BASE_URL not set"}, 500
        try:
            res = _bpp.publish_to_portal(payload, base_url=base, console_key=key)
        except Exception as e:
            return {"ok": False, "error": str(e)[:300]}, 502
        return {"ok": True, "url": res.get("url", ""), "unresolved": []}
```

Then add the button to the report/author view. Find where the report view template renders its action buttons (search the file for the audio button / `make_audio` UI, e.g. `grep -n "audio" biofield_local_app.py` near the HTML), and add alongside it:

```html
<button onclick="publishPortal()">Publish to portal</button>
<span id="portal-url"></span>
<script>
async function publishPortal() {
  const cents = parseInt(prompt("Courtesy price per bottle, in cents (e.g. 5000 = $50)", "5000"), 10);
  if (!cents) return;
  const r = await fetch(`/test/${TEST_ID}/publish-portal`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({special_price_cents: cents})});
  const d = await r.json();
  const el = document.getElementById("portal-url");
  if (d.ok) { el.innerHTML = `<a href="${d.url}" target="_blank">${d.url}</a> (copy into her email)`; }
  else if (d.unresolved) { el.textContent = "Unresolved remedies (fix names): " + d.unresolved.join(", "); }
  else { el.textContent = "Error: " + (d.error || "publish failed"); }
}
</script>
```

Use the same `TEST_ID` JS variable the existing report-view buttons use; if the view uses a different accessor for the test id, match it. The button is exercised manually at go-live; the route is the tested unit.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_portal_publish_route.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the whole module suite + commit**

```bash
~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_portal_publish_resolve.py tests/test_biofield_portal_publish_text.py tests/test_biofield_portal_publish_build.py tests/test_biofield_portal_publish_post.py tests/test_biofield_portal_publish_route.py -v
git add biofield_local_app.py tests/test_biofield_portal_publish_route.py
git commit -m "feat(portal-publish): local publish route + report-view button"
```

---

## Self-Review

**1. Spec coverage:**
- Portal content shape → Task 3 (`build_portal_content`). ✅
- Module `biofield_portal_publish` (load_catalog, ALIAS_SLUGS, resolve_remedy_slug, _dosing, segment_narrative, build_portal_content, publish_to_portal) → Tasks 1–4. ✅
- Local route + button, 409-on-unresolved, no partial publish → Task 5. ✅
- biofield_status confirmed, send:false, flat courtesy price, alias overrides, dedup-by-slug, narrative segmentation + fallback, unresolved surfaced → covered in Tasks 1/3/4 + Global Constraints. ✅
- Non-goals (no auto-email, no findings, no prod-app change) → respected (findings `[]`, send `False`, only calls existing endpoint). ✅

**2. Placeholder scan:** No TBD/TODO; every code step has complete code. The two "match the existing pattern" notes (add_chain_row signature in Task 3; report-view button accessor + app-test pattern in Task 5) point at concrete files to read, not vague instructions. ✅

**3. Type consistency:** `resolve_remedy_slug(name, catalog)`, `_dosing(layer)`, `segment_narrative(narrative, layers)`, `build_portal_content(cx, test_id, *, special_price_cents, catalog=None)`, `publish_to_portal(payload, *, base_url, console_key, http_post=None)` — names/signatures identical across tasks and the route. `content` keys match the spec's target shape and what `/api/portal` + `/admin/portal/upsert` consume. ✅
