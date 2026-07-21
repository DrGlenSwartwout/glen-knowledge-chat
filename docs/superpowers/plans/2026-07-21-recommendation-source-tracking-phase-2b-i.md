# Recommendation Source Tracking — Phase 2b-i (portal backend + operator) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the backend + operator half of the client recommendation portal — per-client preferences (operator note + client note per product, section-collapse state), the token-authed portal read/write endpoints that a client UI will consume, the console operator-note editing, and the process-strip switch to prefer per-line order source. The client-facing portal UI (2b-ii) is a separate follow-on plan that consumes these endpoints.

**Architecture:** A new pure prefs module `dashboard/recommendation_prefs.py` (notes + section-state tables, alongside the existing `recommendation_hidden`). A pure transform `dashboard/portal_recommendations.py::build_sections` groups Phase-1's `product_sources` into collapsible per-source sections (product appears under each source it has, ranked by that source's count desc then recency, top-5 + remainder). Thin Flask endpoints follow the established portal patterns: read via `_portal_record_for` (token→email), writes under `_db_lock`. The console client-360 hub gains operator-note display + edit; `process_strip` reads the latest order's per-line `source` (captured in 2a) in preference to presence detection.

**Tech Stack:** Python 3 / Flask (`app.py`), `dashboard/*.py`, SQLite (`LOG_DB`), pytest (app tests under `doppler run -- python3`), vanilla-JS console page.

## Global Constraints

- **Client identity comes from the portal token**, never an email in the URL. Portal endpoints resolve email via `_portal_record_for(cx, token)` → `portal["email"]` (lowercased); a missing token → `404 {"ok": False, "error": "not found"}`. Writes run under `with _db_lock, sqlite3.connect(LOG_DB) as cx:`.
- **Console endpoints are owner/console-key gated** via `_bos_actor()` (401 when None), like the sibling per-client console endpoints.
- **`client_360.bundle` stays READ-ONLY** — it may READ notes to display them, never write.
- **Notes are separate from the hide flag.** Hide stays in `recommendation_hidden` (Phase 1, via `recommendation_events.set_hidden`); notes + section-state live in the new `recommendation_prefs` tables. Do not migrate or alter `recommendation_hidden` / `product_sources`.
- **Operator note is console-authored only** (read-only to the client); **client note is portal-authored only** (by the client). Two independent fields; setting one must preserve the other.
- **Grouping:** a product appears in EACH source-category section it has an event for; within a section, sort **count DESC, then last_touch DESC**, show **top 5** + a remainder count. Hidden products are excluded from the client view. Section order follows the registry order.
- **CI known_failures ratchet; never run the bare full suite (sends live email).** Run named feature tests via `doppler run -- python3 -m pytest ...` (pure-sqlite tests can use plain `python3`).

---

### Task 1: `recommendation_prefs` module (notes + section-state)

**Files:**
- Create: `dashboard/recommendation_prefs.py`
- Test: `tests/test_recommendation_prefs.py`

**Interfaces:**
- Produces: `init_recommendation_prefs(cx)`; `get_notes(cx, email) -> {product_key: {operator_note, client_note}}`; `set_operator_note(cx, email, product_key, note)`; `set_client_note(cx, email, product_key, note)` (each preserves the other note); `get_section_state(cx, email) -> {section_key: bool}`; `set_section_state(cx, email, section_key, collapsed)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recommendation_prefs.py
import sqlite3
from dashboard import recommendation_prefs as rp


def _cx():
    cx = sqlite3.connect(":memory:")
    rp.init_recommendation_prefs(cx)
    return cx


def test_operator_and_client_notes_independent():
    cx = _cx()
    rp.set_operator_note(cx, "A@B.com", "neuro-magnesium", "take at night")
    rp.set_client_note(cx, "a@b.com", "neuro-magnesium", "helped my sleep")
    n = rp.get_notes(cx, "a@b.com")["neuro-magnesium"]
    assert n["operator_note"] == "take at night"
    assert n["client_note"] == "helped my sleep"
    # updating one preserves the other
    rp.set_operator_note(cx, "a@b.com", "neuro-magnesium", "morning now")
    n = rp.get_notes(cx, "a@b.com")["neuro-magnesium"]
    assert n["operator_note"] == "morning now" and n["client_note"] == "helped my sleep"


def test_notes_empty_and_blank_guards():
    cx = _cx()
    assert rp.get_notes(cx, "nobody@x.com") == {}
    rp.set_client_note(cx, "", "x", "n")          # blank email -> no-op
    rp.set_client_note(cx, "a@b.com", "", "n")    # blank product -> no-op
    assert rp.get_notes(cx, "a@b.com") == {}


def test_section_state_roundtrip():
    cx = _cx()
    assert rp.get_section_state(cx, "a@b.com") == {}
    rp.set_section_state(cx, "A@B.com", "biofield", True)
    rp.set_section_state(cx, "a@b.com", "purchased", False)
    assert rp.get_section_state(cx, "a@b.com") == {"biofield": True, "purchased": False}
    rp.set_section_state(cx, "a@b.com", "biofield", False)   # toggle
    assert rp.get_section_state(cx, "a@b.com")["biofield"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_recommendation_prefs.py -q`
Expected: FAIL — `ModuleNotFoundError: dashboard.recommendation_prefs`.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/recommendation_prefs.py
"""Per-client recommendation preferences: per-(client, product) notes (operator + client)
and per-(client, section) collapse state. Separate from recommendation_events (the log) and
recommendation_hidden (the hide flag). Pure: functions take an open sqlite connection."""
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(e):
    return (e or "").strip().lower()


def init_recommendation_prefs(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_notes (
            client_email  TEXT NOT NULL,
            product_key   TEXT NOT NULL,
            operator_note TEXT NOT NULL DEFAULT '',
            client_note   TEXT NOT NULL DEFAULT '',
            updated_at    TEXT,
            PRIMARY KEY (client_email, product_key)
        )""")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_section_state (
            client_email TEXT NOT NULL,
            section_key  TEXT NOT NULL,
            collapsed    INTEGER NOT NULL DEFAULT 0,
            updated_at   TEXT,
            PRIMARY KEY (client_email, section_key)
        )""")
    cx.commit()


def get_notes(cx, email):
    rows = cx.execute(
        "SELECT product_key, operator_note, client_note FROM recommendation_notes "
        "WHERE client_email=?", (_norm(email),)).fetchall()
    return {r[0]: {"operator_note": r[1] or "", "client_note": r[2] or ""} for r in rows}


def set_operator_note(cx, email, product_key, note):
    _set_note(cx, email, product_key, "operator_note", note)


def set_client_note(cx, email, product_key, note):
    _set_note(cx, email, product_key, "client_note", note)


def _set_note(cx, email, product_key, field, note):
    # field is a fixed internal literal ("operator_note" | "client_note"), never user input.
    assert field in ("operator_note", "client_note")
    e = _norm(email)
    pk = (product_key or "").strip()
    if not e or not pk:
        return
    val = (note or "").strip()
    cx.execute(
        f"INSERT INTO recommendation_notes (client_email, product_key, {field}, updated_at) "
        f"VALUES (?,?,?,?) ON CONFLICT(client_email, product_key) "
        f"DO UPDATE SET {field}=excluded.{field}, updated_at=excluded.updated_at",
        (e, pk, val, _now()))
    cx.commit()


def get_section_state(cx, email):
    rows = cx.execute(
        "SELECT section_key, collapsed FROM recommendation_section_state WHERE client_email=?",
        (_norm(email),)).fetchall()
    return {r[0]: bool(r[1]) for r in rows}


def set_section_state(cx, email, section_key, collapsed):
    e = _norm(email)
    sk = (section_key or "").strip()
    if not e or not sk:
        return
    cx.execute(
        "INSERT INTO recommendation_section_state (client_email, section_key, collapsed, updated_at) "
        "VALUES (?,?,?,?) ON CONFLICT(client_email, section_key) "
        "DO UPDATE SET collapsed=excluded.collapsed, updated_at=excluded.updated_at",
        (e, sk, 1 if collapsed else 0, _now()))
    cx.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_recommendation_prefs.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add dashboard/recommendation_prefs.py tests/test_recommendation_prefs.py
git commit -m "feat(rec): recommendation_prefs (operator/client notes + section-collapse state)"
```

---

### Task 2: `build_sections` transform + portal read endpoint

**Files:**
- Create: `dashboard/portal_recommendations.py`
- Modify: `app.py` (new `GET /api/portal/<token>/recommendations`, near the other `/api/portal/<token>/...` routes ~app.py:19140)
- Test: `tests/test_portal_recommendations_build.py`, `tests/test_portal_recommendations_endpoint.py`

**Interfaces:**
- Consumes: `recommendation_sources.RECOMMENDATION_SOURCES`.
- Produces: `portal_recommendations.build_sections(product_sources, notes, section_state, resolve_product, *, top_n=5) -> list[section]` where a section is `{source, label, icon, collapsed, total, shown, products: [{product_key, name, url, icons:[{source,count,icon,first_touch}], operator_note, client_note}]}`. Hidden products excluded; a product appears in each of its source sections; sorted count desc then last_touch desc; sections in registry order.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_portal_recommendations_build.py
from dashboard import portal_recommendations as pr


def _resolve(slug):
    return {"name": {"neuro-magnesium": "Neuro Magnesium"}.get(slug, slug), "url": "/begin/buy/" + slug}


def test_product_appears_in_each_source_section_sorted():
    product_sources = [
        {"product_key": "neuro-magnesium", "hidden": False, "sources": [
            {"source": "self", "count": 1, "first_touch": "2026-06-01", "last_touch": "2026-06-01"},
            {"source": "biofield", "count": 3, "first_touch": "2026-07-01", "last_touch": "2026-07-20"}]},
        {"product_key": "immune-modulation", "hidden": False, "sources": [
            {"source": "biofield", "count": 1, "first_touch": "2026-07-05", "last_touch": "2026-07-05"}]},
        {"product_key": "hidden-one", "hidden": True, "sources": [
            {"source": "biofield", "count": 9, "first_touch": "2026-07-01", "last_touch": "2026-07-01"}]},
    ]
    secs = pr.build_sections(product_sources, {}, {}, _resolve)
    by = {s["source"]: s for s in secs}
    # biofield section: neuro-magnesium (count 3) before immune-modulation (count 1); hidden excluded
    assert [p["product_key"] for p in by["biofield"]["products"]] == ["neuro-magnesium", "immune-modulation"]
    # self section present with neuro-magnesium
    assert [p["product_key"] for p in by["self"]["products"]] == ["neuro-magnesium"]
    # neuro-magnesium's icon row carries BOTH its sources with counts
    icons = {i["source"]: i["count"] for i in by["biofield"]["products"][0]["icons"]}
    assert icons == {"self": 1, "biofield": 3}
    # section order follows the registry (biofield before self)
    assert [s["source"] for s in secs].index("biofield") < [s["source"] for s in secs].index("self")


def test_notes_and_collapse_attached_top_n():
    product_sources = [{"product_key": f"p{i}", "hidden": False,
                        "sources": [{"source": "purchased", "count": 10 - i,
                                     "first_touch": "d", "last_touch": "d"}]} for i in range(7)]
    notes = {"p0": {"operator_note": "take daily", "client_note": "works"}}
    secs = pr.build_sections(product_sources, notes, {"purchased": True}, lambda s: {"name": s, "url": ""}, top_n=5)
    sec = next(s for s in secs if s["source"] == "purchased")
    assert sec["collapsed"] is True
    assert sec["total"] == 7 and sec["shown"] == 5 and len(sec["products"]) == 5
    assert sec["products"][0]["product_key"] == "p0"
    assert sec["products"][0]["operator_note"] == "take daily" and sec["products"][0]["client_note"] == "works"
```

```python
# tests/test_portal_recommendations_endpoint.py
import sqlite3, hashlib
import app as app_module
from dashboard import recommendation_events as re, client_portal as cp


def test_portal_recommendations_endpoint(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db)
    cp.init_client_portal_table(cx); re.init_recommendation_events(cx)
    # a portal token for the client
    cp.upsert_portal(cx, email="a@b.com", name="Al")   # mints a token; fetch it
    # seed a purchased event
    re.record_event(cx, "a@b.com", "neuro-magnesium", "purchased", occurred_at="2026-07-10", origin_ref="7")
    cx.commit()
    token = cp.get_token_for_email(cx, "a@b.com") if hasattr(cp, "get_token_for_email") else None
    cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    # If a token accessor isn't available, the implementer resolves the token via the upsert return
    # value or the portal test helpers; assert the endpoint returns sections for the client.
    r = c.get(f"/api/portal/{token}/recommendations")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert any(s["source"] == "purchased" and
               any(p["product_key"] == "neuro-magnesium" for p in s["products"]) for s in data["sections"])
```

Note to implementer: mint/resolve the portal token using the real `dashboard/client_portal.py` API (`upsert_portal` returns or stores the raw token; use whatever the existing portal endpoint tests use to obtain a usable token). If no helper exists, insert a `client_portals` row with a known `token_hash = sha256(token)` directly, mirroring `client_portal._hash`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_portal_recommendations_build.py tests/test_portal_recommendations_endpoint.py -q`
Expected: FAIL — module/endpoint missing.

- [ ] **Step 3: Implement**

`dashboard/portal_recommendations.py`:

```python
"""Pure transform: a client's product_sources + prefs -> portal sections. One collapsible
section per source category; a product appears in each source it has an event for, ranked by
that source's count (desc) then recency (desc); top_n shown + a remainder count. Hidden
products excluded. Sections ordered by the registry."""
from dashboard.recommendation_sources import RECOMMENDATION_SOURCES


def build_sections(product_sources, notes, section_state, resolve_product, *, top_n=5):
    by_source = {}
    for p in product_sources:
        if p.get("hidden"):
            continue
        pk = p["product_key"]
        n = notes.get(pk, {})
        prod = resolve_product(pk) or {}
        icons = [{"source": s["source"], "count": s["count"],
                  "icon": (RECOMMENDATION_SOURCES.get(s["source"]) or {}).get("icon", "•"),
                  "first_touch": s.get("first_touch", "")} for s in p["sources"]]
        for s in p["sources"]:
            by_source.setdefault(s["source"], []).append({
                "product_key": pk, "name": prod.get("name") or pk, "url": prod.get("url") or "",
                "icons": icons,
                "operator_note": n.get("operator_note", ""), "client_note": n.get("client_note", ""),
                "_count": s["count"], "_recent": s.get("last_touch", "") or ""})
    out = []
    for key, meta in RECOMMENDATION_SOURCES.items():
        prods = by_source.get(key)
        if not prods:
            continue
        prods.sort(key=lambda e: (e["_count"], e["_recent"]), reverse=True)   # count desc, recency desc
        shown = prods[:top_n]
        out.append({
            "source": key, "label": meta["label"], "icon": meta["icon"],
            "collapsed": bool(section_state.get(key, False)),
            "total": len(prods), "shown": len(shown),
            "products": [{k: v for k, v in e.items() if not k.startswith("_")} for e in shown]})
    return out
```

`app.py` — add near the other `/api/portal/<token>/...` routes (import the modules at the top-of-file dashboard import block or lazily inside):

```python
@app.route("/api/portal/<token>/recommendations", methods=["GET"])
def api_portal_recommendations(token):
    from dashboard import (client_portal as _cp, recommendation_events as _re,
                           recommendation_prefs as _rp, portal_recommendations as _pr,
                           products as _products)
    with sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        _re.init_recommendation_events(cx)
        _rp.init_recommendation_prefs(cx)
        portal = _portal_record_for(cx, token)
        if not portal:
            return jsonify({"ok": False, "error": "not found"}), 404
        email = (portal.get("email") or "").strip().lower()
        ps = _re.product_sources(cx, email)
        notes = _rp.get_notes(cx, email)
        state = _rp.get_section_state(cx, email)
    catalog = _products.load_products()

    def resolve(slug):
        p = catalog.get(slug) or {}
        return {"name": p.get("name"), "url": p.get("url")}

    return jsonify({"ok": True, "sections": _pr.build_sections(ps, notes, state, resolve)})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_portal_recommendations_build.py tests/test_portal_recommendations_endpoint.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add dashboard/portal_recommendations.py app.py tests/test_portal_recommendations_build.py tests/test_portal_recommendations_endpoint.py
git commit -m "feat(rec): portal recommendations read endpoint + section grouping transform"
```

---

### Task 3: Portal write endpoints (hide, client-note, section-collapse)

**Files:**
- Modify: `app.py` (three new `POST /api/portal/<token>/...` routes near Task 2's)
- Test: `tests/test_portal_recommendation_writes.py`

**Interfaces:**
- Produces: `POST /api/portal/<token>/recommendation/hide` `{product_key, hidden}` → `recommendation_events.set_hidden`; `POST /api/portal/<token>/recommendation/client-note` `{product_key, note}` → `recommendation_prefs.set_client_note`; `POST /api/portal/<token>/recommendation/section` `{section_key, collapsed}` → `recommendation_prefs.set_section_state`. Each returns `{"ok": True}` or `404` for an unknown token.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_portal_recommendation_writes.py
import sqlite3
import app as app_module
from dashboard import recommendation_events as re, recommendation_prefs as rp, client_portal as cp


def _seed(tmp_path, monkeypatch):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db)
    cp.init_client_portal_table(cx); re.init_recommendation_events(cx); rp.init_recommendation_prefs(cx)
    cp.upsert_portal(cx, email="a@b.com", name="Al")
    re.record_event(cx, "a@b.com", "neuro-magnesium", "purchased", occurred_at="d", origin_ref="1")
    cx.commit(); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    app_module.app.config["TESTING"] = True
    # implementer: obtain a usable raw token for a@b.com via the client_portal API/helpers
    return db


def test_hide_client_note_and_section_writes(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    token = ...  # implementer resolves the raw token (see Task 2 note)
    assert c.post(f"/api/portal/{token}/recommendation/hide",
                  json={"product_key": "neuro-magnesium", "hidden": True}).get_json()["ok"]
    assert c.post(f"/api/portal/{token}/recommendation/client-note",
                  json={"product_key": "neuro-magnesium", "note": "great"}).get_json()["ok"]
    assert c.post(f"/api/portal/{token}/recommendation/section",
                  json={"section_key": "purchased", "collapsed": True}).get_json()["ok"]
    cx = sqlite3.connect(db)
    assert rp.get_notes(cx, "a@b.com")["neuro-magnesium"]["client_note"] == "great"
    assert rp.get_section_state(cx, "a@b.com")["purchased"] is True
    # hidden product no longer appears in the portal sections
    r = c.get(f"/api/portal/{token}/recommendations").get_json()
    assert all(p["product_key"] != "neuro-magnesium"
               for s in r["sections"] for p in s["products"])


def test_unknown_token_404(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post("/api/portal/not-a-real-token/recommendation/hide", json={"product_key": "x", "hidden": True})
    assert r.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_portal_recommendation_writes.py -q`
Expected: FAIL (routes missing).

- [ ] **Step 3: Implement** (mirror `api_portal_cc_pref`, app.py:19941)

```python
@app.route("/api/portal/<token>/recommendation/hide", methods=["POST"])
def api_portal_rec_hide(token):
    from dashboard import client_portal as _cp, recommendation_events as _re
    data = request.get_json(silent=True) or {}
    pk = (data.get("product_key") or "").strip()
    hidden = bool(data.get("hidden"))
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx); _re.init_recommendation_events(cx)
        portal = _portal_record_for(cx, token)
        if not portal:
            return jsonify({"ok": False, "error": "not found"}), 404
        _re.set_hidden(cx, (portal.get("email") or "").strip().lower(), pk, hidden)
    return jsonify({"ok": True})


@app.route("/api/portal/<token>/recommendation/client-note", methods=["POST"])
def api_portal_rec_client_note(token):
    from dashboard import client_portal as _cp, recommendation_prefs as _rp
    data = request.get_json(silent=True) or {}
    pk = (data.get("product_key") or "").strip()
    note = data.get("note") or ""
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx); _rp.init_recommendation_prefs(cx)
        portal = _portal_record_for(cx, token)
        if not portal:
            return jsonify({"ok": False, "error": "not found"}), 404
        _rp.set_client_note(cx, (portal.get("email") or "").strip().lower(), pk, note)
    return jsonify({"ok": True})


@app.route("/api/portal/<token>/recommendation/section", methods=["POST"])
def api_portal_rec_section(token):
    from dashboard import client_portal as _cp, recommendation_prefs as _rp
    data = request.get_json(silent=True) or {}
    sk = (data.get("section_key") or "").strip()
    collapsed = bool(data.get("collapsed"))
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx); _rp.init_recommendation_prefs(cx)
        portal = _portal_record_for(cx, token)
        if not portal:
            return jsonify({"ok": False, "error": "not found"}), 404
        _rp.set_section_state(cx, (portal.get("email") or "").strip().lower(), sk, collapsed)
    return jsonify({"ok": True})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_portal_recommendation_writes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add app.py tests/test_portal_recommendation_writes.py
git commit -m "feat(rec): token-authed portal writes (hide, client-note, section-collapse)"
```

---

### Task 4: Console operator-note (bundle notes + write endpoint + console edit field)

**Files:**
- Modify: `dashboard/client_360.py` (`_recommendations` merges notes into each product), `app.py` (new `POST /api/console/client/recommendation/operator-note`), `static/console-client.html` (`renderRecommendations` gains an operator-note input + save)
- Test: `tests/test_console_operator_note.py`

**Interfaces:**
- Consumes: `recommendation_prefs.get_notes`, `set_operator_note`; `_bos_actor`.
- Produces: `client_360.bundle`'s `recommendations[]` products now carry `operator_note` + `client_note`; `POST /api/console/client/recommendation/operator-note` `{email, product_key, note}` (console-gated) → `set_operator_note`; the console hub Recommendations rows show + edit the operator note.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console_operator_note.py
import sqlite3
import app as app_module
from dashboard import recommendation_events as re, recommendation_prefs as rp, client_360


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    re.init_recommendation_events(cx); rp.init_recommendation_prefs(cx)
    return cx


def test_bundle_recommendations_include_notes():
    cx = _cx()
    re.record_event(cx, "a@b.com", "neuro-magnesium", "purchased", occurred_at="d", origin_ref="1")
    rp.set_operator_note(cx, "a@b.com", "neuro-magnesium", "night dose")
    b = client_360.bundle(cx, "a@b.com")
    rec = next(p for p in b["recommendations"] if p["product_key"] == "neuro-magnesium")
    assert rec["operator_note"] == "night dose"
    assert "client_note" in rec


def test_operator_note_endpoint_writes(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db); re.init_recommendation_events(cx); rp.init_recommendation_prefs(cx); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "_bos_actor", lambda: object())
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    r = c.post("/api/console/client/recommendation/operator-note",
               json={"email": "a@b.com", "product_key": "neuro-magnesium", "note": "am"})
    assert r.get_json()["ok"] is True
    cx = sqlite3.connect(db)
    assert rp.get_notes(cx, "a@b.com")["neuro-magnesium"]["operator_note"] == "am"


def test_operator_note_endpoint_requires_auth(monkeypatch):
    monkeypatch.setattr(app_module, "_bos_actor", lambda: None)
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    r = c.post("/api/console/client/recommendation/operator-note",
               json={"email": "a@b.com", "product_key": "x", "note": "y"})
    assert r.status_code == 401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_console_operator_note.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `dashboard/client_360.py::_recommendations`, merge notes onto each product (still read-only):

```python
def _recommendations(cx, email):
    from dashboard import recommendation_events, recommendation_prefs
    try:
        prods = recommendation_events.product_sources(cx, email)
        recommendation_prefs.init_recommendation_prefs(cx)
        notes = recommendation_prefs.get_notes(cx, email)
        for p in prods:
            n = notes.get(p["product_key"], {})
            p["operator_note"] = n.get("operator_note", "")
            p["client_note"] = n.get("client_note", "")
        return prods
    except Exception:
        return []
```

In `app.py`, add the console endpoint near `console_client_360`:

```python
@app.route("/api/console/client/recommendation/operator-note", methods=["POST"])
def console_rec_operator_note():
    if _bos_actor() is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    from dashboard import recommendation_prefs as _rp
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    pk = (data.get("product_key") or "").strip()
    if not email or not pk:
        return jsonify({"ok": False, "error": "email and product_key required"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _rp.init_recommendation_prefs(cx)
        _rp.set_operator_note(cx, email, pk, data.get("note") or "")
    return jsonify({"ok": True})
```

In `static/console-client.html::renderRecommendations`, add per product row an operator-note `<input>` prefilled from `p.operator_note`, with a save button that POSTs `{email, product_key, note}` to the endpoint (use the page's existing `key()` header + the current `qsEmail()`), and show the client's own `p.client_note` read-only beside it. Keep it minimal and consistent with the page's existing table markup + `esc()` usage.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_console_operator_note.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add dashboard/client_360.py app.py static/console-client.html tests/test_console_operator_note.py
git commit -m "feat(rec): console operator-note edit + notes in client-360 bundle"
```

---

### Task 5: `process_strip` prefers the current order's per-line source

**Files:**
- Modify: `dashboard/client_360.py::process_strip`
- Test: `tests/test_client_360_process.py` (extend)

**Interfaces:**
- Produces: `process_strip` sets the recommendation stage `sources` from the latest non-cancelled order's per-line `source` values (2a) when that order has any; otherwise falls back to the existing `_detect_sources` presence list. Top-level `source` / `sources` semantics unchanged in shape.

- [ ] **Step 1: Write the failing test** (add to `tests/test_client_360_process.py`)

```python
def test_process_strip_prefers_order_line_source():
    import sqlite3, json
    from dashboard import client_360
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, status TEXT, "
               "pay_status TEXT, invoice_sent_at TEXT, items_json TEXT)")
    cx.execute("CREATE TABLE biofield_reveals (id INTEGER PRIMARY KEY, email TEXT, scan_date TEXT)")
    cx.execute("CREATE TABLE ff_match_drafts (email TEXT, scan_date TEXT, status TEXT)")
    cx.execute("CREATE TABLE intake_responses (email TEXT PRIMARY KEY, status TEXT)")
    cx.execute("CREATE TABLE inquiries (id INTEGER PRIMARY KEY, client_email TEXT)")
    # presence would say biofield, but the current order's lines are self + scan -> those win
    cx.execute("INSERT INTO biofield_reveals VALUES (1,'a@b.com','2026-07-01')")
    cx.execute("INSERT INTO orders (email, status, pay_status, invoice_sent_at, items_json) VALUES (?,?,?,?,?)",
               ("a@b.com", "confirmed", "unpaid", "",
                json.dumps([{"slug": "x", "source": "self"}, {"slug": "y", "source": "scan"}])))
    res = client_360.process_strip(cx, "a@b.com")
    assert set(res["sources"]) == {"self", "scan"}
    rec = next(s for s in res["stages"] if s["key"] == "recommendation")
    assert set(rec["sources"]) == {"self", "scan"}


def test_process_strip_falls_back_to_presence_without_line_source():
    import sqlite3, json
    from dashboard import client_360
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, status TEXT, "
               "pay_status TEXT, invoice_sent_at TEXT, items_json TEXT)")
    cx.execute("CREATE TABLE biofield_reveals (id INTEGER PRIMARY KEY, email TEXT, scan_date TEXT)")
    cx.execute("CREATE TABLE ff_match_drafts (email TEXT, scan_date TEXT, status TEXT)")
    cx.execute("CREATE TABLE intake_responses (email TEXT PRIMARY KEY, status TEXT)")
    cx.execute("CREATE TABLE inquiries (id INTEGER PRIMARY KEY, client_email TEXT)")
    cx.execute("INSERT INTO biofield_reveals VALUES (1,'a@b.com','2026-07-01')")
    cx.execute("INSERT INTO orders (email, status, pay_status, invoice_sent_at, items_json) VALUES (?,?,?,?,?)",
               ("a@b.com", "confirmed", "unpaid", "", json.dumps([{"slug": "x"}])))   # no source
    res = client_360.process_strip(cx, "a@b.com")
    assert res["sources"] == ["biofield"]        # presence fallback
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_client_360_process.py::test_process_strip_prefers_order_line_source -q`
Expected: FAIL (sources come from presence, not the order lines).

- [ ] **Step 3: Implement**

In `dashboard/client_360.py::process_strip`, after loading `order` (extend the SELECT to also fetch `items_json`), derive line sources and prefer them:

```python
    order = cx.execute(
        "SELECT id, COALESCE(status,'') status, COALESCE(pay_status,'') pay, "
        "COALESCE(invoice_sent_at,'') sent, COALESCE(items_json,'[]') items FROM orders "
        "WHERE lower(COALESCE(email,''))=? AND COALESCE(status,'')<>'cancelled' "
        "ORDER BY id DESC LIMIT 1", (e,)).fetchone()
    line_sources = []
    if order:
        try:
            for ln in (json.loads(order["items"]) or []):
                sk = (ln.get("source") or "").strip()
                if sk and sk not in line_sources:
                    line_sources.append(sk)
        except Exception:
            line_sources = []
    sources = line_sources or _detect_sources(cx, e)
    source = sources[0] if sources else None
```

(Keep the rest of `process_strip` unchanged; `json` is already imported in the module.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_client_360_process.py -q`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add dashboard/client_360.py tests/test_client_360_process.py
git commit -m "feat(rec): process strip prefers current order's per-line source over presence"
```

---

## Self-review checklist (controller, before dispatch)

- Prefs isolated from Phase 1 (`recommendation_hidden`/`product_sources` untouched); notes independent (setting one preserves the other).
- Portal endpoints resolve identity via token only; console endpoint gated by `_bos_actor`; `bundle` stays read-only.
- Grouping: product per source-section, count-desc/recency, top-5, hidden excluded, registry order.
- `process_strip` prefers line source, falls back to presence — existing tests still green.

## Not in 2b-i (the 2b-ii follow-on plan)

The client-facing portal UI in `static/client-portal.html` — a "My Recommendations" section that fetches `/api/portal/<token>/recommendations` and renders the collapsible sections (remembered state via the section endpoint), icon rows + counts, per-product hide control, operator note (read-only) + client note (editable via the client-note endpoint). Product images are unavailable in the catalog (only `name` + `url`), so the UI is text/icon-based with a buy link. Consider the frontend-design skill for that plan.
