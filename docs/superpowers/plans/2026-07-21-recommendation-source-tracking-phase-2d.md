# Recommendation Source Tracking — Phase 2d (reveal/engagement click capture) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Count `biofield` and `scan` as real client ACTIONS — a `biofield` event when the client clicks a remedy's product link in their biofield reveal or orders from the reveal, and a `scan` event when they click a scan/FF-matched product in their portal — via `recommendation_events` with a per-click (counting) origin_ref. Chat is deferred (blocked — see Not in 2d).

**Architecture:** Both action surfaces already resolve the client email from a token (reveal token for biofield; portal token for scan), so this reuses the established token-authed write pattern (`_portal_record_for` / `_biofield_verify_token` → email → `record_event` under `_db_lock`). Two server hooks (reveal order → biofield; a token-authed click endpoint per surface) plus fire-and-forget `navigator.sendBeacon` POST-before-navigate on the product links. A shared `record_click` helper uses a UNIQUE origin_ref (each action counts), deliberately unlike `self` (sticky) and `purchased` (per-order).

**Tech Stack:** Python 3 / Flask (`app.py`), `dashboard/recommendation_events.py`, SQLite (`LOG_DB`), pytest (app-importing tests under `doppler run -- python3`), vanilla JS in `static/begin-biofield.html` + `static/client-portal.html`.

## Global Constraints

- **Each action COUNTS: unique origin_ref.** `record_click` uses `origin_ref=_now()` (microsecond ISO timestamp) so every genuine click/order records a distinct event (the "actions, not views" counting rule). This is the opposite of `record_self`'s stable `origin_ref="self"`.
- **Email from the surface token only.** biofield: `_biofield_verify_token(th)` → `row["email"]`; scan/portal: `_portal_record_for(cx, token)` → email. Never an email from the request body/query. An unresolved token → no event (and the click endpoint 404s / no-ops).
- **Failure-isolated + non-blocking.** Server emissions are `try/except: pass` (never break a checkout or a page). Client capture is `navigator.sendBeacon` (fire-and-forget — never blocks or delays the client's navigation to the product).
- **Reuse.** `record_event` + the token-authed write template (`api_portal_rec_hide`, `begin_biofield_request_review`). `biofield`/`scan` are already registered engagement sources. No new storage.
- **Distinct from purchased.** Ordering from the reveal already emits a `purchased` event (via the order pipeline) — the `biofield` event 2d adds is the *acted-on-by-ordering* signal, a different `source_key`, so no collision/double-count.
- **CI known_failures ratchet; never the bare full suite (sends live email).** Reviewer: check nothing outside the diff pins the touched handlers/pages (an app.py/HTML wiring change caught a cross-file guard-test break earlier this project — the full-suite ratchet is the backstop).

---

### Task 1: `record_click` helper (per-action, counting)

**Files:**
- Modify: `dashboard/recommendation_events.py`
- Test: `tests/test_recommendation_click.py`

**Interfaces:**
- Produces: `record_click(cx, email, product_key, source_key) -> bool` — `record_event(cx, email, product_key, source_key, occurred_at=_now(), origin_ref=_now())`. Unique origin_ref per call → each action counts.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recommendation_click.py
import sqlite3, time
from dashboard import recommendation_events as re


def _cx():
    cx = sqlite3.connect(":memory:")
    re.init_recommendation_events(cx)
    return cx


def test_each_click_counts():
    cx = _cx()
    assert re.record_click(cx, "A@B.com", "neuro-magnesium", "biofield") is True
    time.sleep(0.001)   # ensure a distinct microsecond timestamp
    assert re.record_click(cx, "a@b.com", "neuro-magnesium", "biofield") is True
    ev = [e for e in re.list_events(cx, "a@b.com") if e["source_key"] == "biofield"]
    assert len(ev) == 2          # two clicks -> two events (unlike sticky self)


def test_click_blank_guard_and_source():
    cx = _cx()
    assert re.record_click(cx, "a@b.com", "", "scan") is False   # blank slug -> no-op
    re.record_click(cx, "a@b.com", "immune-modulation", "scan")
    ev = re.list_events(cx, "a@b.com")
    assert ev[-1]["source_key"] == "scan"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_recommendation_click.py -q`
Expected: FAIL — `AttributeError: ... 'record_click'`.

- [ ] **Step 3: Implement** (append to `dashboard/recommendation_events.py`)

```python
def record_click(cx, email, product_key, source_key):
    """A client took an ACTION on a product from a <source_key> surface (clicked its
    link, or ordered from it). Each action counts — a UNIQUE origin_ref per call
    (timestamp), deliberately unlike record_self's sticky origin_ref."""
    return record_event(cx, email, product_key, source_key,
                        occurred_at=_now(), origin_ref=_now())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_recommendation_click.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add dashboard/recommendation_events.py tests/test_recommendation_click.py
git commit -m "feat(rec): record_click helper (per-action counting event)"
```

---

### Task 2: Ordering from the biofield reveal emits `biofield` events

**Files:**
- Modify: `app.py::begin_biofield_order_checkout` (~line 3796, after a successful `_checkout_cart`)
- Test: `tests/test_biofield_order_self_bridge.py`

**Interfaces:**
- Produces: after a successful reveal checkout, one `biofield` event per ordered slug (`record_click`), email from the reveal token, failure-isolated.

- [ ] **Step 1: Write the failing test**

This handler is deep (Stripe/checkout). Test at the seam: monkeypatch `app._checkout_cart` to return a success dict and `app._biofield_verify_token` to return `(True, {"id":1,"email":"a@b.com"})` and `app._biofield_visible_slugs` to return the requested slugs and `app.is_member` to True; POST the checkout with items; assert a `biofield` event per ordered slug lands in `recommendation_events`. Use a temp `LOG_DB`. (The implementer wires the monkeypatches to the real symbol names — grep the handler for the exact ones.)

```python
# tests/test_biofield_order_self_bridge.py  (sketch — implementer completes the monkeypatch wiring)
import sqlite3
import app as app_module
from dashboard import recommendation_events as re


def test_reveal_order_emits_biofield_events(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db); re.init_recommendation_events(cx); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "BIOFIELD_CART_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_biofield_verify_token", lambda th: (True, {"id": 1, "email": "a@b.com"}))
    monkeypatch.setattr(app_module, "_biofield_visible_slugs", lambda row, email: {"neuro-magnesium", "immune-modulation"})
    monkeypatch.setattr(app_module, "is_member", lambda *a, **k: True)
    monkeypatch.setattr(app_module, "_resolve_ship_address", lambda *a, **k: {})
    monkeypatch.setattr(app_module, "_checkout_cart", lambda email, items, **k: {"out": {}, "stripe_url": "https://x"})
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    r = c.post("/begin/biofield/anytok/order-checkout",
               json={"items": [{"slug": "neuro-magnesium", "qty": 1}, {"slug": "immune-modulation", "qty": 2}]})
    assert r.get_json()["ok"] is True
    cx = sqlite3.connect(db)
    bf = {e["product_key"] for e in re.list_events(cx, "a@b.com") if e["source_key"] == "biofield"}
    assert bf == {"neuro-magnesium", "immune-modulation"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_biofield_order_self_bridge.py -q`
Expected: FAIL (no biofield events).

- [ ] **Step 3: Implement**

In `app.py::begin_biofield_order_checkout`, after the checkout succeeds (right after `out, stripe_url = res["out"], res["stripe_url"]`, ~line 3799), add:

```python
        try:
            from dashboard import recommendation_events as _re
            with _db_lock, db.connect(LOG_DB) as _cx:
                _re.init_recommendation_events(_cx)
                for _it in items:
                    _re.record_click(_cx, email, _it["slug"], "biofield")
        except Exception:
            pass
```

(`items` and `email` are already in scope; `items` is the validated `[{slug, qty}]` list.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_biofield_order_self_bridge.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add app.py tests/test_biofield_order_self_bridge.py
git commit -m "feat(rec): ordering from the biofield reveal emits biofield events"
```

---

### Task 3: Biofield reveal remedy-click → `biofield` event

**Files:**
- Modify: `app.py` (new `POST /begin/biofield/<token>/remedy-click`), `static/begin-biofield.html` (sendBeacon on the remedy name link)
- Test: `tests/test_biofield_remedy_click.py`

**Interfaces:**
- Produces: `POST /begin/biofield/<token>/remedy-click` `{slug}` → resolves email via `_biofield_verify_token`, records a `biofield` click (failure-isolated), returns `{"ok": True}` (or `{"ok": False}` on bad token — never 500). The reveal page fires a `sendBeacon` to it when a remedy name link is clicked, then navigation proceeds.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_remedy_click.py
import sqlite3
import app as app_module
from dashboard import recommendation_events as re


def test_remedy_click_records_biofield(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db); re.init_recommendation_events(cx); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "_biofield_verify_token", lambda th: (True, {"id": 1, "email": "a@b.com"}))
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    r = c.post("/begin/biofield/anytok/remedy-click", json={"slug": "neuro-magnesium"})
    assert r.get_json()["ok"] is True
    cx = sqlite3.connect(db)
    assert any(e["source_key"] == "biofield" and e["product_key"] == "neuro-magnesium"
               for e in re.list_events(cx, "a@b.com"))


def test_remedy_click_bad_token(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db"); import sqlite3 as s; s.connect(db).close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "_biofield_verify_token", lambda th: (False, None))
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    r = c.post("/begin/biofield/bad/remedy-click", json={"slug": "x"})
    assert r.get_json()["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_biofield_remedy_click.py -q`
Expected: FAIL (route missing).

- [ ] **Step 3: Implement**

`app.py` — add near `begin_biofield_request_review` (~line 3696), mirroring its token-auth:

```python
@app.route("/begin/biofield/<token>/remedy-click", methods=["POST"])
def begin_biofield_remedy_click(token):
    """Client clicked a remedy's product link in their reveal — a biofield ACTION.
    Fire-and-forget from the page (sendBeacon); never blocks navigation."""
    try:
        th = _hash_token((token or "").strip())
        valid, row = _biofield_verify_token(th)
        if not valid or row is None:
            return jsonify({"ok": False, "reason": "invalid"})
        email = (row.get("email") or "").strip().lower()
        slug = ((request.get_json(silent=True) or {}).get("slug") or "").strip()
        if email and slug:
            from dashboard import recommendation_events as _re
            with _db_lock, db.connect(LOG_DB) as cx:
                _re.init_recommendation_events(cx)
                _re.record_click(cx, email, slug, "biofield")
        return jsonify({"ok": True})
    except Exception as e:
        print(f"[biofield-remedy-click] {e!r}", flush=True)
        return jsonify({"ok": False, "reason": "error"})
```

`static/begin-biofield.html` — where the remedy name link is built (`nameLink.href = rem.page_url`, ~line 510), add a click handler that beacons before navigating (the token `_token` is already in scope):

```javascript
nameLink.addEventListener("click", function(){
  try {
    navigator.sendBeacon(
      "/begin/biofield/" + encodeURIComponent(_token) + "/remedy-click",
      new Blob([JSON.stringify({slug: rem.slug})], {type: "application/json"}));
  } catch(e) {}
  // do not preventDefault — the link navigates normally
});
```

(Use `rem.slug` — confirm the remedy object's slug field name in `window.__REVEAL__` / `_biofield_remedy_payload`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_biofield_remedy_click.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add app.py static/begin-biofield.html tests/test_biofield_remedy_click.py
git commit -m "feat(rec): biofield reveal remedy-click records a biofield event"
```

---

### Task 4: Scan/FF product-click in the portal → `scan` event

**Files:**
- Modify: `app.py` (new `POST /api/portal/<token>/recommendation/click`), `static/client-portal.html` (sendBeacon on the scan/FF-matched product buy links)
- Test: `tests/test_portal_recommendation_click.py`

**Interfaces:**
- Produces: `POST /api/portal/<token>/recommendation/click` `{slug, source}` → resolves email via `_portal_record_for`, validates `source` against the registry, records a click for that source (failure-isolated), returns `{"ok": True}` / 404 on bad token. The portal's scan/FF matched-product links fire a `sendBeacon` with `source="scan"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_portal_recommendation_click.py
import sqlite3
import app as app_module
from dashboard import recommendation_events as re, client_portal as cp


def test_portal_click_records_scan(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db); cp.init_client_portal_table(cx); re.init_recommendation_events(cx)
    cp.upsert_portal(cx, email="a@b.com", name="Al"); cx.commit(); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    token = ...  # implementer mints via client_portal.upsert_portal return, per prior portal tests
    r = c.post(f"/api/portal/{token}/recommendation/click", json={"slug": "neuro-magnesium", "source": "scan"})
    assert r.get_json()["ok"] is True
    cx = sqlite3.connect(db)
    assert any(e["source_key"] == "scan" and e["product_key"] == "neuro-magnesium"
               for e in re.list_events(cx, "a@b.com"))


def test_portal_click_rejects_unknown_source_and_bad_token(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db); cp.init_client_portal_table(cx); re.init_recommendation_events(cx)
    cp.upsert_portal(cx, email="a@b.com", name="Al"); cx.commit(); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    token = ...  # same token
    # unknown source -> no event (but ok:true or 400, implementer's choice — assert NO event recorded)
    c.post(f"/api/portal/{token}/recommendation/click", json={"slug": "x", "source": "not-a-source"})
    r = c.post("/api/portal/badtoken/recommendation/click", json={"slug": "x", "source": "scan"})
    assert r.status_code == 404
    cx = sqlite3.connect(db)
    assert re.list_events(cx, "a@b.com") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_portal_recommendation_click.py -q`
Expected: FAIL (route missing).

- [ ] **Step 3: Implement**

`app.py` — add near the other `/api/portal/<token>/recommendation/*` writes (mirror `api_portal_rec_hide`):

```python
@app.route("/api/portal/<token>/recommendation/click", methods=["POST"])
def api_portal_rec_click(token):
    from dashboard import client_portal as _cp, recommendation_events as _re, recommendation_sources as _rs
    data = request.get_json(silent=True) or {}
    slug = (data.get("slug") or "").strip()
    source = (data.get("source") or "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx); _re.init_recommendation_events(cx)
        portal = _portal_record_for(cx, token)
        if not portal:
            return jsonify({"ok": False, "error": "not found"}), 404
        email = (portal.get("email") or "").strip().lower()
        if email and slug and _rs.known_source(source):
            try:
                _re.record_click(cx, email, slug, source)
            except Exception:
                pass
    return jsonify({"ok": True})
```

`static/client-portal.html` — locate the portal surface that renders scan/FF-matched products with buy links (grep the render for the scan-recommendations / FF-matches card and its product `<a href>`). Add a click handler that beacons before navigating:

```javascript
// on a scan/FF matched-product link click:
try {
  navigator.sendBeacon(
    "/api/portal/" + encodeURIComponent(token) + "/recommendation/click",
    new Blob([JSON.stringify({slug: <slug>, source: "scan"})], {type: "application/json"}));
} catch(e) {}
```

Note to implementer: if the scan/FF card and its per-product slug/link are not cleanly identifiable, STOP and report — do NOT wire a beacon to a link whose slug you can't reliably read. The endpoint stands alone and is tested regardless; the beacon wiring is the part that needs the real render.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_portal_recommendation_click.py -q`
Expected: PASS.

- [ ] **Step 5: Full feature run + commit**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_recommendation_click.py tests/test_biofield_order_self_bridge.py tests/test_biofield_remedy_click.py tests/test_portal_recommendation_click.py -q`
Expected: PASS (all).

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add app.py static/client-portal.html tests/test_portal_recommendation_click.py
git commit -m "feat(rec): scan/FF product-click in the portal records a scan event"
```

---

## Self-review checklist (controller, before dispatch)

- Each action counts (unique `origin_ref` via `record_click`); distinct from sticky `self` and per-order `purchased`.
- Email from the surface token only (reveal / portal); unknown token → no event / 404.
- Server emissions failure-isolated; client capture is fire-and-forget `sendBeacon` (never blocks navigation).
- biofield: order (Task 2) + click (Task 3); scan: click (Task 4). No double-count with `purchased`.
- Reviewer: check nothing outside the diff pins the touched handlers/pages; run the full 2d set.

## Not in 2d (deferred)

**Chat (`chat` action) is BLOCKED and deferred.** The chat client is mostly anonymous (`amg_session` cookie; `query_log.email` only when volunteered), and the existing `/api/cta-click` → `cta_clicks` log carries neither the product slug nor an email. Attributing a `chat` action needs two prerequisites first: (1) `/api/cta-click` must also send the product slug, and (2) a reliable chat→email identity for the session. Until then, chat cannot be honestly counted. This is the last remaining source; after it, Phase 3 is the external marketing channels (email/newsletter/ads/social).
