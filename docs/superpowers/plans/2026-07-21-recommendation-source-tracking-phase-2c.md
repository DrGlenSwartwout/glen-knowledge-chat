# Recommendation Source Tracking — Phase 2c (self via wishlist bridge) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a client's self-selected products (their **wishlist** — "save a product I'm browsing") flow into their portal's **Self** recommendation section, by emitting a `self` recommendation event whenever a product is added to a client's *email-keyed* wishlist — directly (portal-token toggle, begin-side toggle when the visitor is identified) or when an anonymous session wishlist merges to the client's email at login.

**Architecture:** Product pages (`/begin/product/<slug>`) carry no resolvable client identity (anonymous `amg_session` only), and the portal's product links point off-site — so there is no reliable *product-page* identity for a `self` write. The already-solved mechanism is the **wishlist** (`dashboard/wishlist.py`), which is correctly email-attributed at three moments: the token-authed portal toggle, the begin-side toggle when a `rm_reorder_email`/auth email is present, and `merge_wishlist` (session→email) at login. Phase 2c bridges those three add-moments to `recommendation_events.record_event(email, slug, "self", …)`. All emission is failure-isolated (never breaks a wishlist toggle or a login) and email-only (an anonymous `sess:` add never becomes a `self` event until it merges to an email).

**Tech Stack:** Python 3 / Flask (`app.py`), `dashboard/*.py`, SQLite (`LOG_DB`), pytest (endpoint tests under `doppler run -- python3`; pure-sqlite tests plain).

## Global Constraints

- **Email-only attribution.** A `self` event is emitted ONLY for an email-keyed wishlist add (owner `email:<addr>`), NEVER for an anonymous `sess:<id>` add. Anonymous adds become `self` events at `merge_wishlist` time (session→email at login).
- **Failure-isolated.** The `self` emission is wrapped so it can NEVER break the wishlist toggle or the login/merge flow (mirror 2a's `_emit_source_events` pattern: `try/except: pass`).
- **Idempotent + sticky.** `record_self` uses a STABLE `origin_ref="self"`, so re-adding a product is a no-op (one `self` membership per (client, product)); the `self` event persists even if the client later removes the product from their wishlist (recommendation_events are append-only; the client hides via the existing hide control). `record_event` is `INSERT OR IGNORE` on `(client_email, product_key, source_key, origin_ref)`.
- **Reuse, don't reinvent.** Use the existing `dashboard/wishlist.py` (`toggle` returns True on ADD / False on remove; `merge_wishlist`; `slugs_for`) and `recommendation_events.record_event`. No new wishlist storage.
- The endpoints are already gated by `_WISHLIST_ENABLED`; do not change that gate.
- **CI known_failures ratchet; never run the bare full suite (sends live email).** Run named feature tests. NOTE (learned this session): after any change, the full-suite ratchet may catch a regression a diff-scoped review can't — the reviewer should sanity-check that nothing outside the diff pins the touched code.

---

### Task 1: `record_self` helper

**Files:**
- Modify: `dashboard/recommendation_events.py`
- Test: `tests/test_recommendation_self.py`

**Interfaces:**
- Produces: `recommendation_events.record_self(cx, email, product_key) -> bool` — records one `self` event with `occurred_at=_now()`, `origin_ref="self"` (stable → idempotent per product). Returns `record_event`'s bool.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recommendation_self.py
import sqlite3
from dashboard import recommendation_events as re


def _cx():
    cx = sqlite3.connect(":memory:")
    re.init_recommendation_events(cx)
    return cx


def test_record_self_idempotent_per_product():
    cx = _cx()
    assert re.record_self(cx, "A@B.com", "neuro-magnesium") is True
    # re-add of the same product -> no new event (stable origin_ref)
    assert re.record_self(cx, "a@b.com", "neuro-magnesium") is False
    ev = re.list_events(cx, "a@b.com")
    assert len(ev) == 1
    assert ev[0]["source_key"] == "self" and ev[0]["origin_ref"] == "self"


def test_record_self_distinct_products_and_blank_guard():
    cx = _cx()
    re.record_self(cx, "a@b.com", "neuro-magnesium")
    re.record_self(cx, "a@b.com", "immune-modulation")
    assert len(re.list_events(cx, "a@b.com")) == 2
    assert re.record_self(cx, "a@b.com", "") is False        # blank slug -> no-op (record_event guard)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_recommendation_self.py -q`
Expected: FAIL — `AttributeError: ... 'record_self'`.

- [ ] **Step 3: Implement** (append to `dashboard/recommendation_events.py`)

```python
def record_self(cx, email, product_key):
    """A client self-selected a product (added it to their wishlist). One sticky
    'self' membership per (client, product) — stable origin_ref, so re-adds are a
    no-op and the membership persists even if the product is later un-wishlisted
    (append-only; the client hides via the hide control)."""
    return record_event(cx, email, product_key, "self",
                        occurred_at=_now(), origin_ref="self")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_recommendation_self.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add dashboard/recommendation_events.py tests/test_recommendation_self.py
git commit -m "feat(rec): record_self helper (sticky self recommendation per product)"
```

---

### Task 2: Portal-token wishlist toggle emits `self` on add

**Files:**
- Modify: `app.py::api_portal_wishlist_toggle` (~line 20233)
- Test: `tests/test_portal_wishlist_self_bridge.py`

**Interfaces:**
- Consumes: `recommendation_events.record_self`.
- Produces: when the token-authed wishlist toggle ADDS a product (`_saved` True) for a resolved `_email`, a `self` recommendation event is recorded (failure-isolated). No event on a remove.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_portal_wishlist_self_bridge.py
import sqlite3
import app as app_module
from dashboard import recommendation_events as re, client_portal as cp, wishlist as wl


def _seed(tmp_path, monkeypatch):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db)
    cp.init_client_portal_table(cx); re.init_recommendation_events(cx); wl.init_wishlist_table(cx)
    cp.upsert_portal(cx, email="a@b.com", name="Al")
    cx.commit(); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "_WISHLIST_ENABLED", True, raising=False)
    app_module.app.config["TESTING"] = True
    return db


def test_portal_wishlist_add_emits_self_remove_does_not(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    token = ...  # implementer resolves the raw token via client_portal.upsert_portal return, per prior portal tests
    # add -> self event
    c.post(f"/api/portal/{token}/wishlist/toggle", json={"slug": "neuro-magnesium"})
    cx = sqlite3.connect(db)
    ev = re.list_events(cx, "a@b.com")
    assert len(ev) == 1 and ev[0]["source_key"] == "self"
    cx.close()
    # toggle again (remove) -> no new self event
    c.post(f"/api/portal/{token}/wishlist/toggle", json={"slug": "neuro-magnesium"})
    cx = sqlite3.connect(db)
    assert len([e for e in re.list_events(cx, "a@b.com") if e["source_key"] == "self"]) == 1
```

Note to implementer: obtain the raw token exactly as the existing portal tests do (`client_portal.upsert_portal` return value; see `tests/test_portal_recommendation_writes.py`).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_portal_wishlist_self_bridge.py -q`
Expected: FAIL (no self event emitted).

- [ ] **Step 3: Implement**

In `app.py::api_portal_wishlist_toggle`, after `_saved = _wl.toggle(_cx, "email:" + _email, slug) if _email else False` (~line 20233), before the `with` block closes, add:

```python
            if _saved and _email:
                try:
                    from dashboard import recommendation_events as _re
                    _re.init_recommendation_events(_cx)
                    _re.record_self(_cx, _email, slug)
                except Exception:
                    pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_portal_wishlist_self_bridge.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add app.py tests/test_portal_wishlist_self_bridge.py
git commit -m "feat(rec): portal wishlist add emits a self recommendation event"
```

---

### Task 3: Begin-side wishlist toggle emits `self` on add (email-keyed only)

**Files:**
- Modify: `app.py::begin_wishlist_toggle` (~line 7311)
- Test: `tests/test_begin_wishlist_self_bridge.py`

**Interfaces:**
- Produces: when the begin-side toggle ADDS a product AND the visitor is identified by email (`email` non-empty → owner `email:<addr>`), a `self` event is recorded (failure-isolated). An anonymous (`sess:`) add records NO self event.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_begin_wishlist_self_bridge.py
import sqlite3
import app as app_module
from dashboard import recommendation_events as re, wishlist as wl


def _seed(tmp_path, monkeypatch):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db); re.init_recommendation_events(cx); wl.init_wishlist_table(cx); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "_WISHLIST_ENABLED", True, raising=False)
    app_module.app.config["TESTING"] = True
    return db


def test_identified_add_emits_self_anonymous_does_not(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    # identified via rm_reorder_email cookie -> self event
    c.set_cookie("rm_reorder_email", "a@b.com")   # adapt to the test client's set_cookie signature
    c.post("/begin/wishlist/toggle", json={"slug": "neuro-magnesium"})
    cx = sqlite3.connect(db)
    assert len([e for e in re.list_events(cx, "a@b.com") if e["source_key"] == "self"]) == 1
    cx.close()
    # anonymous (no email cookie) add -> NO self event for anybody
    c2 = app_module.app.test_client()
    c2.post("/begin/wishlist/toggle", json={"slug": "immune-modulation"})
    cx = sqlite3.connect(db)
    rows = cx.execute("SELECT COUNT(*) FROM recommendation_events WHERE source_key='self' AND product_key='immune-modulation'").fetchone()[0]
    assert rows == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_begin_wishlist_self_bridge.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `app.py::begin_wishlist_toggle`, after `saved = _wl.toggle(cx, owner, slug)` (~line 7311), inside the `with` block, add:

```python
            if saved and email:   # email-keyed add only; anonymous sess adds never attribute
                try:
                    from dashboard import recommendation_events as _re
                    _re.init_recommendation_events(cx)
                    _re.record_self(cx, email, slug)
                except Exception:
                    pass
```

(`email` is the local from `_wishlist_ids(request)`; it is empty for an anonymous visitor.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_begin_wishlist_self_bridge.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add app.py tests/test_begin_wishlist_self_bridge.py
git commit -m "feat(rec): identified begin-side wishlist add emits a self event"
```

---

### Task 4: Anonymous session wishlist merge → `self` events at login

**Files:**
- Modify: `app.py` (add `_wishlist_merge_with_self` helper; replace the 3 `_wl.merge_wishlist(...)` call sites at ~3147, ~9978, ~19211)
- Test: `tests/test_wishlist_merge_self_bridge.py`

**Interfaces:**
- Produces: `_wishlist_merge_with_self(cx, session_id, email)` — captures the session's wishlist slugs, calls `wishlist.merge_wishlist`, then emits a `self` event for each merged slug to `email` (failure-isolated). Replaces the 3 direct `merge_wishlist` calls so anonymous product-page adds become `self` recommendations the moment the client is identified.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wishlist_merge_self_bridge.py
import sqlite3
import app as app_module
from dashboard import recommendation_events as re, wishlist as wl


def test_merge_emits_self_for_merged_slugs():
    cx = sqlite3.connect(":memory:")
    re.init_recommendation_events(cx); wl.init_wishlist_table(cx)
    # anonymous session added two products
    wl.toggle(cx, "sess:S1", "neuro-magnesium")
    wl.toggle(cx, "sess:S1", "immune-modulation")
    app_module._wishlist_merge_with_self(cx, "S1", "A@B.com")
    # both moved to the email wishlist AND recorded as self events
    assert wl.slugs_for(cx, "email:a@b.com") == {"neuro-magnesium", "immune-modulation"}
    selfev = {e["product_key"] for e in re.list_events(cx, "a@b.com") if e["source_key"] == "self"}
    assert selfev == {"neuro-magnesium", "immune-modulation"}
    # idempotent: a second merge (nothing in session now) adds nothing
    app_module._wishlist_merge_with_self(cx, "S1", "a@b.com")
    assert len([e for e in re.list_events(cx, "a@b.com") if e["source_key"] == "self"]) == 2


def test_merge_helper_never_raises_on_blank():
    cx = sqlite3.connect(":memory:")
    re.init_recommendation_events(cx); wl.init_wishlist_table(cx)
    app_module._wishlist_merge_with_self(cx, "", "a@b.com")   # no session -> no-op
    app_module._wishlist_merge_with_self(cx, "S1", "")        # no email -> no-op
    assert re.list_events(cx, "a@b.com") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_wishlist_merge_self_bridge.py -q`
Expected: FAIL — `AttributeError: ... '_wishlist_merge_with_self'`.

- [ ] **Step 3: Implement**

Add the helper in `app.py` (near the wishlist helpers, e.g. after `_wishlist_ids`):

```python
def _wishlist_merge_with_self(cx, session_id, email):
    """Merge an anonymous session wishlist into the client's email wishlist AND
    record each merged product as a 'self' recommendation. Failure-isolated:
    never breaks the login/merge flow."""
    from dashboard import wishlist as _wl
    e = (email or "").strip().lower()
    sid = session_id or ""
    if not e or not sid:
        return
    merged = _wl.slugs_for(cx, "sess:" + sid)   # capture BEFORE merge deletes the sess rows
    _wl.merge_wishlist(cx, sid, e)
    if not merged:
        return
    try:
        from dashboard import recommendation_events as _re
        _re.init_recommendation_events(cx)
        for slug in merged:
            _re.record_self(cx, e, slug)
    except Exception:
        pass
```

Then replace each of the three `_wl.merge_wishlist(_cxw, request.cookies.get("amg_session", ""), <email>)` calls (~lines 3147, 9978, 19211) with:

```python
                        _wishlist_merge_with_self(_cxw, request.cookies.get("amg_session", ""), <email>)
```

(keep each site's existing `<email>` expression — `email` at 3147/9978, `email_for_reports` at 19211 — and the surrounding connection/commit handling).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_wishlist_merge_self_bridge.py -q`
Expected: PASS.

- [ ] **Step 5: Full feature run + commit**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_recommendation_self.py tests/test_portal_wishlist_self_bridge.py tests/test_begin_wishlist_self_bridge.py tests/test_wishlist_merge_self_bridge.py -q`
Expected: PASS (all).

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add app.py tests/test_wishlist_merge_self_bridge.py
git commit -m "feat(rec): session wishlist merge records self events at login"
```

---

## Self-review checklist (controller, before dispatch)

- `self` attributed by EMAIL only; anonymous session adds only become self events at merge (Task 4), never at anonymous add (Task 3 guards on `email`).
- Emission failure-isolated at all 3 sites (toggle × 2, merge) — can't break wishlist or login.
- Idempotent + sticky: stable `origin_ref="self"`; the self membership survives a later un-wishlist (append-only).
- Reuses `wishlist.py` + `record_event`; no new storage; `_WISHLIST_ENABLED` gate unchanged.
- Reviewer note: verify nothing outside the diff pins `begin_wishlist_toggle`/`api_portal_wishlist_toggle`/`merge_wishlist` call shapes (the full-suite ratchet is the backstop).

## Not in 2c

The true anonymous product-page button (repointing portal→product links to carry the token, or consulting `rm_portal_session` on `/begin/product`) is a separate, larger change and is NOT needed: the wishlist already captures self-selection and this bridge turns it into `self` recommendations. 2d (reveal/engagement click capture for biofield/scan/chat) remains; then Phase 3 marketing channels.
