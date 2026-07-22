# Recommendation Source Tracking — Chat action capture Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record a `chat` recommendation event when a client clicks a product CTA in chat — for the identified subset of sessions (where the chat session already carries an email). This is the last in-app source; the honest limit is that anonymous chat sessions still can't be attributed.

**Architecture:** The chat CTA button (`static/embed.html::renderCta`) already POSTs `/api/cta-click` `{log_id, cta_type}` on click; its `target` is a `/begin/(buy|product)/<slug>` product URL. Two changes: (1) the button also sends the product `slug`; (2) `/api/cta-click` resolves the session email via `query_log` (by `log_id`) and, when an email and a catalog-valid slug are present, records a `chat` click via `recommendation_events.record_click`. Reuses the per-action `record_click` (unique origin_ref). The existing `cta_clicks` insert is unchanged; the recommendation event is the new output.

**Tech Stack:** Python 3 / Flask (`app.py`), `dashboard/recommendation_events.py` + `dashboard/products.py`, SQLite (`LOG_DB`), pytest (app-importing under `doppler run -- python3`), vanilla JS in `static/embed.html`.

## Global Constraints

- **Identity from the server, by `log_id`.** The email comes from `SELECT email FROM query_log WHERE id=<log_id>` — never from the request body. If that email is empty (anonymous session), NO `chat` event is recorded. This is the documented limitation: only identified chat sessions count.
- **Slug validated against the catalog.** The client-sent `slug` is recorded only if it's a real product (`dashboard/products.py::load_products()` — a `superseded_slug`-resolved key). Prevents a client injecting junk `product_key`s.
- **Failure-isolated + non-blocking.** The `/api/cta-click` handler is already `try/except: pass` returning `{"ok": True}`; the new logic stays inside it and can never break the click response or the existing `cta_clicks` insert. Client capture stays fire-and-forget (the button's `fetch` is not awaited; navigation proceeds).
- **Each action counts.** `record_click` (unique origin_ref) — every genuine CTA click is a distinct `chat` event.
- **Reuse.** `record_click`, `products.load_products`, the existing `/api/cta-click` + `renderCta`. No new table (the `chat` event lives in `recommendation_events`; `cta_clicks` is untouched).
- **CI known_failures ratchet; never the bare full suite (sends live email).** Reviewer: confirm nothing outside the diff pins `/api/cta-click`'s body shape or `embed.html`'s CTA markup.

---

### Task 1: `/api/cta-click` records a `chat` event for identified sessions

**Files:**
- Modify: `app.py::api_cta_click` (~line 11744)
- Test: `tests/test_chat_cta_recommendation.py`

**Interfaces:**
- Consumes: `recommendation_events.record_click`, `products.load_products` (for slug validation), `query_log` (email by `log_id`).
- Produces: when the POST body carries a `slug` that is a valid product AND the `query_log` row for `log_id` has a non-empty email, a `chat` event is recorded for `(email, slug)`. Anonymous session (empty email) or invalid slug → no event. Existing `cta_clicks` insert + `{"ok": True}` response unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chat_cta_recommendation.py
import sqlite3
import app as app_module
from dashboard import recommendation_events as re


def _seed(tmp_path, monkeypatch, *, email):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db)
    re.init_recommendation_events(cx)
    cx.execute("CREATE TABLE IF NOT EXISTS query_log (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, session_id TEXT)")
    cx.execute("CREATE TABLE IF NOT EXISTS cta_clicks (ts TEXT, log_id INTEGER, cta_type TEXT)")
    cx.execute("INSERT INTO query_log (id, email) VALUES (1, ?)", (email,))
    cx.commit(); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    # catalog stub: neuro-magnesium is a real product, junk-slug is not
    monkeypatch.setattr(app_module, "_cta_valid_product", lambda s: s == "neuro-magnesium", raising=False)
    app_module.app.config["TESTING"] = True
    return db


def test_identified_chat_click_records_chat_event(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch, email="a@b.com")
    c = app_module.app.test_client()
    r = c.post("/api/cta-click", json={"log_id": 1, "cta_type": "page", "slug": "neuro-magnesium"})
    assert r.get_json()["ok"] is True
    cx = sqlite3.connect(db)
    assert any(e["source_key"] == "chat" and e["product_key"] == "neuro-magnesium"
               for e in re.list_events(cx, "a@b.com"))


def test_anonymous_session_records_nothing(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch, email="")     # anonymous
    c = app_module.app.test_client()
    c.post("/api/cta-click", json={"log_id": 1, "cta_type": "page", "slug": "neuro-magnesium"})
    cx = sqlite3.connect(db)
    n = cx.execute("SELECT COUNT(*) FROM recommendation_events WHERE source_key='chat'").fetchone()[0]
    assert n == 0


def test_invalid_slug_records_nothing(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch, email="a@b.com")
    c = app_module.app.test_client()
    c.post("/api/cta-click", json={"log_id": 1, "cta_type": "page", "slug": "junk-slug"})
    cx = sqlite3.connect(db)
    assert re.list_events(cx, "a@b.com") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_chat_cta_recommendation.py -q`
Expected: FAIL (no chat event / helper missing).

- [ ] **Step 3: Implement**

Add a small catalog-validation helper near `api_cta_click` (so the test can monkeypatch it, and so it's cheap/cached):

```python
def _cta_valid_product(slug):
    """True if slug is a real catalog product (superseded-resolved)."""
    try:
        from dashboard import products as _p
        cat = _p.load_products()
        s = (slug or "").strip()
        return bool(s) and (s in cat or _p.superseded_slug(s) in cat)
    except Exception:
        return False
```

(Confirm `products.superseded_slug` exists; if not, just `s in cat`.)

In `api_cta_click`, inside the existing `try:`/`with _db_lock ... as cx:` block, after the `cta_clicks` insert, add:

```python
            slug = (str(data.get("slug") or "")).strip()
            if slug and _cta_valid_product(slug):
                erow = cx.execute("SELECT email FROM query_log WHERE id=?", (log_id,)).fetchone()
                email = ((erow[0] if erow else "") or "").strip().lower()
                if email:
                    from dashboard import recommendation_events as _re
                    _re.init_recommendation_events(cx)
                    _re.record_click(cx, email, slug, "chat")
```

(All inside the handler's existing `try/except: pass`, so any failure is swallowed and the click response is unaffected. `log_id` and `data` are already in scope.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_chat_cta_recommendation.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add app.py tests/test_chat_cta_recommendation.py
git commit -m "feat(rec): chat CTA click records a chat event for identified sessions"
```

---

### Task 2: The chat CTA button sends the product slug

**Files:**
- Modify: `static/embed.html::renderCta` (~line 903-920)
- Test: `tests/test_embed_cta_slug.py`

**Interfaces:**
- Produces: the CTA-click `fetch` body includes the product `slug` parsed from the button `target` (`/begin/(buy|product)/<slug>`), so Task 1 can attribute it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embed_cta_slug.py
import app as app_module


def test_embed_cta_click_sends_slug():
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    body = c.get("/embed").get_data(as_text=True)   # confirm the served embed route; adjust if it differs
    assert "cta-click" in body
    # the click payload now carries a slug parsed from the product target
    assert "slug" in body
    assert "/begin/" in body                          # the slug is parsed from a /begin/(buy|product)/<slug> target
```

Note to implementer: confirm the route that serves `static/embed.html` (grep app.py for `embed.html`); if it isn't `/embed`, use the correct path. If the page is only ever iframed and not directly routed, assert against the file contents via a path read (mirror the client-portal UI tests, e.g. `pathlib.Path("static/embed.html").read_text()`).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_embed_cta_slug.py -q`
Expected: FAIL (slug not in the CTA payload yet).

- [ ] **Step 3: Implement**

In `static/embed.html::renderCta`, where the `page`/`action` button posts the click (~line 913-920), parse the slug from `target` and include it:

```javascript
        btn.addEventListener('click', function() {
          try {
            var m = target.match(/\/begin\/(?:buy|product)\/([^\/?#]+)/);
            var slug = m ? decodeURIComponent(m[1]) : '';
            fetch('/api/cta-click', {
              method: 'POST',
              credentials: 'same-origin',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ log_id: logId, cta_type: type, slug: slug })
            });
          } catch (_) {}
        });
```

(Only product targets yield a slug; non-product CTAs send `slug: ''`, which Task 1 ignores. Keep the `fetch` un-awaited — navigation is not blocked.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_embed_cta_slug.py -q`
Expected: PASS.

- [ ] **Step 5: Full run + commit**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_chat_cta_recommendation.py tests/test_embed_cta_slug.py -q`
Expected: PASS (all).

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add static/embed.html tests/test_embed_cta_slug.py
git commit -m "feat(rec): chat CTA button sends the product slug for chat attribution"
```

---

## Self-review checklist (controller, before dispatch)

- Email resolved server-side from `query_log` by `log_id` — never the request body; anonymous session → no event (the honest limitation).
- Slug validated against the catalog (no junk `product_key`s).
- Emission inside the existing failure-isolated `/api/cta-click` handler; `cta_clicks` insert + `{"ok":True}` unchanged; client `fetch` un-awaited (non-blocking).
- Each click counts (`record_click`). Reuses existing infra; no new table.

## Not in this plan — Phase 3 (external marketing channels)

`chat` is the last IN-APP source. The remaining sources are external and each is its own plan:
- **Email / newsletter** — needs a tracked-redirect click service (none exists today): a `/r/<...>` endpoint that logs `(recipient email, slug, source)` then 302s to the product, plus wiring the GHL / email-content-engine product links through it. This is the substantive Phase 3 lift.
- **Ads / social** — most blocked: ad/social clicks land anonymously, so there's no client identity to attribute until the visitor later identifies (login/purchase). Needs an anonymous→identified reconciliation (like the wishlist merge) before it can be honest.
