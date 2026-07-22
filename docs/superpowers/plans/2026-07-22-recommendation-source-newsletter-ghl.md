# Recommendation Source Tracking — Newsletter (GHL click webhook) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record a `newsletter` recommendation event when a client clicks a product link in Glen's GoHighLevel (GHL) newsletter — by receiving a GHL Outbound Webhook (fired by a "Trigger Link Clicked" workflow) that carries the recipient's email and the product slug.

**Architecture:** Glen's newsletter is sent from GHL, not from the app. GHL's Outbound Webhook payload includes the full contact record (email) by default, and we attach the product slug as static CUSTOM DATA per workflow — so the app needs only ONE net-new inbound endpoint: `POST /webhook/ghl-click`. It authenticates with the existing shared-secret scheme (secret in the URL as `?key=`, since GHL outbound webhooks cannot send custom headers; header `X-Webhook-Secret` also accepted for parity with the other `/webhook/*` handlers), reads `email` + `product_slug` (+ optional `source`, default `newsletter`), validates the slug against the live catalog via the existing `_rec_valid_slug` (follows supersession, rejects inactive) and the source against the existing `_EMAIL_LINK_SOURCES` allowlist, then records one engagement click via `recommendation_events.record_click`. No token store, no contactId→email lookup, no URL parsing — the payload is self-describing. All the recording machinery (`record_click`, `newsletter` source, `_rec_valid_slug`, `_EMAIL_LINK_SOURCES`) already shipped in #1120.

**Tech Stack:** Python 3 / Flask (`app.py`), reusing `dashboard/recommendation_events.py` + `dashboard/recommendation_sources.py` + `_rec_valid_slug`/`_EMAIL_LINK_SOURCES` (app.py), SQLite (`LOG_DB`), pytest (app-import test runs under `doppler run -- python3`).

## Global Constraints

- **Auth via shared secret, matching the existing `/webhook/*` pattern.** Compare an incoming secret against the module-level `WEBHOOK_SECRET` (app.py:28409). Accept it from `?key=` OR the `X-Webhook-Secret` header (GHL outbound webhooks can't send headers, so the URL form is the real path). Return `401` on mismatch **only when `WEBHOOK_SECRET` is set** (mirrors `pb_webhook`, app.py:28440 — if the env secret is unset, auth is a no-op, same as siblings). Use the same plain `!=` compare the sibling webhooks use (`hmac` is not imported in app.py; do not add it — stay consistent with the neighbors).
- **Identity + product come only from the GHL payload, server-to-server.** Email from the payload's `email` (fallback `contact.email`); slug from `product_slug` (fallback `slug`). No client-facing URL is generated here, so there is no PII-in-URL concern (the email arrives in GHL's POST body). Normalize email `strip().lower()`.
- **Slug catalog-validated; source allowlisted.** Record only if `_rec_valid_slug(slug)` returns a live survivor slug AND `source ∈ _EMAIL_LINK_SOURCES` AND `recommendation_sources.known_source(source)`. Record under the RESOLVED survivor slug (what `_rec_valid_slug` returns), not the raw input.
- **Failure-isolated; always 200 on authed calls.** All parsing/recording inside a `try/except: pass`. Return `{"ok": True}, 200` for any authed call (even if nothing is recorded) so GHL does not retry-storm. Only a bad/missing secret returns 401. Each click counts (`record_click`, unique origin_ref).
- **Capture the raw payload for diagnosis.** Call the existing `_record_webhook_debug("ghl-click", <raw body>, headers=...)` AFTER auth passes (never record unauthenticated bodies) — we have not yet seen GHL's exact payload shape live, and this is how the other webhooks are debugged (`app.py:28104`).
- **Reuse, don't rebuild.** `_rec_valid_slug` (app.py:7457), `_EMAIL_LINK_SOURCES` (app.py:7469), `record_click`/`init_recommendation_events`, `known_source`, the `with _db_lock, sqlite3.connect(LOG_DB) as cx:` pattern (`api_portal_rec_click`, app.py:19895).
- **CI known_failures ratchet; never the bare full suite (it sends live email).** App-import test runs via `doppler run -- python3 -m pytest`.

---

### Task 1: `POST /webhook/ghl-click` records a `newsletter` event

**Files:**
- Modify: `app.py` — add the route next to the other `/webhook/*` handlers (grep `@app.route("/webhook/groovekart"`, place after it, module-level)
- Test: `tests/test_ghl_click_webhook.py`

**Interfaces:**
- Consumes: `WEBHOOK_SECRET`, `_rec_valid_slug`, `_EMAIL_LINK_SOURCES`, `_record_webhook_debug`, `recommendation_events.record_click`+`init_recommendation_events`, `recommendation_sources.known_source`, `_db_lock`/`sqlite3`/`LOG_DB` (all module-level in app.py already).
- Produces: a `POST` route that records ONE `newsletter` (or `email`) click for `(payload email, resolved slug)` when authed + slug valid + source allowlisted; returns 200 on any authed call, 401 on bad secret.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ghl_click_webhook.py
import sqlite3
import app as app_module
from dashboard import recommendation_events as re


def _seed(tmp_path, monkeypatch, *, secret="s3cret"):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db)
    re.init_recommendation_events(cx)
    cx.commit(); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "WEBHOOK_SECRET", secret, raising=False)
    # catalog stub: terrain-restore resolves to itself; anything else is not a product
    monkeypatch.setattr(app_module, "_rec_valid_slug",
                        lambda s: ("terrain-restore" if s == "terrain-restore" else None),
                        raising=False)
    app_module.app.config["TESTING"] = True
    return db


def test_authed_click_records_newsletter_event(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post("/webhook/ghl-click?key=s3cret",
               json={"email": "A@B.com", "product_slug": "terrain-restore"})
    assert r.status_code == 200
    cx = sqlite3.connect(db)
    assert any(e["source_key"] == "newsletter" and e["product_key"] == "terrain-restore"
               for e in re.list_events(cx, "a@b.com"))   # normalized lower


def test_bad_secret_is_401_and_records_nothing(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post("/webhook/ghl-click?key=wrong",
               json={"email": "a@b.com", "product_slug": "terrain-restore"})
    assert r.status_code == 401
    cx = sqlite3.connect(db)
    assert re.list_events(cx, "a@b.com") == []


def test_secret_via_header_also_accepted(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post("/webhook/ghl-click",
               headers={"X-Webhook-Secret": "s3cret"},
               json={"email": "a@b.com", "product_slug": "terrain-restore"})
    assert r.status_code == 200
    cx = sqlite3.connect(db)
    assert len(re.list_events(cx, "a@b.com")) == 1


def test_invalid_slug_records_nothing_but_200(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post("/webhook/ghl-click?key=s3cret",
               json={"email": "a@b.com", "product_slug": "junk-slug"})
    assert r.status_code == 200
    cx = sqlite3.connect(db)
    assert re.list_events(cx, "a@b.com") == []


def test_missing_email_records_nothing_but_200(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post("/webhook/ghl-click?key=s3cret",
               json={"product_slug": "terrain-restore"})
    assert r.status_code == 200
    cx = sqlite3.connect(db)
    n = cx.execute("SELECT COUNT(*) FROM recommendation_events").fetchone()[0]
    assert n == 0


def test_disallowed_source_records_nothing_but_200(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post("/webhook/ghl-click?key=s3cret",
               json={"email": "a@b.com", "product_slug": "terrain-restore", "source": "chat"})
    assert r.status_code == 200
    cx = sqlite3.connect(db)
    assert re.list_events(cx, "a@b.com") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_ghl_click_webhook.py -q`
Expected: FAIL (route does not exist → 404, so the 200/401 assertions fail).

- [ ] **Step 3: Implement**

Add the route after `groovekart_webhook` (module-level):

```python
@app.route("/webhook/ghl-click", methods=["POST"])
def ghl_click_webhook():
    """GHL newsletter product-link click -> `newsletter` recommendation event.

    Glen's newsletter is sent from GoHighLevel using Trigger Links; a
    'Trigger Link Clicked' workflow fires a GHL Outbound Webhook here. GHL's payload
    carries the full contact record (email) plus CUSTOM DATA we set per workflow
    (`product_slug`, optional `source`). We validate the slug against the live catalog
    and record ONE engagement click. Auth: shared secret via ?key= (GHL outbound
    webhooks can't send custom headers) or X-Webhook-Secret header. Always 200 on an
    authed call so GHL does not retry-storm; 401 only on a bad/missing secret."""
    if WEBHOOK_SECRET:
        incoming = request.args.get("key") or request.headers.get("X-Webhook-Secret", "")
        if incoming != WEBHOOK_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    _record_webhook_debug("ghl-click", request.get_data(as_text=True),
                          headers=str(dict(request.headers)))
    try:
        data = request.get_json(force=True, silent=True) or {}
        contact = data.get("contact") if isinstance(data.get("contact"), dict) else {}
        email = (data.get("email") or contact.get("email") or "").strip().lower()
        slug_raw = (data.get("product_slug") or data.get("slug") or "").strip()
        source = (data.get("source") or "newsletter").strip().lower()
        resolved = _rec_valid_slug(slug_raw)
        if email and resolved and source in _EMAIL_LINK_SOURCES:
            from dashboard import (recommendation_events as _re,
                                   recommendation_sources as _rs)
            if _rs.known_source(source):
                with _db_lock, sqlite3.connect(LOG_DB) as cx:
                    _re.init_recommendation_events(cx)
                    _re.record_click(cx, email, resolved, source)
    except Exception:
        pass
    return jsonify({"ok": True}), 200
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_ghl_click_webhook.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add app.py tests/test_ghl_click_webhook.py
git commit -m "feat(rec): /webhook/ghl-click records newsletter clicks from GHL trigger-link workflow"
```

---

## Self-review checklist (controller, before dispatch)

- Auth matches the sibling `/webhook/*` pattern (plain compare vs `WEBHOOK_SECRET`, 401 on mismatch, no-op when unset); `?key=` supported because GHL can't send headers; no `hmac` import added.
- Email + product from the GHL payload only; email normalized; no PII in any URL we emit.
- Slug validated + resolved via `_rec_valid_slug` (survivor/active); source allowlisted via `_EMAIL_LINK_SOURCES` + `known_source`. Records under the resolved slug.
- Failure-isolated; authed calls always 200 (no GHL retry-storm); raw payload captured post-auth via `_record_webhook_debug`.
- Reuses shipped infra; no new table, no token store, no contactId resolver.

## Not in this plan (GHL-side setup — Glen, one-time per featured product)

Delivered separately as a short setup recipe, not code:
1. Create a GHL **Trigger Link** whose destination is the product page (e.g. `https://illtowell.com/begin/product/terrain-restore`); put it in the newsletter.
2. Build a GHL **Workflow**: Trigger = "Trigger Link Clicked" (filtered to that link) → Action = **Outbound Webhook**, `POST` to `https://illtowell.com/webhook/ghl-click?key=<WEBHOOK_SECRET>`, with CUSTOM DATA `product_slug = <slug>` (and optionally `source = newsletter`). GHL includes the contact email automatically.
3. Repeat the trigger-link + workflow per product featured.

Ads/social remain deferred (blocked on anonymous→identified reconciliation).
