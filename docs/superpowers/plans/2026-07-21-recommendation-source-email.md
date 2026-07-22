# Recommendation Source Tracking — Email/newsletter click capture Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record an `email` recommendation event when a client clicks the product link in one of Glen's personal emails — via a new app-owned tracked-redirect that resolves a PII-free per-recipient token to the recipient's email, records the click, then 302s to the product page.

**Architecture:** No app-owned redirect exists today (product links are external Rebrandly `truly.vip` shortlinks that 302 off-app, so the app never sees the click). This plan adds one. (1) `dashboard/email_click_tokens.py` mints a durable, opaque, per-recipient token (random `secrets.token_urlsafe`, stored `token → email`), so a URL can carry recipient identity with **no PII in the URL** and **without exposing the portal credential**. (2) `GET /r/<token>/<source>/<slug>` resolves `token → email`, validates the `source` (allowlist `{email, newsletter}`) and `slug` (catalog), records a click via `recommendation_events.record_click`, then **302s to `/begin/product/<slug>` regardless** (the redirect must never break the click). (3) The live personal-email engine (`incentive_engine._process_one_user`) builds its product link through this route with `source=email`. The `newsletter` source has **no live producer today** (`generate_newsletter` has no non-test caller) — the same URL shape (`/r/<token>/newsletter/<slug>`) is ready for it, wired later at zero extra cost.

**Tech Stack:** Python 3 / Flask (`app.py`), `dashboard/email_click_tokens.py` (new) + `dashboard/recommendation_events.py` + `dashboard/recommendation_sources.py` + `dashboard/products.py`, SQLite (`LOG_DB`), Jinja email templates, pytest (app-importing tests run under `doppler run -- python3`).

## Global Constraints

- **Identity from the server-resolved token, never the request.** The recipient email comes ONLY from `email_click_tokens.email_for(cx, token)`. No email is ever read from a query string, path, or body. If the token doesn't resolve, NO event is recorded (but the redirect still happens).
- **No PII in URLs.** The tracked link carries an opaque random token, never the email (encoded or plain). This is deliberately better than the existing `unsubscribe?email=…` links. Do NOT reuse the client's portal token (`client_portal.ensure_token`) as the tracked-link token — a forwarded email link must not grant portal access.
- **Source is an allowlist.** Only `email` and `newsletter` are accepted from the URL (validated via `recommendation_sources.known_source` AND membership in `{"email","newsletter"}`). Any other source value → no event (still redirect).
- **Slug validated against the catalog.** Record only if the slug resolves to a real product (`products.load_products()` / `products.superseded_slug()`), preventing a token-holder from injecting junk `product_key`s.
- **The redirect never breaks the click.** All recording is inside a failure-isolated `try/except`; the route ALWAYS returns a 302. Destination is internal-only: `/begin/product/<resolved-slug>` when the slug is valid, else `/` — never a URL derived from the token or params (no open redirect).
- **Each action counts.** `record_click` (unique `origin_ref` per call) — every genuine click is a distinct `email` event.
- **Reuse.** `record_click`, `products.load_products`/`superseded_slug`, `recommendation_sources.known_source`, the `with _db_lock, sqlite3.connect(LOG_DB) as cx:` route pattern (see `api_portal_rec_click`, `app.py:19873`). `email`/`newsletter` are already-registered sources (`recommendation_sources.py:13-14`).
- **CI known_failures ratchet; never the bare full suite (it sends live email).** App-importing tests run via `doppler run -- python3 -m pytest`. Reviewer: confirm nothing outside the diff pins `_process_one_user`'s product URL (the existing `generate_personal_email` tests pass their own product dict and are unaffected — `tests/test_incentive_engine.py:149,173,222`).

---

### Task 1: `dashboard/email_click_tokens.py` — PII-free per-recipient token store

**Files:**
- Create: `dashboard/email_click_tokens.py`
- Test: `tests/test_email_click_tokens.py`

**Interfaces:**
- Produces:
  - `init_email_click_tokens(cx)` — idempotent DDL (`CREATE TABLE IF NOT EXISTS` + unique index on `email`).
  - `token_for(cx, email) -> str` — returns the recipient's durable token, minting one on first call and REUSING it on subsequent calls (stable links, one token per email). Normalizes email (`strip().lower()`). Commits.
  - `email_for(cx, token) -> str | None` — resolves a token to its normalized email, or `None` if unknown/blank.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_email_click_tokens.py
import sqlite3
from dashboard import email_click_tokens as ect


def _cx():
    cx = sqlite3.connect(":memory:")
    ect.init_email_click_tokens(cx)
    return cx


def test_token_is_opaque_and_contains_no_email():
    cx = _cx()
    t = ect.token_for(cx, "Alice@Example.com")
    assert t and isinstance(t, str) and len(t) >= 16
    assert "alice" not in t.lower() and "@" not in t and "example" not in t.lower()


def test_token_is_stable_and_idempotent_per_email():
    cx = _cx()
    t1 = ect.token_for(cx, "alice@example.com")
    t2 = ect.token_for(cx, "  Alice@example.com  ")   # normalized to same email
    assert t1 == t2
    n = cx.execute("SELECT COUNT(*) FROM email_click_tokens WHERE email=?",
                   ("alice@example.com",)).fetchone()[0]
    assert n == 1


def test_distinct_emails_get_distinct_tokens():
    cx = _cx()
    assert ect.token_for(cx, "a@x.com") != ect.token_for(cx, "b@x.com")


def test_email_for_resolves_and_normalizes():
    cx = _cx()
    t = ect.token_for(cx, "Bob@Example.com")
    assert ect.email_for(cx, t) == "bob@example.com"


def test_email_for_unknown_or_blank_is_none():
    cx = _cx()
    assert ect.email_for(cx, "nope") is None
    assert ect.email_for(cx, "") is None
    assert ect.email_for(cx, None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_email_click_tokens.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Implement**

```python
# dashboard/email_click_tokens.py
"""Durable, opaque, per-recipient tokens for email/newsletter tracked links.

A tracked email link carries a random token (never the recipient's email), so
the redirect can resolve identity server-side with NO PII in the URL. This is a
DEDICATED token type — deliberately NOT the client portal token — so a forwarded
email link can never grant portal access.
"""
import secrets
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def init_email_click_tokens(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS email_click_tokens ("
        "token TEXT PRIMARY KEY, email TEXT NOT NULL, created_at TEXT)"
    )
    cx.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_email_click_tokens_email "
        "ON email_click_tokens(email)"
    )
    cx.commit()


def token_for(cx, email):
    """Return the recipient's durable token, minting once and reusing after."""
    e = _norm(email)
    if not e:
        return ""
    row = cx.execute(
        "SELECT token FROM email_click_tokens WHERE email=?", (e,)
    ).fetchone()
    if row and row[0]:
        return row[0]
    token = secrets.token_urlsafe(24)
    cx.execute(
        "INSERT OR IGNORE INTO email_click_tokens (token, email, created_at) "
        "VALUES (?, ?, ?)", (token, e, _now())
    )
    cx.commit()
    # Re-read in case of a concurrent insert winning the unique(email) race.
    row = cx.execute(
        "SELECT token FROM email_click_tokens WHERE email=?", (e,)
    ).fetchone()
    return row[0] if row else token


def email_for(cx, token):
    """Resolve a token to its normalized email, or None."""
    t = (token or "").strip()
    if not t:
        return None
    row = cx.execute(
        "SELECT email FROM email_click_tokens WHERE token=?", (t,)
    ).fetchone()
    return _norm(row[0]) if row and row[0] else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && python3 -m pytest tests/test_email_click_tokens.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add dashboard/email_click_tokens.py tests/test_email_click_tokens.py
git commit -m "feat(rec): PII-free per-recipient email click-token store"
```

---

### Task 2: `GET /r/<token>/<source>/<slug>` tracked-redirect route

**Files:**
- Modify: `app.py` (add the route; place it near the other public `/begin`/redirect routes — grep `@app.route("/begin/product` for a neighborhood)
- Test: `tests/test_email_click_redirect.py`

**Interfaces:**
- Consumes: `email_click_tokens.email_for` + `.init_email_click_tokens`, `recommendation_events.record_click` + `.init_recommendation_events`, `recommendation_sources.known_source`, `products.load_products`/`superseded_slug`.
- Produces: a `GET` route that ALWAYS 302s. Records one `email` (or `newsletter`) click iff the token resolves to an email AND `source ∈ {email, newsletter}` AND the slug is a catalog product. Redirect target: `/begin/product/<resolved-slug>` for a valid slug, else `/`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_email_click_redirect.py
import sqlite3
import app as app_module
from dashboard import email_click_tokens as ect
from dashboard import recommendation_events as re


def _seed(tmp_path, monkeypatch, *, email="a@b.com"):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db)
    ect.init_email_click_tokens(cx)
    re.init_recommendation_events(cx)
    cx.commit(); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    # catalog stub: terrain-restore is real, junk-slug is not
    monkeypatch.setattr(app_module, "_rec_valid_slug", lambda s: (s if s == "terrain-restore" else None), raising=False)
    app_module.app.config["TESTING"] = True
    cx = sqlite3.connect(db)
    token = ect.token_for(cx, email)
    cx.close()
    return db, token


def test_valid_email_click_records_and_redirects(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.get(f"/r/{token}/email/terrain-restore")
    assert r.status_code in (301, 302)
    assert "/begin/product/terrain-restore" in r.headers["Location"]
    cx = sqlite3.connect(db)
    assert any(e["source_key"] == "email" and e["product_key"] == "terrain-restore"
               for e in re.list_events(cx, "a@b.com"))


def test_unknown_token_records_nothing_but_still_redirects(tmp_path, monkeypatch):
    db, _ = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.get("/r/bogustoken/email/terrain-restore")
    assert r.status_code in (301, 302)
    cx = sqlite3.connect(db)
    n = cx.execute("SELECT COUNT(*) FROM recommendation_events").fetchone()[0]
    assert n == 0


def test_disallowed_source_records_nothing(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.get(f"/r/{token}/chat/terrain-restore")   # 'chat' not allowed via email links
    assert r.status_code in (301, 302)
    cx = sqlite3.connect(db)
    assert re.list_events(cx, "a@b.com") == []


def test_invalid_slug_records_nothing_and_redirects_home(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.get(f"/r/{token}/email/junk-slug")
    assert r.status_code in (301, 302)
    assert r.headers["Location"].endswith("/")
    cx = sqlite3.connect(db)
    assert re.list_events(cx, "a@b.com") == []


def test_newsletter_source_is_allowed(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    c.get(f"/r/{token}/newsletter/terrain-restore")
    cx = sqlite3.connect(db)
    assert any(e["source_key"] == "newsletter" for e in re.list_events(cx, "a@b.com"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_email_click_redirect.py -q`
Expected: FAIL (route + `_rec_valid_slug` helper missing).

- [ ] **Step 3: Implement**

Add a small catalog-resolution helper (module-level, near the route, so the test can monkeypatch it):

```python
def _rec_valid_slug(slug):
    """Return the catalog-resolved slug for `slug`, or None if not a product."""
    try:
        from dashboard import products as _p
        cat = _p.load_products()
        s = (slug or "").strip()
        if not s:
            return None
        if s in cat:
            return s
        r = _p.superseded_slug(s)
        return r if r in cat else None
    except Exception:
        return None
```

Add the route (allowlist the two email-channel sources explicitly; `<slug>` uses the default string converter — storefront slugs never contain `/`):

```python
_EMAIL_LINK_SOURCES = ("email", "newsletter")


@app.route("/r/<token>/<source>/<slug>", methods=["GET"])
def email_click_redirect(token, source, slug):
    """Tracked redirect for product links in Glen's emails/newsletters. Resolves the
    opaque per-recipient token -> email (never PII in the URL), records ONE engagement
    click for the given source, then 302s to the product page. ALWAYS redirects — the
    click is never blocked by a recording failure. Identity is server-resolved from the
    token only; source is an allowlist; slug is catalog-validated."""
    resolved = _rec_valid_slug(slug)
    dest = f"/begin/product/{resolved}" if resolved else "/"
    try:
        src = (source or "").strip().lower()
        if resolved and src in _EMAIL_LINK_SOURCES:
            from dashboard import (email_click_tokens as _ect,
                                   recommendation_events as _re,
                                   recommendation_sources as _rs)
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                _ect.init_email_click_tokens(cx)
                _re.init_recommendation_events(cx)
                email = _ect.email_for(cx, token)
                if email and _rs.known_source(src):
                    _re.record_click(cx, email, resolved, src)
    except Exception:
        pass
    return redirect(dest, code=302)
```

(Confirm `redirect` is imported from Flask at the top of `app.py` — it is used widely already. `_db_lock` and `sqlite3` and `LOG_DB` are module-level, as in `api_portal_rec_click`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_email_click_redirect.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add app.py tests/test_email_click_redirect.py
git commit -m "feat(rec): /r tracked-redirect records email/newsletter product clicks"
```

---

### Task 3: Route the personal-email product link through the tracked redirect

**Files:**
- Modify: `incentive_engine.py::_process_one_user` (~line 677-685)
- Test: `tests/test_incentive_engine_tracked_link.py`

**Interfaces:**
- Consumes: `email_click_tokens.token_for`, `_public_base()` (existing, `incentive_engine.py:708`), `LOG_DB` / `_sqlite3` (existing module aliases).
- Produces: the `product["url"]` handed to `generate_personal_email` is now `{_public_base()}/r/{token}/email/{slug}` (slug = `terrain-restore`), so a click on the personal-email product link records an `email` event. `generate_personal_email` / the template are UNCHANGED (they render whatever `product["url"]` they're given).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_incentive_engine_tracked_link.py
import sqlite3
import incentive_engine as ie
from dashboard import email_click_tokens as ect


def test_process_one_user_builds_tracked_email_link(tmp_path, monkeypatch):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db); ect.init_email_click_tokens(cx); cx.commit(); cx.close()
    monkeypatch.setattr(ie, "LOG_DB", db, raising=False)
    monkeypatch.setattr(ie, "_public_base", lambda: "https://illtowell.com", raising=False)
    monkeypatch.setattr(ie, "_load_incentive_config", lambda: {"beta_shared_code": "BETA5"}, raising=False)

    captured = {}
    def fake_generate(user, topic, topic_source_text, product, is_beta, audience, **kw):
        captured["product"] = product
        return {"subject": "s", "body": "b"}
    monkeypatch.setattr(ie, "generate_personal_email", fake_generate, raising=False)
    # short-circuit topic selection + send/record so we only exercise link-building
    monkeypatch.setattr(ie, "_select_candidate_topics", lambda *a, **k: ["leaky-gut"], raising=False)  # adjust to real helper if named differently
    monkeypatch.setattr(ie, "_send_email", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(ie, "_record_send", lambda *a, **k: None, raising=False)

    ie._process_one_user({"id": 1, "email": "a@b.com", "name": "A"},
                         ie._load_incentive_config(), audience="client", is_beta=True)

    url = captured["product"]["url"]
    assert url.startswith("https://illtowell.com/r/")
    assert url.endswith("/email/terrain-restore")
    assert "a@b.com" not in url            # no PII in the link
    # the token in the URL resolves back to the recipient
    token = url.split("/r/")[1].split("/email/")[0]
    cx = sqlite3.connect(db)
    assert ect.email_for(cx, token) == "a@b.com"
```

Note to implementer: `_process_one_user` does topic selection before building `product`. The monkeypatches above short-circuit the parts that need Pinecone/LLM/DB so the test reaches the `product = {...}` construction and the `generate_personal_email` call. **Confirm the real helper names** for candidate-topic selection and the send/record calls in `_process_one_user` (read the function first) and patch those exact names; the goal is simply to run `_process_one_user` far enough to capture the `product` dict passed to `generate_personal_email`. If short-circuiting proves brittle, instead extract the link-building into a tiny helper `_tracked_product_url(email, slug, source="email")` and unit-test THAT directly, then call it from `_process_one_user` — either satisfies the interface.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_incentive_engine_tracked_link.py -q`
Expected: FAIL (url still the raw `truly.vip` shortlink).

- [ ] **Step 3: Implement**

Add a small helper near `_process_one_user`, then use it when building `product`:

```python
def _tracked_product_url(email, slug, source="email"):
    """Build the app-owned tracked-redirect URL for an email/newsletter product
    link. Mints a durable per-recipient token (no PII in the URL) and points at
    /r/<token>/<source>/<slug>, which records the click then 302s to the product."""
    from dashboard import email_click_tokens as _ect
    with _sqlite3.connect(LOG_DB) as cx:
        _ect.init_email_click_tokens(cx)
        token = _ect.token_for(cx, email)
    return f"{_public_base()}/r/{token}/{source}/{slug}"
```

In `_process_one_user`, change the `product` dict (was `"url": "https://truly.vip/terrain-restore"`):

```python
    _slug = "terrain-restore"
    product = {
        "name": "Terrain Restore",
        "url":  _tracked_product_url(user["email"], _slug, "email"),
        "code": config.get("beta_shared_code", "BETA5"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_incentive_engine_tracked_link.py -q`
Expected: PASS.

- [ ] **Step 5: Full run of touched suites + commit**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_email_click_tokens.py tests/test_email_click_redirect.py tests/test_incentive_engine_tracked_link.py tests/test_incentive_engine.py -q`
Expected: PASS (all — including the pre-existing `test_incentive_engine.py`, which is unaffected because it calls `generate_personal_email` with its own product dict).

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add incentive_engine.py tests/test_incentive_engine_tracked_link.py
git commit -m "feat(rec): personal-email product link routes through tracked redirect (email source)"
```

---

## Self-review checklist (controller, before dispatch)

- Identity resolved server-side from the opaque token only; no email in any URL (encoded or plain). Dedicated token, NOT the portal credential.
- Source is a hard allowlist `{email, newsletter}`; slug catalog-validated; redirect is internal-only (`/begin/product/<slug>` or `/`) — no open redirect.
- Recording is failure-isolated; the route ALWAYS 302s (the click is never broken).
- Each click counts (`record_click`, unique origin_ref). Reuses existing infra; one new small module + one route + one link-builder.
- Newsletter source: infra ready (`/r/<token>/newsletter/<slug>`), no live producer wired (honest — `generate_newsletter` has no non-test caller). Ads/social remain out of scope (blocked on anonymous identity).
- No test outside the diff pins `_process_one_user`'s URL; the `generate_personal_email` tests pass their own product dict (unaffected).

## Not in this plan

- **Live newsletter send** — `generate_newsletter` has no non-test caller today; when a live newsletter broadcast is built, its product links use the same `/r/<token>/newsletter/<slug>` shape (zero redirect changes).
- **Ads / social** — clicks land anonymously (no recipient identity), so they need an anonymous→identified reconciliation (like the wishlist-merge bridge) before they can be attributed honestly. Separate future plan.
- **Transactional emails** (welcome, biofield report) — could also route product links through `/r/<token>/email/<slug>`; deferred unless Glen wants it, to keep this slice focused on the live 3×/daily engine.
