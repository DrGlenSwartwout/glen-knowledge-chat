# Funnel ToS Consistency — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Require ToS-membership before the four ungated actionable funnel surfaces transact/advise/grant: affiliate signup, reorder, referral my-code, and the concierge.

**Architecture:** Each gated server action adds the existing `is_member` check returning `{need_optin:true}` 403 (fail-safe); the page hands `need_optin` to the existing `OptinGate` (`static/optin-gate.js`) which captures name+email+ToS, fires `unlock('tos')`, and retries. The affiliate form additionally gets a visible required ToS checkbox. No new gate machinery, no flag.

**Tech Stack:** Flask (Python 3.11), `is_member` (app.py ~341), `record_unlock`/`unlock('tos')`, `static/optin-gate.js`, pytest + Flask test client.

## Global Constraints

- No emoji, no em dashes. No feature flag - these are live correctness fixes; `main` auto-deploys; stage surface-by-surface as each merges.
- The gate is **fail-safe**: `is_member(session_id, email)` returns False on internal error -> gate (never bypass). Already-members pass through unchanged - a member's order/code/chat behaves exactly as today.
- The gate sits IN FRONT of the existing logic (return before the transact/advise/grant); do NOT change the order/pricing/Stripe/QBO internals.
- The exact existing gate response shape (copy verbatim): `return jsonify({"ok": False, "need_optin": True, "error": "<msg>"}), 403`.
- Affiliate: a non-member cannot become an affiliate; agreeing ToS in the request (the form checkbox or JSON `tos:true`) sets membership via `record_unlock(... trigger="tos", tos=True)` BEFORE creating the affiliate, so the affiliate is always a member.
- Test harness: `importlib.import_module("app")` (skip if not importable); `monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path/"chat_log.db"))`; `begin_funnel.init_journey_tables(cx)`; `app_module.app.test_client()`; mock `ghl_onboard_contact` + `_capture_concierge_referral` on free-tier transitions; set membership in a test by `client` posting `/begin/unlock {trigger:"tos", email, tos:true}` OR `begin_funnel.record_unlock(cx, session_id=..., trigger="tos", email=..., tos=True)`. Assert the gate returns BEFORE Stripe/QBO by NOT mocking them and expecting the 403 (a non-member never reaches them).
- Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest <target> -v`. Spec: `docs/superpowers/specs/2026-06-20-funnel-tos-consistency-design.md`.

## Critical files / anchors
- `app.py`: `is_member` (341); `affiliate_apply_form` (6315), `affiliate_apply` (6399); `api_reorder_items` (9926), `reorder_checkout` (9981), `reorder_subscribe` (10103), `_reorder_email_from_cookie` (8132); `api_referral_my_code` (8089); `begin_concierge_chat` (4367), `begin_concierge_add` (4438).
- `static/`: `affiliate.html` (add ToS checkbox), `reorder.html`, the referral-calling page, the concierge page; `optin-gate.js` (`OptinGate.show({base, onAgree})`).
- Reference gate already in use: app.py:3703 (studio claim) and begin-buy.html:707 (`if (data.need_optin && window.OptinGate) OptinGate.show({base, onAgree: retry})`).

---

### Task 1: Affiliate / Ambassador gate (CRITICAL)

**Files:** Modify `app.py` (`affiliate_apply`, `affiliate_apply_form`); Modify `static/affiliate.html`; Create `tests/test_tos_consistency.py`.

**Interfaces:** Consumes `is_member`, `begin_funnel.record_unlock`. Produces the gated affiliate routes.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tos_consistency.py
"""Funnel ToS consistency - gate affiliate/reorder/referral/concierge."""
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
    import begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    monkeypatch.setattr(app_module, "ghl_onboard_contact", lambda *a, **k: {"contact_id": "x"})
    monkeypatch.setattr(app_module, "_capture_concierge_referral", lambda *a, **k: None)
    return db


def _make_member(app_module, db, email, session="m1"):
    import begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.record_unlock(cx, session_id=session, trigger="tos", email=email, tos=True)


def test_affiliate_apply_blocked_for_non_member(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    r = client.post("/affiliate/apply", json={"name": "Ann B", "email": "ann@x.com"})
    assert r.status_code == 403 and r.get_json().get("need_optin") is True
    # not created
    with sqlite3.connect(db) as cx:
        n = cx.execute("SELECT COUNT(*) FROM affiliate_signups WHERE LOWER(email)='ann@x.com'").fetchone()[0]
    assert n == 0


def test_affiliate_apply_with_tos_creates_and_sets_membership(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    r = client.post("/affiliate/apply", json={"name": "Ann B", "email": "ann@x.com", "tos": True})
    assert r.status_code == 200
    assert app_module.is_member(email="ann@x.com") is True
    with sqlite3.connect(db) as cx:
        n = cx.execute("SELECT COUNT(*) FROM affiliate_signups WHERE LOWER(email)='ann@x.com'").fetchone()[0]
    assert n == 1


def test_affiliate_apply_member_passes_through(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _make_member(app_module, db, "lee@x.com")
    client = app_module.app.test_client()
    r = client.post("/affiliate/apply", json={"name": "Lee X", "email": "lee@x.com"})
    assert r.status_code == 200


def test_affiliate_form_without_tos_redirects_to_error(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    r = client.post("/affiliate/apply-form", data={"name": "Ann B", "email": "ann@x.com"})
    assert r.status_code in (302, 303)
    assert "error=" in r.headers.get("Location", "")
```

- [ ] **Step 2: Run to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_tos_consistency.py -k affiliate -v`
Expected: FAIL (affiliate created without ToS today).

- [ ] **Step 3: Gate `affiliate_apply` (JSON)**

In `app.py` `affiliate_apply`, right after `email` is resolved and the `if not name or not email:` 400 check, insert:

```python
    _tos = bool(data.get("tos"))
    _sid = (request.cookies.get("amg_session") or "").strip()
    if not is_member(_sid, email) and not _tos:
        return jsonify({"ok": False, "need_optin": True,
                        "error": "Please agree to our Terms to become an Ambassador."}), 403
    if _tos and not is_member(_sid, email):
        try:
            with _db_lock, sqlite3.connect(LOG_DB) as _cx:
                begin_funnel.record_unlock(
                    _cx, session_id=(_sid or uuid.uuid4().hex), trigger="tos",
                    email=email, first_name=(name.split(None, 1)[0] if name else ""),
                    tos=True)
        except Exception as e:
            print(f"[affiliate-tos] {e!r}", flush=True)
```

(`is_member`, `_db_lock`, `sqlite3`, `begin_funnel`, `uuid` already exist in app.py.)

- [ ] **Step 4: Gate `affiliate_apply_form` (form)**

In `app.py` `affiliate_apply_form`, after the `if not name or not email:` redirect, insert (form fields):

```python
    _tos = (request.form.get("tos") or "").strip().lower() in ("1", "true", "on", "yes")
    _sid = (request.cookies.get("amg_session") or "").strip()
    if not is_member(_sid, email) and not _tos:
        return _redirect("/affiliate?error=" + _urlparse.quote(
            "Please agree to our Terms to become an Ambassador."))
    if _tos and not is_member(_sid, email):
        try:
            with _db_lock, sqlite3.connect(LOG_DB) as _cx:
                begin_funnel.record_unlock(
                    _cx, session_id=(_sid or uuid.uuid4().hex), trigger="tos",
                    email=email, first_name=_first, tos=True)
        except Exception as e:
            print(f"[affiliate-tos] {e!r}", flush=True)
```

(`_redirect`, `_urlparse`, `_first` already exist in that function.)

- [ ] **Step 5: Add the ToS checkbox to `static/affiliate.html`**

In the signup form (grep `apply-form` / the `<form>` in affiliate.html), add a required checkbox before the submit button:

```html
        <label class="tos-line" style="display:flex;gap:8px;align-items:flex-start;margin:14px 0;font-size:13px;">
          <input type="checkbox" name="tos" value="true" required />
          <span>I agree to the <a href="https://remedymatch.com/info/terms-and-conditions" target="_blank" rel="noopener">Terms</a> and will share ethically.</span>
        </label>
```

(If the form posts to `/affiliate/apply` via JS rather than `/affiliate/apply-form`, ensure the JS includes `tos: true` when the box is checked; grep the form's submit handler. The `required` attribute blocks an unchecked submit client-side; the server check is the real gate.)

- [ ] **Step 6: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_tos_consistency.py -k affiliate -v`
Expected: PASS (4 affiliate tests).

- [ ] **Step 7: Commit**

```bash
git add app.py static/affiliate.html tests/test_tos_consistency.py
git commit -m "feat: ToS-gate affiliate/Ambassador signup (server + form checkbox)"
```

Note for the reviewer: manual visual pass - the affiliate form ToS checkbox blocks submit until checked; agreeing creates an affiliate who is now a member.

---

### Task 2: Reorder gate

**Files:** Modify `app.py` (`api_reorder_items`, `reorder_checkout`, `reorder_subscribe`); Modify `static/reorder.html`; add tests to `tests/test_tos_consistency.py`.

**Interfaces:** Consumes `is_member`, `_reorder_email_from_cookie`.

- [ ] **Step 1: Write the failing tests**

```python
def _reorder_client(app_module, email):
    c = app_module.app.test_client()
    c.set_cookie("rm_reorder_email", email)
    return c


def test_reorder_checkout_blocked_for_non_member(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    c = _reorder_client(app_module, "ann@x.com")
    r = c.post("/reorder/checkout", json={"items": []})
    assert r.status_code == 403 and r.get_json().get("need_optin") is True


def test_reorder_items_blocked_for_non_member(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    c = _reorder_client(app_module, "ann@x.com")
    r = c.get("/api/reorder/items")
    assert r.status_code == 403 and r.get_json().get("need_optin") is True


def test_reorder_subscribe_blocked_for_non_member(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    c = _reorder_client(app_module, "ann@x.com")
    r = c.post("/reorder/subscribe", json={"items": []})
    assert r.status_code == 403 and r.get_json().get("need_optin") is True


def test_reorder_items_member_not_gated(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _make_member(app_module, db, "lee@x.com")
    c = _reorder_client(app_module, "lee@x.com")
    r = c.get("/api/reorder/items")
    assert r.status_code != 403  # passes the gate (200 or its normal non-gate response)
```

- [ ] **Step 2: Run to verify they fail**

Run: `... -m pytest tests/test_tos_consistency.py -k reorder -v` -> FAIL.

- [ ] **Step 3: Gate the three reorder endpoints**

In each of `api_reorder_items`, `reorder_checkout`, `reorder_subscribe`, immediately after `email = _reorder_email_from_cookie()` (and the existing "not signed in" 401 if any), add:

```python
    _sid = (request.cookies.get("amg_session") or "").strip()
    if not is_member(_sid, email):
        return jsonify({"ok": False, "need_optin": True,
                        "error": "Please agree to our Terms to continue your order."}), 403
```

Place it BEFORE any pricing/QBO/Stripe call so a non-member never reaches them.

- [ ] **Step 4: Wire `static/reorder.html` need_optin -> OptinGate**

Ensure `reorder.html` loads `optin-gate.js` (`<script src="/static/optin-gate.js"></script>`; add if missing). On each reorder action's response, add the standard handler (mirror begin-buy.html:707):

```javascript
        if (data && data.need_optin && window.OptinGate) {
          OptinGate.show({ base: '', onAgree: function () { /* re-call the same action */ } });
          return;
        }
```

Apply it to the items-load, checkout, and subscribe calls (the `onAgree` retries that specific call).

- [ ] **Step 5: Run + commit**

Run: `... -m pytest tests/test_tos_consistency.py -k reorder -v` -> PASS.

```bash
git add app.py static/reorder.html tests/test_tos_consistency.py
git commit -m "feat: ToS-gate reorder checkout/subscribe/items"
```

---

### Task 3: Referral my-code gate

**Files:** Modify `app.py` (`api_referral_my_code`); Modify the page that calls `/api/referral/my-code`; add tests.

- [ ] **Step 1: Write the failing tests**

```python
def test_referral_mycode_blocked_for_non_member(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    c = app_module.app.test_client()
    c.set_cookie("rm_reorder_email", "ann@x.com")
    r = c.get("/api/referral/my-code")
    assert r.status_code == 403 and r.get_json().get("need_optin") is True


def test_referral_mycode_member_not_gated(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _make_member(app_module, db, "lee@x.com")
    c = app_module.app.test_client()
    c.set_cookie("rm_reorder_email", "lee@x.com")
    r = c.get("/api/referral/my-code")
    assert r.status_code != 403
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Gate `api_referral_my_code`**

In `api_referral_my_code`, after the caller email is resolved (the existing `get_authenticated_user` OR `_reorder_email_from_cookie` resolution), add before minting the code:

```python
    _sid = (request.cookies.get("amg_session") or "").strip()
    if not is_member(_sid, email):
        return jsonify({"ok": False, "need_optin": True,
                        "error": "Please agree to our Terms to get your referral code."}), 403
```

(Match the existing local variable name for the resolved email in that function - grep it; if it is not `email`, use that name.)

- [ ] **Step 4: Wire the calling page need_optin -> OptinGate** (grep which static page fetches `/api/referral/my-code`; add the same `need_optin` handler + ensure `optin-gate.js` is loaded). If it is the reorder page, this is already covered by Task 2's script include - just add the handler on the my-code fetch.

- [ ] **Step 5: Run + commit**

Run: `... -m pytest tests/test_tos_consistency.py -k referral -v` -> PASS.

```bash
git add app.py static/<referral-page>.html tests/test_tos_consistency.py
git commit -m "feat: ToS-gate referral my-code minting"
```

---

### Task 4: Concierge gate

**Files:** Modify `app.py` (`begin_concierge_chat`, `begin_concierge_add`); Modify the concierge page; add tests.

- [ ] **Step 1: Write the failing tests**

```python
def test_concierge_add_blocked_for_non_member(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    c = app_module.app.test_client()
    r = c.post("/begin/concierge/add", json={"email": "ann@x.com", "slug": "x", "invoice_id": "1"})
    assert r.status_code == 403 and r.get_json().get("need_optin") is True


def test_concierge_add_member_not_gated(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _make_member(app_module, db, "lee@x.com")
    c = app_module.app.test_client()
    r = c.post("/begin/concierge/add", json={"email": "lee@x.com", "slug": "x", "invoice_id": "1"})
    assert r.status_code != 403  # passes the gate (may 400 for a bad invoice, but not the ToS 403)
```

(Note: the concierge chat is an SSE stream; gate it the same way `/begin/match/chat` does - emit the gate signal or 403 before streaming. The plan's test targets `/begin/concierge/add` which is a plain JSON endpoint; for the chat, mirror the match-chat gate pattern and add a serve/stream assertion if practical, else gate it server-side and rely on the manual pass.)

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Gate the concierge endpoints**

In `begin_concierge_add`, after the order/email is resolved, before modifying the invoice:

```python
    _sid = (request.cookies.get("amg_session") or "").strip()
    if not is_member(_sid, email):
        return jsonify({"ok": False, "need_optin": True,
                        "error": "Please agree to our Terms to continue."}), 403
```

In `begin_concierge_chat` (SSE), mirror the `/begin/match/chat` gate: resolve `email`/`session_id` and, for a non-member, yield the gate signal the front end already understands (grep how `begin_match_chat` emits `{"gate": True}` / or returns need_optin) and apply the same here BEFORE producing individualized advice. Use the EXISTING pattern from `begin_match_chat` verbatim so the concierge page's existing SSE handling works.

- [ ] **Step 4: Wire the concierge page** need_optin / gate handling (the page already renders concierge chat; add `OptinGate.onSSE`/`need_optin` handling mirroring `begin-match.html`; ensure `optin-gate.js` is loaded).

- [ ] **Step 5: Run the focused tests + the full begin sweep**

Run: `... -m pytest tests/test_tos_consistency.py -v`
Then: `... -m pytest tests/ -k "begin or tos or affiliate or reorder or referral" -v`
Expected: all PASS; no regressions.

- [ ] **Step 6: Commit**

```bash
git add app.py static/<concierge-page>.html tests/test_tos_consistency.py
git commit -m "feat: ToS-gate post-purchase concierge chat/add"
```

---

## Self-Review

**1. Spec coverage:** affiliate (T1, server + form checkbox + sets-membership), reorder checkout/subscribe/items (T2), referral my-code (T3), concierge chat/add (T4). Journey-signal page-loads deliberately NOT gated (out of scope, none added). The uniform `is_member`->`need_optin`->`OptinGate` pattern + the affiliate checkbox -> every task. Fail-safe (is_member False on error) -> inherent to `is_member`. Members pass through -> each task's `*_member_not_gated`/`passes_through` test.

**2. Placeholder scan:** No TBD/handle-edge-cases. Two grep-to-locate instructions (T3/T4: which page fetches my-code / the concierge SSE gate emit) are concrete, named lookups against existing patterns, not placeholders. The concierge-chat SSE gate says "use the EXISTING begin_match_chat pattern verbatim" - a located, real anchor.

**3. Type consistency:** The gate snippet `{"ok": False, "need_optin": True, "error": ...}, 403` is identical across T1-T4 (matches the in-use shape at app.py:3704). `is_member(_sid, email)` signature consistent. The affiliate `tos`-sets-membership uses `record_unlock(... trigger="tos", tos=True)` consistent with the spec. `_reorder_email_from_cookie()` reused in T2/T3. The test helpers `_fresh`/`_make_member`/`_load_app` defined once in T1 and reused T2-T4.
