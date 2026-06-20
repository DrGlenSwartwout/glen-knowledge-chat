# Begin #4b — $1 Biofield Unlock -> $99/mo Trial — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** The Biofield reveal's unlock CTA becomes a $1 Stripe checkout that vaults the card and, on return, creates the $99/mo membership (subscription + access grant) so the deeper remedies release and the visitor becomes a full paid member; +30d auto-charge via the existing cron; one-click cancel. Behind `BIOFIELD_TRIAL_ENABLED` (dark).

**Architecture:** New endpoints `POST /begin/biofield/<token>/unlock-checkout` and a `kind=="biofield_trial"` branch in `begin_checkout_return` (mirrors the existing `kind=subscribe`/group-bundle branches: read the PaymentIntent -> customer+pm -> create). On return: `subscriptions.create_membership` (billing) + a `memberships` access grant (so `_active_membership_for_email` goes active), idempotent on the checkout session id. The reveal page-data releases the full remedies when paid. A tokened `/membership/cancel/<token>` cancels via `subscriptions.set_status`.

**Tech Stack:** Flask, Stripe (`dashboard/stripe_pay.py`), `dashboard/subscriptions.py`, the `memberships` grant table, the #4a reveal (`biofield_reveals`, `auth_tokens`), pytest.

## Global Constraints

- No emoji, no em dashes. **Live money** -> the whole feature is gated by `BIOFIELD_TRIAL_ENABLED` (default off) AND `_STRIPE_ACTIVE`; with either off, the unlock CTA stays the #4a "unlocking soon" stub and no charge path is reachable.
- **Idempotent membership creation:** keyed on the Stripe checkout session id (a `biofield_trial_grants(session_id PK)` marker, mirroring `group_bundle_grants`). A reloaded return URL never double-creates the subscription or grant. Never double-charge (Stripe charges the $1 once on a completed checkout).
- The return handler is best-effort and NEVER 500s (wrap; log + redirect on any failure), mirroring `studio_claim_return`.
- **Anti-bypass preserved:** the deeper remedy details leave the server ONLY when `_active_membership_for_email(email)` is active. A non-paid visitor cannot obtain them via page-data or `reveal-top`.
- Reuse: `stripe_pay.create_checkout_session(amount_cents, *, customer_email, description, metadata, success_url, cancel_url, save_card=True)`, `get_session`, `get_payment_intent` (returns `{customer, payment_method, ...}`); `subscriptions.create_membership` / `set_status` / `add_months` / `init_subscriptions_table` / `migrate_add_membership_columns`; `group_bundle.MEMBERSHIP_AMOUNT_CENTS` (9900); `_active_membership_for_email`; the admin `memberships` grant INSERT shape (app.py ~18943); `_biofield_verify_token` / `dashboard.biofield_reveals.get_by_token_hash`; `_hash_token` + `secrets.token_urlsafe` for the cancel token.
- Tests MOCK Stripe (no live charge): monkeypatch `stripe_pay.create_checkout_session` / `get_session` / `get_payment_intent`. deploy-chat isolation (tmp `LOG_DB`; init subscriptions + memberships + biofield_reveals + auth_tokens tables). Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest <target> -v`. Spec: `docs/superpowers/specs/2026-06-20-biofield-trial-design.md`.

## Critical files / anchors
- `app.py`: `begin_checkout_return` (~4268, the kind-branch); `begin_biofield_reveal` (~1472, page-data) + `_biofield_top_payload` (~1425) + the `reveal-top` route (~1544); `_biofield_verify_token` (~1440); the admin `memberships` grant INSERT (~18943); `_active_membership_for_email` (~5905); `_STRIPE_ACTIVE` (2536); `PUBLIC_BASE_URL`; `_hash_token` (229).
- `dashboard/stripe_pay.py` (checkout/session/PI), `dashboard/subscriptions.py` (`create_membership`/`set_status`/`add_months`), `dashboard/group_bundle.py` (`MEMBERSHIP_AMOUNT_CENTS`).
- `static/begin-biofield.html` (the CTA + remedy rendering, from #4a).

---

### Task 1: Flag + the $1 checkout endpoint

**Files:** Modify `app.py` (flag + `POST /begin/biofield/<token>/unlock-checkout`); Create `tests/test_biofield_trial.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_biofield_trial.py
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
    from dashboard import biofield_reveals, subscriptions
    with sqlite3.connect(db) as cx:
        biofield_reveals.init_table(cx)
        subscriptions.init_subscriptions_table(cx)
        subscriptions.migrate_add_membership_columns(cx)
    return db


def _approved_reveal(app_module, db, email="t@x.com"):
    """Create an approved reveal + an auth_tokens biofield_reveal token; return the plaintext token."""
    import secrets as _s
    from datetime import datetime, timezone, timedelta
    from dashboard import biofield_reveals as br
    token = "tk_" + _s.token_urlsafe(8)
    th = app_module._hash_token(token)
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        rid, _ = br.upsert(cx, email, "2026-06-20", {"greeting": "Hi", "body": "b"},
                           [{"name": "Top", "slug": "top", "meaning": "m"},
                            {"name": "Deep1", "slug": "deep1", "meaning": "m2"},
                            {"name": "Deep2", "slug": "deep2", "meaning": "m3"}], "s")
        br.set_token(cx, rid, th)
        br.approve_first(cx, rid, "glen")
        cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
                   (th, email, "biofield_reveal", datetime.now(timezone.utc).isoformat(),
                    (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()))
        cx.commit()
    return token


def test_unlock_checkout_flag_off(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", False, raising=False)
    token = _approved_reveal(app_module, db)
    r = app_module.app.test_client().post(f"/begin/biofield/{token}/unlock-checkout")
    assert r.get_json().get("ok") is False


def test_unlock_checkout_creates_dollar_session(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: None)
    captured = {}
    from dashboard import stripe_pay
    def _fake(amount_cents, **kw):
        captured["amount"] = amount_cents; captured.update(kw)
        return {"id": "cs_1", "url": "https://stripe.test/cs_1"}
    monkeypatch.setattr(stripe_pay, "create_checkout_session", _fake)
    token = _approved_reveal(app_module, db)
    r = app_module.app.test_client().post(f"/begin/biofield/{token}/unlock-checkout")
    body = r.get_json()
    assert body["ok"] is True and body["url"] == "https://stripe.test/cs_1"
    assert captured["amount"] == 100 and captured["save_card"] is True
    assert captured["metadata"]["kind"] == "biofield_trial" and captured["metadata"]["email"] == "t@x.com"


def test_unlock_checkout_already_member(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    token = _approved_reveal(app_module, db)
    r = app_module.app.test_client().post(f"/begin/biofield/{token}/unlock-checkout")
    assert r.get_json() == {"ok": True, "already": True}
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Add the flag + endpoint** in `app.py`:

```python
BIOFIELD_TRIAL_ENABLED = os.environ.get("BIOFIELD_TRIAL_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


@app.route("/begin/biofield/<token>/unlock-checkout", methods=["POST"])
def begin_biofield_unlock_checkout(token):
    if not (BIOFIELD_TRIAL_ENABLED and _STRIPE_ACTIVE):
        return jsonify({"ok": False, "error": "unavailable"}), 200
    from dashboard import biofield_reveals as _br, stripe_pay as _sp
    th = _hash_token((token or "").strip())
    valid, row = _biofield_verify_token(th)
    if not valid or row is None:
        return jsonify({"ok": False, "error": "invalid"}), 200
    email = (row.get("email") or "").strip().lower()
    if _active_membership_for_email(email):
        return jsonify({"ok": True, "already": True})
    base = PUBLIC_BASE_URL.rstrip("/")
    try:
        sess = _sp.create_checkout_session(
            100, customer_email=email,
            description="Biofield Analysis - full unlock",
            metadata={"email": email, "kind": "biofield_trial", "token": token},
            success_url=f"{base}/begin/checkout-return?kind=biofield_trial&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{base}/begin/biofield/{token}",
            save_card=True)
        return jsonify({"ok": True, "url": sess.get("url")})
    except Exception as e:
        print(f"[biofield-trial] checkout failed: {e!r}", flush=True)
        return jsonify({"ok": False, "error": "checkout_failed"}), 200
```

- [ ] **Step 4: Run -> PASS.**

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_biofield_trial.py
git commit -m "feat: begin #4b $1 biofield unlock-checkout endpoint (flag-gated)"
```

---

### Task 2: Return branch creates membership (subscription + grant), idempotent

**Files:** Modify `app.py` (`begin_checkout_return` + a `_grant_membership` helper + the idempotency marker); add tests.

**Interfaces produced:** `_grant_membership(cx, email, days, source) -> str` (inserts a `memberships` grant row, returns its id).

- [ ] **Step 1: Write the failing tests**

```python
def _mock_paid_session(app_module, monkeypatch, email="t@x.com", sid="cs_1"):
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda s: {"metadata": {"kind": "biofield_trial", "email": email}, "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi: {"customer": "cus_1", "payment_method": "pm_1", "status": "succeeded"})


def test_return_creates_membership_and_grant(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    _mock_paid_session(app_module, monkeypatch)
    c = app_module.app.test_client()
    c.get("/begin/checkout-return?kind=biofield_trial&session_id=cs_1")
    from dashboard import subscriptions
    with sqlite3.connect(db) as cx:
        subs = cx.execute("SELECT amount_cents, status, kind FROM subscriptions WHERE email='t@x.com'").fetchall()
        grants = cx.execute("SELECT source FROM memberships WHERE email='t@x.com'").fetchall()
    assert len(subs) == 1 and subs[0] == (9900, "active", "membership")
    assert len(grants) == 1 and grants[0][0] == "biofield_trial"
    assert app_module._active_membership_for_email("t@x.com") is not None


def test_return_idempotent(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    _mock_paid_session(app_module, monkeypatch)
    c = app_module.app.test_client()
    c.get("/begin/checkout-return?kind=biofield_trial&session_id=cs_1")
    c.get("/begin/checkout-return?kind=biofield_trial&session_id=cs_1")
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM subscriptions WHERE email='t@x.com'").fetchone()[0] == 1
        assert cx.execute("SELECT COUNT(*) FROM memberships WHERE email='t@x.com'").fetchone()[0] == 1


def test_return_unpaid_creates_nothing(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session", lambda s: {"metadata": {"kind": "biofield_trial", "email": "t@x.com"}, "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent", lambda pi: {"customer": "", "payment_method": "", "status": "requires_payment_method"})
    app_module.app.test_client().get("/begin/checkout-return?kind=biofield_trial&session_id=cs_1")
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM subscriptions WHERE email='t@x.com'").fetchone()[0] == 0
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Add the `_grant_membership` helper + the return branch.** Helper (near `_active_membership_for_email`):

```python
def _grant_membership(cx, email, days, source):
    import uuid as _uuid
    mid = str(_uuid.uuid4())
    now = datetime.utcnow()
    cx.execute(
        "INSERT INTO memberships (id, email, granted_at, expires_at, granted_by, source, truly_vip_ref, notes) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (mid, email, now.isoformat() + "Z", (now + timedelta(days=days)).isoformat() + "Z",
         source, source, "", ""))
    return mid
```

In `begin_checkout_return`, add a branch (after the `_kind`/`md`/`pi_id` are resolved, mirroring the `kind=subscribe` branch):

```python
                if _kind == "biofield_trial" and pi_id:
                    try:
                        from dashboard import subscriptions as _bt_subs, group_bundle as _bt_gb
                        import datetime as _bt_dt
                        pi = _sp.get_payment_intent(pi_id)
                        if (pi.get("status") == "succeeded") and pi.get("customer") and pi.get("payment_method"):
                            bt_email = (md.get("email") or "").strip().lower()
                            with _db_lock, sqlite3.connect(LOG_DB) as _bc:
                                _bc.execute("CREATE TABLE IF NOT EXISTS biofield_trial_grants (session_id TEXT PRIMARY KEY, email TEXT, granted_at TEXT)")
                                already = _bc.execute("SELECT 1 FROM biofield_trial_grants WHERE session_id=?", (sid,)).fetchone()
                                if not already and bt_email:
                                    _bt_subs.init_subscriptions_table(_bc)
                                    _bt_subs.migrate_add_membership_columns(_bc)
                                    _bt_subs.create_membership(
                                        _bc, email=bt_email, stripe_customer_id=pi["customer"],
                                        stripe_payment_method_id=pi["payment_method"],
                                        amount_cents=_bt_gb.MEMBERSHIP_AMOUNT_CENTS,
                                        next_charge_date=_bt_subs.add_months(_bt_dt.date.today().isoformat(), 1))
                                    _grant_membership(_bc, bt_email, 31, "biofield_trial")
                                    _bc.execute("INSERT INTO biofield_trial_grants (session_id, email, granted_at) VALUES (?,?,?)",
                                                (sid, bt_email, datetime.utcnow().isoformat() + "Z"))
                                    _bc.commit()
                    except Exception as e:
                        print(f"[biofield-trial] grant failed: {e!r}", flush=True)
                    return _redir(f"/begin/biofield/{md.get('token', '')}")
```

(Place this branch alongside the existing `_kind ==` branches; `_sp`, `_redir`, `_db_lock`, `datetime`, `timedelta` are in scope.)

- [ ] **Step 4: Run -> PASS.**

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_biofield_trial.py
git commit -m "feat: begin #4b checkout-return creates membership + grant (idempotent)"
```

---

### Task 3: Release the deeper remedies for paid members (page-data)

**Files:** Modify `app.py` (`begin_biofield_reveal` payload); add tests.

- [ ] **Step 1: Write the failing tests** (a PAID member's reveal page-data -> full remedies + blurred_count 0; a NON-paid member -> unchanged #4a). Use `_make_reveal` style helper from the #4a tests but assert the new payload. Concretely: monkeypatch `_active_membership_for_email` to return active, GET `/begin/biofield/<token>`, assert the served HTML / injected `__REVEAL__` payload contains the deeper remedy names and `blurred_count` 0; with it returning None, assert deeper names absent and `blurred_count` > 0.

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Modify the member branch** of `begin_biofield_reveal` (the payload built when `member` is true, ~line 1515): compute `paid = bool(_active_membership_for_email(email))`. When `paid`, set:

```python
        remedies = row.get("remedies") or []
        full = [_biofield_remedy_payload(r) for r in remedies]   # top + deep, each {name, meaning, buy_url, page_url}
        payload = {
            "interpretation": row.get("interpretation") or {},
            "blurred_count": 0,
            "first_approved": first_approved,
            "free_available": False,
            "top_unlocked": True,
            "paid": True,
            "trial_enabled": BIOFIELD_TRIAL_ENABLED,
            "remedies": full,
        }
```

(Add a small `_biofield_remedy_payload(r)` helper that returns `{name, meaning, buy_url, page_url}` for any remedy dict, reusing the slug -> `/begin/buy/<slug>` and `/begin/product/<slug>` rules already in `_biofield_top_payload`; refactor `_biofield_top_payload` to call it for remedies[0].) When NOT paid, the existing branch stands, but ALSO add `"paid": False` and `"trial_enabled": BIOFIELD_TRIAL_ENABLED` to that payload so the page can render the $1 CTA. Do not emit any deeper remedy content in the non-paid branch (anti-bypass).

- [ ] **Step 4: Run -> PASS.**

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_biofield_trial.py
git commit -m "feat: begin #4b release full remedies for paid members + trial flag in payload"
```

---

### Task 4: One-click cancel

**Files:** Modify `app.py` (mint a cancel token at grant time + `GET /membership/cancel/<token>`); Create `static/membership-cancel.html` (or render inline); add tests.

- [ ] **Step 1: Write the failing tests** (valid cancel token -> the email's active `kind='membership'` subscription becomes `cancelled`; invalid token -> friendly 200 page, no change; double-cancel -> still cancelled).

- [ ] **Step 2-4: Implement.** At grant time (Task 2's branch), also mint a cancel token: `cancel_tok = secrets.token_urlsafe(32)`; INSERT `auth_tokens (token_hash=_hash_token(cancel_tok), email=bt_email, purpose="membership_cancel", created_at, expires_at=+60d)`. (Expose it to the page via the post-unlock confirmation; for the reveal, the verified owner can be given a cancel link minted on demand - simplest: a `POST /begin/biofield/<token>/cancel-link` that mints+returns a cancel URL for the reveal's email when they are a paid member. The plan's minimal version mints the cancel token at grant and the front-end confirmation links it.) Add:

```python
@app.route("/membership/cancel/<token>", methods=["GET"])
def membership_cancel(token):
    from dashboard import subscriptions as _subs
    th = _hash_token((token or "").strip())
    email = None
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        row = cx.execute("SELECT email, expires_at FROM auth_tokens WHERE token_hash=? AND purpose='membership_cancel'", (th,)).fetchone()
        if row:
            try:
                if datetime.fromisoformat((row[1] or "").replace("Z", "+00:00")) >= datetime.now(timezone.utc):
                    email = row[0]
            except Exception:
                email = None
        if email:
            _subs.init_subscriptions_table(cx)
            sub = cx.execute("SELECT id FROM subscriptions WHERE email=? AND kind='membership' AND status='active' ORDER BY id DESC LIMIT 1", (email,)).fetchone()
            if sub:
                _subs.set_status(cx, sub[0], "cancelled")
    resp = send_from_directory(STATIC, "membership-cancel.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp
```

`static/membership-cancel.html`: a calm "Your trial is cancelled - no further charge. Your access continues until the end of your paid period." (static, no PHI). No emoji/em dash.

- [ ] **Step 5: Commit**

```bash
git add app.py static/membership-cancel.html tests/test_biofield_trial.py
git commit -m "feat: begin #4b one-click membership cancel (tokened)"
```

---

### Task 5: Reveal page CTA wiring + confirmation

**Files:** Modify `static/begin-biofield.html`; add a serve assertion.

- [ ] **Step 1: Write the serve test** (the page references the unlock-checkout POST + reads `paid`/`trial_enabled`).

- [ ] **Step 2-4: Update `static/begin-biofield.html`** (from #4a):
  - When `__REVEAL__.paid` is true: render ALL `remedies` (top + deep), each name a `_blank` link to `page_url` + an Order button to `buy_url` (reuse the top-remedy rendering for every entry); no blurred stack, no $1 CTA.
  - When NOT paid AND `__REVEAL__.trial_enabled`: the "Unlock your full Biofield Analysis" CTA becomes an active button "Unlock your full analysis ($1)" that `POST`s to `/begin/biofield/<token>/unlock-checkout` (token from `location.pathname`), and on `{ok:true,url}` does `location.href = url` (or `{ok:true,already:true}` -> `location.reload()`).
  - When NOT paid AND NOT trial_enabled: the disabled "unlocking soon" stub (current #4a).
  - All dynamic text via textContent; links via setAttribute. No emoji/em dash.

- [ ] **Step 5: Run the focused tests + the begin sweep**

Run: `... -m pytest tests/test_biofield_trial.py tests/test_biofield_reveal_routes.py -v`
Then: `... -m pytest tests/ -k "biofield or begin" -v` - no regressions.

- [ ] **Step 6: Commit**

```bash
git add static/begin-biofield.html tests/test_biofield_trial.py
git commit -m "feat: begin #4b reveal page - \$1 unlock CTA + paid full-remedy render"
```

---

## Self-Review

**1. Spec coverage:** flag + $1 checkout (T1); return creates subscription + grant idempotently (T2); deeper-remedy release for paid (T3); one-click cancel (T4); CTA + paid render (T5). Both membership records (subscription + grant) -> T2. Anti-bypass (deep content only when paid) -> T3 (non-paid branch emits no deep content). `BIOFIELD_TRIAL_ENABLED` dark + `_STRIPE_ACTIVE` -> T1 guard + T5 CTA. Auto-convert $99 at +30d -> T2 `next_charge_date=add_months(...,1)` + the existing cron (untouched).

**2. Placeholder scan:** No TBD. T3/T4/T5 describe the exact payload/route/page deltas with code or named-anchor edits; the cancel-token delivery has a concrete minimal path (mint at grant + the confirmation links it).

**3. Type consistency:** `_grant_membership(cx, email, days, source)` defined in T2, used in T2 (+ optionally T4 mint). The `biofield_trial_grants(session_id PK)` idempotency marker consistent T2. `_biofield_remedy_payload(r) -> {name, meaning, buy_url, page_url}` (T3) reused for top + deep; `_biofield_top_payload` refactored to call it. Payload keys `paid`/`trial_enabled`/`remedies`/`blurred_count` consistent between the route (T3) and the page (T5). `create_membership(amount_cents=MEMBERSHIP_AMOUNT_CENTS=9900, next_charge_date=+1mo)` matches the spec. The cancel route reads `auth_tokens purpose="membership_cancel"` minted in T2.
