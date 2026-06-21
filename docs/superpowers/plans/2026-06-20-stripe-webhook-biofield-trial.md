# Stripe Webhook for $1 Biofield Trial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the $1-trial membership from whichever of the Stripe success-redirect OR a new `checkout.session.completed` webhook arrives first, idempotently, so a closed tab never leaves a paid customer without a membership.

**Architecture:** Extract the `biofield_trial` membership-creation block out of `begin_checkout_return` into a shared, idempotent `_fulfill_biofield_trial(session_id)` (re-fetches the session + PaymentIntent from Stripe, claim-then-creates on the existing `biofield_trial_grants(session_id PK)` marker). Add `POST /webhook/stripe` that calls it on a `checkout.session.completed` event, plus a `stripe_pay.verify_webhook` signature helper. The redirect and the webhook call the same function; the marker dedupes.

**Tech Stack:** Python 3.11 / Flask (single `app.py`), `dashboard/stripe_pay.py`, SQLite (`LOG_DB`), pytest. Stripe HMAC-SHA256 webhook signatures.

## Global Constraints

- No emoji, no em dashes (code, comments, commit messages).
- Scope is the `biofield_trial` checkout kind ONLY. Do not change other checkout kinds, the $1 checkout creation, the pricing/billing engine, the daily charge cron, or the reveal.
- Idempotent: the redirect and the webhook call the SAME `_fulfill_biofield_trial`; the existing `biofield_trial_grants(session_id PK)` claim-then-create marker guarantees exactly one membership (live money - no double-create).
- Security via independent re-fetch: `_fulfill_biofield_trial` re-fetches the session + PaymentIntent from Stripe and only creates the membership on `status=="succeeded"` with a `customer` and `payment_method`. Correctness holds even before `STRIPE_WEBHOOK_SECRET` is set; signature verification is defense-in-depth on top.
- The redirect path (`begin_checkout_return`) stays behavior-identical and best-effort / never-500.
- No new feature flag. The webhook is inert until Stripe is configured to deliver events; go-live is a Stripe-dashboard config step + `STRIPE_WEBHOOK_SECRET` in Doppler.
- Test runner: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest <target> -v`. Mock Stripe; tmp `LOG_DB` via `monkeypatch.setattr(app_module, "LOG_DB", db)`.

---

## Critical files

- `dashboard/stripe_pay.py` - add `verify_webhook(payload, sig_header, secret, tolerance=300) -> dict|None`.
- `app.py`
  - Add `_fulfill_biofield_trial(session_id) -> dict` (place it just above `def begin_checkout_return` ~4654).
  - Replace the inline `biofield_trial` block in `begin_checkout_return` (~4901-4940) with a call to it.
  - Add `POST /webhook/stripe` (near the other `/webhook/*` routes, e.g. after `/webhook/groovekart` ~12724).
- Tests: `tests/test_stripe_webhook.py` (new). The existing `tests/test_biofield_trial.py` must stay green (the redirect path is behavior-identical).

---

## Task 1: `verify_webhook` signature helper (`dashboard/stripe_pay.py`)

**Files:**
- Modify: `dashboard/stripe_pay.py` (add `verify_webhook`)
- Test: `tests/test_stripe_webhook.py`

**Interfaces:**
- Produces: `verify_webhook(payload, sig_header, secret, tolerance=300) -> dict|None` - parses + verifies a Stripe `Stripe-Signature` header (`t=<ts>,v1=<sig>`) via HMAC-SHA256 over `f"{ts}.".encode()+payload`; returns the parsed JSON event on success, `None` on any failure (bad/missing sig, wrong secret, stale timestamp, unparseable body). Pure, no network.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stripe_webhook.py`:

```python
# tests/test_stripe_webhook.py
import importlib, sqlite3, sys, json, time, hmac, hashlib
from pathlib import Path
import pytest


def _load(mod):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module(mod)
    except Exception as e:
        pytest.skip(f"{mod} not importable: {e}")


def _sign(payload: bytes, secret: str, ts: int) -> str:
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    return f"t={ts},v1={sig}"


def test_verify_webhook_accepts_valid():
    sp = _load("dashboard.stripe_pay")
    body = json.dumps({"type": "checkout.session.completed", "data": {"object": {"id": "cs_1"}}}).encode()
    ts = int(time.time())
    ev = sp.verify_webhook(body, _sign(body, "whsec_test", ts), "whsec_test")
    assert ev and ev["type"] == "checkout.session.completed" and ev["data"]["object"]["id"] == "cs_1"


def test_verify_webhook_rejects_tampered_body():
    sp = _load("dashboard.stripe_pay")
    body = b'{"type":"checkout.session.completed"}'
    ts = int(time.time())
    sig = _sign(body, "whsec_test", ts)
    assert sp.verify_webhook(b'{"type":"evil"}', sig, "whsec_test") is None


def test_verify_webhook_rejects_wrong_secret():
    sp = _load("dashboard.stripe_pay")
    body = b'{"a":1}'
    ts = int(time.time())
    assert sp.verify_webhook(body, _sign(body, "whsec_test", ts), "whsec_other") is None


def test_verify_webhook_rejects_stale():
    sp = _load("dashboard.stripe_pay")
    body = b'{"a":1}'
    old = int(time.time()) - 10000
    assert sp.verify_webhook(body, _sign(body, "whsec_test", old), "whsec_test", tolerance=300) is None


def test_verify_webhook_rejects_malformed_header():
    sp = _load("dashboard.stripe_pay")
    assert sp.verify_webhook(b'{}', "garbage", "whsec_test") is None
    assert sp.verify_webhook(b'{}', "", "whsec_test") is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_stripe_webhook.py -k verify_webhook -v`
Expected: FAIL (`verify_webhook` does not exist).

- [ ] **Step 3: Add `verify_webhook`**

In `dashboard/stripe_pay.py`, append:

```python
def verify_webhook(payload, sig_header, secret, tolerance=300):
    """Verify a Stripe webhook signature. payload = the raw request body (bytes or str).
    Returns the parsed event dict on success, None on any failure (bad/missing/stale
    signature, wrong secret, unparseable body). Pure; no network."""
    import hmac, hashlib, json, time
    try:
        payload_b = payload.encode("utf-8") if isinstance(payload, str) else payload
        items = dict(p.split("=", 1) for p in (sig_header or "").split(",") if "=" in p)
        ts, v1 = items.get("t"), items.get("v1")
        if not ts or not v1:
            return None
        expected = hmac.new(secret.encode("utf-8"), f"{ts}.".encode("utf-8") + payload_b,
                            hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, v1):
            return None
        if tolerance and abs(time.time() - int(ts)) > tolerance:
            return None
        return json.loads(payload_b.decode("utf-8"))
    except Exception:
        return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_stripe_webhook.py -k verify_webhook -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/stripe_pay.py tests/test_stripe_webhook.py
git commit -m "feat: stripe_pay.verify_webhook signature helper"
```

---

## Task 2: Extract `_fulfill_biofield_trial` + rewire the redirect (`app.py`)

**Files:**
- Modify: `app.py` (add `_fulfill_biofield_trial`; replace the inline `biofield_trial` block in `begin_checkout_return`)
- Test: `tests/test_stripe_webhook.py`

**Interfaces:**
- Consumes: `dashboard.stripe_pay` (`get_session`, `get_payment_intent`), `dashboard.subscriptions` (`init_subscriptions_table`, `migrate_add_membership_columns`, `create_membership`, `add_months`), `dashboard.group_bundle.MEMBERSHIP_AMOUNT_CENTS`, `_grant_membership`, `init_membership_tables`, `_hash_token`, `secrets`, `_db_lock`, `LOG_DB`, `datetime`, `timezone`, `timedelta`.
- Produces: `_fulfill_biofield_trial(session_id) -> dict` - re-fetches the session + PI from Stripe; creates the `subscriptions` membership ($9900, next +1mo) + `memberships` grant + `membership_cancel` token, idempotently on `biofield_trial_grants(session_id PK)`. Returns `{ok, created|already, email}` or `{ok:False, reason}`. Never raises.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_stripe_webhook.py` (mirrors the harness in `tests/test_biofield_trial.py`):

```python
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
    from dashboard import subscriptions
    with sqlite3.connect(db) as cx:
        subscriptions.init_subscriptions_table(cx)
        subscriptions.migrate_add_membership_columns(cx)
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        cx.commit()
    return db


def _mock_paid_trial(app_module, monkeypatch, email="t@x.com", succeeded=True):
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda s: {"metadata": {"kind": "biofield_trial", "email": email, "token": "tk"}, "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi: {"customer": "cus_1", "payment_method": "pm_1",
                    "status": "succeeded" if succeeded else "requires_payment_method"})


def test_fulfill_creates_membership_and_grant(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _mock_paid_trial(app_module, monkeypatch)
    res = app_module._fulfill_biofield_trial("cs_x")
    assert res["ok"] is True
    with sqlite3.connect(db) as cx:
        subs = cx.execute("SELECT amount_cents, status, kind FROM subscriptions WHERE email='t@x.com'").fetchall()
        grants = cx.execute("SELECT source FROM memberships WHERE email='t@x.com'").fetchall()
    assert subs == [(9900, "active", "membership")] and grants == [("biofield_trial",)]
    assert app_module._active_membership_for_email("t@x.com") is not None


def test_fulfill_idempotent(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _mock_paid_trial(app_module, monkeypatch)
    app_module._fulfill_biofield_trial("cs_x")
    r2 = app_module._fulfill_biofield_trial("cs_x")
    assert r2.get("already") is True
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM subscriptions WHERE email='t@x.com'").fetchone()[0] == 1
        assert cx.execute("SELECT COUNT(*) FROM memberships WHERE email='t@x.com'").fetchone()[0] == 1


def test_fulfill_unpaid_creates_nothing(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _mock_paid_trial(app_module, monkeypatch, succeeded=False)
    res = app_module._fulfill_biofield_trial("cs_x")
    assert res["ok"] is False
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM subscriptions WHERE email='t@x.com'").fetchone()[0] == 0


def test_fulfill_non_trial_noop(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session", lambda s: {"metadata": {"kind": "reorder"}})
    res = app_module._fulfill_biofield_trial("cs_x")
    assert res["ok"] is False and res.get("reason") == "not_trial"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_stripe_webhook.py -k fulfill -v`
Expected: FAIL (`_fulfill_biofield_trial` does not exist).

- [ ] **Step 3: Add `_fulfill_biofield_trial`**

In `app.py`, immediately above `def begin_checkout_return():` (~line 4654), add:

```python
def _fulfill_biofield_trial(session_id):
    """Create the $1-trial membership (subscription + access grant + cancel token) from a
    paid biofield_trial Stripe session, idempotently. Callable from the redirect return AND
    the webhook. Re-fetches the session + PaymentIntent from Stripe (the security guarantee);
    only proceeds on a succeeded payment with a vaulted card. Never raises."""
    try:
        from dashboard import stripe_pay as _sp, subscriptions as _bt_subs, group_bundle as _bt_gb
        import datetime as _bt_dt
        sess = _sp.get_session(session_id)
        md = sess.get("metadata") or {}
        if md.get("kind") != "biofield_trial":
            return {"ok": False, "reason": "not_trial"}
        bt_email = (md.get("email") or "").strip().lower()
        pi_id = sess.get("payment_intent")
        if not pi_id:
            return {"ok": False, "reason": "unpaid"}
        pi = _sp.get_payment_intent(pi_id)
        if not (pi.get("status") == "succeeded" and pi.get("customer") and pi.get("payment_method")):
            return {"ok": False, "reason": "unpaid"}
        with _db_lock, sqlite3.connect(LOG_DB) as _bc:
            _bc.execute("CREATE TABLE IF NOT EXISTS biofield_trial_grants (session_id TEXT PRIMARY KEY, email TEXT, granted_at TEXT)")
            # Claim-then-create: write the idempotency marker FIRST and commit, so the redirect
            # and the webhook (in any order, even simultaneously) create exactly one membership.
            claimed = bool(bt_email) and _bc.execute(
                "INSERT OR IGNORE INTO biofield_trial_grants (session_id, email, granted_at) VALUES (?,?,?)",
                (session_id, bt_email, datetime.utcnow().isoformat() + "Z")).rowcount == 1
            _bc.commit()
            if not claimed:
                return {"ok": True, "already": True, "email": bt_email}
            _bt_subs.init_subscriptions_table(_bc)
            _bt_subs.migrate_add_membership_columns(_bc)
            init_membership_tables(_bc)
            _bt_subs.create_membership(
                _bc, email=bt_email, stripe_customer_id=pi["customer"],
                stripe_payment_method_id=pi["payment_method"],
                amount_cents=_bt_gb.MEMBERSHIP_AMOUNT_CENTS,
                next_charge_date=_bt_subs.add_months(_bt_dt.date.today().isoformat(), 1))
            _grant_membership(_bc, bt_email, 31, "biofield_trial")
            cancel_tok = secrets.token_urlsafe(32)
            _bc.execute(
                "INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
                (_hash_token(cancel_tok), bt_email, "membership_cancel",
                 datetime.now(timezone.utc).isoformat(),
                 (datetime.now(timezone.utc) + timedelta(days=60)).isoformat()))
            _bc.commit()
            return {"ok": True, "created": True, "email": bt_email}
    except Exception as e:
        print(f"[biofield-trial] fulfill failed: {e!r}", flush=True)
        return {"ok": False, "reason": "error"}
```

- [ ] **Step 4: Rewire `begin_checkout_return` to call it**

In `app.py`, in `begin_checkout_return`, replace the entire inline `biofield_trial` block (the `# -- Biofield trial: create subscription + access grant (idempotent) --` comment + the `if _kind == "biofield_trial":` try/except, app.py ~4900-4940) with:

```python
            # Biofield trial: membership creation is shared with the Stripe webhook so a
            # closed tab still gets fulfilled. Idempotent via biofield_trial_grants.
            if _kind == "biofield_trial":
                _fulfill_biofield_trial(sid)
```

(Leave everything else in `begin_checkout_return` unchanged, including the `if _kind == "biofield_trial": return _redir(f"/begin/biofield/{_bt_token}")` near the end.)

- [ ] **Step 5: Run the tests + the existing trial suite (no regression)**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_stripe_webhook.py tests/test_biofield_trial.py -v`
Expected: PASS (the Task 1 verify_webhook tests + the 4 new fulfill tests + all existing biofield_trial tests, including the redirect-path `test_return_creates_membership_and_grant` / `test_return_idempotent` / `test_return_unpaid_creates_nothing` which now exercise the extracted function).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_stripe_webhook.py
git commit -m "refactor: extract _fulfill_biofield_trial; redirect calls it (shared with webhook)"
```

---

## Task 3: `POST /webhook/stripe` (`app.py`)

**Files:**
- Modify: `app.py` (add the webhook route)
- Test: `tests/test_stripe_webhook.py`

**Interfaces:**
- Consumes: `_fulfill_biofield_trial`, `dashboard.stripe_pay.verify_webhook`, `request`, `os`, `json`.
- Produces: `POST /webhook/stripe` - on a `checkout.session.completed` event, calls `_fulfill_biofield_trial(session_id)`; 200 for handled (ignored type / fulfilled / already), 400 on a present-but-invalid signature, 500 only on an unexpected fulfillment exception.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_stripe_webhook.py`:

```python
def _event(session_id="cs_evt", etype="checkout.session.completed"):
    return json.dumps({"type": etype, "data": {"object": {"id": session_id}}}).encode()


def test_webhook_completed_calls_fulfill(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    seen = {}
    monkeypatch.setattr(app_module, "_fulfill_biofield_trial", lambda sid: seen.setdefault("sid", sid) or {"ok": True})
    r = app_module.app.test_client().post("/webhook/stripe", data=_event("cs_42"), content_type="application/json")
    assert r.status_code == 200 and seen.get("sid") == "cs_42"


def test_webhook_other_event_noop(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    called = {"n": 0}
    monkeypatch.setattr(app_module, "_fulfill_biofield_trial", lambda sid: called.__setitem__("n", called["n"] + 1))
    r = app_module.app.test_client().post("/webhook/stripe", data=_event(etype="payment_intent.created"), content_type="application/json")
    assert r.status_code == 200 and called["n"] == 0


def test_webhook_bad_signature_400(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    called = {"n": 0}
    monkeypatch.setattr(app_module, "_fulfill_biofield_trial", lambda sid: called.__setitem__("n", called["n"] + 1))
    r = app_module.app.test_client().post("/webhook/stripe", data=_event("cs_42"),
                                          headers={"Stripe-Signature": "t=1,v1=bad"}, content_type="application/json")
    assert r.status_code == 400 and called["n"] == 0


def test_webhook_valid_signature_200(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    monkeypatch.setattr(app_module, "_fulfill_biofield_trial", lambda sid: {"ok": True})
    body = _event("cs_42")
    ts = int(time.time())
    sig = _sign(body, "whsec_test", ts)
    r = app_module.app.test_client().post("/webhook/stripe", data=body,
                                          headers={"Stripe-Signature": sig}, content_type="application/json")
    assert r.status_code == 200
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_stripe_webhook.py -k webhook -v`
Expected: FAIL (route 404).

- [ ] **Step 3: Add the webhook route**

In `app.py`, after the `/webhook/groovekart` route (~line 12724, near the other webhooks), add:

```python
@app.route("/webhook/stripe", methods=["POST"])
def webhook_stripe():
    """Stripe webhook: create the $1-trial membership on checkout.session.completed,
    independent of the success-redirect (a closed tab still gets fulfilled). Idempotent
    via _fulfill_biofield_trial. Signature-verified when STRIPE_WEBHOOK_SECRET is set;
    otherwise the body is parsed directly (the re-fetch in fulfillment is the guarantee)."""
    from dashboard import stripe_pay as _sp
    raw = request.get_data()
    secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    if secret:
        event = _sp.verify_webhook(raw, request.headers.get("Stripe-Signature", ""), secret)
        if event is None:
            return ("", 400)
    else:
        try:
            event = json.loads(raw.decode("utf-8"))
        except Exception:
            return ("", 400)
    try:
        if (event or {}).get("type") == "checkout.session.completed":
            session_id = (((event.get("data") or {}).get("object") or {}).get("id") or "").strip()
            if session_id:
                _fulfill_biofield_trial(session_id)
        return ("", 200)
    except Exception as e:
        print(f"[webhook-stripe] {e!r}", flush=True)
        return ("", 500)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_stripe_webhook.py -v`
Expected: PASS (the whole file).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_stripe_webhook.py
git commit -m "feat: POST /webhook/stripe creates biofield-trial membership on checkout.session.completed"
```

---

## Verification

- Per task: the named `pytest` target passes (doppler + venv).
- Full sweep after Task 3: `tests/test_stripe_webhook.py` all green; plus `tests/test_biofield_trial.py -v` (the redirect path is behavior-identical through the extracted function) green.
- Final Opus whole-branch review (focus: the extraction is behavior-identical to the old inline block - same claim-then-create idempotency, same succeeded-PI gate, same subscription + grant + cancel-token; the redirect and the webhook share `_fulfill_biofield_trial` so they cannot double-create; `verify_webhook` is correct - HMAC over `f"{ts}."+payload`, constant-time compare, timestamp tolerance, returns None on any failure; the webhook gates on the signature only when the secret is set, returns 200 for handled / 400 for bad-sig / 500 only on unexpected exception; never-raises contracts hold; the re-fetch is the security guarantee so a forged event cannot mint a membership; no emoji/em-dash; YAGNI - biofield_trial only).
- Manual go-live (documented, after merge): in the Stripe dashboard add a webhook endpoint -> `https://illtowell.com/webhook/stripe`, subscribe to `checkout.session.completed`, copy the signing secret into Doppler `remedy-match/prd` as `STRIPE_WEBHOOK_SECRET`. Then a tab-closed $1 payment is fulfilled by the webhook within seconds. (Until configured, the endpoint is inert because Stripe sends nothing to it; the redirect path is unchanged.)
- Ship via PR + merge to `main` (auto-deploys). Gentle probe: `POST /webhook/stripe` with an unhandled event type -> 200 (no-op), per the warm-up rule. Update memory.

## Build order
Task 1 (verify_webhook) -> Task 2 (extract _fulfill_biofield_trial + rewire redirect) -> Task 3 (webhook route). Task 3 depends on Tasks 1 + 2. After merge, configure the Stripe dashboard webhook + set STRIPE_WEBHOOK_SECRET to activate delivery.
