# Biofield Trial One-Click Cancel Email Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the already-built `/membership/cancel/<token>` link to the $1-trial member by emailing it at membership creation, and extend the cancel token's TTL so the link outlives the recurring membership.

**Architecture:** A single function (`_fulfill_biofield_trial` in `app.py`) already mints the `membership_cancel` token but discards the plaintext. We capture the plaintext, extend its TTL via a named constant, and send one best-effort welcome/cancel email after the DB lock is released. The send sits inside the won-claim path so it fires exactly once across the redirect and webhook fulfillment callers.

**Tech Stack:** Python 3.11, Flask, SQLite, the existing `_send_inquiry_email` SMTP helper.

## Global Constraints

- No emoji. No em dashes (use a plain hyphen). Copy is provisional (BNSN later).
- This is LIVE money (#4b `BIOFIELD_TRIAL_ENABLED=true` in prod). No new feature flag; the change only adds an email and lengthens a token.
- The email send is best-effort: a send failure must never undo or block membership creation.
- The email must fire exactly once per membership (across redirect + webhook), riding the existing `biofield_trial_grants` claim-then-create idempotency.
- The SMTP send must happen AFTER the `_db_lock` / sqlite connection block is released (do not hold the global DB lock during an SMTP round-trip).
- Test harness: `importlib` app load; tmp `LOG_DB` via a monkeypatched attribute; Stripe (`stripe_pay.get_session` / `get_payment_intent`) and SMTP (`_send_inquiry_email`) mocked. Run:
  `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest <target> -v`

---

## File Structure

- `app.py`
  - New module constant `MEMBERSHIP_CANCEL_TTL_DAYS` near `AUTH_TOKEN_TTL_MIN` (~line 171).
  - `_fulfill_biofield_trial` (~line 4685): lift the next-charge date into a local, extend the token TTL, move the success `return` out of the `with` block, and send the welcome/cancel email after the block.
- `tests/test_membership_cancel_email.py` (new): the 4 tests for this change.

No other files change. `/membership/cancel/<token>`, the Stripe webhook, billing, and the cron are untouched.

---

## Task 1: Email the one-click cancel link + extend its TTL

**Files:**
- Modify: `app.py:171` (add constant), `app.py:4685-4733` (`_fulfill_biofield_trial`)
- Test: `tests/test_membership_cancel_email.py` (create)

**Interfaces:**
- Consumes: `_send_inquiry_email(to_email, subject, body, reply_to=None) -> bool` (never raises; prints in dev when SMTP env unset); `PUBLIC_BASE_URL` (module global, no trailing slash); `secrets`, `_hash_token`, `datetime`, `timezone`, `timedelta` (already imported in `app.py`); `dashboard.subscriptions.add_months(yyyy_mm_dd, n)`.
- Produces: no new public symbol other than `MEMBERSHIP_CANCEL_TTL_DAYS`. `_fulfill_biofield_trial`'s return values are unchanged (`{"ok": True, "created": True, "email": ...}` etc.).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_membership_cancel_email.py`:

```python
# tests/test_membership_cancel_email.py
import importlib, re, sqlite3, sys
from datetime import datetime
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
    from dashboard import subscriptions
    with sqlite3.connect(db) as cx:
        subscriptions.init_subscriptions_table(cx)
        subscriptions.migrate_add_membership_columns(cx)
        cx.execute(
            "CREATE TABLE IF NOT EXISTS auth_tokens "
            "(token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        cx.commit()
    return db


def _mock_paid_session(app_module, monkeypatch, email="t@x.com"):
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda s: {"metadata": {"kind": "biofield_trial", "email": email}, "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi: {"customer": "cus_1", "payment_method": "pm_1", "status": "succeeded"})


def test_fulfill_sends_welcome_email_with_working_cancel_link(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _mock_paid_session(app_module, monkeypatch)
    sent = []
    monkeypatch.setattr(app_module, "_send_inquiry_email",
                        lambda to, subj, body, **kw: sent.append((to, subj, body)) or True)
    res = app_module._fulfill_biofield_trial("cs_1")
    assert res.get("created") is True
    assert len(sent) == 1, f"expected exactly one email, got {len(sent)}"
    to, subj, body = sent[0]
    assert to == "t@x.com"
    m = re.search(r"/membership/cancel/(\S+)", body)
    assert m, "no cancel link in email body"
    tok = m.group(1)
    # the emailed link actually cancels the membership
    r = app_module.app.test_client().get(f"/membership/cancel/{tok}")
    assert r.status_code == 200
    with sqlite3.connect(db) as cx:
        status = cx.execute(
            "SELECT status FROM subscriptions WHERE email='t@x.com' AND kind='membership'").fetchone()
    assert status is not None and status[0] == "cancelled", f"got {status}"


def test_welcome_email_sent_exactly_once(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    _mock_paid_session(app_module, monkeypatch)
    sent = []
    monkeypatch.setattr(app_module, "_send_inquiry_email",
                        lambda to, subj, body, **kw: sent.append(to) or True)
    app_module._fulfill_biofield_trial("cs_1")
    app_module._fulfill_biofield_trial("cs_1")
    assert len(sent) == 1, f"expected one email across two fulfills, got {len(sent)}"


def test_email_failure_does_not_break_fulfillment(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _mock_paid_session(app_module, monkeypatch)
    def _boom(*a, **k):
        raise RuntimeError("smtp down")
    monkeypatch.setattr(app_module, "_send_inquiry_email", _boom)
    res = app_module._fulfill_biofield_trial("cs_1")
    assert res.get("created") is True, f"fulfillment should still succeed, got {res}"
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM subscriptions WHERE email='t@x.com'").fetchone()[0] == 1
        assert cx.execute("SELECT COUNT(*) FROM memberships WHERE email='t@x.com'").fetchone()[0] == 1


def test_cancel_token_ttl_extended(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _mock_paid_session(app_module, monkeypatch)
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)
    app_module._fulfill_biofield_trial("cs_1")
    with sqlite3.connect(db) as cx:
        row = cx.execute(
            "SELECT created_at, expires_at FROM auth_tokens "
            "WHERE purpose='membership_cancel' AND email='t@x.com'").fetchone()
    assert row, "no membership_cancel token minted"
    created = datetime.fromisoformat(row[0])
    expires = datetime.fromisoformat(row[1])
    days = (expires - created).days
    assert days >= 365, f"cancel token TTL too short: {days} days"
    assert days == app_module.MEMBERSHIP_CANCEL_TTL_DAYS
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_membership_cancel_email.py -v`
Expected: FAIL - `test_cancel_token_ttl_extended` errors on `AttributeError: module 'app' has no attribute 'MEMBERSHIP_CANCEL_TTL_DAYS'`; the email tests fail because no email is sent (`len(sent) == 0`).

- [ ] **Step 3: Add the TTL constant**

In `app.py`, after `AUTH_TOKEN_TTL_LABEL` (~line 172), add:

```python
MEMBERSHIP_CANCEL_TTL_DAYS = 1095  # ~3 years: the emailed one-click cancel link must outlive the recurring membership
```

- [ ] **Step 4: Capture the next-charge date, extend the TTL, send the email outside the lock**

In `_fulfill_biofield_trial` (~line 4717-4730), edit the won-claim block. Replace this:

```python
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

with this (note the success `return` moves out of the `with` block, and the email send follows it):

```python
            next_charge = _bt_subs.add_months(_bt_dt.date.today().isoformat(), 1)
            _bt_subs.create_membership(
                _bc, email=bt_email, stripe_customer_id=pi["customer"],
                stripe_payment_method_id=pi["payment_method"],
                amount_cents=_bt_gb.MEMBERSHIP_AMOUNT_CENTS,
                next_charge_date=next_charge)
            _grant_membership(_bc, bt_email, 31, "biofield_trial")
            cancel_tok = secrets.token_urlsafe(32)
            _bc.execute(
                "INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
                (_hash_token(cancel_tok), bt_email, "membership_cancel",
                 datetime.now(timezone.utc).isoformat(),
                 (datetime.now(timezone.utc) + timedelta(days=MEMBERSHIP_CANCEL_TTL_DAYS)).isoformat()))
            _bc.commit()
        # Lock released. Send the welcome / one-click-cancel email best-effort: it must
        # never undo the committed membership. Inside the won-claim path, so exactly once.
        try:
            if bt_email and PUBLIC_BASE_URL:
                cancel_url = f"{PUBLIC_BASE_URL}/membership/cancel/{cancel_tok}"
                subject = "You're in - your membership is active"
                body = (
                    "Aloha,\n\n"
                    "Your $1 unlocked your full Biofield Analysis and started your membership. "
                    f"Your first monthly payment of $99 will run on {next_charge}. Everything "
                    "stays unlocked in the meantime, and you can order your matched remedies anytime.\n\n"
                    "No pressure, ever. If you want to cancel before your first payment, it is one "
                    "click, no charge, no reply needed:\n\n"
                    f"{cancel_url}\n\n"
                    "In wellness,\n"
                    "Dr. Glen and Rae\n"
                )
                _send_inquiry_email(bt_email, subject, body)
            else:
                print("[biofield-trial] welcome-email skipped (missing email or base url)", flush=True)
        except Exception as _e:
            print(f"[biofield-trial] welcome-email failed: {_e!r}", flush=True)
        return {"ok": True, "created": True, "email": bt_email}
    except Exception as e:
        print(f"[biofield-trial] fulfill failed: {e!r}", flush=True)
        return {"ok": False, "reason": "error"}
```

Dedent matters: the `try:`/email block and the final `return {"ok": True, "created": True, ...}` are at the same indentation as the `with _db_lock, ...` statement (one level inside the function's outer `try`), so they run after the connection closes. The `not claimed` early return and the other guards inside the `with` are unchanged.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_membership_cancel_email.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Run the existing trial suite for regressions**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_trial.py -v`
Expected: PASS (all existing trial tests still green - membership/grant/idempotent/cancel-route unchanged).

- [ ] **Step 7: Commit**

```bash
git add app.py tests/test_membership_cancel_email.py
git commit -m "feat: email the one-click cancel link at biofield-trial grant; extend cancel-token TTL"
```

---

## Verification

- `pytest tests/test_membership_cancel_email.py tests/test_biofield_trial.py -v` (doppler + venv) all pass.
- Final Opus whole-branch review. Focus: email fires exactly once across redirect + webhook (won-claim path only); send is best-effort and outside the DB lock; membership stays committed when the email raises; TTL constant applied; cancel link in the email actually reaches the working `/membership/cancel/<token>`; no emoji / no em dashes in the copy.
- Ship via PR + merge to `main` (auto-deploys); gentle `/begin` probe per the warm-up rule.
- Operational note (not a code step): no Stripe/Doppler change needed; the next real $1 trial payer will receive the email and have a working cancel link.

## Build order

Single task. After merge, the live-money trust gap (charged member with no working self-cancel) is closed.
