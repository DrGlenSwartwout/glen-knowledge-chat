# Spec 2b-2 — Referrer Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a referee's referral order is paid, credit the referrer store-credit points = a configurable percent of the referee's product spend, once per referee, idempotently.

**Architecture:** Extend the 2b-1 `referral_redemptions` table (additive `rewarded_at`/`reward_cents`) + two lookup/stamp functions; a `_settle_referrer_reward(cx, order, order_ref)` helper called from the existing paid-order hook `_settle_order_points`. Gated by `REFERRALS` + `REFERRER_REWARD_PCT > 0`. No checkout/pricing change.

**Tech Stack:** Python 3.11, Flask, SQLite (`chat_log.db`/`LOG_DB`), the points ledger (`dashboard/points.py`), pytest.

## Global Constraints

- **Reward = `REFERRER_REWARD_PCT`% of the referee's product spend** = `max(0, total_cents - shipping_cents - get_cents)`; credited as points to the referrer (`owner_email`).
- **Config = on-switch:** `REFERRER_REWARD_PCT` (env, default **0 = off**). Gated by `_REFERRALS` AND pct > 0; fully inert otherwise.
- **Idempotent, once per referee, referrers uncapped:** `points.credit(... reason="referral_reward", order_ref=f"referral:{referee_email}")` + a `rewarded_at` stamp.
- The reward step is **wrapped** so a failure never affects the buyer's settle or the payment-return handler. A 0-value reward is still stamped (no pointless retry).
- No new boolean flag; no checkout/pricing change; no new external dependency.
- NO emoji; no em dashes in generated text.
- **Test command (every task):** `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_referrer_reward_spec2b2.py -v`
- Tests reload `app` via `importlib`; emails lowercased.

---

### Task 1: `referral_redemptions` reward columns + lookups

**Files:**
- Modify: `dashboard/referrals.py`
- Test: `tests/test_referrer_reward_spec2b2.py` (create)

**Interfaces:**
- Consumes: existing `referrals.init_tables`, `record_redemption`, `_norm`, `_now`.
- Produces:
  - `redemption_by_order_ref(cx, order_ref) -> dict|None`
  - `mark_rewarded(cx, referee_email, reward_cents) -> None`
  - `init_tables` now guarantees `rewarded_at` + `reward_cents` columns.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_referrer_reward_spec2b2.py`:

```python
import sqlite3
from dashboard import referrals as rf


def _cx():
    return sqlite3.connect(":memory:")


def test_redemption_by_order_ref():
    cx = _cx()
    rf.record_redemption(cx, "CODE1", "owner@x.com", "Friend@x.com", "INV-9")
    r = rf.redemption_by_order_ref(cx, "INV-9")
    assert r["referee_email"] == "friend@x.com" and r["owner_email"] == "owner@x.com"
    assert r["code"] == "CODE1" and (r.get("rewarded_at") or "") == ""
    assert rf.redemption_by_order_ref(cx, "NOPE") is None


def test_mark_rewarded():
    cx = _cx()
    rf.record_redemption(cx, "CODE1", "owner@x.com", "friend@x.com", "INV-9")
    rf.mark_rewarded(cx, "Friend@x.com", 250)
    r = rf.redemption_by_order_ref(cx, "INV-9")
    assert r["rewarded_at"] and r["reward_cents"] == 250
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (`redemption_by_order_ref` missing / no `rewarded_at` column).

- [ ] **Step 3: Implement in `dashboard/referrals.py`**

In `init_tables`, after the two `CREATE TABLE` statements and before `cx.commit()`, add the additive columns:

```python
    for _col in ("rewarded_at TEXT DEFAULT ''", "reward_cents INTEGER DEFAULT 0"):
        try:
            cx.execute(f"ALTER TABLE referral_redemptions ADD COLUMN {_col}")
        except sqlite3.IntegrityError:
            pass
        except sqlite3.OperationalError:
            pass  # column already exists
```

(Confirm `import sqlite3` is at the top of `dashboard/referrals.py` — it was added in the 2b-1 polish; if absent, add it.)

Append the two functions:

```python
def redemption_by_order_ref(cx, order_ref):
    init_tables(cx)
    cur = cx.cursor(); cur.row_factory = sqlite3.Row
    r = cur.execute("SELECT * FROM referral_redemptions WHERE order_ref=?",
                    (order_ref,)).fetchone()
    return dict(r) if r else None


def mark_rewarded(cx, referee_email, reward_cents):
    init_tables(cx)
    cx.execute("UPDATE referral_redemptions SET rewarded_at=?, reward_cents=? WHERE referee_email=?",
               (_now(), int(reward_cents), _norm(referee_email)))
    cx.commit()
```

- [ ] **Step 4: Run to verify they pass**

Run the test command. Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/referrals.py tests/test_referrer_reward_spec2b2.py
git commit -m "feat(referrals-2b2): rewarded_at/reward_cents columns + redemption_by_order_ref + mark_rewarded"
```

---

### Task 2: `REFERRER_REWARD_PCT` config + reward helper + settle-hook wiring

**Files:**
- Modify: `app.py`
- Test: `tests/test_referrer_reward_spec2b2.py` (append)

**Interfaces:**
- Consumes: Task 1 `referrals.{redemption_by_order_ref, mark_rewarded}`; `points.credit/balance/init_points_table`; existing `_REFERRALS`, `_settle_order_points`, `LOG_DB`.
- Produces: `_referrer_reward_pct() -> int`; `_settle_referrer_reward(cx, order, order_ref) -> int` (cents credited); a call to it inside `_settle_order_points`.

- [ ] **Step 1: Write the failing tests**

Append:

```python
import importlib


def _reload_reward_app(monkeypatch, tmp_path, pct="10", referrals="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REFERRALS", referrals)
    monkeypatch.setenv("REFERRER_REWARD_PCT", pct)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _seed_redemption(appmod, order_ref="INV-1", owner="owner@x.com", referee="friend@x.com"):
    import sqlite3
    from dashboard import referrals as rf
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rf.record_redemption(cx, "CODE1", owner, referee, order_ref)


def _order(order_ref="INV-1", referee="friend@x.com", total=7000, shipping=1300, get=0):
    return {"email": referee, "total_cents": total, "shipping_cents": shipping, "get_cents": get}


def test_settle_referrer_reward_credits_pct(monkeypatch, tmp_path):
    appmod = _reload_reward_app(monkeypatch, tmp_path, pct="10")
    _seed_redemption(appmod)
    import sqlite3
    from dashboard import points, referrals as rf
    with sqlite3.connect(appmod.LOG_DB) as cx:
        credited = appmod._settle_referrer_reward(cx, _order(), "INV-1")
    # product spend = 7000 - 1300 - 0 = 5700; 10% = 570
    assert credited == 570
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "owner@x.com") == 570
        assert rf.redemption_by_order_ref(cx, "INV-1")["rewarded_at"]
    # idempotent: a second settle credits nothing more
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert appmod._settle_referrer_reward(cx, _order(), "INV-1") == 0
        assert points.balance(cx, "owner@x.com") == 570


def test_no_reward_when_pct_zero(monkeypatch, tmp_path):
    appmod = _reload_reward_app(monkeypatch, tmp_path, pct="0")
    _seed_redemption(appmod)
    import sqlite3
    from dashboard import points
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert appmod._settle_referrer_reward(cx, _order(), "INV-1") == 0
        assert points.balance(cx, "owner@x.com") == 0


def test_no_reward_without_redemption(monkeypatch, tmp_path):
    appmod = _reload_reward_app(monkeypatch, tmp_path, pct="10")
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert appmod._settle_referrer_reward(cx, _order(order_ref="OTHER"), "OTHER") == 0


def test_zero_product_cents_stamps_no_credit(monkeypatch, tmp_path):
    appmod = _reload_reward_app(monkeypatch, tmp_path, pct="10")
    _seed_redemption(appmod)
    import sqlite3
    from dashboard import points, referrals as rf
    with sqlite3.connect(appmod.LOG_DB) as cx:
        # all shipping -> product_cents 0
        credited = appmod._settle_referrer_reward(cx, _order(total=1300, shipping=1300), "INV-1")
        assert credited == 0 and points.balance(cx, "owner@x.com") == 0
        assert rf.redemption_by_order_ref(cx, "INV-1")["rewarded_at"]   # stamped, won't retry
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (`_settle_referrer_reward` undefined).

- [ ] **Step 3: Implement in `app.py`**

Add the config reader near `_referral_pct` (search `def _referral_pct`):

```python
def _referrer_reward_pct():
    try:
        return max(0, int(os.environ.get("REFERRER_REWARD_PCT", "0")))
    except (TypeError, ValueError):
        return 0
```

Add the reward helper near `_settle_order_points` (place it just above that function):

```python
def _settle_referrer_reward(cx, order, order_ref):
    """On a paid referral order: credit the referrer pct% of the referee's product spend.
    Returns cents credited (0 if none). Idempotent per referee; never raises into the caller."""
    if not _REFERRALS:
        return 0
    pct = _referrer_reward_pct()
    if pct <= 0:
        return 0
    from dashboard import referrals as _rf, points as _points
    red = _rf.redemption_by_order_ref(cx, order_ref)
    if not red or (red.get("rewarded_at") or "") or not (red.get("owner_email") or "").strip():
        return 0
    product_cents = max(0, int(order.get("total_cents") or 0)
                        - int(order.get("shipping_cents") or 0)
                        - int(order.get("get_cents") or 0))
    reward = product_cents * pct // 100
    if reward > 0:
        _points.init_points_table(cx)
        _points.credit(cx, red["owner_email"], value_cents=reward, reason="referral_reward",
                       order_ref=f"referral:{red['referee_email']}")
    _rf.mark_rewarded(cx, red["referee_email"], reward)
    return reward
```

Call it from `_settle_order_points`, inside its `with sqlite3.connect(LOG_DB) as cx:` block, right after the Phase-4 image-pick `try/except` block (app.py ~2729, before the function ends):

```python
        # ── 2b-2: referrer reward (pct of referee's product spend on a paid referral order) ──
        try:
            _settle_referrer_reward(cx, order, order_ref)
        except Exception as _rre:
            print(f"[referrals] referrer reward skipped: {_rre!r}", flush=True)
```

- [ ] **Step 4: Run to verify they pass**

Run the test command, then the broader sweep:
`doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/ -k "referr or settle or points" -q`
Expected: all pass, no regressions.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_referrer_reward_spec2b2.py
git commit -m "feat(referrals-2b2): REFERRER_REWARD_PCT + referrer reward on paid referral order"
```

---

## Self-Review (plan author)

- **Spec coverage:** reward columns + lookups (T1) → spec Store; config + helper + settle wiring (T2) → spec Reward helper + config. Both gated by `_REFERRALS` + pct.
- **Decisions honored:** points reward = pct% × product_cents (`total - shipping - get`) (T2); config `REFERRER_REWARD_PCT` default 0 = off (T2); idempotent per referee via `points.credit` order_ref + `rewarded_at` stamp (T1/T2); referrers uncapped (no cap logic); 0-value still stamped (T2 test); wrapped so it never affects buyer settle (T2 call site).
- **Type consistency:** `redemption_by_order_ref(cx, order_ref) -> dict|None`, `mark_rewarded(cx, referee_email, reward_cents)`, `_settle_referrer_reward(cx, order, order_ref) -> int`, `points.credit(... reason="referral_reward", order_ref="referral:{referee}")` — used identically across tasks.
- **Confirm-then-use flagged in-task:** `import sqlite3` at the top of `dashboard/referrals.py` (T1); the exact insertion point in `_settle_order_points` after the image-pick block (T2).
