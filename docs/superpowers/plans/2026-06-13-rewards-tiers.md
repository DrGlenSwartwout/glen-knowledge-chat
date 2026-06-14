# Rewards Tiers + Referral Attribution + Cash-Out — Implementation Plan (Plan 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Reward referrers when a referred buyer purchases — **points** for client-affiliates + doctors (spend on their own orders), **cash** (owner-approved) for pro-influencers — suppress a buyer's own points on an affiliate-acquired first order, and raise an owner **review task** when a balance crosses a cash-out threshold. Behind `REWARDS_TIERS_ENABLED` (default off — it moves real money).

**Architecture:** A `dashboard/rewards.py` (tier resolution from `people.tags`, the `affiliate_earnings` cash ledger, threshold logic, settings). A `_settle_referral(order, order_ref)` hook called next to the Plan 4 `_settle_order_points` in `/begin/checkout-return`: look up the referrer (existing attribution), credit them (points or cash), and record the conversion — idempotent per order. The Plan 4 buyer-earn gate gains a first-order-affiliate suppression. Cash-out is a **review task** (todo) + a `MONEY_SEND` dispatch action (manual owner approval via the existing spine) — no new payment integration.

**Tech Stack:** Python 3.11, Flask, sqlite, `dashboard/points`, `dashboard/events`, `dashboard/dispatch`, the `todos` table, pytest.

**Decisions (spec §A.9 + this plan's defaults):** client-affiliate + doctor → points; pro-influencer → cash (owner-approved). Points worth more as product than cash (cash-out at `cash_out_face_pct`=0.70). Suppress buyer points on an affiliate-acquired **first** order (decision b). Cash-out is **not automatic** — threshold → review task. **Defaults to confirm before enabling:** `referral_reward_pct`=0.05, `cash_out_threshold_cents`=10000 ($100), `cash_out_face_pct`=0.70.

**Map facts (exact):**
- Buyer→referrer: `SELECT re.utm_source FROM referral_events re JOIN affiliate_signups a ON a.slug=re.utm_source AND a.status='approved' WHERE LOWER(re.email)=? ORDER BY re.received_at DESC LIMIT 1` (app.py ~6393). Wrapped as `_attribute_conversion_by_email(email, conversion_type, ...) -> slug|None` (app.py:6383).
- `affiliate_signups(slug UNIQUE, email, status, ...)` — slug↔referrer email.
- `people(email UNIQUE, tags TEXT='[]' JSON array, ...)`; tags incl. `type:client`, `type:practitioner`, `ref:<slug>`. NO get-by-email helper (write one). NO `tier:pro-influencer` tag yet (this plan introduces it).
- `points.earn/redeem/balance/has_entry(cx, *, order_ref, reason)`.
- `todos(created_at, owner, category, title, body, priority, status, source, dedup_key UNIQUE, ...)` — INSERT … ON CONFLICT(dedup_key) for idempotent tasks.
- `dispatch.dispatch_action(cx, key, params, actor, *, source, confirmed=False)`; register via `@action(key, module, title, description, risk_tier=MONEY_SEND, permission=(OWNER,OPS,VA))`.
- Plan 4 `_settle_order_points(order, *, order_ref)` is the buyer-earn hook (in `/begin/checkout-return`, paid block); order dict has email/total_cents/discount_cents/points_redeemed_cents/shipping_cents/get_cents.
- `orders.list_orders_by_email(cx, email, limit)` for first-order detection.

**Run tests:** route tests via `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest <path>`; pure via plain venv pytest.

---

### Task 1: `dashboard/rewards.py` — tiers, earnings ledger, settings, threshold

**Files:** Create `dashboard/rewards.py`; Test `tests/test_rewards_model.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_rewards_model.py
import json, sqlite3
from dashboard import rewards

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE people (email TEXT UNIQUE, tags TEXT DEFAULT '[]')")
    cx.execute("CREATE TABLE affiliate_signups (slug TEXT UNIQUE, email TEXT, status TEXT)")
    rewards.init_affiliate_earnings_table(cx)
    return cx

def _person(cx, email, tags):
    cx.execute("INSERT INTO people (email, tags) VALUES (?,?)", (email, json.dumps(tags)))
    cx.commit()

def test_reward_mode_pro_influencer_is_cash():
    cx = _cx()
    cx.execute("INSERT INTO affiliate_signups VALUES ('jane','jane@x.com','approved')")
    _person(cx, "jane@x.com", ["type:client", "tier:pro-influencer"])
    assert rewards.reward_mode_for_slug(cx, "jane") == "cash"

def test_reward_mode_doctor_is_points():
    cx = _cx()
    cx.execute("INSERT INTO affiliate_signups VALUES ('doc','doc@x.com','approved')")
    _person(cx, "doc@x.com", ["type:practitioner"])
    assert rewards.reward_mode_for_slug(cx, "doc") == "points"

def test_reward_mode_default_client_is_points():
    cx = _cx()
    cx.execute("INSERT INTO affiliate_signups VALUES ('cl','cl@x.com','approved')")
    _person(cx, "cl@x.com", ["type:client"])
    assert rewards.reward_mode_for_slug(cx, "cl") == "points"

def test_referrer_email_for_slug():
    cx = _cx()
    cx.execute("INSERT INTO affiliate_signups VALUES ('jane','jane@x.com','approved')")
    assert rewards.referrer_email_for_slug(cx, "jane") == "jane@x.com"
    assert rewards.referrer_email_for_slug(cx, "nope") is None

def test_affiliate_earnings_accrue_and_pending_total():
    cx = _cx()
    rewards.accrue_cash(cx, slug="jane", email="jane@x.com", order_ref="INV1", amount_cents=500)
    rewards.accrue_cash(cx, slug="jane", email="jane@x.com", order_ref="INV2", amount_cents=700)
    assert rewards.pending_cash_total(cx, "jane") == 1200
    # idempotent per order_ref
    rewards.accrue_cash(cx, slug="jane", email="jane@x.com", order_ref="INV1", amount_cents=500)
    assert rewards.pending_cash_total(cx, "jane") == 1200

def test_settings_defaults():
    s = rewards.load_settings({})
    assert s["referral_reward_pct"] == 0.05
    assert s["cash_out_threshold_cents"] == 10000
    assert s["cash_out_face_pct"] == 0.70
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement `dashboard/rewards.py`**
- `DEFAULTS = {"referral_reward_pct":0.05, "cash_out_threshold_cents":10000, "cash_out_face_pct":0.70}`; `load_settings(overrides)` merges (skip None).
- `tags_for_email(cx, email)` → `set(json.loads(row["tags"] or "[]"))` for `SELECT tags FROM people WHERE lower(email)=?`, else empty set.
- `referrer_email_for_slug(cx, slug)` → `SELECT email FROM affiliate_signups WHERE slug=? AND status='approved'` → email or None.
- `reward_mode_for_slug(cx, slug)`: email = referrer_email_for_slug; tags = tags_for_email(email); `"cash"` if `"tier:pro-influencer"` in tags else `"points"`.
- `init_affiliate_earnings_table(cx)`: `affiliate_earnings(id, slug, email, order_ref, amount_cents, status DEFAULT 'pending', created_at, paid_at)` + UNIQUE(slug, order_ref) + index on slug.
- `accrue_cash(cx, *, slug, email, order_ref, amount_cents)`: INSERT OR IGNORE (idempotent on UNIQUE(slug, order_ref)).
- `pending_cash_total(cx, slug)` → SUM(amount_cents) WHERE slug=? AND status='pending'.
- `mark_paid(cx, slug)` → UPDATE … SET status='paid', paid_at=now WHERE slug=? AND status='pending'.

- [ ] **Step 4: Run → pass (7).**
- [ ] **Step 5: Commit** — `feat(rewards): tier resolution + affiliate cash-earnings ledger + settings`

---

### Task 2: Referral crediting + buyer first-order suppression

**Files:** Modify `app.py` (`_settle_referral` + extend `_settle_order_points` + call both in `/begin/checkout-return`); Test `tests/test_referral_settlement.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_referral_settlement.py
import json, sqlite3, app as appmod
from dashboard import points, rewards

def _db(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db"); monkeypatch.setattr(appmod, "LOG_DB", db)
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE people (email TEXT UNIQUE, tags TEXT DEFAULT '[]')")
    cx.execute("CREATE TABLE affiliate_signups (slug TEXT UNIQUE, email TEXT, status TEXT)")
    cx.execute("""CREATE TABLE referral_events (received_at TEXT, email TEXT, utm_source TEXT)""")
    cx.execute("""CREATE TABLE orders (email TEXT, created_at TEXT, source TEXT, external_ref TEXT)""")
    cx.execute("""CREATE TABLE todos (id INTEGER PRIMARY KEY, created_at TEXT, owner TEXT,
                  category TEXT, title TEXT, body TEXT, priority TEXT, status TEXT DEFAULT 'open',
                  source TEXT, dedup_key TEXT UNIQUE)""")
    points.init_points_table(cx); rewards.init_affiliate_earnings_table(cx)
    monkeypatch.setenv("REWARDS_TIERS_ENABLED", "true")
    return cx

def _refer(cx, buyer, slug, ref_email, tags):
    cx.execute("INSERT INTO affiliate_signups VALUES (?,?,?)", (slug, ref_email, "approved"))
    cx.execute("INSERT INTO people (email, tags) VALUES (?,?)", (ref_email, json.dumps(tags)))
    cx.execute("INSERT INTO referral_events VALUES ('2026-01-01', ?, ?)", (buyer, slug))
    cx.commit()

def test_points_referrer_credited(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "doc", "doc@x.com", ["type:practitioner"])
    order = {"email":"buyer@x.com","total_cents":6000,"shipping_cents":0,"get_cents":0,
             "discount_cents":0,"points_redeemed_cents":0}
    appmod._settle_referral(order, order_ref="INV1")
    assert points.balance(cx, "doc@x.com") == 300        # 5% of 6000 product
    # idempotent
    appmod._settle_referral(order, order_ref="INV1")
    assert points.balance(cx, "doc@x.com") == 300

def test_cash_referrer_accrues_not_points(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "jane", "jane@x.com", ["tier:pro-influencer"])
    order = {"email":"buyer@x.com","total_cents":6000,"shipping_cents":0,"get_cents":0,
             "discount_cents":0,"points_redeemed_cents":0}
    appmod._settle_referral(order, order_ref="INV2")
    assert rewards.pending_cash_total(cx, "jane") == 300
    assert points.balance(cx, "jane@x.com") == 0

def test_buyer_first_order_affiliate_suppresses_buyer_points(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "doc", "doc@x.com", ["type:practitioner"])
    # no prior orders for buyer → first order; attributed to an affiliate → suppress buyer earn
    order = {"email":"buyer@x.com","total_cents":6000,"shipping_cents":0,"get_cents":0,
             "discount_cents":0,"points_redeemed_cents":0}
    appmod._settle_order_points(order, order_ref="INV3")
    assert points.balance(cx, "buyer@x.com") == 0        # suppressed (affiliate-acquired first order)
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement**
- `_rewards_enabled()`: truthy `REWARDS_TIERS_ENABLED`.
- `_referrer_slug_for_email(cx, email)`: the attribution SELECT (referral_events JOIN affiliate_signups approved, most recent). Return slug or None.
- `_settle_referral(order, *, order_ref)`: if not `_rewards_enabled()` return. email=buyer; product_cents = total−shipping−get. Only credit on a full-price referred sale (`discount_cents==0` — match the buyer rule; a discounted order doesn't generate referral reward). slug = `_referrer_slug_for_email(cx, email)`; if None return. Don't credit if the referrer IS the buyer (self-referral). reward = round(product_cents × referral_reward_pct). mode = `rewards.reward_mode_for_slug(cx, slug)`. If "points": referrer_email = rewards.referrer_email_for_slug; if not `points.has_entry(cx, order_ref=order_ref, reason="referral")` → `points.earn(referrer_email, full_price_cents=reward/earn_pct...)` — NO: earn computes pct internally; instead add a direct ledger entry. Use a new `points.credit(cx, email, value_cents, reason, order_ref)` (a thin wrapper around `_add` that's idempotent-checked by caller) OR reuse `_add`. Simplest: add `points.credit(cx, email, *, value_cents, reason, order_ref)` to dashboard/points.py = `_add(cx, email, value_cents, reason, order_ref)` guarded by has_entry. Credit `reason="referral"`. If "cash": `rewards.accrue_cash(slug, referrer_email, order_ref, reward)`. Then call `_maybe_raise_cashout_review(cx, slug, mode)` (Task 3). Best-effort; never raise.
- Extend `_settle_order_points`: before earning, compute suppression — if `_rewards_enabled()` AND it's the buyer's FIRST order (`len(orders.list_orders_by_email(cx, email, limit=2)) <= 1` — the just-created order may already be in the table; treat ≤1 prior as first) AND `_referrer_slug_for_email(cx, email)` is not None → skip the buyer earn. (Add `points.credit` helper.)
- In `/begin/checkout-return` paid block, after `_settle_order_points(_o, order_ref=inv)`, add `_settle_referral(_o, order_ref=inv)` (own try/except, never breaks redirect).

- [ ] **Step 4: Run → pass (3).**
- [ ] **Step 5: Commit** — `feat(rewards): referral crediting (points/cash) + buyer first-order suppression`

---

### Task 3: Cash-out review task + payout action

**Files:** Modify `app.py` (`_maybe_raise_cashout_review`); Create `dashboard/actions_rewards.py`; Test `tests/test_cashout_review.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_cashout_review.py
import sqlite3, app as appmod
from dashboard import points, rewards

def _db(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db"); monkeypatch.setattr(appmod, "LOG_DB", db)
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    cx.execute("""CREATE TABLE todos (id INTEGER PRIMARY KEY, created_at TEXT, owner TEXT,
                  category TEXT, title TEXT, body TEXT, priority TEXT, status TEXT DEFAULT 'open',
                  source TEXT, dedup_key TEXT UNIQUE)""")
    points.init_points_table(cx); rewards.init_affiliate_earnings_table(cx)
    return cx

def test_cashout_review_raised_over_threshold_idempotent(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    rewards.accrue_cash(cx, slug="jane", email="jane@x.com", order_ref="INV1", amount_cents=12000)
    appmod._maybe_raise_cashout_review(cx, "jane", "cash")   # 12000 >= 10000 threshold
    n = cx.execute("SELECT COUNT(*) FROM todos WHERE source='affiliate-cashout'").fetchone()[0]
    assert n == 1
    appmod._maybe_raise_cashout_review(cx, "jane", "cash")   # idempotent (dedup_key)
    assert cx.execute("SELECT COUNT(*) FROM todos WHERE source='affiliate-cashout'").fetchone()[0] == 1

def test_no_review_under_threshold(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    rewards.accrue_cash(cx, slug="jane", email="jane@x.com", order_ref="INV1", amount_cents=500)
    appmod._maybe_raise_cashout_review(cx, "jane", "cash")
    assert cx.execute("SELECT COUNT(*) FROM todos WHERE source='affiliate-cashout'").fetchone()[0] == 0
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement**
- `_maybe_raise_cashout_review(cx, slug, mode)`: settings = rewards.load_settings(_REWARDS_SETTINGS if defined else {}). amount = `rewards.pending_cash_total(cx, slug)` (cash) or `points.balance(cx, referrer_email)` (points). If `amount >= cash_out_threshold_cents`: INSERT a todo `owner='glen', category='Finance', priority='high', source='affiliate-cashout', dedup_key=f"cashout:{slug}:{amount//threshold}"` (so it re-raises once per threshold band, not every order), title=`Review $X cash-out for {slug}`, body explaining mode, amount, and (points) the 70%-face conversion value `round(amount*cash_out_face_pct)`. ON CONFLICT(dedup_key) DO NOTHING.
- `dashboard/actions_rewards.py`: register `@action(key="rewards.process_payout", module="money", title="Process affiliate cash-out", risk_tier=MONEY_SEND, permission=(OWNER, OPS, VA))` — params `{slug, mode}`; on execute: cash → `rewards.mark_paid(cx, slug)` (records the payout intent; the actual money-send is the owner's existing finance flow); points → record the conversion at `cash_out_face_pct` (a `points.redeem(referrer_email, value_cents=balance, reason='cashout', order_ref=f'cashout:{slug}:{ts}')` and log the cash value). Return the amount + cash value. (The MONEY_SEND risk tier routes VA submissions to owner-approval automatically.)
- Import `actions_rewards` where the other action modules are registered (search how `actions_tasks`/`finance` get imported into the ACTION_REGISTRY) so the action is live.

- [ ] **Step 4: Run → pass (2).**
- [ ] **Step 5: Commit** — `feat(rewards): cash-out review task at threshold + MONEY_SEND payout action`

---

### Task 4: Full suite + doc

**Files:** Create `docs/rewards-tiers.md`
- [ ] **Step 1:** Run all rewards + points + checkout suites; green.
- [ ] **Step 2:** Write `docs/rewards-tiers.md`: the three tiers + how tier is read (`tier:pro-influencer` tag = cash, else points), the `referral_reward_pct`/`cash_out_threshold_cents`/`cash_out_face_pct` settings (and that they're DEFAULTS to confirm), the `REWARDS_TIERS_ENABLED` flag, buyer first-order suppression, and the cash-out review→`rewards.process_payout` (MONEY_SEND, owner-approved) flow. Note the go-live needs: confirm the rates, tag pro-influencers with `tier:pro-influencer`, enable the flag.
- [ ] **Step 3:** Commit.

---

## Self-review
- **Spec coverage:** tiers (Task 1); referral crediting points-vs-cash (Task 2); buyer first-order suppression = decision (b) (Task 2); cash-out review threshold + 70%-face + MONEY_SEND owner approval (Task 3); all behind `REWARDS_TIERS_ENABLED`; tunable settings.
- **Deferred:** automated payout execution (Stripe Connect/ACH — the review+action records intent, owner pays via existing finance tools); 1099 threshold tracking (capture W-9 at the review task — operational, noted in doc); syncing `tier:pro-influencer` to GHL; referral reward on subscription cycles (only one-time full-price referred orders credit in v1).
- **Risk:** distributes money → flag-gated off + cash-out is owner-approved via MONEY_SEND, never automatic; crediting idempotent per order; self-referral excluded; only full-price referred orders credit.
- **Type consistency:** `rewards.reward_mode_for_slug(cx,slug)->'cash'|'points'`, `referrer_email_for_slug`, `accrue_cash(cx,*,slug,email,order_ref,amount_cents)`, `pending_cash_total`, `points.credit(cx,email,*,value_cents,reason,order_ref)`, `_settle_referral(order,*,order_ref)`, `_maybe_raise_cashout_review(cx,slug,mode)` used identically across tasks.

## Next
Optional Plan 6 — automated payout execution + 1099/W-9 capture; begin-funnel checkout on `_price_cart`; Products-console per-SKU floor + months/eligibility edit UI; auto-points-redeem on subscription orders.
