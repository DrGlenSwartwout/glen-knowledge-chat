# Referral-Attribution Part 2 (Capture) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bridge the dispensary link (System C) into the referral graph (System A): when a patient orders through a practitioner's dispensary link, write a durable `referral_redemptions` row that attributes all their future orders to that practitioner and pays L2 points only, never L1.

**Architecture:** Add a `kind` column to `referral_redemptions` so a practitioner-portal redemption is distinguishable from an Ambassador referral. `record_redemption` writes the kind. `_settle_referrer_reward` skips the L1 credit for `kind='dispensary_portal'` but keeps the L2 override and the paid-stamp. A capture hook on the patient-paid dispensary checkout writes the row (owner = practitioner email, resolved pid→email via Supabase). A one-time backfill materializes rows for existing dispensary clients.

**Tech Stack:** Python, Flask, SQLite (`chat_log.db` via `LOG_DB`/`DATA_DIR`) for `referral_redemptions`/`dispensary_orders`/`orders`/`points_ledger`; Supabase/Postgres for the `practitioners` table (pid→email). pytest.

## Global Constraints

- `referral_redemptions.referee_email` is the PRIMARY KEY; `record_redemption` uses `INSERT OR IGNORE` — first-touch, single-owner, idempotent. Part 2 must NOT change this (a patient already owned by an Ambassador keeps that owner).
- The new kind value is the exact string `dispensary_portal`. Existing rows and normal Ambassador referrals are kind `referral` (the column DEFAULT).
- "No L1, L2-points-only" applies ONLY to `kind='dispensary_portal'` redemptions. The Ambassador L1 flow (`kind='referral'`) is unchanged and must keep paying L1.
- The L1 suppression must still call `mark_rewarded` so the row is stamped (`rewarded_at`) and never pays L1 on a later settlement.
- `reward_cents` is a paid-stamp, not a trigger; the payout guard stays `rewarded_at IS NULL` + row existence. Do not change that guard.
- Backfill establishes attribution only — it writes NO reward and stamps NO `rewarded_at` (so go-forward L2 can still fire on the next paid order).
- `dispensary_orders` and `referral_redemptions` are SQLite (`LOG_DB`); `practitioners` is Supabase (`%s` placeholders, `db_supabase.supabase_cursor`). Never query them in the same connection.
- Do not touch System B (affiliate cash, `rewards.py`/`affiliate_earnings`).

---

### Task 1: `kind` column on `referral_redemptions` + `record_redemption(kind=)`

**Files:**
- Modify: `dashboard/referrals.py` (`init_tables` lines 20-26; `record_redemption` lines 68-75)
- Test: `tests/test_referral_kind.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `record_redemption(cx, code, owner_email, referee_email, order_ref, *, kind='referral') -> bool` — writes `kind` into the new column. `redemption_by_order_ref` / `owner_of_referee` rows now carry a `kind` key (via existing `SELECT *`). New rows default to `kind='referral'`; existing rows read `kind='referral'` (column DEFAULT).

- [ ] **Step 1: Write the failing test**

Create `tests/test_referral_kind.py`:

```python
import sqlite3
from dashboard import referrals as rf


def _cx():
    cx = sqlite3.connect(":memory:")
    rf.init_tables(cx)
    return cx


def test_kind_column_exists_and_defaults_to_referral():
    cx = _cx()
    cols = {r[1] for r in cx.execute("PRAGMA table_info(referral_redemptions)")}
    assert "kind" in cols
    rf.record_redemption(cx, "C1", "owner@x.com", "friend@x.com", "INV-1")
    row = rf.redemption_by_order_ref(cx, "INV-1")
    assert row["kind"] == "referral"   # default preserves the Ambassador flow


def test_record_redemption_writes_explicit_kind():
    cx = _cx()
    rf.record_redemption(cx, "C1", "doc@x.com", "patient@x.com", "INV-2",
                         kind="dispensary_portal")
    row = rf.redemption_by_order_ref(cx, "INV-2")
    assert row["kind"] == "dispensary_portal"
    assert row["owner_email"] == "doc@x.com"


def test_first_touch_preserves_original_owner_and_kind():
    cx = _cx()
    rf.record_redemption(cx, "C1", "ambassador@x.com", "patient@x.com", "INV-3")  # referral
    wrote = rf.record_redemption(cx, "C2", "doc@x.com", "patient@x.com", "INV-4",
                                 kind="dispensary_portal")  # same referee PK
    assert wrote is False   # INSERT OR IGNORE dropped the second
    row = rf.redemption_by_order_ref(cx, "INV-3")
    assert row["owner_email"] == "ambassador@x.com" and row["kind"] == "referral"


def test_init_tables_idempotent_adds_kind_once():
    cx = _cx()
    rf.init_tables(cx)   # second call must not raise
    cols = [r[1] for r in cx.execute("PRAGMA table_info(referral_redemptions)")]
    assert cols.count("kind") == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_referral_kind.py -v`
Expected: FAIL — `kind` column absent / `record_redemption` has no `kind` kwarg.

- [ ] **Step 3: Add the lazy ALTER**

In `dashboard/referrals.py`, extend the additive-columns list in `init_tables` (line 21):

```python
    # Additive columns: lazy ALTER, idempotent (OperationalError = column already exists)
    for col, typedef in [("rewarded_at", "TEXT"), ("reward_cents", "INTEGER"),
                         ("kind", "TEXT DEFAULT 'referral'")]:
        try:
            cx.execute(f"ALTER TABLE referral_redemptions ADD COLUMN {col} {typedef}")
        except Exception:
            pass  # column already present
```

- [ ] **Step 4: Add the `kind` param to `record_redemption`**

Replace `record_redemption` (lines 68-75) with:

```python
def record_redemption(cx, code, owner_email, referee_email, order_ref, *, kind="referral"):
    init_tables(cx)
    cur = cx.execute(
        "INSERT OR IGNORE INTO referral_redemptions "
        "(referee_email, code, owner_email, order_ref, created_at, kind) "
        "VALUES (?,?,?,?,?,?)",
        (_norm(referee_email), code, _norm(owner_email), order_ref or "", _now(), kind))
    cx.commit()
    return cur.rowcount > 0
```

- [ ] **Step 5: Run to verify it passes**

Run: `python3 -m pytest tests/test_referral_kind.py tests/test_referrer_reward_spec2b2.py -v`
Expected: PASS (new file green; the existing reward suite still green — `record_redemption`'s new kwarg is keyword-only with a default, so its positional callers are unaffected).

- [ ] **Step 6: Commit**

```bash
git add dashboard/referrals.py tests/test_referral_kind.py
git commit -m "feat(referrals): add kind column + record_redemption(kind=) for portal attribution"
```

---

### Task 2: Suppress L1 for `kind='dispensary_portal'` in `_settle_referrer_reward`

**Files:**
- Modify: `app.py` (`_settle_referrer_reward` lines 5172-5204)
- Test: `tests/test_portal_referral_reward.py` (create)

**Interfaces:**
- Consumes: `redemption_by_order_ref` rows now carry `kind` (Task 1).
- Produces: `_settle_referrer_reward` credits NO L1 points when `red["kind"] == "dispensary_portal"`, still credits L2 (when tier-2 enabled + a valid upline), still stamps `rewarded_at`. Return value = L1 cents credited (0 for a suppressed portal redemption).

- [ ] **Step 1: Write the failing test**

Create `tests/test_portal_referral_reward.py`:

```python
import importlib
import sqlite3
from dashboard import referrals as rf, points


def _reload(monkeypatch, tmp_path, pct="20", tier2=True):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REFERRALS", "true")
    monkeypatch.setenv("REFERRER_REWARD_PCT", pct)
    import app as appmod
    importlib.reload(appmod)
    monkeypatch.setattr(appmod, "REFERRAL_TIER2_ENABLED", tier2)
    return appmod


def _order(referee="patient@x.com", total=7000, shipping=1300, get=0):
    return {"email": referee, "total_cents": total, "shipping_cents": shipping, "get_cents": get}


def test_dispensary_portal_pays_no_l1_but_stamps(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path, tier2=False)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rf.record_redemption(cx, "DISP", "doc@x.com", "patient@x.com", "INV-1",
                             kind="dispensary_portal")
        credited = appmod._settle_referrer_reward(cx, _order(), "INV-1")
    assert credited == 0
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "doc@x.com") == 0        # no L1 to the practitioner
        assert rf.redemption_by_order_ref(cx, "INV-1")["rewarded_at"]  # stamped: no replay


def test_dispensary_portal_still_pays_l2_to_upline(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path, tier2=True)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        # upline@x.com referred the doctor into the system
        rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
        rf.record_redemption(cx, "DISP", "doc@x.com", "patient@x.com", "INV-1",
                             kind="dispensary_portal")
        appmod._settle_referrer_reward(cx, _order(), "INV-1")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        # product 5700; L1 suppressed = 0; L2 = 5700 * 20 // 200 = 570
        assert points.balance(cx, "doc@x.com") == 0
        assert points.balance(cx, "upline@x.com") == 570
        assert points.earned_by_reason(cx, "upline@x.com", "referral_reward_l2") == 570


def test_ambassador_referral_still_pays_l1(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path, tier2=False)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rf.record_redemption(cx, "AMB", "amb@x.com", "friend@x.com", "INV-2")  # kind='referral'
        credited = appmod._settle_referrer_reward(cx, _order(referee="friend@x.com"), "INV-2")
    assert credited == 1140   # 5700 * 20 // 100 — Ambassador flow unchanged
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "amb@x.com") == 1140
```

Note: `points.earned_by_reason` is used by `tests/test_two_tier_referral.py::test_earned_by_reason` — confirm the signature there (`earned_by_reason(cx, email, reason)`); if it differs, assert on `points.balance` alone.

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_portal_referral_reward.py -v`
Expected: FAIL — `test_dispensary_portal_pays_no_l1_but_stamps` fails because L1 currently credits `doc@x.com` 1140.

- [ ] **Step 3: Add the suppression gate**

In `app.py`, inside `_settle_referrer_reward`, replace the L1 credit block (lines 5188-5191):

```python
    reward = product_cents * pct // 100
    if reward > 0:
        _points.credit(cx, red["owner_email"], value_cents=reward, reason="referral_reward",
                       order_ref=f"referral:{red['referee_email']}")
```

with:

```python
    reward = product_cents * pct // 100
    # Drop-ship / portal sales are paid at wholesale (the practitioner's markup IS their pay),
    # so NO L1 is credited on them — only the L2 override below. Ambassador referrals
    # (kind='referral') keep L1. The row is still stamped (mark_rewarded) so it never pays later.
    l1_suppressed = (red.get("kind") == "dispensary_portal")
    if reward > 0 and not l1_suppressed:
        _points.credit(cx, red["owner_email"], value_cents=reward, reason="referral_reward",
                       order_ref=f"referral:{red['referee_email']}")
```

Then change the return so a suppressed portal redemption returns 0 L1 cents. Replace the final `return reward` (line 5204) with:

```python
    return 0 if l1_suppressed else reward
```

Leave the L2 block (lines 5195-5202) and `mark_rewarded` (line 5203) unchanged — L2 still fires, the row is still stamped with `reward_cents=reward` (the computed value, harmless as a stamp).

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_portal_referral_reward.py tests/test_referrer_reward_spec2b2.py tests/test_two_tier_referral.py -v`
Expected: PASS (portal suppression works; existing Ambassador L1 and two-tier suites still green).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_portal_referral_reward.py
git commit -m "feat(referrals): suppress L1 on dispensary_portal redemptions, keep L2 + stamp"
```

---

### Task 3: Capture hook on the patient-paid dispensary checkout

**Files:**
- Modify: `dashboard/practitioner_portal.py` (add `practitioner_email_by_id`, near `practitioner_id_by_dispensary_code` line 232)
- Modify: `app.py` (`api_client_checkout` — after the `_ingest_order(source="dispensary", …)` at line 12544-12552; add a module-level helper `_capture_portal_referral`)
- Test: `tests/test_portal_referral_capture.py` (create)

**Interfaces:**
- Consumes: `record_redemption(..., kind='dispensary_portal')` (Task 1); `practitioner_id_by_dispensary_code(code) -> pid` (existing).
- Produces:
  - `practitioner_portal.practitioner_email_by_id(pid) -> str` — Supabase `SELECT email FROM practitioners WHERE id=%s`; `''` if none/error.
  - `app._capture_portal_referral(code, patient_email, practitioner_email, order_ref) -> None` — best-effort; opens `LOG_DB` and writes a `kind='dispensary_portal'` redemption (first-touch via `record_redemption`). Never raises.

- [ ] **Step 1: Write the failing test**

Create `tests/test_portal_referral_capture.py`:

```python
import importlib
import sqlite3
from dashboard import referrals as rf


def _reload(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REFERRALS", "true")
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_capture_writes_dispensary_portal_row(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    appmod._capture_portal_referral("DISPCODE", "Patient@X.com", "doc@x.com", "INV-9")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = rf.redemption_by_order_ref(cx, "INV-9")
    assert row["referee_email"] == "patient@x.com"   # normalized
    assert row["owner_email"] == "doc@x.com"
    assert row["kind"] == "dispensary_portal"


def test_capture_is_first_touch(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rf.record_redemption(cx, "AMB", "ambassador@x.com", "patient@x.com", "INV-A")
    appmod._capture_portal_referral("DISPCODE", "patient@x.com", "doc@x.com", "INV-9")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert rf.owner_of_referee(cx, "patient@x.com") == "ambassador@x.com"  # unchanged


def test_capture_noops_without_practitioner_email(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    appmod._capture_portal_referral("DISPCODE", "patient@x.com", "", "INV-9")  # unresolved pid
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert rf.redemption_by_order_ref(cx, "INV-9") is None


def test_client_checkout_records_referral(monkeypatch, tmp_path):
    """The route resolves pid->email and captures the referral after ingest."""
    appmod = _reload(monkeypatch, tmp_path)
    appmod.app.config["TESTING"] = True
    monkeypatch.setattr(appmod._pp, "practitioner_id_by_dispensary_code", lambda code: "p1")
    monkeypatch.setattr(appmod._pp, "practitioner_email_by_id", lambda pid: "doc@x.com")
    monkeypatch.setattr(appmod._pp, "portal_data", lambda pid, **kw: {"modules_completed": 1})
    monkeypatch.setattr(appmod, "is_member", lambda session_id, email: True)
    monkeypatch.setattr(appmod._dropship, "build_client_order",
                        lambda *a, **k: {"ok": True, "invoice_id": "INV-77", "total": 70.0,
                                         "get_cents": 0})
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", False)
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)
    c = appmod.app.test_client()
    r = c.post("/api/client/DISPCODE/checkout",
               json={"email": "patient@x.com", "name": "Pat", "method": "zelle",
                     "items": [{"slug": "bone-builder", "qty": 1}],
                     "address": {"line1": "1 A St", "city": "Hilo", "state": "HI",
                                 "zip": "96720", "country": "US", "name": "Pat"}})
    assert r.status_code == 200
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = rf.redemption_by_order_ref(cx, "INV-77")
    assert row and row["owner_email"] == "doc@x.com" and row["kind"] == "dispensary_portal"
```

(If the checkout's consent/address validation rejects this payload, mirror the exact happy-path body from `tests/test_client_routes.py::test_happy_path_zelle_returns_200_and_records_order` — it is the source of truth for a passing request shape.)

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_portal_referral_capture.py -v`
Expected: FAIL — `_capture_portal_referral` and `practitioner_email_by_id` don't exist.

- [ ] **Step 3: Add `practitioner_email_by_id`**

In `dashboard/practitioner_portal.py`, after `practitioner_id_by_dispensary_code` (ends ~line 241), add:

```python
def practitioner_email_by_id(practitioner_id) -> str:
    """Resolve a practitioner's login email from their id (Supabase). '' if none/error.
    Used to fill referral_redemptions.owner_email for portal attribution."""
    if not practitioner_id:
        return ""
    try:
        from db_supabase import supabase_cursor
        with supabase_cursor() as cur:
            cur.execute("SELECT email FROM practitioners WHERE id=%s LIMIT 1",
                        (str(practitioner_id),))
            row = cur.fetchone()
        if not row:
            return ""
        email = row["email"] if isinstance(row, dict) else row[0]
        return (email or "").strip().lower()
    except Exception:
        return ""
```

(Confirm the cursor's row shape against `portal_data` lines 807-826 — it indexes `row["email"]`, so `supabase_cursor` yields dict rows; the `isinstance` guard covers both.)

- [ ] **Step 4: Add `_capture_portal_referral` and wire the route**

In `app.py`, add a module-level helper near `_settle_referrer_reward` (after line 5204):

```python
def _capture_portal_referral(dispensary_code, patient_email, practitioner_email, order_ref):
    """Bridge a dispensary-link sale into the referral graph: first-touch write of a
    kind='dispensary_portal' redemption so all this patient's future orders attribute to
    the practitioner (L2 points only, never L1). Best-effort — never raises into checkout."""
    if not _REFERRALS or not (patient_email or "").strip() or not (practitioner_email or "").strip():
        return
    try:
        from dashboard import referrals as _rf
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            _rf.record_redemption(cx, dispensary_code or "", practitioner_email, patient_email,
                                  order_ref or "", kind="dispensary_portal")
    except Exception as _e:
        print(f"[referrals] portal capture skipped: {_e!r}", flush=True)
```

Then in `api_client_checkout`, immediately after the `_ingest_order(source="dispensary", …)` call (ends line 12552), add:

```python
    # Bridge this dispensary sale into the referral graph for durable attribution + L2.
    _capture_portal_referral(code, email, _pp.practitioner_email_by_id(pid),
                             str(out.get("invoice_id") or ""))
```

(`code`, `email`, `pid`, and `out` are all in scope at that point in the route.)

- [ ] **Step 5: Run to verify it passes**

Run: `python3 -m pytest tests/test_portal_referral_capture.py tests/test_client_routes.py -v`
Expected: PASS (capture + first-touch + route; existing client-route suite still green).

- [ ] **Step 6: Commit**

```bash
git add app.py dashboard/practitioner_portal.py tests/test_portal_referral_capture.py
git commit -m "feat(referrals): capture dispensary-link sales into the referral graph"
```

---

### Task 4: One-time backfill of `dispensary_orders` into the referral graph

**Files:**
- Create: `scripts/backfill_dispensary_referrals.py`
- Test: `tests/test_backfill_dispensary_referrals.py` (create)

**Interfaces:**
- Consumes: `record_redemption(..., kind='dispensary_portal')` (Task 1); reads `dispensary_orders` (SQLite).
- Produces: `backfill(db_path, email_for_pid, *, dry_run=False) -> dict` — for each distinct `(practitioner_id, customer_email)` in `dispensary_orders`, resolves the practitioner email via the injected `email_for_pid(pid) -> str` callable and `INSERT OR IGNORE`s a `kind='dispensary_portal'` row (order_ref = one of that pair's dispensary invoice_ids). Returns `{"written": n, "skipped": m, "unresolved": k}`. Writes NO reward and stamps NO `rewarded_at`. Idempotent on re-run.

- [ ] **Step 1: Write the failing test**

Create `tests/test_backfill_dispensary_referrals.py`:

```python
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import backfill_dispensary_referrals as bf
from dashboard import referrals as rf


def _seed(tmp_path):
    p = os.path.join(str(tmp_path), "chat_log.db")
    cx = sqlite3.connect(p)
    cx.executescript(
        "CREATE TABLE dispensary_orders(invoice_id TEXT PRIMARY KEY, practitioner_id TEXT, "
        "customer_email TEXT, bottles INT, credit_earned_cents INT, created_at TEXT);")
    cx.execute("INSERT INTO dispensary_orders VALUES('D1','p1','a@x.com',1,0,'t')")
    cx.execute("INSERT INTO dispensary_orders VALUES('D2','p1','a@x.com',1,0,'t')")  # dup pair
    cx.execute("INSERT INTO dispensary_orders VALUES('D3','p2','b@x.com',1,0,'t')")
    cx.execute("INSERT INTO dispensary_orders VALUES('D4','p9','c@x.com',1,0,'t')")  # unresolved
    cx.commit(); cx.close()
    return p


EMAILS = {"p1": "doc1@x.com", "p2": "doc2@x.com", "p9": ""}   # p9 has no email


def test_backfill_writes_one_row_per_pair(tmp_path):
    p = _seed(tmp_path)
    res = bf.backfill(p, EMAILS.get)
    assert res["written"] == 2 and res["unresolved"] == 1
    with sqlite3.connect(p) as cx:
        assert rf.owner_of_referee(cx, "a@x.com") == "doc1@x.com"
        assert rf.owner_of_referee(cx, "b@x.com") == "doc2@x.com"
        arow = rf.redemption_by_order_ref(cx, "D1") or rf.redemption_by_order_ref(cx, "D2")
        assert arow["kind"] == "dispensary_portal"
        assert not (arow["rewarded_at"] or "")   # attribution only, no reward stamp


def test_backfill_idempotent(tmp_path):
    p = _seed(tmp_path)
    bf.backfill(p, EMAILS.get)
    res2 = bf.backfill(p, EMAILS.get)
    assert res2["written"] == 0 and res2["skipped"] >= 2   # PK already present


def test_backfill_dry_run_writes_nothing(tmp_path):
    p = _seed(tmp_path)
    res = bf.backfill(p, EMAILS.get, dry_run=True)
    assert res["written"] == 2   # would-write count
    with sqlite3.connect(p) as cx:
        assert rf.redemption_by_order_ref(cx, "D1") is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_backfill_dispensary_referrals.py -v`
Expected: FAIL — `scripts/backfill_dispensary_referrals.py` does not exist.

- [ ] **Step 3: Write the script**

Create `scripts/backfill_dispensary_referrals.py`:

```python
"""One-time backfill: materialize a kind='dispensary_portal' referral row for each
existing dispensary client, so current clients are durably attributed and L2-eligible.

Attribution only — writes NO reward, stamps NO rewarded_at. Idempotent (referee PK).

Usage:
    doppler run -p remedy-match -c prd -- env DATA_DIR=/path python3 \\
        scripts/backfill_dispensary_referrals.py [--dry-run]
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dashboard import referrals as rf


def backfill(db_path, email_for_pid, *, dry_run=False):
    written = skipped = unresolved = 0
    with sqlite3.connect(db_path) as cx:
        rf.init_tables(cx)
        pairs = cx.execute(
            "SELECT practitioner_id, lower(customer_email) AS email, MIN(invoice_id) AS ref "
            "FROM dispensary_orders "
            "WHERE customer_email IS NOT NULL AND customer_email != '' "
            "GROUP BY practitioner_id, lower(customer_email)").fetchall()
        cache = {}
        for pid, email, ref in pairs:
            if pid not in cache:
                cache[pid] = (email_for_pid(pid) or "").strip().lower()
            owner = cache[pid]
            if not owner:
                unresolved += 1
                continue
            if dry_run:
                written += 1
                continue
            wrote = rf.record_redemption(cx, "", owner, email, ref, kind="dispensary_portal")
            if wrote:
                written += 1
            else:
                skipped += 1
    return {"written": written, "skipped": skipped, "unresolved": unresolved}


def _email_for_pid(pid):
    from dashboard.practitioner_portal import practitioner_email_by_id
    return practitioner_email_by_id(pid)


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    base = os.environ.get("DATA_DIR") or "."
    path = os.path.join(base, "chat_log.db")
    result = backfill(path, _email_for_pid, dry_run=dry)
    print(f"backfill {'(dry-run) ' if dry else ''}complete: {result}", flush=True)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_backfill_dispensary_referrals.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/backfill_dispensary_referrals.py tests/test_backfill_dispensary_referrals.py
git commit -m "feat(referrals): one-time backfill of dispensary clients into referral graph"
```

---

## Post-implementation (controller, not a task)

- Run the full relevant suite once green:
  `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/pptest python3 -m pytest tests/test_referral_kind.py tests/test_portal_referral_reward.py tests/test_portal_referral_capture.py tests/test_backfill_dispensary_referrals.py tests/test_referrer_reward_spec2b2.py tests/test_two_tier_referral.py tests/test_client_routes.py tests/test_dispensary_stats.py -q`
- The backfill is a manual, owner-run step against prod (NOT auto-run on deploy): dry-run first, review counts, then run for real. It does not send email or touch GHL.
- After merge, Part 1's `patient_portal_items` union still stands; over time the referral graph becomes the single source and the dispensary-clients arm of the union becomes redundant (a future cleanup, out of scope here).

## Self-review notes

- **Spec coverage:** kind column (Task 1), L1 suppression keeping L2 + stamp (Task 2), capture at first dispensary order with first-touch + pid→email bridge (Task 3), idempotent no-reward backfill (Task 4) — all four approved decisions covered.
- **Type consistency:** `record_redemption(..., *, kind=...)` defined in Task 1 and consumed identically in Tasks 2/3/4; `practitioner_email_by_id(pid) -> str` defined in Task 3 and injected into Task 4's `backfill`.
- **Cross-DB guard:** every SQLite write uses `LOG_DB`; the only Supabase read is `practitioner_email_by_id`, injected as a callable into the backfill so its test needs no live Postgres.
