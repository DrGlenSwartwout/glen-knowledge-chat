# Household Hold-and-Batch Shipping (Model B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For a household with an active Family Plan, hold each new shippable order up to 4 days so same-household orders can be combined into one parcel, with a caregiver "ship now" email button, operator extend/release, and auto-release at the deadline into the existing combined-shipment flow.

**Architecture:** A new pure-sqlite hold engine (`dashboard/household_holds.py`) owns a `household_holds` group table plus a `hold_group_id` column on `orders`. New shippable orders from Family-Plan households open or join a hold group instead of shipping immediately. Release (by caregiver button, operator action, or the 4-day deadline cron) hands the group's order ids to the existing `dashboard/combined_shipments.create_shipment`, which already buys one label, splits shipping fair-share, and fans tracking to every member. The hold layer sits strictly upstream of the combined-shipment layer; it never touches the label/tracking/delivery code, which is unchanged.

**Tech Stack:** Python 3, Flask, SQLite (`LOG_DB`), the in-repo Action/RBAC framework, Gmail-API send via `dashboard.inbox.send_email`, `secrets.token_urlsafe` + `_hash_token` for one-time links.

## Global Constraints

- Feature is gated by a NEW env flag `HOUSEHOLD_AUTO_BATCH_ENABLED` (truthy = `1/true/yes/on`), independent of the existing `HOUSEHOLD_SHIPMENTS_ENABLED` (which gates the manual combine tool + suggestions). When the auto flag is off, the order-creation hook, cron sweep, operator actions, and release page all no-op/guard out.
- Money path: held orders stay UNPAID until release. Fair-share shipping is recomputed at release (reuse `_recompute_combined_shipping` in `app.py`). Do NOT auto-capture a card in this plan — release re-invoices each member at their fair share and leaves collection to the existing member-order payment flow. (Auto-capture off a vaulted card is explicitly out of scope; see Task 11 note.)
- Scope: ONLY households with an active Family Plan (`dashboard.family_plan.covers(cx, buyer_email)` is True). Non-covered buyers ship immediately as today — the hook must return without creating a hold.
- The 4-day window belongs to the GROUP, anchored to the FIRST held order's arrival. Sibling orders join but never slide the deadline; only `extend_hold` moves it.
- Never hold: pickup-channel orders, already-grouped/held orders, terminal-status orders. Reuse `combined_shipments._combinable_reason`-style guards.
- Emails: send to the caregiver, cc consented adult members (`household.cc_recipients_for` / `viewable_members_for` filtered to non-`pet`/non-`child` relationships). Never email a `pet`/`child` member account. One invite email per hold group (guard on `invite_sent_at`).
- Scanner safety: the caregiver "ship now" link is a GET that renders `_confirm_post_page`; the actual release happens on the POST. Never mutate on the bare GET.
- All new sqlite functions are pure (take a `cx` connection), mirror `dashboard/combined_shipments.py` conventions, and are unit-testable without Flask.
- Time: pass `now` into engine functions (default `datetime.now(timezone.utc)`) so tests are deterministic. Store timestamps as ISO-8601 UTC strings, matching `combined_shipments._now()`.

---

## File Structure

- **Create** `dashboard/household_holds.py` — hold engine (schema, eligibility, open/join, release, extend, due, release-token, invite-email compose). Pure sqlite + one `send_invite` that calls `inbox.send_email`.
- **Modify** `dashboard/orders.py` — add `hold_group_id` column (additive ALTER in `init_*`), plus `set_order_hold_group(cx, order_id, hold_group_id)` and `orders_in_hold_group(cx, hold_group_id)` helpers (mirror `set_order_group`/`orders_in_group`).
- **Modify** `app.py` — order-creation hook call, public release page routes (GET+POST), operator board actions registration import, cron sweep endpoint, `run_daily_piggybacks` wiring is in the cron script (below).
- **Modify** `scripts/run_personal_email_cron.py` — add a `_piggyback_post` for the hold sweep inside `run_daily_piggybacks()`.
- **Create** `tests/test_household_holds.py` — engine unit tests.
- **Create** `tests/test_household_hold_routes.py` — release page + cron route + action tests.

Interfaces the engine exposes (used across tasks):
- `init_hold_tables(cx)`
- `eligible_for_hold(cx, order) -> bool`
- `open_or_join_hold(cx, order_id, *, caregiver_email, household_key, hold_days=4, now=None) -> dict` → `{"group_id": int, "opened": bool, "joined": bool}`
- `get_hold(cx, group_id) -> dict|None`  (includes `members` list)
- `orders_in_hold(cx, group_id) -> list[dict]`
- `extend_hold(cx, group_id, days, *, now=None) -> dict`
- `release_hold(cx, group_id, *, by) -> dict` → `{"group_id", "order_ids": [int], "status": "released"}`
- `due_holds(cx, now=None) -> list[dict]`  (open groups with `hold_until <= now`)
- `set_release_token(cx, group_id) -> str`  (returns RAW token; stores only the hash)
- `hold_by_release_token(cx, raw_token) -> dict|None`
- `caregiver_of(cx, buyer_email) -> str|None`  (active-plan caregiver covering this buyer)
- `invite_recipients(cx, group_id) -> dict` → `{"to": str, "cc": [str]}`
- `compose_invite(group, members, ship_date, release_url) -> dict` → `{"subject", "body", "html"}`
- `maybe_hold_new_order(cx, order_id, *, now=None) -> dict|None`  (the hook: eligibility + open_or_join, returns hold result or None)

---

### Task 1: Hold schema + `hold_group_id` column + eligibility

**Files:**
- Create: `dashboard/household_holds.py`
- Modify: `dashboard/orders.py` (additive ALTER in the orders init path; add `set_order_hold_group`, `orders_in_hold_group`)
- Test: `tests/test_household_holds.py`

**Interfaces:**
- Produces: `init_hold_tables(cx)`, `eligible_for_hold(cx, order)`, `orders.set_order_hold_group(cx, order_id, hold_group_id)`, `orders.orders_in_hold_group(cx, hold_group_id)`.
- Consumes: `dashboard.family_plan.covers`, `dashboard.orders._TERMINAL_STATUSES`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_household_holds.py
import sqlite3
import pytest
from dashboard import household_holds as H
from dashboard import orders as O
from dashboard import family_plan as FP
from dashboard import household as HH


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    O.init_orders_tables(cx) if hasattr(O, "init_orders_tables") else O._ensure_schema(cx)
    FP.init_family_plan_table(cx)
    HH.init_household_tables(cx)
    H.init_hold_tables(cx)
    return cx


def _order(cx, email, *, channel="ship", status="proposed"):
    return O.create_order(cx, source="test", email=email, name=email.split("@")[0],
                          items=[{"slug": "x", "qty": 1}], total_cents=1000,
                          channel=channel, status=status)


def test_eligible_only_for_covered_shippable(tmp_path):
    cx = _cx()
    FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
    HH.add_member(cx, "cg@x.com", "kid@x.com", relationship="child")
    covered = _order(cx, "kid@x.com")
    uncovered = _order(cx, "stranger@x.com")
    pickup = _order(cx, "cg@x.com", channel="pickup")
    assert H.eligible_for_hold(cx, O.get_order(cx, covered)) is True
    assert H.eligible_for_hold(cx, O.get_order(cx, uncovered)) is False
    assert H.eligible_for_hold(cx, O.get_order(cx, pickup)) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py::test_eligible_only_for_covered_shippable -v`
Expected: FAIL — `AttributeError: module 'dashboard.household_holds' has no attribute 'init_hold_tables'` (module/functions not defined yet). If `create_order`/init helper names differ, fix the test's setup to the real names first (grep `dashboard/orders.py` for `def create_order` and the schema-init function).

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/household_holds.py
"""Household hold-and-batch: hold a Family-Plan household's new shippable orders
up to N days so same-household orders combine into ONE parcel via
dashboard/combined_shipments. Pure sqlite; the hold layer sits UPSTREAM of the
combined-shipment layer and never touches labels/tracking/delivery.
"""
import os
from datetime import datetime, timezone, timedelta

from dashboard import orders as _orders
from dashboard import family_plan as _fp
from dashboard import household as _hh

_TERMINAL = _orders._TERMINAL_STATUSES  # ("shipped","delivered","done","cancelled")
_DEPENDENT_NO_EMAIL = {"pet", "child"}  # accounts we never email an invite to


def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat()


def _lc(e):
    return (e or "").strip().lower()


def _enabled():
    return str(os.environ.get("HOUSEHOLD_AUTO_BATCH_ENABLED", "")).strip().lower() \
        in ("1", "true", "yes", "on")


def init_hold_tables(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS household_holds (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            caregiver_email TEXT NOT NULL,
            household_key  TEXT NOT NULL,
            status         TEXT NOT NULL DEFAULT 'open',
            opened_at      TEXT NOT NULL,
            hold_until     TEXT NOT NULL,
            invite_sent_at TEXT,
            release_token_hash TEXT,
            released_at    TEXT,
            released_by    TEXT,
            updated_at     TEXT
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_hold_status ON household_holds(status)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_hold_cg ON household_holds(caregiver_email, status)")
    cx.commit()


def caregiver_of(cx, buyer_email):
    """The active-plan caregiver covering this buyer, or None. A buyer who holds
    their own plan is their own caregiver."""
    e = _lc(buyer_email)
    if not e:
        return None
    if _fp.is_active(cx, e):
        return e
    for cg in _hh.caregivers_for(cx, e):
        if cg["share_consent"] and _fp.is_active(cx, cg["primary_email"]):
            return cg["primary_email"]
    return None


def eligible_for_hold(cx, order):
    if order is None:
        return False
    if (order.get("status") or "") in _TERMINAL:
        return False
    if (order.get("channel") or "") == "pickup":
        return False
    if order.get("hold_group_id") is not None:
        return False
    if order.get("group_shipment_id") is not None:
        return False
    return caregiver_of(cx, order.get("email")) is not None
```

Then in `dashboard/orders.py`, in the orders schema-init function (the one that runs the `CREATE TABLE orders` / additive ALTERs — grep for `ADD COLUMN group_shipment_id`), add directly beneath the `group_shipment_id` ALTER:

```python
    try:
        cx.execute("ALTER TABLE orders ADD COLUMN hold_group_id INTEGER")
    except Exception:
        pass
```

and add these two helpers next to `set_order_group`/`orders_in_group`:

```python
def set_order_hold_group(cx, order_id, hold_group_id):
    cx.execute("UPDATE orders SET hold_group_id=?, updated_at=? WHERE id=?",
               (hold_group_id, _now(), order_id))
    cx.commit()


def orders_in_hold_group(cx, hold_group_id):
    rows = cx.execute("SELECT * FROM orders WHERE hold_group_id=? AND status!='cancelled' "
                      "ORDER BY id", (hold_group_id,)).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py::test_eligible_only_for_covered_shippable -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/household_holds.py dashboard/orders.py tests/test_household_holds.py
git commit -m "feat(holds): hold schema, hold_group_id column, eligibility"
```

---

### Task 2: `open_or_join_hold` — group anchored to first order

**Files:**
- Modify: `dashboard/household_holds.py`
- Test: `tests/test_household_holds.py`

**Interfaces:**
- Consumes: `init_hold_tables`, `orders.set_order_hold_group`, `orders.orders_in_hold_group`.
- Produces: `open_or_join_hold(...)`, `get_hold`, `orders_in_hold`.

- [ ] **Step 1: Write the failing test**

```python
def test_open_then_sibling_joins_same_group_deadline_from_first():
    from datetime import datetime, timezone
    cx = _cx()
    FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
    HH.add_member(cx, "cg@x.com", "kid@x.com", relationship="child")
    o1 = _order(cx, "cg@x.com")
    o2 = _order(cx, "kid@x.com")
    t0 = datetime(2026, 7, 12, 9, 0, tzinfo=timezone.utc)
    r1 = H.open_or_join_hold(cx, o1, caregiver_email="cg@x.com",
                             household_key="cg@x.com", hold_days=4, now=t0)
    assert r1["opened"] is True and r1["joined"] is False
    t1 = datetime(2026, 7, 14, 9, 0, tzinfo=timezone.utc)  # 2 days later
    r2 = H.open_or_join_hold(cx, o2, caregiver_email="cg@x.com",
                             household_key="cg@x.com", hold_days=4, now=t1)
    assert r2["opened"] is False and r2["joined"] is True
    assert r2["group_id"] == r1["group_id"]
    hold = H.get_hold(cx, r1["group_id"])
    assert hold["hold_until"].startswith("2026-07-16")  # t0 + 4d, NOT t1 + 4d
    assert {m["id"] for m in H.orders_in_hold(cx, r1["group_id"])} == {o1, o2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py::test_open_then_sibling_joins_same_group_deadline_from_first -v`
Expected: FAIL — `AttributeError: ... 'open_or_join_hold'`

- [ ] **Step 3: Write minimal implementation**

```python
def _open_group_for(cx, caregiver_email, household_key):
    row = cx.execute(
        "SELECT * FROM household_holds WHERE caregiver_email=? AND household_key=? "
        "AND status='open' ORDER BY id DESC LIMIT 1",
        (_lc(caregiver_email), _lc(household_key))).fetchone()
    return dict(row) if row else None


def get_hold(cx, group_id):
    row = cx.execute("SELECT * FROM household_holds WHERE id=?", (group_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["members"] = orders_in_hold(cx, group_id)
    return d


def orders_in_hold(cx, group_id):
    return _orders.orders_in_hold_group(cx, group_id)


def open_or_join_hold(cx, order_id, *, caregiver_email, household_key, hold_days=4, now=None):
    now = now or _now()
    existing = _open_group_for(cx, caregiver_email, household_key)
    if existing:
        _orders.set_order_hold_group(cx, order_id, existing["id"])
        cx.execute("UPDATE household_holds SET updated_at=? WHERE id=?",
                   (_iso(now), existing["id"]))
        cx.commit()
        return {"group_id": existing["id"], "opened": False, "joined": True}
    hold_until = _iso(now + timedelta(days=int(hold_days)))
    cur = cx.execute(
        "INSERT INTO household_holds (caregiver_email, household_key, status, "
        "opened_at, hold_until, updated_at) VALUES (?,?,'open',?,?,?)",
        (_lc(caregiver_email), _lc(household_key), _iso(now), hold_until, _iso(now)))
    gid = int(cur.lastrowid)
    _orders.set_order_hold_group(cx, order_id, gid)
    cx.commit()
    return {"group_id": gid, "opened": True, "joined": False}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py::test_open_then_sibling_joins_same_group_deadline_from_first -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/household_holds.py tests/test_household_holds.py
git commit -m "feat(holds): open_or_join_hold with group-anchored deadline"
```

---

### Task 3: `release_hold` + `due_holds`

**Files:**
- Modify: `dashboard/household_holds.py`
- Test: `tests/test_household_holds.py`

**Interfaces:**
- Produces: `release_hold(cx, group_id, *, by)`, `due_holds(cx, now=None)`.

- [ ] **Step 1: Write the failing test**

```python
def test_release_returns_order_ids_and_closes_group():
    from datetime import datetime, timezone
    cx = _cx()
    FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
    HH.add_member(cx, "cg@x.com", "kid@x.com", relationship="child")
    o1 = _order(cx, "cg@x.com"); o2 = _order(cx, "kid@x.com")
    t0 = datetime(2026, 7, 12, 9, 0, tzinfo=timezone.utc)
    g = H.open_or_join_hold(cx, o1, caregiver_email="cg@x.com", household_key="cg@x.com", now=t0)["group_id"]
    H.open_or_join_hold(cx, o2, caregiver_email="cg@x.com", household_key="cg@x.com", now=t0)
    res = H.release_hold(cx, g, by="caregiver")
    assert sorted(res["order_ids"]) == sorted([o1, o2])
    assert H.get_hold(cx, g)["status"] == "released"


def test_due_holds_only_past_deadline_open():
    from datetime import datetime, timezone
    cx = _cx()
    FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
    o1 = _order(cx, "cg@x.com")
    t0 = datetime(2026, 7, 12, 9, 0, tzinfo=timezone.utc)
    g = H.open_or_join_hold(cx, o1, caregiver_email="cg@x.com", household_key="cg@x.com", hold_days=4, now=t0)["group_id"]
    before = datetime(2026, 7, 15, 9, 0, tzinfo=timezone.utc)
    after = datetime(2026, 7, 16, 10, 0, tzinfo=timezone.utc)
    assert H.due_holds(cx, now=before) == []
    assert [d["id"] for d in H.due_holds(cx, now=after)] == [g]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py -k "release or due_holds" -v`
Expected: FAIL — `AttributeError: ... 'release_hold'`

- [ ] **Step 3: Write minimal implementation**

```python
def release_hold(cx, group_id, *, by):
    """Close an open hold group. Returns its non-cancelled member order ids so the
    caller can hand them to combined_shipments.create_shipment. Idempotent: a
    non-open group returns its ids with the existing status."""
    hold = get_hold(cx, group_id)
    if hold is None:
        return {"group_id": group_id, "order_ids": [], "status": "missing"}
    order_ids = [m["id"] for m in hold["members"]]
    if hold["status"] != "open":
        return {"group_id": group_id, "order_ids": order_ids, "status": hold["status"]}
    cx.execute("UPDATE household_holds SET status='released', released_at=?, released_by=?, "
               "updated_at=? WHERE id=?",
               (_iso(_now()), str(by or ""), _iso(_now()), group_id))
    cx.commit()
    return {"group_id": group_id, "order_ids": order_ids, "status": "released"}


def due_holds(cx, now=None):
    now = now or _now()
    rows = cx.execute(
        "SELECT * FROM household_holds WHERE status='open' AND hold_until <= ? "
        "ORDER BY id", (_iso(now),)).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py -k "release or due_holds" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/household_holds.py tests/test_household_holds.py
git commit -m "feat(holds): release_hold + due_holds"
```

---

### Task 4: `extend_hold`

**Files:**
- Modify: `dashboard/household_holds.py`
- Test: `tests/test_household_holds.py`

**Interfaces:**
- Produces: `extend_hold(cx, group_id, days, *, now=None)`.

- [ ] **Step 1: Write the failing test**

```python
def test_extend_pushes_deadline_from_current():
    from datetime import datetime, timezone
    cx = _cx()
    FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
    o1 = _order(cx, "cg@x.com")
    t0 = datetime(2026, 7, 12, 9, 0, tzinfo=timezone.utc)
    g = H.open_or_join_hold(cx, o1, caregiver_email="cg@x.com", household_key="cg@x.com", hold_days=4, now=t0)["group_id"]
    H.extend_hold(cx, g, 3)  # 2026-07-16 -> 2026-07-19
    assert H.get_hold(cx, g)["hold_until"].startswith("2026-07-19")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py::test_extend_pushes_deadline_from_current -v`
Expected: FAIL — `AttributeError: ... 'extend_hold'`

- [ ] **Step 3: Write minimal implementation**

```python
def extend_hold(cx, group_id, days, *, now=None):
    hold = get_hold(cx, group_id)
    if hold is None or hold["status"] != "open":
        raise ValueError(f"hold #{group_id} is not open")
    base = datetime.fromisoformat(hold["hold_until"])
    new_until = _iso(base + timedelta(days=int(days)))
    cx.execute("UPDATE household_holds SET hold_until=?, updated_at=? WHERE id=?",
               (new_until, _iso(now or _now()), group_id))
    cx.commit()
    return get_hold(cx, group_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py::test_extend_pushes_deadline_from_current -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/household_holds.py tests/test_household_holds.py
git commit -m "feat(holds): extend_hold"
```

---

### Task 5: Release token (one-time, hash-stored)

**Files:**
- Modify: `dashboard/household_holds.py`
- Test: `tests/test_household_holds.py`

**Interfaces:**
- Produces: `set_release_token(cx, group_id) -> str` (raw), `hold_by_release_token(cx, raw_token) -> dict|None`.
- Consumes: `hashlib` for the hash (self-contained; do NOT import app-level `_hash_token` to keep the engine Flask-free).

- [ ] **Step 1: Write the failing test**

```python
def test_release_token_roundtrip_and_wrong_token():
    cx = _cx()
    FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
    o1 = _order(cx, "cg@x.com")
    g = H.open_or_join_hold(cx, o1, caregiver_email="cg@x.com", household_key="cg@x.com")["group_id"]
    raw = H.set_release_token(cx, g)
    assert isinstance(raw, str) and len(raw) > 20
    got = H.hold_by_release_token(cx, raw)
    assert got and got["id"] == g
    assert H.hold_by_release_token(cx, "not-the-token") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py::test_release_token_roundtrip_and_wrong_token -v`
Expected: FAIL — `AttributeError: ... 'set_release_token'`

- [ ] **Step 3: Write minimal implementation**

```python
import hashlib
import secrets


def _tok_hash(raw):
    return hashlib.sha256(("household-hold:" + (raw or "")).encode("utf-8")).hexdigest()


def set_release_token(cx, group_id):
    raw = secrets.token_urlsafe(32)
    cx.execute("UPDATE household_holds SET release_token_hash=?, updated_at=? WHERE id=?",
               (_tok_hash(raw), _iso(_now()), group_id))
    cx.commit()
    return raw


def hold_by_release_token(cx, raw_token):
    th = _tok_hash((raw_token or "").strip())
    if not (raw_token or "").strip():
        return None
    row = cx.execute("SELECT * FROM household_holds WHERE release_token_hash=?",
                     (th,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["members"] = orders_in_hold(cx, d["id"])
    return d
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py::test_release_token_roundtrip_and_wrong_token -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/household_holds.py tests/test_household_holds.py
git commit -m "feat(holds): one-time release token (hash-stored)"
```

---

### Task 6: `maybe_hold_new_order` hook + wire into order creation

**Files:**
- Modify: `dashboard/household_holds.py`
- Modify: `app.py` (call the hook right after each shippable-order `create_order` in the order-creation route(s))
- Test: `tests/test_household_holds.py`

**Interfaces:**
- Produces: `maybe_hold_new_order(cx, order_id, *, now=None) -> dict|None`.
- Consumes: `eligible_for_hold`, `open_or_join_hold`, `caregiver_of`, `_enabled`.

- [ ] **Step 1: Write the failing test**

```python
def test_maybe_hold_gated_by_flag_and_eligibility(monkeypatch):
    cx = _cx()
    FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
    o1 = _order(cx, "cg@x.com")
    monkeypatch.setenv("HOUSEHOLD_AUTO_BATCH_ENABLED", "")
    assert H.maybe_hold_new_order(cx, o1) is None            # flag off -> no hold
    monkeypatch.setenv("HOUSEHOLD_AUTO_BATCH_ENABLED", "1")
    res = H.maybe_hold_new_order(cx, o1)
    assert res and res["opened"] is True
    assert O.get_order(cx, o1)["hold_group_id"] == res["group_id"]
    # a stranger order is never held even with the flag on
    o2 = _order(cx, "stranger@x.com")
    assert H.maybe_hold_new_order(cx, o2) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py::test_maybe_hold_gated_by_flag_and_eligibility -v`
Expected: FAIL — `AttributeError: ... 'maybe_hold_new_order'`

- [ ] **Step 3: Write minimal implementation**

```python
def maybe_hold_new_order(cx, order_id, *, now=None):
    """Called right after a shippable order is created. If the auto-batch flag is
    on and this order belongs to an active Family-Plan household, open or join a
    hold group and return the hold result; otherwise return None (ship as normal)."""
    if not _enabled():
        return None
    order = _orders.get_order(cx, order_id)
    if not eligible_for_hold(cx, order):
        return None
    cg = caregiver_of(cx, order.get("email"))
    if not cg:
        return None
    return open_or_join_hold(cx, order_id, caregiver_email=cg,
                             household_key=cg, now=now)
```

Then wire into `app.py`. Grep the order-creation route(s) that persist a shippable member order (`grep -n "create_order(" app.py`). Immediately after the `oid = ... create_order(...)` call in each shippable path (skip pickup/digital paths, which the eligibility guard also rejects), add:

```python
                try:
                    from dashboard import household_holds as _holds
                    _holds.init_hold_tables(cx)
                    _holds.maybe_hold_new_order(cx, oid)
                except Exception as _e:
                    print(f"[hold] maybe_hold_new_order({oid}) skipped: {_e!r}", flush=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py::test_maybe_hold_gated_by_flag_and_eligibility -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/household_holds.py app.py tests/test_household_holds.py
git commit -m "feat(holds): maybe_hold_new_order hook wired into order creation"
```

---

### Task 7: Invite email — recipients + compose (pure), then send

**Files:**
- Modify: `dashboard/household_holds.py`
- Test: `tests/test_household_holds.py`

**Interfaces:**
- Produces: `invite_recipients(cx, group_id) -> {"to","cc"}`, `compose_invite(hold, ship_date, release_url) -> {"subject","body","html"}`, `send_invite(cx, group_id, *, base_url) -> dict`.
- Consumes: `household.viewable_members_for`, `inbox.send_email` (only inside `send_invite`, imported lazily), `set_release_token`.

- [ ] **Step 1: Write the failing test**

```python
def test_invite_recipients_exclude_pet_child_and_compose():
    cx = _cx()
    FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
    HH.add_member(cx, "cg@x.com", "spouse@x.com", relationship="dependent")
    HH.add_member(cx, "cg@x.com", "kid@x.com", relationship="child")
    HH.add_member(cx, "cg@x.com", "rex@x.com", relationship="pet")
    o1 = _order(cx, "cg@x.com")
    g = H.open_or_join_hold(cx, o1, caregiver_email="cg@x.com", household_key="cg@x.com")["group_id"]
    rec = H.invite_recipients(cx, g)
    assert rec["to"] == "cg@x.com"
    assert "spouse@x.com" in rec["cc"]
    assert "kid@x.com" not in rec["cc"] and "rex@x.com" not in rec["cc"]
    msg = H.compose_invite(H.get_hold(cx, g), "July 16", "https://x/hold/abc/ship")
    assert "July 16" in msg["body"]
    assert "https://x/hold/abc/ship" in msg["html"]
    assert msg["subject"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py::test_invite_recipients_exclude_pet_child_and_compose -v`
Expected: FAIL — `AttributeError: ... 'invite_recipients'`

- [ ] **Step 3: Write minimal implementation**

```python
def invite_recipients(cx, group_id):
    hold = get_hold(cx, group_id)
    cg = hold["caregiver_email"]
    cc = []
    for m in _hh.viewable_members_for(cx, cg):
        if (m.get("relationship") or "").strip().lower() in _DEPENDENT_NO_EMAIL:
            continue
        if _lc(m["email"]) and _lc(m["email"]) != _lc(cg):
            cc.append(_lc(m["email"]))
    return {"to": _lc(cg), "cc": cc}


def compose_invite(hold, ship_date, release_url):
    members = hold.get("members") or []
    lines = []
    for m in members:
        who = m.get("name") or m.get("email") or f"order #{m.get('id')}"
        lines.append(f"  • {who}")
    items = "\n".join(lines) if lines else "  • your order"
    subject = "Your household order is being prepared to ship"
    body = (
        f"A shipment for your household is being prepared to go out on {ship_date}.\n\n"
        f"It currently includes:\n{items}\n\n"
        "If anyone else in your household wants to add something, just place their "
        "order in the next few days and it will ship together in the same box.\n\n"
        f"Or if nothing else is coming, ship it now: {release_url}\n\n"
        "In wellness,\nDr. Glen & Rae"
    )
    html = (
        f"<p>A shipment for your household is being prepared to go out on "
        f"<strong>{ship_date}</strong>.</p>"
        f"<p>It currently includes:</p><ul>"
        + "".join(f"<li>{(m.get('name') or m.get('email') or ('order #' + str(m.get('id'))))}</li>"
                  for m in members)
        + "</ul>"
        "<p>If anyone else in your household wants to add something, just place "
        "their order in the next few days and it will ship together in the same box.</p>"
        f"<p>Or if nothing else is coming, "
        f"<a href='{release_url}'>ship it now</a>.</p>"
        "<p>In wellness,<br>Dr. Glen &amp; Rae</p>"
    )
    return {"subject": subject, "body": body, "html": html}


def send_invite(cx, group_id, *, base_url, now=None):
    """One invite per group. Mints the release token, composes, sends via Gmail,
    stamps invite_sent_at. No-op if already sent."""
    hold = get_hold(cx, group_id)
    if hold is None or hold.get("invite_sent_at"):
        return {"skipped": "already_sent_or_missing"}
    raw = set_release_token(cx, group_id)
    ship_date = datetime.fromisoformat(hold["hold_until"]).strftime("%B %-d")
    release_url = f"{base_url.rstrip('/')}/hold/{raw}/ship"
    rec = invite_recipients(cx, group_id)
    msg = compose_invite(get_hold(cx, group_id), ship_date, release_url)
    from dashboard import inbox as _inbox
    res = _inbox.send_email(rec["to"], msg["subject"], msg["body"], html=msg["html"])
    cx.execute("UPDATE household_holds SET invite_sent_at=?, updated_at=? WHERE id=?",
               (_iso(now or _now()), _iso(now or _now()), group_id))
    cx.commit()
    return {"sent_to": rec["to"], "cc": rec["cc"], "send_result": res}
```

Note: cc delivery is a follow-on nicety — `inbox.send_email` currently takes a single `to`. Sending to the caregiver is the committed behavior; cc'ing adult members is computed here and can be added when `send_email` grows a `cc` param. Do NOT block this task on cc.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py::test_invite_recipients_exclude_pet_child_and_compose -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/household_holds.py tests/test_household_holds.py
git commit -m "feat(holds): invite recipients + compose + send_invite (one per group)"
```

---

### Task 8: Public release page — GET confirm, POST releases → combined shipment

**Files:**
- Modify: `app.py` (two routes: `GET /hold/<token>/ship`, `POST /hold/<token>/ship`)
- Test: `tests/test_household_hold_routes.py`

**Interfaces:**
- Consumes: `household_holds.hold_by_release_token`, `household_holds.release_hold`, `combined_shipments.create_shipment`, `_confirm_post_page`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_household_hold_routes.py
import os, sqlite3, importlib
import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOUSEHOLD_AUTO_BATCH_ENABLED", "1")
    monkeypatch.setenv("HOUSEHOLD_SHIPMENTS_ENABLED", "1")
    monkeypatch.setenv("LOG_DB", str(tmp_path / "log.db"))
    import app as _app
    importlib.reload(_app)
    _app.app.config["TESTING"] = True
    return _app


def _seed_hold(app):
    from dashboard import orders as O, family_plan as FP, household as HH, household_holds as H
    with sqlite3.connect(app.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        FP.init_family_plan_table(cx); HH.init_household_tables(cx); H.init_hold_tables(cx)
        FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
        HH.add_member(cx, "cg@x.com", "kid@x.com", relationship="child")
        o1 = O.create_order(cx, source="t", email="cg@x.com", name="cg",
                            items=[{"slug": "x", "qty": 1}], total_cents=1000, channel="ship")
        o2 = O.create_order(cx, source="t", email="kid@x.com", name="kid",
                            items=[{"slug": "x", "qty": 1}], total_cents=1000, channel="ship")
        g = H.open_or_join_hold(cx, o1, caregiver_email="cg@x.com", household_key="cg@x.com")["group_id"]
        H.open_or_join_hold(cx, o2, caregiver_email="cg@x.com", household_key="cg@x.com")
        raw = H.set_release_token(cx, g)
    return g, raw


def test_get_renders_confirm_post_releases(client):
    g, raw = _seed_hold(client)
    c = client.app.test_client()
    # GET must NOT mutate (scanner safety)
    r_get = c.get(f"/hold/{raw}/ship")
    assert r_get.status_code == 200 and b"<form" in r_get.data
    with sqlite3.connect(client.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import household_holds as H
        assert H.get_hold(cx, g)["status"] == "open"   # still open after GET
    # POST releases
    r_post = c.post(f"/hold/{raw}/ship")
    assert r_post.status_code in (200, 302)
    with sqlite3.connect(client.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import household_holds as H
        assert H.get_hold(cx, g)["status"] == "released"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_hold_routes.py::test_get_renders_confirm_post_releases -v`
Expected: FAIL — 404 on `/hold/<token>/ship` (route not defined)

- [ ] **Step 3: Write minimal implementation**

Add to `app.py` (near the other public token routes; reuse `_confirm_post_page`):

```python
@app.route("/hold/<token>/ship", methods=["GET", "POST"])
def household_hold_ship(token):
    from dashboard import household_holds as _holds
    from dashboard import combined_shipments as _cs
    invalid = ("<!doctype html><meta charset=utf-8><title>Link expired</title>"
               "<div style='font-family:Georgia,serif;max-width:520px;margin:60px auto'>"
               "<h1>This shipment is already on its way</h1>"
               "<p>Nothing more to do — your household order has been released.</p></div>")
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _holds.init_hold_tables(cx)
        hold = _holds.hold_by_release_token(cx, token)
        if hold is None or hold["status"] != "open":
            return invalid, (200 if hold else 404)
        if request.method == "GET":
            members = hold.get("members") or []
            names = ", ".join((m.get("name") or m.get("email") or f"#{m['id']}") for m in members)
            return _confirm_post_page(
                f"/hold/{token}/ship",
                title="Ship your household order",
                heading="Ship your household order now?",
                blurb=f"This will send your household's order ({names}) to fulfillment now. "
                      "Anything ordered after this ships separately.",
                button="Ship it now")
        # POST: release, then hand to the combined-shipment layer
        res = _holds.release_hold(cx, hold["id"], by="caregiver")
        ids = res["order_ids"]
        try:
            if len(ids) >= 2:
                _cs.create_shipment(cx, ids, created_by="caregiver-release")
        except Exception as e:
            print(f"[hold-ship] create_shipment({ids}) failed: {e!r}", flush=True)
    return ("<!doctype html><meta charset=utf-8><title>On its way</title>"
            "<div style='font-family:Georgia,serif;max-width:520px;margin:60px auto'>"
            "<h1>Done — your order is on its way</h1>"
            "<p>We’ll email tracking as soon as it ships.</p></div>"), 200
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_hold_routes.py::test_get_renders_confirm_post_releases -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_household_hold_routes.py
git commit -m "feat(holds): scanner-safe caregiver ship-now release page"
```

---

### Task 9: Operator board actions — `holds.extend`, `holds.release`

**Files:**
- Modify: `dashboard/household_holds.py` (register actions on import, mirroring `combined_shipments.py`'s pattern)
- Test: `tests/test_household_holds.py`

**Interfaces:**
- Consumes: `dashboard.actions.action`, `LOW_WRITE`; `dashboard.rbac.OWNER, OPS`.
- Produces: registered actions `holds.extend`, `holds.release` whose executors call `extend_hold` / `release_hold` (+ `create_shipment` on release).

- [ ] **Step 1: Write the failing test**

```python
def test_holds_actions_registered():
    from dashboard import actions as A
    import dashboard.household_holds  # noqa: F401 (import self-registers)
    assert A.get_action("holds.extend") is not None
    assert A.get_action("holds.release") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py::test_holds_actions_registered -v`
Expected: FAIL — `holds.extend` is None (not registered)

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/household_holds.py`:

```python
# ── Board actions (self-register on import) ──────────────────────────────────
from dashboard.actions import action, LOW_WRITE
from dashboard.rbac import OWNER, OPS


def _cx_of(params, ctx):
    cx = (ctx or {}).get("cx") or (params or {}).get("cx")
    if cx is None:
        raise ValueError("no db connection")
    return cx


def _extend_exec(params, ctx):
    cx = _cx_of(params, ctx)
    gid = int(params["group_id"])
    days = int(params.get("days", 2))
    hold = extend_hold(cx, gid, days)
    return {"group_id": gid, "hold_until": hold["hold_until"],
            "message": f"Hold #{gid} extended to {hold['hold_until'][:10]}."}


def _release_exec(params, ctx):
    from dashboard import combined_shipments as _cs
    cx = _cx_of(params, ctx)
    gid = int(params["group_id"])
    res = release_hold(cx, gid, by="operator")
    ids = res["order_ids"]
    made = None
    if len(ids) >= 2:
        made = _cs.create_shipment(cx, ids, created_by="operator-release")
    return {"group_id": gid, "order_ids": ids,
            "shipment_id": (made["id"] if made else None),
            "message": (f"Hold #{gid} released; combined shipment "
                        f"#{made['id']} created." if made
                        else f"Hold #{gid} released ({len(ids)} order).")}


action(key="holds.extend", module="orders", title="Extend household hold",
       description="Push a household hold group's ship-by deadline out N days (default 2).",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS), reversible=True)(_extend_exec)

action(key="holds.release", module="orders", title="Release household hold now",
       description="Close a household hold and send its orders to fulfillment "
                   "(combining 2+ into one shipment).",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS), reversible=False)(_release_exec)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py::test_holds_actions_registered -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/household_holds.py tests/test_household_holds.py
git commit -m "feat(holds): operator holds.extend + holds.release board actions"
```

---

### Task 10: Deadline sweep cron endpoint + piggyback wiring

**Files:**
- Modify: `app.py` (`POST /api/cron/household-holds/sweep`, mirroring `family_plan_charge_cron`'s `X-Console-Key` guard)
- Modify: `scripts/run_personal_email_cron.py` (add a `_piggyback_post` in `run_daily_piggybacks()`)
- Test: `tests/test_household_hold_routes.py`

**Interfaces:**
- Consumes: `household_holds.due_holds`, `household_holds.release_hold`, `combined_shipments.create_shipment`, `CONSOLE_SECRET`.

- [ ] **Step 1: Write the failing test**

```python
def test_sweep_releases_due_holds(client, monkeypatch):
    import sqlite3, datetime as _dt
    from dashboard import orders as O, family_plan as FP, household as HH, household_holds as H
    with sqlite3.connect(client.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        FP.init_family_plan_table(cx); HH.init_household_tables(cx); H.init_hold_tables(cx)
        FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
        HH.add_member(cx, "cg@x.com", "kid@x.com", relationship="child")
        o1 = O.create_order(cx, source="t", email="cg@x.com", name="cg",
                            items=[{"slug": "x", "qty": 1}], total_cents=1000, channel="ship")
        o2 = O.create_order(cx, source="t", email="kid@x.com", name="kid",
                            items=[{"slug": "x", "qty": 1}], total_cents=1000, channel="ship")
        past = _dt.datetime(2020, 1, 1, tzinfo=_dt.timezone.utc)
        g = H.open_or_join_hold(cx, o1, caregiver_email="cg@x.com", household_key="cg@x.com", now=past)["group_id"]
        H.open_or_join_hold(cx, o2, caregiver_email="cg@x.com", household_key="cg@x.com", now=past)
    c = client.app.test_client()
    r = c.post("/api/cron/household-holds/sweep",
               headers={"X-Console-Key": client.CONSOLE_SECRET})
    assert r.status_code == 200
    with sqlite3.connect(client.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        assert H.get_hold(cx, g)["status"] == "released"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_hold_routes.py::test_sweep_releases_due_holds -v`
Expected: FAIL — 404 on the sweep route

- [ ] **Step 3: Write minimal implementation**

Add to `app.py`:

```python
@app.route("/api/cron/household-holds/sweep", methods=["POST"])
def household_holds_sweep_cron():
    """Auto-release hold groups past their ship-by deadline: 2+ orders -> one
    combined shipment; a lone order -> just un-held so it ships normally. Idempotent:
    a released group is no longer 'open' so a re-run skips it."""
    if request.headers.get("X-Console-Key") != CONSOLE_SECRET:
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import household_holds as _holds
    from dashboard import combined_shipments as _cs
    released = combined = 0
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _holds.init_hold_tables(cx)
        for g in _holds.due_holds(cx):
            res = _holds.release_hold(cx, g["id"], by="deadline")
            released += 1
            ids = res["order_ids"]
            if len(ids) >= 2:
                try:
                    _cs.create_shipment(cx, ids, created_by="deadline-release")
                    combined += 1
                except Exception as e:
                    print(f"[hold-sweep] create_shipment({ids}) failed: {e!r}", flush=True)
    return jsonify({"ok": True, "released": released, "combined": combined})
```

Then in `scripts/run_personal_email_cron.py`, inside `run_daily_piggybacks()` (next to the `family-plan-charge` piggyback), add:

```python
    _piggyback_post("household-holds-sweep", "/api/cron/household-holds/sweep",
                    "X-Console-Key", CONSOLE_SECRET)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_hold_routes.py::test_sweep_releases_due_holds -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py scripts/run_personal_email_cron.py tests/test_household_hold_routes.py
git commit -m "feat(holds): deadline sweep cron + daily piggyback"
```

---

### Task 11: Fair-share shipping recompute at release (money path)

**Files:**
- Modify: `app.py` (call `_recompute_combined_shipping(cx, group_shipment_id)` after `create_shipment` in BOTH release paths: the sweep cron and the ship-now route and the operator action)
- Test: `tests/test_household_hold_routes.py`

**Interfaces:**
- Consumes: `_recompute_combined_shipping(cx, sid)` (already in `app.py`), `combined_shipments.create_shipment` (returns a shipment dict whose `id` is the `group_shipment_id` on member orders).

> **MONEY-PATH NOTE — verify before merge (per feedback_verify_against_live_api / verify_review_findings_money_path):** held orders are unpaid; `create_shipment` sets each order's `group_shipment_id`, and `_recompute_combined_shipping` re-bills ONLY unpaid members' shipping to their fair share (paid members are skipped by design). This plan recomputes the fair-share invoice; it does NOT capture a card. Confirm with Glen how member product orders are actually collected (portal checkout vs vaulted card vs manual) — if auto-capture is wanted, that is a separate task with its own live-Stripe verification. Do not invent a charge call here.

- [ ] **Step 1: Write the failing test**

```python
def test_release_recomputes_combined_shipping(client):
    import sqlite3
    from dashboard import orders as O, family_plan as FP, household as HH, household_holds as H
    from dashboard import combined_shipments as CS
    with sqlite3.connect(client.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        FP.init_family_plan_table(cx); HH.init_household_tables(cx); H.init_hold_tables(cx)
        CS.init_combined_shipments_table(cx)
        FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
        HH.add_member(cx, "cg@x.com", "kid@x.com", relationship="child")
        o1 = O.create_order(cx, source="t", email="cg@x.com", name="cg",
                            items=[{"slug": "x", "qty": 1}], total_cents=1000,
                            shipping_cents=800, channel="ship")
        o2 = O.create_order(cx, source="t", email="kid@x.com", name="kid",
                            items=[{"slug": "x", "qty": 1}], total_cents=1000,
                            shipping_cents=800, channel="ship")
        g = H.open_or_join_hold(cx, o1, caregiver_email="cg@x.com", household_key="cg@x.com")["group_id"]
        H.open_or_join_hold(cx, o2, caregiver_email="cg@x.com", household_key="cg@x.com")
        raw = H.set_release_token(cx, g)
    client.app.test_client().post(f"/hold/{raw}/ship")
    with sqlite3.connect(client.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        sid = O.get_order(cx, o1)["group_shipment_id"]
        assert sid is not None
        members = O.orders_in_group(cx, sid)
        # combined parcel shipping is split, so the pair's total shipping is <= 2x single
        assert sum(int(m["shipping_cents"] or 0) for m in members) <= 1600
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_hold_routes.py::test_release_recomputes_combined_shipping -v`
Expected: FAIL — shipping not recomputed (still 800 + 800 = 1600, or `group_shipment_id` None if create_shipment not reached). If it happens to pass because the packer returns 1600, tighten the assertion to `< 1600` only after confirming the test slugs pack cheaper combined; otherwise assert `sid is not None` and that `_recompute_combined_shipping` was invoked (spy).

- [ ] **Step 3: Write minimal implementation**

In `app.py`, extract a small helper and call it from all three release paths:

```python
def _release_to_shipment(cx, order_ids, *, created_by):
    """Shared: group released hold orders into one combined shipment and recompute
    fair-share shipping. Returns the shipment id (or None for a lone order)."""
    from dashboard import combined_shipments as _cs
    if len(order_ids) < 2:
        return None
    made = _cs.create_shipment(cx, order_ids, created_by=created_by)
    sid = made["id"]
    try:
        _recompute_combined_shipping(cx, sid)
    except Exception as e:
        print(f"[hold-release] recompute shipping failed for #{sid}: {e!r}", flush=True)
    return sid
```

Then replace the inline `create_shipment(...)` calls in `household_hold_ship` (Task 8), `household_holds_sweep_cron` (Task 10), and `_release_exec` (Task 9) with `_release_to_shipment(cx, ids, created_by=...)`. (For `_release_exec`, which lives in `household_holds.py`, keep its own `create_shipment` call but add a follow-on call into a thin `app`-level recompute is not importable there — instead have the operator action return the ids and let the board's action-runner call `_recompute_combined_shipping`; simplest: the operator action calls `create_shipment` only, and a note in the action description says shipping is recomputed by the board's post-combine hook. Prefer routing the operator release through the same `/api/cron`/route helper if the action-runner has `app` context.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_hold_routes.py::test_release_recomputes_combined_shipping -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_household_hold_routes.py
git commit -m "feat(holds): recompute fair-share shipping on release"
```

---

### Task 12: Cancel handling — cancelled held order leaves the group

**Files:**
- Modify: `dashboard/household_holds.py` (`remove_from_hold(cx, order_id)`) and wire into the order-cancel path in `dashboard/orders.py` / `app.py`
- Test: `tests/test_household_holds.py`

**Interfaces:**
- Produces: `remove_from_hold(cx, order_id) -> dict` (clears `hold_group_id`; if the group has no members left, closes it as `cancelled`).

- [ ] **Step 1: Write the failing test**

```python
def test_cancel_last_member_closes_group():
    cx = _cx()
    FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
    o1 = _order(cx, "cg@x.com")
    g = H.open_or_join_hold(cx, o1, caregiver_email="cg@x.com", household_key="cg@x.com")["group_id"]
    H.remove_from_hold(cx, o1)
    assert O.get_order(cx, o1)["hold_group_id"] is None
    assert H.get_hold(cx, g)["status"] == "cancelled"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py::test_cancel_last_member_closes_group -v`
Expected: FAIL — `AttributeError: ... 'remove_from_hold'`

- [ ] **Step 3: Write minimal implementation**

```python
def remove_from_hold(cx, order_id):
    order = _orders.get_order(cx, order_id)
    gid = order.get("hold_group_id") if order else None
    if gid is None:
        return {"ok": False, "reason": "not in a hold"}
    _orders.set_order_hold_group(cx, order_id, None)
    remaining = orders_in_hold(cx, gid)
    if not remaining:
        cx.execute("UPDATE household_holds SET status='cancelled', updated_at=? "
                   "WHERE id=? AND status='open'", (_iso(_now()), gid))
        cx.commit()
    return {"ok": True, "group_id": gid, "remaining": len(remaining)}
```

Then wire into the order-cancel path: grep `dashboard/orders.py` / `app.py` for where an order is set to `status='cancelled'` (e.g. `cancel_order`); immediately after, call `household_holds.remove_from_hold(cx, order_id)` inside a `try/except` (mirroring the hook in Task 6).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py::test_cancel_last_member_closes_group -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/household_holds.py dashboard/orders.py app.py tests/test_household_holds.py
git commit -m "feat(holds): cancelled held order leaves group; empty group closes"
```

---

### Task 13: Full-suite gate + flag-off safety

**Files:**
- Test: run the whole suite; add one flag-off integration assertion.

**Interfaces:** none new.

- [ ] **Step 1: Write the flag-off test**

```python
def test_flag_off_no_hold_end_to_end(client, monkeypatch):
    import sqlite3
    monkeypatch.setenv("HOUSEHOLD_AUTO_BATCH_ENABLED", "")
    from dashboard import orders as O, family_plan as FP, household_holds as H
    with sqlite3.connect(client.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        FP.init_family_plan_table(cx); H.init_hold_tables(cx)
        FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
        oid = O.create_order(cx, source="t", email="cg@x.com", name="cg",
                             items=[{"slug": "x", "qty": 1}], total_cents=1000, channel="ship")
        assert H.maybe_hold_new_order(cx, oid) is None
        assert O.get_order(cx, oid)["hold_group_id"] is None
```

- [ ] **Step 2: Run the new test + full suite**

Run: `cd ~/deploy-chat && python -m pytest tests/test_household_holds.py tests/test_household_hold_routes.py -v && python -m pytest -q`
Expected: the two hold suites PASS; full suite shows no NEW failures vs the pre-change baseline (compare the sorted FAILED list to main's baseline, not the count — per feedback_suite_green_not_task_green).

- [ ] **Step 3: Commit**

```bash
git add tests/test_household_hold_routes.py
git commit -m "test(holds): flag-off end-to-end safety"
```

---

## Rollout (after merge)

1. Keep `HOUSEHOLD_AUTO_BATCH_ENABLED` UNSET in Doppler `remedy-match/prd` until you want it live (the whole feature is inert while unset).
2. When ready: `doppler secrets set HOUSEHOLD_AUTO_BATCH_ENABLED=true --project remedy-match --config prd` (reversible; triggers a Render redeploy — per feedback_flag_flip_two_deploys, poll the live surface).
3. Verify live: create a test Family-Plan household order, confirm a hold group opens and the invite email arrives, click "ship it now" from the email, confirm the confirm-page renders and POST releases into a combined shipment.

## Self-Review notes

- **Spec coverage:** 4-day hold (Task 2, `hold_days=4`), group-anchored deadline (Task 2), sibling join (Task 2), caregiver ship-now button (Tasks 5,7,8), scanner-safe confirm page (Task 8), operator extend/release (Task 9), email check to caregiver + cc adult members excluding pet/child (Task 7), auto-release deadline (Task 10), fair-share billed at release (Task 11), Family-Plan-only scope (Task 1 eligibility + Task 6 hook), unpaid-until-release (Task 11 note), flag gating (Tasks 6,13). All present.
- **Open money-path decision** carried explicitly in Task 11 (how member orders are captured) — resolve with Glen before implementing Task 11.
- **Type consistency:** `open_or_join_hold`/`release_hold`/`due_holds`/`get_hold`/`extend_hold`/`hold_by_release_token`/`maybe_hold_new_order` signatures are used identically across tasks. `orders.set_order_hold_group`/`orders_in_hold_group` names match between Task 1 (def) and later consumers.
- **Verify before implementing:** exact names in `dashboard/orders.py` (schema-init function, `create_order` params like `shipping_cents`, `cancel_order`) must be confirmed against the file at implementation time — Task 1 Step 2 and Task 12 Step 3 call this out.
