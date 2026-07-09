# Per-Client Pickup Flag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mark a client as collecting in person, so in-house order entry pre-checks the pickup box and the Biofield hand-off resolves it server-side — while the client's own checkout ignores it.

**Architecture:** One additive `people.pickup_default` column behind a single migration (collapsing today's two divergent copies). Two pure helpers in `dashboard/customers.py`. Exactly one resolution point — `/api/orders/manual` — where an explicit `pickup` in the request body always beats the flag. The edit route never consults the flag.

**Tech Stack:** Python 3, Flask, sqlite3, pytest, vanilla JS.

**Spec:** `docs/superpowers/specs/2026-07-08-per-client-pickup-flag-design.md`

**Branch:** `sess/5629cdf8-pickup-flag`, worktree `/tmp/wt-deploy-chat-5629cdf8`, off `b405022c`.

## Global Constraints

- **Test command for pure modules:** `python3 -m pytest tests/<file> -q`
- **Test command for app-importing tests:** `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) python3 -m pytest tests/<file> -q`
  Both pieces are required. `import app` validates the Pinecone key over the network at import (needs real prod secrets via Doppler); Doppler-prd also sets `DATA_DIR=/data`, which is unwritable locally, so it MUST be overridden or you get `sqlite3.OperationalError: unable to open database file`. Skipping the override does **not** mean the failure is environmental.
- **`main` is red.** A full-suite run shows ~97 pre-existing failures and is flaky. Never judge by failure counts; diff `FAILED <nodeid>` names against an `origin/main` baseline.
- **Fail toward charging shipping.** Any ambiguity (unknown email, missing column, error) resolves to `pickup = False`. Guessing `True` gives goods away silently.
- **The edit route (`api_orders_edit`, `app.py:32802`) must never read `pickup_default`.** Re-resolving on edit would rebuild the PR #734 latch.
- Never write `if (x) el.checked = true;` — always `el.checked = !!x;` so the box also clears.

---

### Task 1: One migration, not two

Today `people` has **two** migrations: an inline `ALTER TABLE` loop in `app.py:_init_people_table()` (what prod runs) and `dashboard/customers.py:add_people_address_columns()` (referenced only by `tests/test_inhouse_order_entry.py:32`). Adding a column to the customers.py copy alone leaves every test green while prod never gains the column. Collapse them, then add `pickup_default`.

**Files:**
- Modify: `dashboard/customers.py:11-29` (`_ADDR_COLS`, `add_people_address_columns`)
- Modify: `app.py:23949-23955` (inline ALTER loop in `_init_people_table`)
- Modify: `tests/test_inhouse_order_entry.py:32` (caller rename)
- Test: `tests/test_people_pickup_column.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `dashboard.customers.add_people_columns(cx) -> None` (idempotent). Module constant `_PEOPLE_COLS: tuple[tuple[str, str], ...]` of `(column_name, sql_type_and_default)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_people_pickup_column.py`:

```python
import sqlite3

import dashboard.customers as C


def _people_db():
    cx = sqlite3.connect(":memory:")
    cx.execute("""CREATE TABLE people (
        id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,
        name TEXT DEFAULT '')""")
    cx.commit()
    return cx


def _cols(cx):
    return {r[1] for r in cx.execute("PRAGMA table_info(people)").fetchall()}


def test_migration_adds_pickup_default_and_address_columns():
    cx = _people_db()
    C.add_people_columns(cx)
    cols = _cols(cx)
    assert "pickup_default" in cols
    assert {"address1", "address2", "zip"} <= cols


def test_migration_is_idempotent():
    cx = _people_db()
    C.add_people_columns(cx)
    C.add_people_columns(cx)          # must not raise
    assert "pickup_default" in _cols(cx)


def test_pickup_default_defaults_to_zero():
    cx = _people_db()
    C.add_people_columns(cx)
    cx.execute("INSERT INTO people (email) VALUES ('a@b.com')")
    cx.commit()
    row = cx.execute("SELECT pickup_default FROM people WHERE email='a@b.com'").fetchone()
    assert row[0] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_people_pickup_column.py -q`
Expected: FAIL — `AttributeError: module 'dashboard.customers' has no attribute 'add_people_columns'`

- [ ] **Step 3: Write minimal implementation**

In `dashboard/customers.py`, replace `_ADDR_COLS` and `add_people_address_columns` with:

```python
# Columns additively migrated onto `people`. THE migration — app.py calls this too,
# so tests and prod exercise one code path (they used to have divergent copies).
_PEOPLE_COLS = (
    ("address1", "TEXT DEFAULT ''"),
    ("address2", "TEXT DEFAULT ''"),
    ("zip", "TEXT DEFAULT ''"),
    ("pickup_default", "INTEGER DEFAULT 0"),
)


def add_people_columns(cx):
    """Additively migrate `people` (shipping address + pickup default). Idempotent."""
    for col, decl in _PEOPLE_COLS:
        try:
            cx.execute(f"ALTER TABLE people ADD COLUMN {col} {decl}")
        except Exception:
            pass  # already present
    cx.commit()
```

In `app.py`, inside `_init_people_table()`, replace the inline loop:

```python
        # In-house customer records (order-entry Phase 1): full shipping address.
        for _col in ("address1", "address2", "zip"):
            try:
                cx.execute(f"ALTER TABLE people ADD COLUMN {_col} TEXT DEFAULT ''")
            except Exception:
                pass  # already present
        cx.commit()
```

with:

```python
        # Additive columns (address + pickup_default) live in ONE place so prod and
        # tests run the same migration. See dashboard/customers.py:_PEOPLE_COLS.
        from dashboard import customers as _cust_mig
        _cust_mig.add_people_columns(cx)
```

In `tests/test_inhouse_order_entry.py:32`, change `C.add_people_address_columns(cx)` to `C.add_people_columns(cx)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_people_pickup_column.py -q`
Expected: `3 passed`

Run: `python3 -m pytest tests/test_inhouse_order_entry.py -q`
Expected: PASS (no `AttributeError` on the renamed function)

- [ ] **Step 5: Commit**

```bash
git add dashboard/customers.py app.py tests/test_inhouse_order_entry.py tests/test_people_pickup_column.py
git commit -m "refactor(customers): one people migration; add pickup_default column"
```

---

### Task 2: Prove prod's boot path gains the column

Task 1's tests exercise `customers.add_people_columns` directly. That is exactly the mistake we are correcting — the test-only path passing while prod's differs. This task asserts against the function prod actually runs at import: `app._init_people_table()`.

**Files:**
- Test: `tests/test_init_people_table_pickup.py` (create)

**Interfaces:**
- Consumes: `dashboard.customers.add_people_columns` (Task 1); `app._init_people_table`.
- Produces: nothing.

- [ ] **Step 1: Write the failing test**

Create `tests/test_init_people_table_pickup.py`:

```python
"""Guard the two-migrations divergence: prod's boot path must produce the column,
not just dashboard.customers.add_people_columns() in isolation."""
import sqlite3

import pytest


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    return appmod


def test_init_people_table_creates_pickup_default(appmod):
    appmod._init_people_table()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cols = {r[1] for r in cx.execute("PRAGMA table_info(people)").fetchall()}
    assert "pickup_default" in cols, f"prod boot path missing pickup_default: {sorted(cols)}"


def test_init_people_table_is_idempotent(appmod):
    appmod._init_people_table()
    appmod._init_people_table()          # must not raise
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cols = {r[1] for r in cx.execute("PRAGMA table_info(people)").fetchall()}
    assert "pickup_default" in cols
```

- [ ] **Step 2: Run test to verify it fails**

First stash Task 1's `app.py` change to watch it fail for the right reason:

Run: `git stash push app.py && doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) python3 -m pytest tests/test_init_people_table_pickup.py -q; git stash pop`
Expected: FAIL — `AssertionError: prod boot path missing pickup_default: [...]`

(This proves the test catches the divergence, not merely that the column exists.)

- [ ] **Step 3: No implementation needed**

Task 1 already wired `_init_people_table()` to `add_people_columns`. This task's deliverable is the guard.

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) python3 -m pytest tests/test_init_people_table_pickup.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_init_people_table_pickup.py
git commit -m "test(customers): guard that prod's boot path adds pickup_default"
```

---

### Task 3: `customers` read/write helpers

**Files:**
- Modify: `dashboard/customers.py` (add helpers; extend `PICKER_COLS`)
- Test: `tests/test_customers_pickup_default.py` (create)

**Interfaces:**
- Consumes: `add_people_columns(cx)` (Task 1).
- Produces:
  - `set_pickup_default(cx, person_id: int, on: bool) -> None`
  - `pickup_default_for_email(cx, email: str | None) -> bool`
  - `PICKER_COLS` now includes `"pickup_default"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_customers_pickup_default.py`:

```python
import sqlite3

import dashboard.customers as C


def _db():
    cx = sqlite3.connect(":memory:")
    cx.execute("""CREATE TABLE people (
        id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,
        name TEXT DEFAULT '', first_name TEXT DEFAULT '', last_name TEXT DEFAULT '',
        phone TEXT DEFAULT '', city TEXT DEFAULT '', state TEXT DEFAULT '',
        country TEXT DEFAULT '', order_count INTEGER DEFAULT 0,
        last_order_date TEXT DEFAULT '')""")
    cx.commit()
    C.add_people_columns(cx)
    return cx


def _person(cx, email):
    cx.execute("INSERT INTO people (email) VALUES (?)", (email,))
    cx.commit()
    return cx.execute("SELECT id FROM people WHERE email=?", (email,)).fetchone()[0]


def test_unknown_email_is_not_pickup():
    """Fail toward CHARGING shipping: never free-ship on a guess."""
    cx = _db()
    assert C.pickup_default_for_email(cx, "nobody@example.com") is False
    assert C.pickup_default_for_email(cx, "") is False
    assert C.pickup_default_for_email(cx, None) is False


def test_set_and_read_round_trip():
    cx = _db()
    pid = _person(cx, "d@x.com")
    assert C.pickup_default_for_email(cx, "d@x.com") is False
    C.set_pickup_default(cx, pid, True)
    assert C.pickup_default_for_email(cx, "d@x.com") is True
    C.set_pickup_default(cx, pid, False)
    assert C.pickup_default_for_email(cx, "d@x.com") is False


def test_email_lookup_is_case_insensitive():
    cx = _db()
    pid = _person(cx, "d@x.com")
    C.set_pickup_default(cx, pid, True)
    assert C.pickup_default_for_email(cx, "D@X.COM") is True


def test_missing_column_reads_false_not_raise():
    """A pre-migration DB must resolve False, never explode a checkout."""
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL)")
    cx.execute("INSERT INTO people (email) VALUES ('d@x.com')")
    cx.commit()
    assert C.pickup_default_for_email(cx, "d@x.com") is False


def test_picker_cols_exposes_pickup_default():
    """A column absent from PICKER_COLS never reaches the browser."""
    assert "pickup_default" in C.PICKER_COLS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_customers_pickup_default.py -q`
Expected: FAIL — `AttributeError: module 'dashboard.customers' has no attribute 'pickup_default_for_email'`

- [ ] **Step 3: Write minimal implementation**

In `dashboard/customers.py`, extend `PICKER_COLS`:

```python
PICKER_COLS = ("id", "name", "first_name", "last_name", "email", "phone",
               "address1", "address2", "city", "state", "zip", "country",
               "pickup_default")
```

Add below `add_people_columns`:

```python
def set_pickup_default(cx, person_id, on):
    """Mark/unmark a client as collecting in person (no shipping on their orders)."""
    cx.execute("UPDATE people SET pickup_default=?, updated_at=? WHERE id=?",
               (1 if on else 0, _now(), int(person_id)))
    cx.commit()


def pickup_default_for_email(cx, email):
    """True when this client collects in person. Unknown email, blank email, or a
    pre-migration table (no such column) -> False. Never raises: the safe direction
    is to CHARGE shipping, because guessing True ships goods for free."""
    e = (email or "").strip().lower()
    if not e:
        return False
    try:
        row = cx.execute(
            "SELECT pickup_default FROM people WHERE lower(email)=?", (e,)).fetchone()
    except Exception:
        return False          # column/table absent
    return bool(row and row[0])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_customers_pickup_default.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add dashboard/customers.py tests/test_customers_pickup_default.py
git commit -m "feat(customers): pickup_default read/write helpers + expose in PICKER_COLS"
```

---

### Task 4: Resolve the flag in `/api/orders/manual` only

**Files:**
- Modify: `app.py:32916` (inside `api_orders_manual`, which starts at `app.py:32904`)
- Test: `tests/test_orders_manual_pickup_default.py` (create)

**Interfaces:**
- Consumes: `dashboard.customers.pickup_default_for_email(cx, email) -> bool` (Task 3).
- Produces: request contract — an explicit `pickup` key in the body wins; an **absent** key falls back to the client's flag.

**Do NOT touch `api_orders_edit` (`app.py:32802`).** Its `pickup = bool(body.get("pickup"))` stays exactly as it is. Task 5 adds the guard test.

- [ ] **Step 1: Write the failing test**

These drive the **real route** and assert on the **stored order** (`channel`, `shipping_cents`). Do not assert on `inspect.getsource(...)` — source text passes on a route that merely mentions the right words while behaving wrongly.

Harness copied from `tests/test_inhouse_volume_pricing.py:94-105` (monkeypatch `LOG_DB` + `_get_product`, POST with `X-Console-Key`).

⚠️ `_app()` calls `pytest.skip(f"app not importable: {e}")`. If you forget the `DATA_DIR` override you get **`1 passed, N skipped` — green, with nothing executed.** Always read the skip count.

Create `tests/test_orders_manual_pickup_default.py`:

```python
"""A Biofield hand-off posts NO `pickup` key -> the client's saved flag decides.
Order entry always posts the checkbox -> the checkbox decides, always."""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

BOTTLE = {"slug": "mix", "price_cents": 7000, "name": "Drink Mix"}
_CAT = {"mix": BOTTLE}


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture
def env(monkeypatch, tmp_path):
    appmod = _app()
    db = str(tmp_path / "m.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setattr(appmod, "_get_product", _CAT.get)
    appmod._init_people_table()
    from dashboard import orders as O
    cx = sqlite3.connect(db)
    O.init_orders_table(cx)
    cx.close()
    return appmod, db


def _flag(db, email, on):
    from dashboard import customers as C
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT OR IGNORE INTO people (email) VALUES (?)", (email,))
        cx.commit()
        pid = cx.execute("SELECT id FROM people WHERE email=?", (email,)).fetchone()[0]
        C.set_pickup_default(cx, pid, on)


def _post(appmod, body):
    key = appmod.dashboard.CONSOLE_SECRET or ""
    base = {"customer": {"name": "T", "email": body.pop("email", ""),
                         "address": {"address1": "1", "city": "Hilo", "state": "HI",
                                     "zip": "96720", "country": "US"}},
            "lines": [{"slug": "mix", "qty": 2}], "method": "Zelle"}
    base.update(body)
    r = appmod.app.test_client().post("/api/orders/manual", json=base,
                                      headers={"X-Console-Key": key})
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()


def _stored(db, order_id):
    from dashboard import orders as O
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    try:
        return O.get_order(cx, order_id)
    finally:
        cx.close()


def test_absent_pickup_key_uses_client_flag(env):
    """The Biofield hand-off path: no `pickup` key, flagged client -> pickup."""
    appmod, db = env
    _flag(db, "pick@x.com", True)
    j = _post(appmod, {"email": "pick@x.com"})
    o = _stored(db, j["order_id"])
    assert o["channel"] == "pickup"
    assert o["shipping_cents"] == 0


def test_absent_pickup_key_unflagged_client_charges_shipping(env):
    appmod, db = env
    _flag(db, "ship@x.com", False)
    j = _post(appmod, {"email": "ship@x.com"})
    o = _stored(db, j["order_id"])
    assert o["channel"] == "retail"
    assert o["shipping_cents"] > 0


def test_absent_pickup_key_unknown_client_charges_shipping(env):
    """Fail toward charging: never free-ship a client we've never seen."""
    appmod, db = env
    j = _post(appmod, {"email": "stranger@x.com"})
    o = _stored(db, j["order_id"])
    assert o["channel"] == "retail"
    assert o["shipping_cents"] > 0


def test_explicit_false_beats_flagged_client(env):
    """Order entry always posts the checkbox; unticking it wins over the flag."""
    appmod, db = env
    _flag(db, "pick@x.com", True)
    j = _post(appmod, {"email": "pick@x.com", "pickup": False})
    o = _stored(db, j["order_id"])
    assert o["channel"] == "retail"
    assert o["shipping_cents"] > 0


def test_explicit_true_beats_unflagged_client(env):
    appmod, db = env
    _flag(db, "ship@x.com", False)
    j = _post(appmod, {"email": "ship@x.com", "pickup": True})
    o = _stored(db, j["order_id"])
    assert o["channel"] == "pickup"
    assert o["shipping_cents"] == 0


def test_edit_never_resurrects_the_flag(env):
    """PR #734 latch guard: unticking pickup on a FLAGGED client's order must stick.
    If the edit route consulted pickup_default, this would snap back to 'pickup'."""
    appmod, db = env
    _flag(db, "pick@x.com", True)
    j = _post(appmod, {"email": "pick@x.com"})          # created as pickup via flag
    assert _stored(db, j["order_id"])["channel"] == "pickup"
    key = appmod.dashboard.CONSOLE_SECRET or ""
    r = appmod.app.test_client().post(
        f"/api/orders/{j['order_id']}/edit",
        json={"lines": [{"slug": "mix", "qty": 2}], "pickup": False},
        headers={"X-Console-Key": key})
    assert r.status_code == 200, r.get_data(as_text=True)
    o = _stored(db, j["order_id"])
    assert o["channel"] == "retail", "edit route re-resolved the client flag — latch rebuilt"
    assert o["shipping_cents"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) python3 -m pytest tests/test_orders_manual_pickup_default.py -q`
Expected: FAIL on `test_absent_pickup_key_uses_client_flag` — `assert 'retail' == 'pickup'`. Today `pickup = bool(body.get("pickup"))` yields `False` when the key is absent, so the order stores `retail`.

Confirm the skip count is `0`. `1 passed, 6 skipped` means `DATA_DIR` was not overridden and nothing ran.

- [ ] **Step 3: Write minimal implementation**

In `app.py`, inside `api_orders_manual`, replace line 32916:

```python
    pickup = bool(body.get("pickup"))
```

with:

```python
    # An explicit `pickup` always wins (order entry always posts the checkbox).
    # Only when the key is ABSENT — the Biofield hand-off — does the client's
    # saved default decide. Unknown client -> False -> shipping is charged.
    # The EDIT route must never do this: re-resolving would re-latch the channel.
    if "pickup" in body:
        pickup = bool(body.get("pickup"))
    else:
        _pcx = _sqlite3.connect(LOG_DB)
        try:
            pickup = _cust.pickup_default_for_email(_pcx, customer.get("email"))
        finally:
            _pcx.close()
```

(`_cust` is already imported at `app.py:32912` — `from dashboard import customers as _cust`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) python3 -m pytest tests/test_orders_manual_pickup_default.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_orders_manual_pickup_default.py
git commit -m "feat(orders): /api/orders/manual resolves the client's pickup default"
```

---

### Task 5: Console endpoint to set the flag

**Files:**
- Modify: `app.py` — insert after `api_console_customer_rename` (ends `app.py:33105`)
- Test: `tests/test_console_customer_pickup.py` (create)

**Interfaces:**
- Consumes: `dashboard.customers.set_pickup_default`, `pickup_default_for_email` (Task 3).
- Produces: `POST /api/console/customers/pickup` body `{email, pickup: bool}` → `{"ok": true, "email": str, "pickup": bool}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_console_customer_pickup.py`:

```python
import sqlite3

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_people_table()
    appmod.app.config["TESTING"] = True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("INSERT INTO people (email) VALUES ('d@x.com')")
        cx.commit()
    monkeypatch.setattr(appmod, "_bos_actor",
                        lambda: type("A", (), {"role": appmod._bos_rbac.OWNER})())
    return appmod.app.test_client(), appmod


def test_sets_and_clears_the_flag(client):
    c, appmod = client
    from dashboard import customers as C
    r = c.post("/api/console/customers/pickup", json={"email": "d@x.com", "pickup": True})
    assert r.get_json()["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert C.pickup_default_for_email(cx, "d@x.com") is True
    c.post("/api/console/customers/pickup", json={"email": "d@x.com", "pickup": False})
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert C.pickup_default_for_email(cx, "d@x.com") is False


def test_unknown_email_is_400_not_silent_noop(client):
    c, _ = client
    r = c.post("/api/console/customers/pickup", json={"email": "nope@x.com", "pickup": True})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_blank_email_is_400(client):
    c, _ = client
    r = c.post("/api/console/customers/pickup", json={"email": "", "pickup": True})
    assert r.status_code == 400


def test_non_owner_is_401(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_bos_actor", lambda: None)
    r = c.post("/api/console/customers/pickup", json={"email": "d@x.com", "pickup": True})
    assert r.status_code == 401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) python3 -m pytest tests/test_console_customer_pickup.py -q`
Expected: FAIL — all four return `404` (route not registered).

- [ ] **Step 3: Write minimal implementation**

In `app.py`, immediately after `api_console_customer_rename` returns, add:

```python
@app.route("/api/console/customers/pickup", methods=["POST"])
def api_console_customer_pickup():
    """Owner: mark a client as collecting in person. Orders Glen creates for them
    default to pickup (no shipping); the client's own portal/funnel checkout is
    unaffected. Body: {email, pickup: bool}. Email-keyed, like /rename."""
    actor = _bos_actor()
    if actor is None or actor.role != _bos_rbac.OWNER:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip()
    if not email:
        return jsonify({"ok": False, "error": "email required"}), 400
    on = bool(body.get("pickup"))
    from dashboard import customers as _cust
    cx = _sqlite3.connect(LOG_DB)
    try:
        row = cx.execute("SELECT id FROM people WHERE lower(email)=?",
                         (email.lower(),)).fetchone()
        if not row:
            return jsonify({"ok": False, "error": f"no client with email {email}"}), 400
        _cust.set_pickup_default(cx, row[0], on)
    finally:
        cx.close()
    return jsonify({"ok": True, "email": email, "pickup": on})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) python3 -m pytest tests/test_console_customer_pickup.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_console_customer_pickup.py
git commit -m "feat(console): POST /api/console/customers/pickup sets a client's pickup default"
```

---

### Task 6: Order-entry UI — pre-check and toggle

**Files:**
- Modify: `static/order-new.html:114` (checkbox markup — add the toggle beside it)
- Modify: `static/order-new.html:188-196` (`pickPerson`)
- Test: `tests/test_order_new_pickup_ui.py` (create — static-source assertions, matching the repo's existing HTML-wiring test style)

**Interfaces:**
- Consumes: `pickup_default` on the picker payload (Task 3, via `PICKER_COLS`); `POST /api/console/customers/pickup` (Task 5).
- Produces: no server interface.

- [ ] **Step 1: Write the failing test**

Create `tests/test_order_new_pickup_ui.py`:

```python
"""order-new.html must (a) pre-check pickup from the client's flag on CREATE,
(b) clear it when a non-pickup client is picked, (c) never do either in EDIT mode."""
import pathlib
import re

HTML = pathlib.Path("static/order-new.html").read_text()


def test_pick_person_sets_checkbox_both_ways():
    """`= !!p.pickup_default` — never `if (x) ... = true`, which cannot clear."""
    assert re.search(r'\$\("pickup"\)\.checked\s*=\s*!!p\.pickup_default', HTML)


def test_pick_person_guards_edit_mode():
    """Edit mode prefills from the ORDER's channel, never the client's flag."""
    m = re.search(r"function pickPerson\(p\)\{(.*?)\n\}", HTML, re.S)
    assert m, "pickPerson not found"
    body = m.group(1)
    assert "pickup_default" in body
    assert "EDIT_OID" in body, "pickPerson must not touch pickup in edit mode"


def test_always_picks_up_toggle_exists_and_posts():
    assert 'id="pickup-default"' in HTML
    assert "/api/console/customers/pickup" in HTML
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_order_new_pickup_ui.py -q`
Expected: FAIL — `assert re.search(...)` is `None` (no `pickup_default` in the file).

- [ ] **Step 3: Write minimal implementation**

In `static/order-new.html`, replace line 114:

```html
    <div style="margin-top:10px"><label style="display:inline-flex;align-items:center;gap:6px;text-transform:none;font-size:14px;color:var(--text);cursor:pointer"><input type="checkbox" id="pickup"> Pickup (no shipping)</label></div>
```

with:

```html
    <div style="margin-top:10px"><label style="display:inline-flex;align-items:center;gap:6px;text-transform:none;font-size:14px;color:var(--text);cursor:pointer"><input type="checkbox" id="pickup"> Pickup (no shipping)</label>
      <label style="display:inline-flex;align-items:center;gap:6px;margin-left:16px;text-transform:none;font-size:13px;color:var(--muted);cursor:pointer"><input type="checkbox" id="pickup-default"> Always picks up (save to client)</label></div>
```

In `pickPerson(p)`, after the `$("a-country")` line, add:

```javascript
  // Create mode only: a client marked "always picks up" pre-checks the box. Set it
  // BOTH ways so choosing a normal client afterwards clears it. Edit mode prefills
  // from the order's own channel — re-resolving there would re-latch the channel.
  if (!EDIT_OID) $("pickup").checked = !!p.pickup_default;
  $("pickup-default").checked = !!p.pickup_default;
```

Add near the other listeners (after the `cust-search` listener block):

```javascript
$("pickup-default").addEventListener("change", async function(){
  const email = $("c-email").value.trim();
  if (!email){ toast("Pick a client first","error"); this.checked = !this.checked; return; }
  try{
    const r = await fetch("/api/console/customers/pickup", {method:"POST", headers:HEADERS,
      body: JSON.stringify({email: email, pickup: this.checked})});
    const j = await r.json();
    if (!j.ok){ toast(j.error||"failed","error"); this.checked = !this.checked; return; }
    if (!EDIT_OID) $("pickup").checked = j.pickup;
    toast(j.pickup ? "Client marked as pickup" : "Client no longer pickup");
  }catch(e){ toast(e.message,"error"); this.checked = !this.checked; }
});
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_order_new_pickup_ui.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add static/order-new.html tests/test_order_new_pickup_ui.py
git commit -m "feat(order-entry): pre-check pickup from the client's flag + save toggle"
```

---

### Task 7: Regression gate before PR

**Files:** none modified.

**Interfaces:**
- Consumes: everything above.
- Produces: a clean branch-vs-baseline diff.

- [ ] **Step 1: Run the touched suites**

```bash
doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) python3 -m pytest \
  tests/test_people_pickup_column.py tests/test_init_people_table_pickup.py \
  tests/test_customers_pickup_default.py tests/test_orders_manual_pickup_default.py \
  tests/test_console_customer_pickup.py tests/test_order_new_pickup_ui.py \
  tests/test_inhouse_order_entry.py tests/test_biofield_invoice_net.py \
  tests/test_orders_effective_shipping.py -q
```
Expected: all pass.

- [ ] **Step 2: Full suite on this branch, captured by name**

```bash
doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) python3 -m pytest tests/ -q -p no:cacheprovider 2>&1 \
  | grep -E "^FAILED" | sed -E 's/^(FAILED [^ ]+).*/\1/' | sort -u > /tmp/branch.set
```

- [ ] **Step 3: Full suite on the `origin/main` baseline**

```bash
git -C ~/deploy-chat worktree add -q --detach /tmp/wt-baseline origin/main
cd /tmp/wt-baseline && doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) \
  python3 -m pytest tests/ -q -p no:cacheprovider 2>&1 \
  | grep -E "^FAILED" | sed -E 's/^(FAILED [^ ]+).*/\1/' | sort -u > /tmp/base.set
```

- [ ] **Step 4: Diff by NAME, not count**

```bash
comm -13 /tmp/base.set /tmp/branch.set   # must be EMPTY = no new failures
```
Expected: no output. `main` carries ~97 pre-existing failures and is flaky — counts prove nothing.

- [ ] **Step 5: Open the PR (do not merge)**

```bash
git push -u origin sess/5629cdf8-pickup-flag
gh pr create --title "feat(orders): per-client pickup flag" --body "See docs/superpowers/specs/2026-07-08-per-client-pickup-flag-design.md"
```

**Merging is a production deploy (auto-deploy on merge to `main`) and requires Glen's explicit authorization. Do not merge.**

---

## Manual verification (Glen, after deploy)

1. Order entry → pick a normal client → "Pickup (no shipping)" is **unchecked**.
2. Tick "Always picks up (save to client)" → toast confirms; "Pickup (no shipping)" checks itself.
3. Pick a different, normal client → both boxes **clear** (proves `= !!x`, not `if (x) … = true`).
4. Re-pick the flagged client → both boxes check again (flag persisted).
5. Stage a Biofield hand-off for the flagged client → invoice arrives with **$0 shipping**.
6. Stage a Biofield hand-off for a normal client → invoice arrives with **shipping charged** on the remedy bottles.
7. Edit the flagged client's order, untick "Pickup (no shipping)", save → shipping is charged and it **stays** unticked on reload. (This is the PR #734 latch guard; if it snaps back, the edit route is reading the flag.)

## Follow-ups (not this plan)

- Backfill of already-latched `channel='pickup'` rows. Counting script: `/tmp/pickup-count.sh` (aggregate only, no PII).
- `dashboard/portal_identity.py:46` creates a narrower `people` table under the same name; `_init_people_table()` only wins by import order.
- Bundles (`dry-eye-relief-program`, `glucose-tolerance-program`, `macular-wellness-program`) resolve to the `"default"` bottle type but hold physical components; their packing is likely wrong.
