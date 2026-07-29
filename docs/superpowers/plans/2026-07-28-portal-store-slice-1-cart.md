# Portal Store Slice 1: Persistent Cart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the app a persistent, server-side cart that an anonymous visitor can fill from a product page and that becomes theirs, in their portal, when they become a Tier-1 member at checkout.

**Architecture:** Two new tables keyed by an opaque cart token, one new pure module (`dashboard/cart_store.py`) that never imports Flask, thin routes in `app.py`, and no change to the pricing or checkout engine. `_checkout_cart()` is called exactly as it is today, with its `cart` argument read from `cart_store` instead of posted by the page. Everything is gated on `PORTAL_CART_ENABLED`, which ships off.

**Tech Stack:** Python 3, Flask, the repo's `db` adapter (SQLite locally, Postgres in prod via the pgcompat adapter), pytest, vanilla JS in `static/`.

**Spec:** `docs/superpowers/specs/2026-07-28-portal-store-design.md`

## Global Constraints

- `PORTAL_CART_ENABLED` defaults **off**. With the flag off, the portal view payload and every existing page must be **byte-identical** to today. This is the same discipline `support_programs`, `PORTAL_OASIS_ENABLED` and `PORTAL_REMEDIES_ENABLED` follow.
- **Never use `cur.lastrowid`.** It raises on the Postgres adapter. This schema avoids autoincrement entirely: `carts` is keyed by a TEXT token the app generates, `cart_items` by a composite primary key.
- **Carts never store prices.** Only `slug`, `qty`, `fmt`. `_price_cart()` recomputes at checkout.
- `cart_store` is pure: the caller passes `cx`. No Flask import, no `LOG_DB` reference. This mirrors `dashboard/repertoire.py`.
- Item dicts handed to `_checkout_cart` use the key **`format`**, not `fmt`, because `_price_cart` reads `c.get("format")` (see `app.py:9950`, `begin_checkout`). The database column is `fmt`; the boundary translates.
- **Do not run the bare full test suite locally.** It sends real email. Run named test files, and let CI run the suite.
- Tests that touch the catalog must pin `app._get_product` or the products file explicitly. `$DATA_DIR` strips `products.json` under the full suite.
- Pricing policy is untouched by this slice. No change to `dashboard/pricing.py`.
- **`abort` is not imported** in `app.py` (see the flask import at line 29). Return a 404 tuple, never `abort(404)`.
- Cookies in this repo are set with `secure=request.is_secure`, not `secure=True`. Using `secure=True` breaks the pytest test client, which runs over http and would refuse to send the cookie back.
- **Line anchors below were verified against this worktree**, which is ~1,090 lines ahead of the `~/deploy-chat` working checkout. Do not trust anchors from that checkout; re-grep if anything has moved again.

---

## File Structure

**Create:**
- `dashboard/cart_store.py` — persistence only: schema, add, set quantity, list, merge, mark ordered. No prices, no Flask.
- `dashboard/cart_block.py` — the portal view block, mirroring `dashboard/oasis_block.py`.
- `tests/test_cart_store.py` — unit tests for the module.
- `tests/test_cart_routes.py` — route tests for the cart API.
- `tests/test_cart_checkout.py` — the anonymous-to-member merge and checkout path.
- `tests/test_cart_block.py` — the portal view block: inert when off, counts when on, never raises.

**Modify:**
- `app.py` — one flag constant beside the other portal flags (after line 6174), a `_cart_email()` helper and four routes beside the reorder routes (`_reorder_email_from_cookie` is at 18605, `@app.route("/reorder")` at 18609), the product route at 7678, and the `get_portal_view` call site at 27899.
- `dashboard/portal_view.py` — a `cart_enabled` kwarg on `get_portal_view` (line 365) and a `"cart"` entry in the view dict (line 408).
- `static/begin-product.html` — an Add to cart control next to the existing buy control.
- `static/client-portal.html` — a Cart tile in `buildHubHtml` (`actTiles` at line 949) and a cart panel.

---

## Task 1: cart_store schema and item operations

**Files:**
- Create: `dashboard/cart_store.py`
- Test: `tests/test_cart_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `init_cart_tables(cx) -> None`
  - `get_or_create(cx, token, email="") -> str` (returns the token of an open cart, **which may differ from the token passed in** — see the resolution order in the module docstring; callers must use the return value)
  - `add_item(cx, token, slug, qty=1, fmt="", source="") -> int` (returns the new quantity for that line)
  - `set_qty(cx, token, slug, fmt, qty) -> None` (qty <= 0 removes the line)
  - `items(cx, token) -> list[dict]` with keys `slug`, `qty`, `format`, `source`
  - `open_token_for_email(cx, email) -> str | ""`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cart_store.py`:

```python
import sqlite3

import pytest

from dashboard import cart_store as CS


@pytest.fixture()
def cx(tmp_path):
    c = sqlite3.connect(str(tmp_path / "cart.db"))
    CS.init_cart_tables(c)
    yield c
    c.close()


def test_get_or_create_returns_token_and_is_idempotent(cx):
    t = CS.get_or_create(cx, "tok1")
    assert t == "tok1"
    assert CS.get_or_create(cx, "tok1") == "tok1"
    assert CS.items(cx, "tok1") == []


def test_add_item_then_list(cx):
    CS.get_or_create(cx, "tok1")
    assert CS.add_item(cx, "tok1", "brain-boost", qty=2, source="product") == 2
    assert CS.items(cx, "tok1") == [
        {"slug": "brain-boost", "qty": 2, "format": "", "source": "product"}
    ]


def test_add_same_slug_and_format_increments(cx):
    CS.get_or_create(cx, "tok1")
    CS.add_item(cx, "tok1", "brain-boost", qty=1)
    assert CS.add_item(cx, "tok1", "brain-boost", qty=2) == 3
    assert len(CS.items(cx, "tok1")) == 1


def test_same_slug_different_format_is_a_separate_line(cx):
    CS.get_or_create(cx, "tok1")
    CS.add_item(cx, "tok1", "brain-boost", qty=1, fmt="bottle")
    CS.add_item(cx, "tok1", "brain-boost", qty=1, fmt="refill")
    assert len(CS.items(cx, "tok1")) == 2


def test_set_qty_updates_and_zero_removes(cx):
    CS.get_or_create(cx, "tok1")
    CS.add_item(cx, "tok1", "brain-boost", qty=5)
    CS.set_qty(cx, "tok1", "brain-boost", "", 2)
    assert CS.items(cx, "tok1")[0]["qty"] == 2
    CS.set_qty(cx, "tok1", "brain-boost", "", 0)
    assert CS.items(cx, "tok1") == []


def test_qty_is_clamped_to_1_99(cx):
    CS.get_or_create(cx, "tok1")
    CS.add_item(cx, "tok1", "brain-boost", qty=500)
    assert CS.items(cx, "tok1")[0]["qty"] == 99


def test_open_token_for_email(cx):
    CS.get_or_create(cx, "tokA", email="A@X.com")
    assert CS.open_token_for_email(cx, "a@x.com") == "tokA"
    assert CS.open_token_for_email(cx, "nobody@x.com") == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-4bb91433 && python -m pytest tests/test_cart_store.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'dashboard.cart_store'`

- [ ] **Step 3: Write the implementation**

Create `dashboard/cart_store.py`:

```python
"""Persistent shopping cart (token-keyed). Pure: the caller passes cx.

Deliberately stores NO prices -- only slug/qty/fmt. `_price_cart` recomputes at
checkout, which is what makes a price change between adding and paying resolve
correctly by construction.

Schema note: `carts` is keyed by an app-generated TEXT token and `cart_items` by a
composite primary key, so nothing here needs an autoincrement id. `cur.lastrowid`
RAISES on the Postgres adapter, and this shape never reaches for it.
"""
from datetime import datetime, timezone

MAX_QTY = 99


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _norm_email(email):
    return (email or "").strip().lower()


def _clamp(qty):
    try:
        qty = int(qty)
    except (TypeError, ValueError):
        qty = 1
    return max(1, min(qty, MAX_QTY))


def init_cart_tables(cx):
    cx.execute(
        """CREATE TABLE IF NOT EXISTS carts (
             token        TEXT PRIMARY KEY,
             email        TEXT NOT NULL DEFAULT '',
             status       TEXT NOT NULL DEFAULT 'open',
             checkout_ref TEXT NOT NULL DEFAULT '',
             created_at   TEXT NOT NULL,
             updated_at   TEXT NOT NULL
           )"""
    )
    cx.execute(
        """CREATE TABLE IF NOT EXISTS cart_items (
             token    TEXT NOT NULL,
             slug     TEXT NOT NULL,
             fmt      TEXT NOT NULL DEFAULT '',
             qty      INTEGER NOT NULL,
             source   TEXT NOT NULL DEFAULT '',
             added_at TEXT NOT NULL,
             PRIMARY KEY (token, slug, fmt)
           )"""
    )
    # One open cart per identified member. Partial index works on SQLite and Postgres.
    cx.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_carts_open_email "
        "ON carts(email) WHERE status='open' AND email<>''"
    )
    cx.commit()


def get_or_create(cx, token, email=""):
    token = (token or "").strip()
    if not token:
        raise ValueError("token required")
    row = cx.execute(
        "SELECT token FROM carts WHERE token=? AND status='open'", (token,)
    ).fetchone()
    if row:
        return token
    now = _now_iso()
    cx.execute(
        "INSERT INTO carts(token, email, status, checkout_ref, created_at, updated_at) "
        "VALUES (?,?,'open','',?,?)",
        (token, _norm_email(email), now, now),
    )
    cx.commit()
    return token


def open_token_for_email(cx, email):
    email = _norm_email(email)
    if not email:
        return ""
    row = cx.execute(
        "SELECT token FROM carts WHERE email=? AND status='open' LIMIT 1", (email,)
    ).fetchone()
    return row[0] if row else ""


def _touch(cx, token):
    cx.execute("UPDATE carts SET updated_at=? WHERE token=?", (_now_iso(), token))


def add_item(cx, token, slug, qty=1, fmt="", source=""):
    slug = (slug or "").strip().lower()
    if not slug:
        raise ValueError("slug required")
    fmt = (fmt or "").strip().lower()
    qty = _clamp(qty)
    row = cx.execute(
        "SELECT qty FROM cart_items WHERE token=? AND slug=? AND fmt=?", (token, slug, fmt)
    ).fetchone()
    new_qty = _clamp((row[0] if row else 0) + qty)
    if row:
        cx.execute(
            "UPDATE cart_items SET qty=? WHERE token=? AND slug=? AND fmt=?",
            (new_qty, token, slug, fmt),
        )
    else:
        cx.execute(
            "INSERT INTO cart_items(token, slug, fmt, qty, source, added_at) VALUES (?,?,?,?,?,?)",
            (token, slug, fmt, new_qty, (source or "").strip(), _now_iso()),
        )
    _touch(cx, token)
    cx.commit()
    return new_qty


def set_qty(cx, token, slug, fmt, qty):
    slug = (slug or "").strip().lower()
    fmt = (fmt or "").strip().lower()
    try:
        qty = int(qty)
    except (TypeError, ValueError):
        qty = 0
    if qty <= 0:
        cx.execute(
            "DELETE FROM cart_items WHERE token=? AND slug=? AND fmt=?", (token, slug, fmt)
        )
    else:
        cx.execute(
            "UPDATE cart_items SET qty=? WHERE token=? AND slug=? AND fmt=?",
            (_clamp(qty), token, slug, fmt),
        )
    _touch(cx, token)
    cx.commit()


def items(cx, token):
    rows = cx.execute(
        "SELECT slug, qty, fmt, source FROM cart_items WHERE token=? ORDER BY added_at, slug",
        (token,),
    ).fetchall()
    return [
        {"slug": r[0], "qty": int(r[1]), "format": r[2] or "", "source": r[3] or ""}
        for r in rows
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-4bb91433 && python -m pytest tests/test_cart_store.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-4bb91433
git add dashboard/cart_store.py tests/test_cart_store.py
git commit -m "feat(cart): cart_store schema and item operations"
```

---

## Task 2: cart_store merge and mark_ordered

**Files:**
- Modify: `dashboard/cart_store.py`
- Test: `tests/test_cart_store_merge.py`

**Interfaces:**
- Consumes: Task 1's `init_cart_tables`, `get_or_create`, `add_item`, `items`, `open_token_for_email`.
- Produces:
  - `merge(cx, anon_token, email) -> str` (returns the surviving member cart token)
  - `mark_ordered(cx, token, checkout_ref) -> None`

The merge rule, decided in the spec: when the same slug and format exist in both carts, the surviving quantity is **the higher of the two, never the sum**. Adding the same bottle on a phone and then a laptop is one intent repeated, and summing would charge the customer double.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cart_store_merge.py`:

```python
import sqlite3

import pytest

from dashboard import cart_store as CS


@pytest.fixture()
def cx(tmp_path):
    c = sqlite3.connect(str(tmp_path / "cart.db"))
    CS.init_cart_tables(c)
    yield c
    c.close()


def test_merge_claims_anon_cart_when_member_has_none(cx):
    CS.get_or_create(cx, "anon1")
    CS.add_item(cx, "anon1", "brain-boost", qty=2)
    surviving = CS.merge(cx, "anon1", "A@X.com")
    assert surviving == "anon1"
    assert CS.open_token_for_email(cx, "a@x.com") == "anon1"
    assert CS.items(cx, "anon1")[0]["qty"] == 2


def test_merge_folds_into_existing_member_cart_higher_qty_wins(cx):
    CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.add_item(cx, "mem1", "brain-boost", qty=1)
    CS.add_item(cx, "mem1", "wholomega", qty=4)
    CS.get_or_create(cx, "anon1")
    CS.add_item(cx, "anon1", "brain-boost", qty=3)   # higher than member's 1
    CS.add_item(cx, "anon1", "neuroprotect", qty=1)  # new line

    surviving = CS.merge(cx, "anon1", "a@x.com")

    assert surviving == "mem1"
    got = {i["slug"]: i["qty"] for i in CS.items(cx, "mem1")}
    assert got == {"brain-boost": 3, "wholomega": 4, "neuroprotect": 1}


def test_merge_never_sums(cx):
    CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.add_item(cx, "mem1", "brain-boost", qty=2)
    CS.get_or_create(cx, "anon1")
    CS.add_item(cx, "anon1", "brain-boost", qty=2)
    CS.merge(cx, "anon1", "a@x.com")
    assert CS.items(cx, "mem1")[0]["qty"] == 2


def test_merge_closes_the_anon_cart(cx):
    CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.get_or_create(cx, "anon1")
    CS.add_item(cx, "anon1", "brain-boost", qty=1)
    CS.merge(cx, "anon1", "a@x.com")
    assert CS.items(cx, "anon1") == []
    row = cx.execute("SELECT status FROM carts WHERE token=?", ("anon1",)).fetchone()
    assert row[0] == "merged"


def test_merge_is_idempotent(cx):
    CS.get_or_create(cx, "anon1")
    CS.add_item(cx, "anon1", "brain-boost", qty=2)
    first = CS.merge(cx, "anon1", "a@x.com")
    second = CS.merge(cx, "anon1", "a@x.com")
    assert first == second == "anon1"
    assert CS.items(cx, first)[0]["qty"] == 2


def test_merge_with_unknown_anon_token_returns_member_cart(cx):
    CS.get_or_create(cx, "mem1", email="a@x.com")
    assert CS.merge(cx, "nosuchtoken", "a@x.com") == "mem1"


def test_merge_requires_an_email(cx):
    CS.get_or_create(cx, "anon1")
    with pytest.raises(ValueError):
        CS.merge(cx, "anon1", "")


def test_mark_ordered_closes_cart_and_records_ref(cx):
    CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.add_item(cx, "mem1", "brain-boost", qty=1)
    CS.mark_ordered(cx, "mem1", "ref123")
    row = cx.execute(
        "SELECT status, checkout_ref FROM carts WHERE token=?", ("mem1",)
    ).fetchone()
    assert row[0] == "ordered"
    assert row[1] == "ref123"
    assert CS.open_token_for_email(cx, "a@x.com") == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-4bb91433 && python -m pytest tests/test_cart_store_merge.py -v`
Expected: FAIL, `AttributeError: module 'dashboard.cart_store' has no attribute 'merge'`

- [ ] **Step 3: Write the implementation**

Append to `dashboard/cart_store.py`:

```python
def merge(cx, anon_token, email):
    """Fold an anonymous cart onto a member email and return the surviving token.

    Quantity rule: the HIGHER of the two wins, never the sum. The same bottle added
    on a phone and then a laptop is one intent repeated; summing would charge double.

    Idempotent: merging an already-merged or unknown token is a no-op that still
    returns the member's open cart token.
    """
    email = _norm_email(email)
    if not email:
        raise ValueError("email required")
    anon_token = (anon_token or "").strip()

    member_token = open_token_for_email(cx, email)

    anon_open = False
    if anon_token:
        row = cx.execute(
            "SELECT status, email FROM carts WHERE token=?", (anon_token,)
        ).fetchone()
        anon_open = bool(row) and row[0] == "open"
        if anon_open and (row[1] or "") == email:
            return anon_token          # already this member's cart

    if not anon_open:
        return member_token or get_or_create(cx, _new_token_for(email), email=email)

    if not member_token:
        cx.execute(
            "UPDATE carts SET email=?, updated_at=? WHERE token=?",
            (email, _now_iso(), anon_token),
        )
        cx.commit()
        return anon_token

    for it in items(cx, anon_token):
        row = cx.execute(
            "SELECT qty FROM cart_items WHERE token=? AND slug=? AND fmt=?",
            (member_token, it["slug"], it["format"]),
        ).fetchone()
        if row:
            if int(it["qty"]) > int(row[0]):
                cx.execute(
                    "UPDATE cart_items SET qty=? WHERE token=? AND slug=? AND fmt=?",
                    (_clamp(it["qty"]), member_token, it["slug"], it["format"]),
                )
        else:
            cx.execute(
                "INSERT INTO cart_items(token, slug, fmt, qty, source, added_at) "
                "VALUES (?,?,?,?,?,?)",
                (member_token, it["slug"], it["format"], _clamp(it["qty"]),
                 it["source"], _now_iso()),
            )
    cx.execute("DELETE FROM cart_items WHERE token=?", (anon_token,))
    cx.execute(
        "UPDATE carts SET status='merged', updated_at=? WHERE token=?",
        (_now_iso(), anon_token),
    )
    _touch(cx, member_token)
    cx.commit()
    return member_token


def _new_token_for(email):
    """Deterministic fallback token when a member needs a cart and has none.
    Never used for anonymous carts, which get a random token from the route layer."""
    import hashlib
    return "cart:" + hashlib.sha1(_norm_email(email).encode()).hexdigest()[:24]


def mark_ordered(cx, token, checkout_ref):
    cx.execute(
        "UPDATE carts SET status='ordered', checkout_ref=?, updated_at=? WHERE token=?",
        ((checkout_ref or "").strip(), _now_iso(), token),
    )
    cx.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-4bb91433 && python -m pytest tests/test_cart_store.py tests/test_cart_store_merge.py -v`
Expected: PASS, 15 tests

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-4bb91433
git add dashboard/cart_store.py tests/test_cart_store_merge.py
git commit -m "feat(cart): anonymous-to-member merge, higher qty wins, and mark_ordered"
```

---

## Task 3: Flag, email resolution, and the cart API

**Files:**
- Modify: `app.py` (flag after line 6174 with the other portal flags; routes beside the reorder routes at 18605-18609)
- Test: `tests/test_cart_routes.py`

**Interfaces:**
- Consumes: `dashboard.cart_store` (Tasks 1 and 2).
- Produces:
  - `app._PORTAL_CART_ENABLED` (bool)
  - `app._cart_email() -> str`
  - `app._cart_token_from_cookie() -> str`
  - Routes: `GET /api/cart`, `POST /api/cart/add`, `POST /api/cart/set-qty`
  - Cookie name `rm_cart`

With the flag off, every route returns 404 so the surface does not exist at all.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cart_routes.py`:

```python
import sqlite3

import pytest

import app
from dashboard import cart_store as CS


@pytest.fixture()
def db(monkeypatch, tmp_path):
    path = str(tmp_path / "log.db")
    monkeypatch.setattr(app, "LOG_DB", path)
    monkeypatch.setattr(app, "_PORTAL_CART_ENABLED", True)
    cx = sqlite3.connect(path)
    CS.init_cart_tables(cx)
    cx.close()
    # a real-shaped catalog entry, pinned so $DATA_DIR cannot strip it
    monkeypatch.setattr(
        app, "_get_product",
        lambda slug: {"slug": "brain-boost", "name": "Brain Boost",
                      "price_cents": 6997} if slug == "brain-boost" else None)
    return path


@pytest.fixture()
def client():
    return app.app.test_client()


def test_routes_404_when_flag_off(monkeypatch, client, db):
    monkeypatch.setattr(app, "_PORTAL_CART_ENABLED", False)
    assert client.get("/api/cart").status_code == 404
    assert client.post("/api/cart/add", json={"slug": "brain-boost"}).status_code == 404


def test_empty_cart_for_a_new_visitor(client, db):
    r = client.get("/api/cart")
    assert r.status_code == 200
    assert r.get_json() == {"ok": True, "items": [], "count": 0}


def test_add_sets_the_cookie_and_persists(client, db):
    r = client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 2})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    assert "rm_cart=" in r.headers.get("Set-Cookie", "")

    r2 = client.get("/api/cart")
    body = r2.get_json()
    assert body["count"] == 2
    assert body["items"][0]["slug"] == "brain-boost"
    assert body["items"][0]["name"] == "Brain Boost"
    assert body["items"][0]["available"] is True


def test_add_rejects_an_unknown_slug(client, db):
    r = client.post("/api/cart/add", json={"slug": "no-such-product"})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_add_does_not_require_membership(client, db):
    """Anonymous adds are the whole point -- no need_optin here, only at checkout."""
    r = client.post("/api/cart/add", json={"slug": "brain-boost"})
    assert r.status_code == 200


def test_set_qty_updates_then_removes(client, db):
    client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 5})
    client.post("/api/cart/set-qty", json={"slug": "brain-boost", "qty": 2})
    assert client.get("/api/cart").get_json()["count"] == 2
    client.post("/api/cart/set-qty", json={"slug": "brain-boost", "qty": 0})
    assert client.get("/api/cart").get_json()["items"] == []


def test_unavailable_item_is_flagged_not_dropped(client, db, monkeypatch):
    client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 1})
    monkeypatch.setattr(app, "_get_product", lambda slug: None)
    body = client.get("/api/cart").get_json()
    assert len(body["items"]) == 1
    assert body["items"][0]["available"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-4bb91433 && python -m pytest tests/test_cart_routes.py -v`
Expected: FAIL, `AttributeError: module 'app' has no attribute '_PORTAL_CART_ENABLED'`

- [ ] **Step 3: Add the flag**

In `app.py`, immediately after the `_PORTAL_OASIS_ENABLED` line (~6174):

```python
# Portal store slice 1: the persistent cart. Ships OFF. With this off, the cart
# routes 404, the product page shows no Add to cart control, and the portal
# payload is byte-identical to pre-cart.
_PORTAL_CART_ENABLED = os.environ.get("PORTAL_CART_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
```

- [ ] **Step 4: Write the helpers and routes**

In `app.py`, beside the reorder routes (`_reorder_email_from_cookie` is defined at 18605, `@app.route("/reorder")` at 18609):

```python
_CART_COOKIE = "rm_cart"


def _cart_token_from_cookie():
    return (request.cookies.get(_CART_COOKIE) or "").strip()


def _cart_email():
    """Best-known email for the current visitor, or "" if anonymous. Tries the
    reorder cookie first (an identified reorder session), then the funnel state
    for this amg_session."""
    email = (_reorder_email_from_cookie() or "").strip().lower()
    if email:
        return email
    sid = (request.cookies.get("amg_session") or "").strip()
    if not sid:
        return ""
    try:
        with db.connect(LOG_DB) as cx:
            state = begin_funnel.get_state(cx, session_id=sid)
        return (state.get("email") or "").strip().lower()
    except Exception as e:
        print(f"[cart] email resolve failed: {e!r}", flush=True)
        return ""


def _cart_open_token(cx):
    """The token for this visitor's open cart, or "" if they have none. A member's
    cart wins over the cookie, which is what makes the cart follow them across
    devices once identified."""
    email = _cart_email()
    if email:
        tok = _cart_store.open_token_for_email(cx, email)
        if tok:
            return tok
    return _cart_token_from_cookie()


def _cart_payload(cx, token):
    if not token:
        return {"ok": True, "items": [], "count": 0}
    out, count = [], 0
    for it in _cart_store.items(cx, token):
        p = _get_product(it["slug"])
        out.append({
            "slug": it["slug"],
            "name": (p or {}).get("name", it["slug"]),
            "qty": it["qty"],
            "format": it["format"],
            "available": bool(p) and not p.get("inactive"),
        })
        # Counts EVERY line, including unavailable ones, so this number always
        # matches the tile badge from dashboard/cart_block.py. An unavailable row
        # is surfaced by its `available` flag, not by silently changing the count.
        count += it["qty"]
    return {"ok": True, "items": out, "count": count}


@app.route("/api/cart", methods=["GET"])
def api_cart():
    if not _PORTAL_CART_ENABLED:
        return jsonify({"ok": False, "error": "not found"}), 404
    with db.connect(LOG_DB) as cx:
        _cart_store.init_cart_tables(cx)
        return jsonify(_cart_payload(cx, _cart_open_token(cx)))


@app.route("/api/cart/add", methods=["POST"])
def api_cart_add():
    if not _PORTAL_CART_ENABLED:
        return jsonify({"ok": False, "error": "not found"}), 404
    data = request.get_json(silent=True) or {}
    slug = (data.get("slug") or "").strip().lower()
    p = _get_product(slug)
    if not p or p.get("info_only") or p.get("inactive"):
        return jsonify({"ok": False, "error": "That product is not available."}), 400
    fmt = (data.get("format") or "").strip().lower()
    try:
        qty = max(1, min(int(data.get("qty", 1) or 1), 99))
    except (TypeError, ValueError):
        qty = 1
    email = _cart_email()
    with db.connect(LOG_DB) as cx:
        _cart_store.init_cart_tables(cx)
        token = _cart_open_token(cx) or _uuid.uuid4().hex
        # get_or_create returns the token of an OPEN cart, which may DIFFER from the
        # one passed: it returns the member's existing open cart, and it mints a fresh
        # token when the one we hold belongs to a cart already merged or ordered (the
        # cookie still holds the old token after a customer's first order). Always use
        # the returned value, and re-cookie whenever it differs from what the browser sent.
        token = _cart_store.get_or_create(cx, token, email=email)
        new_cookie = token if token != _cart_token_from_cookie() else ""
        _cart_store.add_item(cx, token, slug, qty=qty, fmt=fmt,
                             source=(data.get("source") or "").strip())
        payload = _cart_payload(cx, token)
    resp = jsonify(payload)
    if new_cookie:
        # secure=request.is_secure, matching every other cookie in this app. Hard-coding
        # secure=True breaks the pytest test client, which runs over http.
        resp.set_cookie(_CART_COOKIE, new_cookie, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


@app.route("/api/cart/set-qty", methods=["POST"])
def api_cart_set_qty():
    if not _PORTAL_CART_ENABLED:
        return jsonify({"ok": False, "error": "not found"}), 404
    data = request.get_json(silent=True) or {}
    slug = (data.get("slug") or "").strip().lower()
    fmt = (data.get("format") or "").strip().lower()
    with db.connect(LOG_DB) as cx:
        _cart_store.init_cart_tables(cx)
        token = _cart_open_token(cx)
        if not token:
            return jsonify({"ok": True, "items": [], "count": 0})
        _cart_store.set_qty(cx, token, slug, fmt, data.get("qty", 0))
        return jsonify(_cart_payload(cx, token))
```

Add the import near the other `dashboard` imports at the top of `app.py` (alongside `from dashboard import condition_programs, broad_benefit`, ~line 151):

```python
from dashboard import cart_store as _cart_store
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-4bb91433 && python -m pytest tests/test_cart_routes.py -v`
Expected: PASS, 7 tests

- [ ] **Step 6: Commit**

```bash
cd /tmp/wt-deploy-chat-4bb91433
git add app.py tests/test_cart_routes.py
git commit -m "feat(cart): PORTAL_CART_ENABLED flag and the cart API"
```

---

## Task 4: Checkout from the cart, with membership and merge

**Files:**
- Modify: `app.py` (new route beside the cart routes from Task 3)
- Test: `tests/test_cart_checkout.py`

**Interfaces:**
- Consumes: `_cart_store.merge`, `_cart_store.mark_ordered`, `_cart_store.items`, `_checkout_cart`, `is_member`.
- Produces: `POST /api/cart/checkout`

The contract deliberately matches what `reorder.html` and `begin-buy.html` already expect, so the existing `window.OptinGate` is reused rather than a new membership form being written: a non-member gets HTTP 403 with `{"ok": false, "need_optin": true}`, the page shows the gate, the gate posts to `/begin/unlock`, and the page retries the checkout.

`_checkout_cart(email, cart, *, ship, points_to_redeem_cents=0, referral_code=None)` returns `{"out": {...}, "stripe_url": "..."}` and raises `CheckoutError`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cart_checkout.py`:

```python
import sqlite3

import pytest

import app
from dashboard import cart_store as CS


@pytest.fixture()
def db(monkeypatch, tmp_path):
    path = str(tmp_path / "log.db")
    monkeypatch.setattr(app, "LOG_DB", path)
    monkeypatch.setattr(app, "_PORTAL_CART_ENABLED", True)
    cx = sqlite3.connect(path)
    CS.init_cart_tables(cx)
    cx.close()
    monkeypatch.setattr(
        app, "_get_product",
        lambda slug: {"slug": slug, "name": "Brain Boost", "price_cents": 6997}
        if slug in ("brain-boost", "wholomega") else None)
    return path


@pytest.fixture()
def client():
    return app.app.test_client()


ADDRESS = {"name": "A B", "street": "1 Main", "city": "Hilo",
           "state": "HI", "zip": "96720", "country": "US"}


def _stub_checkout(monkeypatch, seen):
    def fake(email, cart, *, ship, points_to_redeem_cents=0, referral_code=None):
        seen.append({"email": email, "cart": cart})
        return {"out": {"invoice_id": "ref123", "total": 69.97},
                "stripe_url": "https://stripe.test/session"}
    monkeypatch.setattr(app, "_checkout_cart", fake)


def test_non_member_gets_need_optin(client, db, monkeypatch):
    monkeypatch.setattr(app, "is_member", lambda sid, email: False)
    client.post("/api/cart/add", json={"slug": "brain-boost"})
    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 403
    assert r.get_json()["need_optin"] is True


def test_empty_cart_is_refused(client, db, monkeypatch):
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")
    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 400
    assert "empty" in r.get_json()["error"].lower()


def test_unavailable_item_blocks_checkout(client, db, monkeypatch):
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    client.post("/api/cart/add", json={"slug": "brain-boost"})
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")
    monkeypatch.setattr(app, "_get_product", lambda slug: None)
    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 400
    assert "no longer available" in r.get_json()["error"].lower()


def test_member_checkout_merges_anon_cart_and_marks_ordered(client, db, monkeypatch):
    seen = []
    _stub_checkout(monkeypatch, seen)
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)

    client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 2})
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")

    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["stripe_url"] == "https://stripe.test/session"

    # the cart reached _checkout_cart in the shape _price_cart reads
    assert seen[0]["email"] == "a@x.com"
    assert seen[0]["cart"] == [{"slug": "brain-boost", "qty": 2, "format": ""}]

    # the cart is now the member's, and closed
    cx = sqlite3.connect(app.LOG_DB)
    try:
        assert CS.open_token_for_email(cx, "a@x.com") == ""
        row = cx.execute(
            "SELECT status, checkout_ref FROM carts WHERE checkout_ref='ref123'").fetchone()
        assert row[0] == "ordered"
    finally:
        cx.close()

    # and the visitor's next GET starts clean
    assert client.get("/api/cart").get_json()["items"] == []


def test_checkout_error_is_surfaced_not_swallowed(client, db, monkeypatch):
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")

    def boom(*a, **k):
        raise app.CheckoutError("We only ship within the US right now.")
    monkeypatch.setattr(app, "_checkout_cart", boom)

    client.post("/api/cart/add", json={"slug": "brain-boost"})
    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 400
    assert "US" in r.get_json()["error"]


def test_no_stripe_url_is_an_error_not_a_silent_success(client, db, monkeypatch):
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")
    monkeypatch.setattr(
        app, "_checkout_cart",
        lambda email, cart, **k: {"out": {"invoice_id": "r1", "total": 1.0},
                                  "stripe_url": ""})
    client.post("/api/cart/add", json={"slug": "brain-boost"})
    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 502
    assert r.get_json()["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-4bb91433 && python -m pytest tests/test_cart_checkout.py -v`
Expected: FAIL, 404 responses because the route does not exist

- [ ] **Step 3: Write the route**

In `app.py`, after `api_cart_set_qty`:

```python
@app.route("/api/cart/checkout", methods=["POST"])
def api_cart_checkout():
    """Cart -> order. The membership gate reuses the existing need_optin contract,
    so the page can show the shared window.OptinGate exactly as reorder.html and
    begin-buy.html already do. Membership itself is written by /begin/unlock."""
    if not _PORTAL_CART_ENABLED:
        return jsonify({"ok": False, "error": "not found"}), 404
    data = request.get_json(silent=True) or {}
    email = _cart_email()
    _sid = (request.cookies.get("amg_session") or "").strip()
    if not is_member(_sid, email):
        return jsonify({"ok": False, "need_optin": True,
                        "error": "Please add your name and agree to our Terms "
                                 "to place your order."}), 403

    with db.connect(LOG_DB) as cx:
        _cart_store.init_cart_tables(cx)
        anon_token = _cart_token_from_cookie()
        token = _cart_store.merge(cx, anon_token, email)
        cart = _cart_store.items(cx, token)

    if not cart:
        return jsonify({"ok": False, "error": "Your cart is empty."}), 400

    unavailable = [c["slug"] for c in cart
                   if not _get_product(c["slug"])
                   or (_get_product(c["slug"]) or {}).get("inactive")]
    if unavailable:
        return jsonify({"ok": False, "unavailable": unavailable,
                        "error": "Some items are no longer available. "
                                 "Please remove them and try again."}), 400

    ship = _normalize_ship_address(data.get("address") or {},
                                   fallback_name=(data.get("name") or ""))
    try:
        redeem = int(data.get("points_to_redeem_cents") or 0)
    except (TypeError, ValueError):
        redeem = 0
    try:
        res = _checkout_cart(email, [{"slug": c["slug"], "qty": c["qty"],
                                      "format": c["format"]} for c in cart],
                             ship=ship, points_to_redeem_cents=redeem,
                             referral_code=(data.get("referral_code") or "").strip())
    except CheckoutError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    if not res.get("stripe_url"):
        # Never confirm an order the customer has no way to pay for.
        print("[cart] checkout produced no stripe_url", flush=True)
        return jsonify({"ok": False,
                        "error": "We could not start payment just now. "
                                 "Please try again in a moment."}), 502

    with db.connect(LOG_DB) as cx:
        _cart_store.mark_ordered(cx, token, res["out"].get("invoice_id", ""))
    return jsonify({"ok": True, "stripe_url": res["stripe_url"],
                    "total": res["out"].get("total")})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-4bb91433 && python -m pytest tests/test_cart_checkout.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Run the neighbouring checkout tests for regressions**

Run: `cd /tmp/wt-deploy-chat-4bb91433 && python -m pytest tests/test_checkout_cart_paid_only.py tests/test_cart_store.py tests/test_cart_store_merge.py tests/test_cart_routes.py -v`
Expected: PASS, all

- [ ] **Step 6: Commit**

```bash
cd /tmp/wt-deploy-chat-4bb91433
git add app.py tests/test_cart_checkout.py
git commit -m "feat(cart): checkout from cart with anon-to-member merge"
```

---

## Task 5: Add to cart on the product page

**Files:**
- Modify: `static/begin-product.html`, `app.py` (`/begin/product/<slug>` at line 7678, to pass the flag into the page)
- Test: `tests/test_cart_product_page.py`

**Interfaces:**
- Consumes: `POST /api/cart/add`, `GET /api/cart` (Task 3).
- Produces: nothing other tasks consume.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cart_product_page.py`:

```python
import pytest

import app


@pytest.fixture()
def client():
    return app.app.test_client()


def _prep(monkeypatch):
    monkeypatch.setattr(
        app, "_get_product",
        lambda slug: {"slug": "brain-boost", "name": "Brain Boost",
                      "price_cents": 6997} if slug == "brain-boost" else None)


def test_product_page_has_no_cart_markup_when_flag_off(client, monkeypatch):
    _prep(monkeypatch)
    monkeypatch.setattr(app, "_PORTAL_CART_ENABLED", False)
    html = client.get("/begin/product/brain-boost").get_data(as_text=True)
    assert "data-cart-add" not in html
    assert "CART_ENABLED = false" in html


def test_product_page_exposes_the_cart_control_when_flag_on(client, monkeypatch):
    _prep(monkeypatch)
    monkeypatch.setattr(app, "_PORTAL_CART_ENABLED", True)
    html = client.get("/begin/product/brain-boost").get_data(as_text=True)
    assert "CART_ENABLED = true" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-4bb91433 && python -m pytest tests/test_cart_product_page.py -v`
Expected: FAIL, `CART_ENABLED` appears nowhere in the page

- [ ] **Step 3: Pass the flag into the page**

In `app.py`, in the `/begin/product/<slug>` handler (line 7678), where the page HTML is read and returned, substitute the flag into the served HTML the same way the handler already injects per-product values:

```python
html = html.replace("__CART_ENABLED__", "true" if _PORTAL_CART_ENABLED else "false")
```

- [ ] **Step 4: Add the control to the page**

In `static/begin-product.html`, in the script block, add near the top of the script:

```javascript
var CART_ENABLED = __CART_ENABLED__;
```

And where the existing buy control is rendered, add the cart control behind the flag:

```javascript
function renderCartControl(slug){
  if (!CART_ENABLED) return;
  var host = document.getElementById('buy-actions');
  if (!host) return;
  var btn = document.createElement('button');
  btn.className = 'btn btn-secondary';
  btn.setAttribute('data-cart-add', slug);
  btn.textContent = 'Add to cart';
  btn.addEventListener('click', function(){
    btn.disabled = true;
    var label = btn.textContent;
    btn.textContent = 'Adding…';
    fetch('/api/cart/add', {
      method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({slug: slug, qty: 1, source: 'product'})
    }).then(function(r){ return r.json(); }).then(function(d){
      btn.disabled = false;
      if (!d || !d.ok) { btn.textContent = label; return; }
      btn.textContent = 'Added';
      setTimeout(function(){ btn.textContent = label; }, 2000);
    }).catch(function(){ btn.disabled = false; btn.textContent = label; });
  });
  host.appendChild(btn);
}
```

Call `renderCartControl(slug)` where the page finishes rendering its buy section. If the buy section has no `id="buy-actions"` container, add that id to the element that wraps the existing buy button rather than introducing a new wrapper.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-4bb91433 && python -m pytest tests/test_cart_product_page.py -v`
Expected: PASS, 2 tests

- [ ] **Step 6: Commit**

```bash
cd /tmp/wt-deploy-chat-4bb91433
git add app.py static/begin-product.html tests/test_cart_product_page.py
git commit -m "feat(cart): Add to cart control on the in-funnel product page"
```

---

## Task 6: Cart tile and panel in the portal

**Files:**
- Create: `dashboard/cart_block.py`
- Modify: `dashboard/portal_view.py` (kwarg on `get_portal_view` at line 365, view-dict entry at line 408), `app.py` (the call site at 27899), `static/client-portal.html` (`actTiles` at line 949, plus a panel)
- Test: `tests/test_cart_block.py`

**Interfaces:**
- Consumes: `dashboard.cart_store` (Tasks 1 and 2), `GET /api/cart` and `POST /api/cart/set-qty` (Task 3), `_PORTAL_CART_ENABLED`.
- Produces: `dashboard/cart_block.py: build_block(cx, email, enabled) -> dict`, surfacing as `v.cart = {"enabled": bool, "count": int}`.

**Follow the local pattern, which is not what "byte-identical" means elsewhere.** `get_portal_view` builds its view dict at `portal_view.py:408` with `"remedies": _rb.build_block(...)` and `"oasis": _ob.build_block(...)`, and each of those returns `{"enabled": False}` when off. So the `cart` key is **always present** and carries `enabled: False` when the flag is off. The inertness that matters is in the UI: `buildHubHtml` only pushes the tile when `v.cart.enabled`, exactly as the Oasis and Remedies tiles do, because `showTab` bounces back to the hub on a tile whose panel is not in the DOM.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cart_block.py`:

```python
import sqlite3

import pytest

from dashboard import cart_block as CB
from dashboard import cart_store as CS


@pytest.fixture()
def cx(tmp_path):
    c = sqlite3.connect(str(tmp_path / "cart.db"))
    CS.init_cart_tables(c)
    yield c
    c.close()


def test_block_is_inert_when_disabled(cx):
    CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.add_item(cx, "mem1", "brain-boost", qty=3)
    assert CB.build_block(cx, "a@x.com", False) == {"enabled": False}


def test_block_counts_the_open_cart(cx):
    CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.add_item(cx, "mem1", "brain-boost", qty=3)
    CS.add_item(cx, "mem1", "wholomega", qty=2)
    assert CB.build_block(cx, "a@x.com", True) == {"enabled": True, "count": 5}


def test_block_is_zero_when_the_member_has_no_cart(cx):
    assert CB.build_block(cx, "nobody@x.com", True) == {"enabled": True, "count": 0}


def test_block_ignores_an_ordered_cart(cx):
    CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.add_item(cx, "mem1", "brain-boost", qty=3)
    CS.mark_ordered(cx, "mem1", "ref1")
    assert CB.build_block(cx, "a@x.com", True) == {"enabled": True, "count": 0}


def test_block_never_raises_into_the_payload(cx):
    """A portal payload must degrade, not 500, when a source fails."""
    cx.execute("DROP TABLE cart_items")
    cx.commit()
    assert CB.build_block(cx, "a@x.com", True) == {"enabled": True, "count": 0}


def test_hub_tile_is_gated_on_enabled():
    html = open("static/client-portal.html", encoding="utf-8").read()
    assert "v.cart && v.cart.enabled" in html
```

The wiring itself (the `cart_enabled` kwarg, the view-dict entry, the call site) is covered by the existing portal view tests continuing to pass, which Step 6 runs. Do **not** add a test-only entry point to `app.py` to make a wiring test easier. If you want an explicit wiring test, seed a `people` row and call the real portal view route, copying the fixtures already in `tests/test_client_portal_routes.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-4bb91433 && python -m pytest tests/test_cart_block.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'dashboard.cart_block'`

- [ ] **Step 3: Write the block module**

Create `dashboard/cart_block.py`, mirroring `dashboard/oasis_block.py`:

```python
"""Portal view block for the persistent cart. {"enabled": False} when the flag is
off. Guarded so a failing source degrades to a zero count rather than raising into
the portal payload."""
from dashboard import cart_store as _cs


def build_block(cx, email, enabled) -> dict:
    if not enabled:
        return {"enabled": False}
    try:
        _cs.init_cart_tables(cx)
        token = _cs.open_token_for_email(cx, email)
        count = sum(i["qty"] for i in _cs.items(cx, token)) if token else 0
    except Exception:
        count = 0
    return {"enabled": True, "count": count}
```

- [ ] **Step 3b: Wire it into the portal view**

In `dashboard/portal_view.py`, add the import beside the other block imports (`_ob` at line 18, `_rb` at line 21):

```python
from dashboard import cart_block as _cb
```

Add the kwarg to `get_portal_view` (line 365), beside `oasis_enabled`:

```python
                    cart_enabled=False,
```

Add the entry to the view dict (line 408, beside `"oasis"`):

```python
        "cart": _cb.build_block(cx, email, cart_enabled),
```

In `app.py`, at the `_pv.get_portal_view(...)` call site (line 27899), pass the flag:

```python
                                   cart_enabled=_PORTAL_CART_ENABLED,
```

- [ ] **Step 4: Add the tile and panel**

In `static/client-portal.html`, in `buildHubHtml` where `actTiles` is assembled (~line 949):

```javascript
if (v && v.cart && v.cart.enabled) {
  actTiles.push(["cart", "My Cart", "What you're ready to order", (v.cart.count > 0) ? v.cart.count : undefined]);
}
```

And add a `data-panel="cart"` panel alongside the other secondary panels, with a Back to hub control matching the neighbouring panels, rendering from `GET /api/cart`:

```javascript
function buildCartHtml(){
  return `<div class="card"><h2>My Cart</h2><div id="cart-body" class="muted">Loading…</div></div>`;
}

function loadCart(){
  fetch('/api/cart', {credentials:'same-origin'})
    .then(r => r.json())
    .then(d => {
      const host = document.getElementById('cart-body');
      if (!host) return;
      if (!d.items.length) { host.innerHTML = '<p class="muted">Your cart is empty.</p>'; return; }
      host.innerHTML = d.items.map(it => `
        <div class="reitem">
          <span>${esc(it.name)}</span>
          ${it.available ? '' : '<span class="small err">No longer available</span>'}
          <input type="number" min="0" max="99" value="${it.qty}" data-cart-qty="${esc(it.slug)}">
        </div>`).join('');
    })
    .catch(() => {});
}
```

Wire the quantity input to `POST /api/cart/set-qty` with `{slug, qty}` and re-run `loadCart()` on success, following the delegated-listener pattern the Oasis panel already uses (`static/client-portal.html` ~line 1187).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-4bb91433 && python -m pytest tests/test_cart_block.py -v`
Expected: PASS, 6 tests

- [ ] **Step 6: Run the portal regression tests**

Run: `cd /tmp/wt-deploy-chat-4bb91433 && python -m pytest tests/test_client_portal.py tests/test_client_portal_routes.py -v`
Expected: PASS, unchanged from before this branch

- [ ] **Step 7: Commit**

```bash
cd /tmp/wt-deploy-chat-4bb91433
git add dashboard/cart_block.py dashboard/portal_view.py app.py \
        static/client-portal.html tests/test_cart_block.py
git commit -m "feat(cart): My Cart tile and panel in the portal hub"
```

---

## Final verification before the PR

- [ ] **Run every test this branch touches**

```bash
cd /tmp/wt-deploy-chat-4bb91433
python -m pytest tests/test_cart_store.py tests/test_cart_store_merge.py \
  tests/test_cart_routes.py tests/test_cart_checkout.py \
  tests/test_cart_product_page.py tests/test_cart_block.py \
  tests/test_client_portal.py tests/test_client_portal_routes.py \
  tests/test_checkout_cart_paid_only.py -v
```

- [ ] **Prove the off-state is inert.** With `PORTAL_CART_ENABLED` unset, confirm `/api/cart` returns 404, the product page contains no `data-cart-add`, and the portal payload's `cart` block is exactly `{"enabled": false}` with no tile rendered. This is the claim the PR rests on, so run it rather than assuming it.

- [ ] **Mutation-check the flag gate.** Temporarily flip the flag default to on, confirm the off-state tests fail, then flip it back. A gate that cannot fail is not a gate.

- [ ] **Open the PR with the flag off.** Merging deploys, so state in the PR body that the branch ships dark and that no flag flip is requested yet.

- [ ] **Do not flip the flag in this slice.** The first live exposure belongs to Slice 2, gated to a single condition aisle, per the spec.

---

## Notes for the implementer

**Why a new cart rather than reusing `/begin/concierge/add`.** That endpoint looks like a cart but appends lines to an already-created order's `qbo_lines_json`. It requires an order to exist first, which is exactly what an anonymous shopper does not have. The two can converge later; they should not be merged now.

**Why the merge lives at checkout, not at add.** Decided in the spec: adding is weightless and anonymous, and the membership ask happens once, at the moment the buyer is already committed.

**Prices.** Nothing in this slice prices anything. The cart deliberately shows names and quantities only; `_price_cart` runs at checkout. Cart-level pricing display is Slice 2 work and needs the paid-membership line described in spec section 7.
