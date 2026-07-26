# Product Purity Ratings — Phase 2a (the on-request spine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the Phase-1 purity engine into the app as an on-request, gated, human-confirmed flow — so a rating can be requested (by a paid client or the console), driven through screen → tier-2 → confirm from the console with an operator-supplied ingredient list, without yet building the automatic acquisition cascade.

**Architecture:** Phase 1 shipped three pure modules (`purity_avoidlist`, `purity_screen`, `product_ratings`). Phase 2a adds a thin gate module and a set of Flask routes that drive the existing `product_ratings` state machine, mirroring the shipped `supplement_reviews` / product-review console flow (console-secret gated writes, portal-token-gated client request). Acquisition is operator-manual in 2a (paste the Other Ingredients); the automatic online scrape (2b) and client-photo fallback (2c) are separate follow-on plans that feed the same spine.

**Tech Stack:** Python 3, Flask, sqlite3 (Postgres via the db adapter in prod), pytest.

**Spec:** `docs/superpowers/specs/2026-07-24-product-purity-ratings-design.md` (Section 2).

## Scope note — why 2a, and what is deferred

Phase 2 as specced spans the gate, two acquisition paths, the tier-2 hand-off, and the console. That is several independent subsystems. This plan is **Phase 2a: the spine** — gate + request + operator-manual screen + tier-2 hand-off + confirm console. It is complete and testable on its own (the console can rate any product by hand end-to-end) and it unblocks Phase 3's readers. The acquisition cascade is deferred to its own plans: **2b** = Step-1 online scrape (search + fetch + guarded text extract, async), **2c** = Step-2 client-photo fallback (portal upload + `document_extract` vision path + guard). Both feed the request rows this plan creates.

## Global Constraints

Every task's requirements implicitly include this section.

- **Reuse the Phase-1 engine; do not reimplement it.** `dashboard/product_ratings.py` already owns the state machine (`request` is added in Task 1; `record_screen`, `set_tier2`, `confirm`, `get` exist). `dashboard/purity_screen.py:screen_label(actives, other_ingredients, avoidlist)` and `dashboard/purity_avoidlist.py:load_avoidlist()` exist and are tested.
- **Never downgrade; reds skip tier-2; unrated never green** — these hold in the engine; routes must not work around them. A route that receives an ingredient list runs it through `screen_label` and `record_screen`; it never sets a color directly.
- **Console writes are console-secret gated** via the existing `_portal_console_ok()` check (401 on failure), exactly like `/api/console/product-review/draft`.
- **Client identity comes from the portal token only** (`_portal_record_for(cx, token)`), never a request field — the isolation rule from the portal routes.
- **Gate:** a client may request a rating only on **paid membership** (`membership_category == 'full'`) OR an explicit access grant. Not default-open.
- **DB access pattern:** `with _db_lock, db.connect(LOG_DB) as cx:` for writes; call the module's `init_tables`/`init_table` at the top of each handler (idempotent), as the sibling routes do.
- **Platform:** no `cur.lastrowid` (Postgres raises); `product_ratings.product_key` is the PK. Timestamps Python-side. No `ON CONFLICT`/`datetime('now')`.
- **Tests that import `app`** need `OPENAI_API_KEY=dummy PINECONE_API_KEY=dummy` or collection fails.
- **Never run the bare full test suite — it sends real email.** Run named files only.
- **Every guard/safety test is mutation-verified** before commit (inject the violation, watch the specific test go red, revert).

---

## File Structure

| File | Responsibility |
| --- | --- |
| `dashboard/purity_ratings_access.py` (create) | the paid-or-explicit request gate (mirrors `supplement_review_access` shape) |
| `dashboard/product_ratings.py` (modify) | add `request(cx, product_key, *, brand, product_name, requested_by)` |
| `app.py` (modify) | request routes (portal + console), operator screen route, tier-2 hand-off route, confirm route, console list route |
| `tests/test_purity_ratings_access.py` (create) | gate logic |
| `tests/test_product_ratings_request.py` (create) | `request()` state + idempotency |
| `tests/test_purity_routes.py` (create) | the five routes: auth, gate, isolation, state transitions |

---

### Task 1: The request gate and `product_ratings.request()`

**Files:**
- Create: `dashboard/purity_ratings_access.py`
- Modify: `dashboard/product_ratings.py`
- Create: `tests/test_purity_ratings_access.py`, `tests/test_product_ratings_request.py`

**Interfaces:**
- Produces:
  - `purity_ratings_access.init_table(cx)`; `purity_ratings_access.can_request(cx, email, membership_category) -> bool` (True if `membership_category == 'full'` OR an explicit `enabled=1` row exists); `purity_ratings_access.set_access(cx, email, enabled, set_by)`.
  - `product_ratings.request(cx, product_key, *, brand, product_name, requested_by) -> dict` — inserts a `requested` row if none exists (returns `{"created": True, "status": "requested"}`); if a row exists at any status, returns it untouched (`{"created": False, "status": <current>}`). Never downgrades.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_purity_ratings_access.py`:

```python
import sqlite3
from dashboard import purity_ratings_access as acc


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    acc.init_table(cx); return cx


def test_paid_membership_can_request():
    cx = _cx()
    assert acc.can_request(cx, "a@b.com", "full") is True


def test_non_paid_without_grant_cannot():
    cx = _cx()
    assert acc.can_request(cx, "a@b.com", "trial") is False
    assert acc.can_request(cx, "a@b.com", "none") is False


def test_explicit_grant_overrides_non_paid():
    cx = _cx()
    acc.set_access(cx, "a@b.com", True, "glen")
    assert acc.can_request(cx, "a@b.com", "none") is True


def test_explicit_revoke_blocks_even_if_row_exists():
    cx = _cx()
    acc.set_access(cx, "a@b.com", True, "glen")
    acc.set_access(cx, "a@b.com", False, "glen")
    assert acc.can_request(cx, "a@b.com", "none") is False
    # but paid membership still passes regardless of the revoke row
    assert acc.can_request(cx, "a@b.com", "full") is True
```

Create `tests/test_product_ratings_request.py`:

```python
import sqlite3
from dashboard import product_ratings as pr

GREEN = {"color": "green", "red_hits": [], "yellow_hits": [], "avoidlist_version": "v1"}


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    pr.init_tables(cx); return cx


def test_request_creates_a_requested_row():
    cx = _cx()
    out = pr.request(cx, "brand-x", brand="Brand", product_name="X", requested_by="a@b.com")
    assert out["created"] is True and out["status"] == "requested"
    assert pr.get(cx, "brand-x")["status"] == "requested"


def test_request_is_idempotent_and_never_downgrades():
    cx = _cx()
    pr.request(cx, "k", brand="B", product_name="N", requested_by="a@b.com")
    pr.record_screen(cx, "k", brand="B", product_name="N",
                     other_ingredients_raw="Magnesium Stearate",
                     other_ingredients_parsed=["Magnesium Stearate"],
                     screen={"color": "red", "red_hits": ["Magnesium Stearate"],
                             "yellow_hits": [], "avoidlist_version": "v1"})
    out = pr.request(cx, "k", brand="B", product_name="N", requested_by="c@d.com")
    assert out["created"] is False and out["status"] == "screened"
    assert pr.get(cx, "k")["status"] == "screened", "request must not walk a screened row back"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/deploy-chat && python -m pytest tests/test_purity_ratings_access.py tests/test_product_ratings_request.py -v`
Expected: FAIL — `No module named 'dashboard.purity_ratings_access'` / `product_ratings has no attribute 'request'`.

- [ ] **Step 3: Write the gate module**

Create `dashboard/purity_ratings_access.py` (mirrors `dashboard/supplement_reviews.py`'s access table — default is CLOSED for non-paid, opposite of the free-review default):

```python
"""Request gate for purity ratings. A client may request a rating on paid
membership OR an explicit access grant. Pure sqlite; caller passes cx.
Default is CLOSED for non-paid clients (unlike the free product review, which
defaults open) -- purity ratings are a paid/explicit-request perk."""


def _norm(email):
    return (email or "").strip().lower()


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS purity_ratings_access (
        email TEXT PRIMARY KEY, enabled INTEGER NOT NULL DEFAULT 1,
        set_by TEXT, updated_at TEXT)""")
    cx.commit()


def can_request(cx, email, membership_category):
    if (membership_category or "").strip().lower() == "full":
        return True
    e = _norm(email)
    if not e:
        return False
    row = cx.execute("SELECT enabled FROM purity_ratings_access WHERE email=?", (e,)).fetchone()
    return bool(row and row[0])


def set_access(cx, email, enabled, set_by):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    e = _norm(email)
    row = cx.execute("SELECT email FROM purity_ratings_access WHERE email=?", (e,)).fetchone()
    if row:
        cx.execute("UPDATE purity_ratings_access SET enabled=?, set_by=?, updated_at=? WHERE email=?",
                   (1 if enabled else 0, set_by, now, e))
    else:
        cx.execute("INSERT INTO purity_ratings_access (email, enabled, set_by, updated_at) "
                   "VALUES (?,?,?,?)", (e, 1 if enabled else 0, set_by, now))
    cx.commit()
```

- [ ] **Step 3b: Add the `requested_by` column and `request()` to `product_ratings.py`**

The Phase-1 `product_ratings` table has no `requested_by` column. Add it the way
`dashboard/supplement_reviews.py` adds late columns — a PRAGMA check + `ALTER TABLE` inside
`init_tables`, so existing prod rows migrate in place (a bare `CREATE ... IF NOT EXISTS` would not
add a column to an already-existing table). Append this to the end of `init_tables`, before its
`cx.commit()`:

```python
    _cols = {r[1] for r in cx.execute("PRAGMA table_info(product_ratings)")}
    if "requested_by" not in _cols:
        cx.execute("ALTER TABLE product_ratings ADD COLUMN requested_by TEXT")
```

Then add `request()` after `get()`:

```python
def request(cx, product_key, *, brand, product_name, requested_by):
    """Create a 'requested' row for a product if none exists. If a row exists at
    ANY status it is returned untouched -- request never creates a duplicate and
    never downgrades a further-along row."""
    existing = get(cx, product_key)
    if existing is not None:
        return {"created": False, "status": existing["status"]}
    now = _now()
    cx.execute("""INSERT INTO product_ratings
        (product_key, brand, product_name, status, requested_by, requested_at, updated_at)
        VALUES (?,?,?,?,?,?,?)""",
        (product_key, brand, product_name, "requested", requested_by, now, now))
    cx.commit()
    return {"created": True, "status": "requested"}
```

Re-run the existing Phase-1 `tests/test_product_ratings.py` after this change to confirm the new
column did not disturb the state-machine tests (they don't touch `requested_by`, so they must still
pass unchanged).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/deploy-chat && python -m pytest tests/test_purity_ratings_access.py tests/test_product_ratings_request.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Mutation-verify the never-downgrade-on-request guard**

In `request()`, change `if existing is not None:` to `if False:` (always insert). Run `tests/test_product_ratings_request.py`. `test_request_is_idempotent_and_never_downgrades` MUST fail (the second request would try to re-insert / walk back). Revert.

- [ ] **Step 6: Commit**

```bash
git add dashboard/purity_ratings_access.py dashboard/product_ratings.py \
        tests/test_purity_ratings_access.py tests/test_product_ratings_request.py
git commit -m "feat(purity): request gate + product_ratings.request()"
```

---

### Task 2: Request routes (portal client + console)

**Files:**
- Modify: `app.py` — add near the product-review routes (search `api_product_review_request`)
- Create: `tests/test_purity_routes.py`

**Interfaces:**
- Consumes: `purity_ratings_access.can_request`, `product_ratings.request`, `_portal_record_for`, `_portal_console_ok`, `membership_category(email)` (existing app helper — grep to confirm its name/signature; the product-review routes already read membership).
- Produces: `POST /api/portal/<token>/purity/request` (client, gated) and `POST /api/console/purity/request` (console).

- [ ] **Step 1: Write the failing test**

Create `tests/test_purity_routes.py`:

```python
"""Phase-2a purity routes: request (portal+console), screen, tier2, confirm.
Console writes are console-secret gated; client identity comes from the portal
token only."""
import sqlite3
import pytest
import app as app_mod
from dashboard import product_ratings as pr


@pytest.fixture
def client(monkeypatch, tmp_path):
    db = str(tmp_path / "c.db")
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    pr.init_tables(cx); cx.commit(); cx.close()
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    # portal token -> email; console ok on a secret header value only
    monkeypatch.setattr(app_mod, "_portal_record_for",
                        lambda cx, tok: {"email": "a@b.com"} if tok == "TOKA" else None)
    monkeypatch.setattr(app_mod, "_portal_console_ok", lambda: True)
    monkeypatch.setattr(app_mod, "membership_category", lambda email: "full", raising=False)
    app_mod.app.config["TESTING"] = True
    return app_mod.app.test_client()


def _get(db, key):
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    r = cx.execute("SELECT * FROM product_ratings WHERE product_key=?", (key,)).fetchone()
    cx.close(); return dict(r) if r else None


def test_console_request_creates_row(client):
    r = client.post("/api/console/purity/request",
                    json={"product_key": "jarrow-mag", "brand": "Jarrow", "product_name": "Mag Taurate"})
    assert r.status_code == 200
    assert _get(app_mod.LOG_DB, "jarrow-mag")["status"] == "requested"


def test_portal_request_uses_token_email_not_a_field(client):
    r = client.post("/api/portal/TOKA/purity/request",
                    json={"product_key": "k", "brand": "B", "product_name": "N",
                          "email": "attacker@evil.com"})
    assert r.status_code == 200
    row = _get(app_mod.LOG_DB, "k")
    assert row["status"] == "requested"
    # requested_by came from the token (a@b.com), never the body's email field
    assert "attacker@evil.com" not in (row.get("requested_by") or "")


def test_portal_request_unknown_token_401(client):
    r = client.post("/api/portal/NOPE/purity/request",
                    json={"product_key": "k", "brand": "B", "product_name": "N"})
    assert r.status_code in (401, 404)
    assert _get(app_mod.LOG_DB, "k") is None


def test_portal_request_not_entitled_403(client, monkeypatch):
    # A non-paid client with no explicit grant is refused, and no row is created.
    monkeypatch.setattr(app_mod, "membership_category", lambda email: "none", raising=False)
    r = client.post("/api/portal/TOKA/purity/request",
                    json={"product_key": "blocked", "brand": "B", "product_name": "N"})
    assert r.status_code == 403
    assert _get(app_mod.LOG_DB, "blocked") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && OPENAI_API_KEY=dummy PINECONE_API_KEY=dummy python -m pytest tests/test_purity_routes.py -v`
Expected: FAIL — routes 404.

- [ ] **Step 3: Add the routes to `app.py`**

Add near the product-review request routes:

```python
@app.route("/api/console/purity/request", methods=["POST"])
def api_console_purity_request():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import product_ratings as _pr
    b = request.get_json(silent=True) or {}
    key = (b.get("product_key") or "").strip()
    if not key:
        return jsonify({"error": "product_key_required"}), 400
    with _db_lock, db.connect(LOG_DB) as cx:
        _pr.init_tables(cx)
        res = _pr.request(cx, key, brand=b.get("brand") or "",
                          product_name=b.get("product_name") or "", requested_by="console")
    return jsonify({"ok": True, **res})


@app.route("/api/portal/<token>/purity/request", methods=["POST"])
def api_portal_purity_request(token):
    from dashboard import product_ratings as _pr, purity_ratings_access as _acc
    b = request.get_json(silent=True) or {}
    key = (b.get("product_key") or "").strip()
    if not key:
        return jsonify({"error": "product_key_required"}), 400
    with _db_lock, db.connect(LOG_DB) as cx:
        _pr.init_tables(cx); _acc.init_table(cx)
        portal = _portal_record_for(cx, token)
        if not portal:
            return jsonify({"error": "not_found"}), 404
        email = (portal.get("email") or "").strip().lower()
        if not _acc.can_request(cx, email, membership_category(email)):
            return jsonify({"error": "not_entitled"}), 403
        res = _pr.request(cx, key, brand=b.get("brand") or "",
                          product_name=b.get("product_name") or "", requested_by=email)
    return jsonify({"ok": True, **res})
```

The membership-tier helper is `membership_category(email)` at `app.py:13909` (module scope; returns `'full'`/`'trial'`/`'none'`/etc.) — confirmed present, so call it directly. The test monkeypatches `app_mod.membership_category`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/deploy-chat && OPENAI_API_KEY=dummy PINECONE_API_KEY=dummy python -m pytest tests/test_purity_routes.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Mutation-verify the token-identity and gate guards**

1. In `api_portal_purity_request`, change `requested_by=email` to `requested_by=b.get("email")` — `test_portal_request_uses_token_email_not_a_field` MUST fail. Revert.
2. Delete the `if not _acc.can_request(...)` block — `test_portal_request_not_entitled_403` MUST fail (a non-paid client would slip through). Revert.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_purity_routes.py
git commit -m "feat(purity): gated request routes (portal token + console)"
```

---

### Task 3: Operator screen route (manual acquisition → screen)

**Files:**
- Modify: `app.py`
- Modify: `tests/test_purity_routes.py`

**Interfaces:**
- Consumes: `purity_screen.screen_label`, `purity_avoidlist.load_avoidlist`, `product_ratings.record_screen`.
- Produces: `POST /api/console/purity/screen` — body `{product_key, brand, product_name, other_ingredients: [str]}` (or `other_ingredients_text` to be split on newlines/commas) → runs the screen → `record_screen`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_purity_routes.py`:

```python
def test_console_screen_runs_the_real_screen(client):
    client.post("/api/console/purity/request",
                json={"product_key": "k", "brand": "B", "product_name": "N"})
    r = client.post("/api/console/purity/screen",
                    json={"product_key": "k",
                          "other_ingredients": ["Hypromellose", "Magnesium Stearate"]})
    assert r.status_code == 200
    row = _get(app_mod.LOG_DB, "k")
    assert row["status"] == "screened" and row["color"] == "red"


def test_console_screen_none_data_is_unrated_not_green(client):
    client.post("/api/console/purity/request",
                json={"product_key": "k2", "brand": "B", "product_name": "N"})
    r = client.post("/api/console/purity/screen",
                    json={"product_key": "k2", "other_ingredients": None})
    assert r.status_code == 200
    assert _get(app_mod.LOG_DB, "k2")["status"] == "unrated"
    assert _get(app_mod.LOG_DB, "k2")["color"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd ~/deploy-chat && OPENAI_API_KEY=dummy PINECONE_API_KEY=dummy python -m pytest tests/test_purity_routes.py -k screen -v`
Expected: FAIL — route 404.

- [ ] **Step 3: Add the route**

```python
@app.route("/api/console/purity/screen", methods=["POST"])
def api_console_purity_screen():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import (product_ratings as _pr, purity_screen as _ps,
                            purity_avoidlist as _pa)
    b = request.get_json(silent=True) or {}
    key = (b.get("product_key") or "").strip()
    if not key:
        return jsonify({"error": "product_key_required"}), 400
    oi = b.get("other_ingredients")   # None -> unrated (never green); list -> screened
    if oi is not None and not isinstance(oi, list):
        return jsonify({"error": "other_ingredients_must_be_list_or_null"}), 400
    avoidlist = _pa.load_avoidlist()
    screen = _ps.screen_label(b.get("actives"), oi, avoidlist)
    raw = "" if oi is None else "\n".join(oi)
    with _db_lock, db.connect(LOG_DB) as cx:
        _pr.init_tables(cx)
        _pr.record_screen(cx, key, brand=b.get("brand") or "",
                          product_name=b.get("product_name") or "",
                          other_ingredients_raw=raw, other_ingredients_parsed=(oi or []),
                          screen=screen)
        row = _pr.get(cx, key)
    return jsonify({"ok": True, "status": row["status"], "color": row["color"]})
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd ~/deploy-chat && OPENAI_API_KEY=dummy PINECONE_API_KEY=dummy python -m pytest tests/test_purity_routes.py -k screen -v`
Expected: PASS, 2 tests.

- [ ] **Step 5: Mutation-verify the unrated guard survives the route**

In the route, change `screen = _ps.screen_label(...)` to hardcode `screen = {"color":"green","red_hits":[],"yellow_hits":[],"avoidlist_version":"x"}`. `test_console_screen_none_data_is_unrated_not_green` MUST fail. Revert. (Proves the route delegates to the real screen, not a shortcut.)

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_purity_routes.py
git commit -m "feat(purity): console screen route (operator ingredient entry)"
```

---

### Task 4: Tier-2 hand-off and confirm routes + console list

**Files:**
- Modify: `app.py`
- Modify: `tests/test_purity_routes.py`

**Interfaces:**
- Consumes: `product_ratings.set_tier2`, `product_ratings.confirm`, `product_ratings.get`.
- Produces: `POST /api/console/purity/tier2` (`{product_key, score, detail_json}` → `set_tier2`), `POST /api/console/purity/confirm` (`{product_key}` → `confirm`), `GET /api/console/purity-ratings` (list rows for the console).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_purity_routes.py`:

```python
def _screen(client, key, oi):
    client.post("/api/console/purity/request", json={"product_key": key, "brand": "B", "product_name": "N"})
    client.post("/api/console/purity/screen", json={"product_key": key, "other_ingredients": oi})


def test_green_tier2_then_confirm(client):
    _screen(client, "g", ["Hypromellose"])   # green
    r = client.post("/api/console/purity/tier2", json={"product_key": "g", "score": 8.5, "detail_json": "{}"})
    assert r.status_code == 200 and _get(app_mod.LOG_DB, "g")["status"] == "ai_draft"
    r = client.post("/api/console/purity/confirm", json={"product_key": "g"})
    assert r.status_code == 200 and _get(app_mod.LOG_DB, "g")["status"] == "confirmed"


def test_red_confirms_without_tier2(client):
    _screen(client, "r", ["Magnesium Stearate"])   # red
    r = client.post("/api/console/purity/confirm", json={"product_key": "r"})
    assert r.status_code == 200 and _get(app_mod.LOG_DB, "r")["status"] == "confirmed"


def test_tier2_on_a_red_is_rejected(client):
    _screen(client, "r2", ["Gelatin"])   # red
    r = client.post("/api/console/purity/tier2", json={"product_key": "r2", "score": 9, "detail_json": "{}"})
    assert r.status_code == 400   # engine raises; route returns 400, does not 500


def test_console_list_returns_rows(client):
    _screen(client, "g2", ["Hypromellose"])
    r = client.get("/api/console/purity-ratings")
    assert r.status_code == 200
    keys = [row["product_key"] for row in r.get_json()["ratings"]]
    assert "g2" in keys


def test_console_routes_require_secret(client, monkeypatch):
    monkeypatch.setattr(app_mod, "_portal_console_ok", lambda: False)
    for path, body in [("/api/console/purity/request", {"product_key": "x"}),
                       ("/api/console/purity/screen", {"product_key": "x"}),
                       ("/api/console/purity/tier2", {"product_key": "x"}),
                       ("/api/console/purity/confirm", {"product_key": "x"})]:
        assert client.post(path, json=body).status_code == 401
    assert client.get("/api/console/purity-ratings").status_code == 401
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd ~/deploy-chat && OPENAI_API_KEY=dummy PINECONE_API_KEY=dummy python -m pytest tests/test_purity_routes.py -k "tier2 or confirm or list or secret" -v`
Expected: FAIL — routes 404.

- [ ] **Step 3: Add the routes**

```python
@app.route("/api/console/purity/tier2", methods=["POST"])
def api_console_purity_tier2():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import product_ratings as _pr
    b = request.get_json(silent=True) or {}
    key = (b.get("product_key") or "").strip()
    if not key:
        return jsonify({"error": "product_key_required"}), 400
    with _db_lock, db.connect(LOG_DB) as cx:
        _pr.init_tables(cx)
        try:
            _pr.set_tier2(cx, key, b.get("score"), b.get("detail_json") or "{}")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        row = _pr.get(cx, key)
    return jsonify({"ok": True, "status": row["status"]})


@app.route("/api/console/purity/confirm", methods=["POST"])
def api_console_purity_confirm():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import product_ratings as _pr
    b = request.get_json(silent=True) or {}
    key = (b.get("product_key") or "").strip()
    if not key:
        return jsonify({"error": "product_key_required"}), 400
    with _db_lock, db.connect(LOG_DB) as cx:
        _pr.init_tables(cx)
        try:
            _pr.confirm(cx, key)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        row = _pr.get(cx, key)
    return jsonify({"ok": True, "status": row["status"]})


@app.route("/api/console/purity-ratings", methods=["GET"])
def api_console_purity_ratings_list():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import product_ratings as _pr
    with db.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _pr.init_tables(cx)
        rows = [dict(r) for r in cx.execute(
            "SELECT product_key, brand, product_name, color, status, avoidlist_version, "
            "updated_at FROM product_ratings ORDER BY updated_at DESC").fetchall()]
    return jsonify({"ok": True, "ratings": rows})
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd ~/deploy-chat && OPENAI_API_KEY=dummy PINECONE_API_KEY=dummy python -m pytest tests/test_purity_routes.py -v`
Expected: PASS (all route tests).

- [ ] **Step 5: Mutation-verify the reject path and the auth gate**

1. In `api_console_purity_tier2`, remove the `try/except ValueError` (let it raise). `test_tier2_on_a_red_is_rejected` will then 500 instead of 400 — confirm it fails, then revert (the route must translate the engine's guard to a 400, not a 500).
2. In one console route, remove the `_portal_console_ok()` check. `test_console_routes_require_secret` MUST fail. Revert.

- [ ] **Step 6: Run the whole purity suite (Phase 1 + 2a), both orders**

Run:
```bash
cd ~/deploy-chat && OPENAI_API_KEY=dummy PINECONE_API_KEY=dummy python -m pytest \
  tests/test_purity_avoidlist.py tests/test_purity_screen.py tests/test_product_ratings.py \
  tests/test_purity_ratings_access.py tests/test_product_ratings_request.py tests/test_purity_routes.py -v
```
Expected: all pass. Then run the six files in reverse order; totals must match. Do NOT run the bare full suite.

- [ ] **Step 7: Commit**

```bash
git add app.py tests/test_purity_routes.py
git commit -m "feat(purity): tier-2 hand-off, confirm, and console list routes"
```

---

## Deferred to follow-on plans

- **Phase 2b — Step-1 online acquisition.** Given a `requested` row, search for the product's Other Ingredients online, fetch the candidate source, extract with the `document_extract.verify_quotes` fabrication guard (accept only quoted text), and call the screen route's logic. Async/operator-triggered. Unverifiable → "not found" → cascade to 2c. Real unknowns: which search/fetch tooling runs from the app vs an operator script, and source reliability — hence its own plan.
- **Phase 2c — Step-2 client-photo fallback.** When Step 1 finds nothing, prompt the client in the portal to upload a facts-panel photo; store via `client_documents.put`; extract via `document_extract.call_model_for_extraction` (image path) with a supplement-specific prompt + the guard; feed the screen.
- **Phase 3 — the two readers.** Fullscript seed-gate (show only confirmed non-red, paired with `best_ff`) and the public aggregate `% fail` stat.

Nothing in 2a builds acquisition automation; it is the gated request + human-confirm spine, drivable end-to-end from the console with operator-entered ingredients.
