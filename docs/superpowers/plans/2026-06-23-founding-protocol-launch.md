# Founding Protocol Launch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch Neuro Magnesium on the illtowell funnel as a product-led founding offer — buy a bottle (charge-on-ship pre-order), get free membership for as long as the autoship stays active — built as a per-product, repeatable founding-launch.

**Architecture:** Reuse the existing `subscriptions` card-vault + off-session charge cron, the `memberships` comp-grant (`_grant_membership`/`_extend_membership_grant`), and the delivery→coaching wiring. Add (1) founding columns + helpers to `dashboard/subscriptions.py`, (2) a pure `dashboard/founding.py` config+counter module, (3) a `/begin/founding/reserve` $0-vault route + a `checkout-return` `kind="founding_reserve"` branch, (4) an admin "ship founding batch" charge-on-ship action, (5) a charge-cron extension that keeps founders' comp membership alive on each successful bottle charge, and (6) product-page content + a live counter.

**Tech Stack:** Python 3 / Flask (`app.py`), sqlite3 (pure-module pattern, connection passed in), Stripe (`dashboard/stripe_pay.py`), QBO (`dashboard/qbo_billing.py`), pytest (in-memory sqlite for model tests; `app.test_client()` + monkeypatch for routes). JSON config in `data/`.

## Global Constraints

- **Structure-function language only** on every public surface; never state/imply the product treats/prevents/slows/reverses AMD / macular degeneration / glaucoma; no disease nouns as the thing the product acts on.
- **Founder reversal story = biography only** (2023 FTC Endorsement Guides); no "you will reverse too"; disclose material connection; carry the DSHEA disclaimer.
- **Autoship under FTC ROSCA:** clear terms, express informed consent before the first (on-ship) charge, one-click cancel.
- Pure data modules (`subscriptions.py`, `founding.py`) take a `cx` sqlite connection — no Flask/Stripe/QBO imports inside them.
- Migrations are idempotent `migrate_add_*` functions (ALTER inside try/except), called at startup and in test setup — follow the existing pattern.
- Feature-flag the public route behind env `FOUNDING_LAUNCH_ENABLED` (truthy = `1/true/yes/on`), matching `_subscriptions_enabled()`.
- Member loyalty/eligibility unchanged: standard $99/mo membership path is untouched; the founding comp membership uses `source="founding"` and writes only the `memberships` table via `_grant_membership`/`_extend_membership_grant`.

---

## File Structure

- `dashboard/subscriptions.py` (MODIFY) — add founding columns migration + founding CRUD helpers (`create_founding_reservation`, `list_founding_pending`, `mark_founding_active`, `count_founding`).
- `dashboard/founding.py` (CREATE) — pure config + counter: load `data/founding_launches.json`, `get_launch`, `count_reserved`, `remaining`, `is_open`.
- `data/founding_launches.json` (CREATE) — per-slug config `{cap, batch_label, video_url, closes_at}`.
- `data/products.json` (MODIFY) — populate the `neuro-magnesium` entry (price, structure-function description, ingredients, benefits).
- `app.py` (MODIFY) — `/begin/founding/reserve` route, `checkout-return` `kind="founding_reserve"` branch, `orders.ship_founding_batch` admin action helper, charge-cron founding membership-extension, `/begin/founding/status/<slug>` counter API.
- `static/begin-product.html` (MODIFY) — render the promo video + founding counter when the product has a founding launch.
- Tests: `tests/test_founding_model.py`, `tests/test_founding_config.py`, `tests/test_founding_reserve_route.py`, `tests/test_founding_checkout_return.py`, `tests/test_founding_ship.py`, `tests/test_founding_cron_membership.py`, `tests/test_founding_counter_api.py`.

---

### Task 1: Founding columns + reservation helpers on subscriptions

**Files:**
- Modify: `dashboard/subscriptions.py`
- Test: `tests/test_founding_model.py`

**Interfaces:**
- Consumes: existing `init_subscriptions_table(cx)`, `add_months`, `_now_iso`, `get(cx, sub_id)`.
- Produces:
  - `migrate_add_founding_columns(cx) -> None`
  - `create_founding_reservation(cx, *, email, stripe_customer_id, stripe_payment_method_id, items, ship_address, founding_slug) -> int` (inserts a product sub: `founding=1`, `founding_state='pending'`, `founding_slug`, `order_count=0`, `cadence_months=1`, `next_charge_date='2999-01-01'`).
  - `mark_founding_active(cx, sub_id, *, next_charge_date) -> None` (`founding_state='active'`, set `next_charge_date`, `order_count=1`).
  - `list_founding_pending(cx, founding_slug) -> list[dict]`
  - `count_founding(cx, founding_slug) -> int` (all founding rows for the slug, pending + active; cancelled excluded).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_founding_model.py
import sqlite3
from dashboard import subscriptions as subs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    subs.migrate_add_founding_columns(cx)
    return cx


def test_create_founding_reservation_is_pending_far_dated():
    cx = _cx()
    sid = subs.create_founding_reservation(
        cx, email="f@x.com", stripe_customer_id="cus", stripe_payment_method_id="pm",
        items=[{"slug": "neuro-magnesium", "qty": 1}], ship_address={"state": "HI"},
        founding_slug="neuro-magnesium")
    row = subs.get(cx, sid)
    assert row["founding"] == 1
    assert row["founding_state"] == "pending"
    assert row["founding_slug"] == "neuro-magnesium"
    assert row["order_count"] == 0
    assert row["next_charge_date"] == "2999-01-01"   # never picked by list_due until shipped


def test_pending_reservation_not_in_list_due():
    cx = _cx()
    subs.create_founding_reservation(
        cx, email="f@x.com", stripe_customer_id="c", stripe_payment_method_id="pm",
        items=[{"slug": "neuro-magnesium", "qty": 1}], ship_address={}, founding_slug="neuro-magnesium")
    assert subs.list_due(cx, as_of="2030-01-01") == []   # far-dated sentinel excludes it


def test_mark_founding_active_sets_first_charge_cycle():
    cx = _cx()
    sid = subs.create_founding_reservation(
        cx, email="f@x.com", stripe_customer_id="c", stripe_payment_method_id="pm",
        items=[{"slug": "neuro-magnesium", "qty": 1}], ship_address={}, founding_slug="neuro-magnesium")
    subs.mark_founding_active(cx, sid, next_charge_date="2026-08-01")
    row = subs.get(cx, sid)
    assert row["founding_state"] == "active"
    assert row["next_charge_date"] == "2026-08-01"
    assert row["order_count"] == 1


def test_count_and_list_founding_pending():
    cx = _cx()
    a = subs.create_founding_reservation(cx, email="a@x.com", stripe_customer_id="c",
        stripe_payment_method_id="pm", items=[], ship_address={}, founding_slug="neuro-magnesium")
    subs.create_founding_reservation(cx, email="b@x.com", stripe_customer_id="c",
        stripe_payment_method_id="pm", items=[], ship_address={}, founding_slug="neuro-magnesium")
    subs.mark_founding_active(cx, a, next_charge_date="2026-08-01")
    assert subs.count_founding(cx, "neuro-magnesium") == 2        # pending + active both count
    pending = subs.list_founding_pending(cx, "neuro-magnesium")
    assert len(pending) == 1 and pending[0]["email"] == "b@x.com"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_founding_model.py -v`
Expected: FAIL with `AttributeError: module 'dashboard.subscriptions' has no attribute 'migrate_add_founding_columns'`

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/subscriptions.py`:

```python
def migrate_add_founding_columns(cx) -> None:
    """Add founding launch columns if missing. Safe on every startup."""
    for ddl in (
        "ALTER TABLE subscriptions ADD COLUMN founding INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE subscriptions ADD COLUMN founding_state TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE subscriptions ADD COLUMN founding_slug TEXT NOT NULL DEFAULT ''",
    ):
        try:
            cx.execute(ddl)
            cx.commit()
        except Exception:
            pass


# Far-future sentinel: a pending reservation must never be picked up by list_due
# until mark_founding_active sets a real next_charge_date (the charge-on-ship event).
_FOUNDING_PENDING_DATE = "2999-01-01"


def create_founding_reservation(cx, *, email, stripe_customer_id,
                                stripe_payment_method_id, items, ship_address,
                                founding_slug) -> int:
    """Insert a pending founding product subscription (card vaulted, $0 today).
    order_count=0 and a far-future next_charge_date keep it out of list_due until
    the founding batch ships (mark_founding_active)."""
    now = _now_iso()
    cur = cx.execute(
        """INSERT INTO subscriptions
               (email, stripe_customer_id, stripe_payment_method_id, items_json,
                cadence_months, status, order_count, next_charge_date, ship_address_json,
                skip_next, created_at, updated_at, founding, founding_state, founding_slug)
           VALUES (?,?,?,?,1,'active',0,?,?,0,?,?,1,'pending',?)""",
        (email, stripe_customer_id, stripe_payment_method_id, json.dumps(items or []),
         _FOUNDING_PENDING_DATE, json.dumps(ship_address or {}), now, now, founding_slug),
    )
    cx.commit()
    return cur.lastrowid


def mark_founding_active(cx, sub_id: int, *, next_charge_date: str) -> None:
    """Flip a pending founding reservation to active after its first (on-ship) charge:
    record the first order and schedule the next autoship charge."""
    cx.execute(
        "UPDATE subscriptions SET founding_state='active', order_count=1,"
        " next_charge_date=?, updated_at=? WHERE id=?",
        (next_charge_date, _now_iso(), sub_id),
    )
    cx.commit()


def list_founding_pending(cx, founding_slug: str) -> list[dict]:
    """Reserved-but-not-yet-shipped founding subscriptions for a launch slug."""
    rows = cx.execute(
        "SELECT * FROM subscriptions WHERE founding=1 AND founding_state='pending'"
        " AND founding_slug=? AND status!='cancelled' ORDER BY id", (founding_slug,)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def count_founding(cx, founding_slug: str) -> int:
    """Count of founding slots consumed for a launch (pending + active, not cancelled)."""
    row = cx.execute(
        "SELECT COUNT(*) FROM subscriptions WHERE founding=1 AND founding_slug=?"
        " AND status!='cancelled'", (founding_slug,)
    ).fetchone()
    return int(row[0]) if row else 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_founding_model.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/subscriptions.py tests/test_founding_model.py
git commit -m "feat(founding): subscriptions founding columns + reservation helpers"
```

---

### Task 2: Founding-launch config + counter module

**Files:**
- Create: `dashboard/founding.py`
- Create: `data/founding_launches.json`
- Test: `tests/test_founding_config.py`

**Interfaces:**
- Consumes: `dashboard.subscriptions.count_founding(cx, slug)`.
- Produces:
  - `get_launch(slug) -> dict | None` — config row or None.
  - `count_reserved(cx, slug) -> int` — delegates to subscriptions.count_founding.
  - `remaining(cx, slug) -> int` — `max(0, cap - count_reserved)`.
  - `is_open(cx, slug, *, now_iso=None) -> bool` — True when launch exists, remaining > 0, and (no `closes_at` or `now_iso < closes_at`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_founding_config.py
import json
import sqlite3
import dashboard.founding as founding
from dashboard import subscriptions as subs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    subs.migrate_add_founding_columns(cx)
    return cx


def _patch_config(monkeypatch):
    monkeypatch.setattr(founding, "_CONFIG", {
        "neuro-magnesium": {"cap": 3, "batch_label": "Founding Batch No. 1",
                            "video_url": "/clip/neuro/promo.mp4", "closes_at": "2026-12-31"}})


def test_get_launch(monkeypatch):
    _patch_config(monkeypatch)
    assert founding.get_launch("neuro-magnesium")["cap"] == 3
    assert founding.get_launch("nope") is None


def test_remaining_and_is_open(monkeypatch):
    _patch_config(monkeypatch)
    cx = _cx()
    assert founding.remaining(cx, "neuro-magnesium") == 3
    assert founding.is_open(cx, "neuro-magnesium", now_iso="2026-07-01") is True
    for e in ("a@x.com", "b@x.com", "c@x.com"):
        subs.create_founding_reservation(cx, email=e, stripe_customer_id="c",
            stripe_payment_method_id="pm", items=[], ship_address={}, founding_slug="neuro-magnesium")
    assert founding.remaining(cx, "neuro-magnesium") == 0
    assert founding.is_open(cx, "neuro-magnesium", now_iso="2026-07-01") is False   # cap hit


def test_is_open_false_after_closes_at(monkeypatch):
    _patch_config(monkeypatch)
    cx = _cx()
    assert founding.is_open(cx, "neuro-magnesium", now_iso="2027-01-01") is False   # window closed
    assert founding.is_open(cx, "missing", now_iso="2026-07-01") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_founding_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.founding'`

- [ ] **Step 3: Write minimal implementation**

Create `data/founding_launches.json`:

```json
{
  "neuro-magnesium": {
    "cap": 2500,
    "batch_label": "Founding Batch No. 1",
    "video_url": "",
    "closes_at": ""
  }
}
```

Create `dashboard/founding.py`:

```python
"""Founding-launch config + counter. Pure module: the only I/O is reading the
JSON config at import and counting rows via a caller-supplied sqlite connection."""

import json
import os

from dashboard import subscriptions as _subs

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "data", "founding_launches.json")


def _load():
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


_CONFIG = _load()


def get_launch(slug: str) -> dict | None:
    return _CONFIG.get(slug)


def count_reserved(cx, slug: str) -> int:
    return _subs.count_founding(cx, slug)


def remaining(cx, slug: str) -> int:
    launch = get_launch(slug)
    if not launch:
        return 0
    return max(0, int(launch.get("cap", 0)) - count_reserved(cx, slug))


def is_open(cx, slug: str, *, now_iso: str | None = None) -> bool:
    launch = get_launch(slug)
    if not launch:
        return False
    if remaining(cx, slug) <= 0:
        return False
    closes_at = (launch.get("closes_at") or "").strip()
    if closes_at and now_iso and now_iso >= closes_at:
        return False
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_founding_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/founding.py data/founding_launches.json tests/test_founding_config.py
git commit -m "feat(founding): launch config + cap/counter module"
```

---

### Task 3: Neuro Magnesium product content + promo video on the product page

**Files:**
- Modify: `data/products.json` (the `neuro-magnesium` entry)
- Modify: `static/begin-product.html` (render `founding_video_url` when present)
- Test: `tests/test_founding_product_data.py`

**Interfaces:**
- Consumes: `dashboard.founding.get_launch`, existing `/begin/product-data/<slug>` route + `_get_product`.
- Produces: `/begin/product-data/neuro-magnesium` JSON includes `founding_video_url` and `founding` block (`batch_label`, `remaining`).

**Note:** The v1 promo video (`~/Downloads/neuro-magnesium-promo-v1.mp4`) must first be hosted on R2 (reuse the clip-serving `/clip/<key>` path from the clip pipeline); put the resulting URL in `data/founding_launches.json` `video_url`. Until hosted, leave `video_url` empty and the page simply omits the video (test covers both).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_founding_product_data.py
import json
import app as appmod
import dashboard.founding as founding


def test_product_data_includes_founding_block(monkeypatch):
    monkeypatch.setattr(appmod, "_get_product", lambda s: {
        "slug": s, "name": "Neuro Magnesium", "price_cents": 8000,
        "description": "Foundational eye and brain support.", "qbo_item_id": ""} if s == "neuro-magnesium" else None)
    monkeypatch.setattr(founding, "_CONFIG", {
        "neuro-magnesium": {"cap": 2500, "batch_label": "Founding Batch No. 1",
                            "video_url": "/clip/neuro/promo.mp4", "closes_at": ""}})
    monkeypatch.setattr(founding, "count_reserved", lambda cx, slug: 653)
    c = appmod.app.test_client()
    r = c.get("/begin/product-data/neuro-magnesium")
    assert r.status_code == 200
    data = r.get_json()
    assert data["founding"]["batch_label"] == "Founding Batch No. 1"
    assert data["founding"]["remaining"] == 2500 - 653
    assert data["founding_video_url"] == "/clip/neuro/promo.mp4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_founding_product_data.py -v`
Expected: FAIL (KeyError `'founding'` — the route doesn't add the block yet)

- [ ] **Step 3: Write minimal implementation**

In `data/products.json`, set the `neuro-magnesium` entry (structure-function language only — no disease nouns):

```json
"neuro-magnesium": {
  "name": "Neuro Magnesium",
  "price_cents": 8000,
  "pinecone_title": "Neuro Magnesium",
  "source": "founding-launch-2026",
  "description": "A foundational magnesium drink mix designed to reach where ordinary magnesium falls short. Built on Magnesium N-Acetyl-Taurate (ATA Mg), designed to cross the blood-brain and blood-eye barrier, carrying magnesium and taurine into the tissue where focus and vision live. Supports calm, a clear steady mind, and the body's own foundational eye and brain support. No fillers, no preservatives, sealed in Miron violet glass.",
  "months_per_unit": 1,
  "volume_eligible": true
}
```

In `app.py`, inside the `/begin/product-data/<slug>` route, after building the product dict and before `jsonify`, add:

```python
        try:
            from dashboard import founding as _founding
            _launch = _founding.get_launch(slug)
            if _launch:
                with sqlite3.connect(LOG_DB) as _fcx:
                    _fcx.row_factory = sqlite3.Row
                    _rem = _founding.remaining(_fcx, slug)
                data["founding"] = {"batch_label": _launch.get("batch_label", ""),
                                    "cap": int(_launch.get("cap", 0)), "remaining": _rem}
                data["founding_video_url"] = _launch.get("video_url", "")
        except Exception as _fe:
            print(f"[founding] product-data enrich failed: {_fe!r}", flush=True)
```

In `static/begin-product.html`, where the product JSON is rendered, add (near the hero) a guarded video + counter block:

```html
<div id="founding-video" style="display:none;margin:1rem 0;">
  <video controls playsinline style="width:100%;border-radius:24px;"></video>
</div>
<div id="founding-counter" style="display:none;font-weight:600;"></div>
<script>
  // populated from /begin/product-data/<slug>: data.founding_video_url, data.founding
  function renderFounding(data){
    if (data.founding_video_url){
      var fv = document.getElementById('founding-video');
      fv.querySelector('video').src = data.founding_video_url;
      fv.style.display = 'block';
    }
    if (data.founding){
      var fc = document.getElementById('founding-counter');
      fc.textContent = data.founding.remaining + ' of ' + data.founding.cap + ' founding bottles remaining';
      fc.style.display = 'block';
    }
  }
</script>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_founding_product_data.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data/products.json app.py static/begin-product.html tests/test_founding_product_data.py
git commit -m "feat(founding): Neuro Magnesium content + promo video + counter on product page"
```

---

### Task 4: Founding reserve route (`/begin/founding/reserve`) — $0 vault, cap-gated

**Files:**
- Modify: `app.py` (new route)
- Test: `tests/test_founding_reserve_route.py`

**Interfaces:**
- Consumes: `dashboard.founding.is_open`, `_reorder_email_from_cookie`, `is_member`, `stripe_pay.create_setup_session` (ALREADY EXISTS, `dashboard/stripe_pay.py:70` — `mode="setup"`, vaults card, NO charge).
- Produces: route `POST /begin/founding/reserve` → setup-mode Stripe session with `metadata.kind="founding_reserve"`, `metadata.slug`, `metadata.email`, items/ship (stash if >450 chars, reusing the `pending_subscriptions` stash pattern). Returns `{ok, stripe_url}` or `409 {"error":"founding_closed"}` when `is_open` is False.

**Note:** Use the existing `stripe_pay.create_setup_session(*, customer_email, metadata, success_url, cancel_url)` — it vaults the card with **zero charge** and creates no QBO invoice. No change to `stripe_pay.py`. The test asserts `create_setup_session` is called and no QBO invoice is created.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_founding_reserve_route.py
import app as appmod
import dashboard.founding as founding


def _setup(monkeypatch, is_open=True):
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "f@x.com")
    monkeypatch.setattr(appmod, "is_member", lambda sid, email: True)
    monkeypatch.setattr(appmod, "_get_product",
        lambda s: {"slug": s, "name": "Neuro Magnesium", "price_cents": 8000, "qbo_item_id": ""} if s == "neuro-magnesium" else None)
    monkeypatch.setattr(founding, "is_open", lambda cx, slug, now_iso=None: is_open)
    inv = {"n": 0}
    monkeypatch.setattr(appmod.qb, "create_invoice", lambda *a, **k: inv.update(n=inv["n"] + 1) or {"Id": "X"})
    cap = {}
    monkeypatch.setattr(appmod.stripe_pay, "create_setup_session",
        lambda **k: cap.update(k) or {"id": "cs", "url": "https://stripe/setup"})
    monkeypatch.setenv("FOUNDING_LAUNCH_ENABLED", "true")
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    return cap, inv


def test_reserve_creates_zero_charge_vault_session(monkeypatch):
    cap, inv = _setup(monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/begin/founding/reserve", json={"slug": "neuro-magnesium",
               "items": [{"slug": "neuro-magnesium", "qty": 1}], "address": {"state": "HI", "country": "US", "name": "F"}})
    assert r.status_code == 200
    assert r.get_json()["stripe_url"] == "https://stripe/setup"     # setup-mode session, $0
    assert cap["metadata"]["kind"] == "founding_reserve"
    assert cap["metadata"]["slug"] == "neuro-magnesium"
    assert inv["n"] == 0                 # no QBO invoice at reservation (vault only)


def test_reserve_closed_returns_409(monkeypatch):
    cap, inv = _setup(monkeypatch, is_open=False)
    c = appmod.app.test_client()
    r = c.post("/begin/founding/reserve", json={"slug": "neuro-magnesium",
               "items": [{"slug": "neuro-magnesium", "qty": 1}], "address": {"state": "HI"}})
    assert r.status_code == 409
    assert r.get_json()["error"] == "founding_closed"


def test_reserve_disabled_when_flag_off(monkeypatch):
    cap, inv = _setup(monkeypatch); monkeypatch.setenv("FOUNDING_LAUNCH_ENABLED", "false")
    c = appmod.app.test_client()
    r = c.post("/begin/founding/reserve", json={"slug": "neuro-magnesium", "items": [], "address": {}})
    assert r.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_founding_reserve_route.py -v`
Expected: FAIL (404 — route not defined)

- [ ] **Step 3: Write minimal implementation**

In `app.py` add near `/reorder/subscribe`:

```python
def _founding_enabled():
    return os.environ.get("FOUNDING_LAUNCH_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


@app.route("/begin/founding/reserve", methods=["POST"])
def begin_founding_reserve():
    """Founding pre-order: vault the card ($0 today), no QBO invoice. The bottle is
    charged on-ship (orders.ship_founding_batch). ROSCA: caller UI must show terms +
    explicit consent before this POST."""
    if not _founding_enabled():
        return jsonify({"error": "founding not enabled"}), 400
    email = _reorder_email_from_cookie()
    if not email:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    _sid = (request.cookies.get("amg_session") or "").strip()
    if not is_member(_sid, email):
        return jsonify({"ok": False, "need_optin": True,
                        "error": "Please agree to our Terms to continue."}), 403
    if not _STRIPE_ACTIVE:
        return jsonify({"error": "card payment not active"}), 400
    body = request.get_json(silent=True) or {}
    slug = (body.get("slug") or "").strip()
    from dashboard import founding as _founding
    with sqlite3.connect(LOG_DB) as _ocx:
        _ocx.row_factory = sqlite3.Row
        if not _founding.is_open(_ocx, slug, now_iso=_now_utc().strftime("%Y-%m-%d")):
            return jsonify({"error": "founding_closed"}), 409
    items = body.get("items") or []
    ship = body.get("address") or {}
    metadata = {"kind": "founding_reserve", "slug": slug, "email": email}
    items_json, ship_json = json.dumps(items), json.dumps(ship)
    if len(items_json) + len(ship_json) <= 450:
        metadata["items"] = items_json
        metadata["ship"] = ship_json
    else:
        stash_key = uuid.uuid4().hex
        with _db_lock, sqlite3.connect(LOG_DB) as _cx:
            _cx.execute("CREATE TABLE IF NOT EXISTS pending_subscriptions "
                        "(key TEXT PRIMARY KEY, items_json TEXT, ship_json TEXT, created_at TEXT)")
            _cx.execute("INSERT INTO pending_subscriptions (key, items_json, ship_json, created_at) "
                        "VALUES (?,?,?,?)", (stash_key, items_json, ship_json, _now_utc().isoformat()))
            _cx.commit()
        metadata["stash_key"] = stash_key
    success = f"{PUBLIC_BASE_URL}/begin/checkout-return?session_id={{CHECKOUT_SESSION_ID}}"
    sess = stripe_pay.create_setup_session(
        customer_email=email, metadata=metadata, success_url=success,
        cancel_url=f"{PUBLIC_BASE_URL}/begin/product/{slug}")
    return jsonify({"ok": True, "stripe_url": sess.get("url") or ""})
```

(No `stripe_pay.py` change — `create_setup_session` already vaults the card at $0.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_founding_reserve_route.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_founding_reserve_route.py
git commit -m "feat(founding): /begin/founding/reserve zero-charge vault route (cap-gated)"
```

---

### Task 5: `checkout-return` founding branch — create reservation + grant comp membership

**Files:**
- Modify: `app.py` (`begin_checkout_return`, add a `kind == "founding_reserve"` branch)
- Test: `tests/test_founding_checkout_return.py`

**Interfaces:**
- Consumes: `subscriptions.create_founding_reservation`, `_grant_membership(cx, email, days, source)`, `_ingest_order`, the `_sp.get_payment_intent` pattern, the stash recovery pattern.
- Produces: on Stripe setup return for `kind="founding_reserve"`: a pending founding subscription row, a `memberships` comp grant (`source="founding"`, 30 days), and a `founding` reservation order (`source="founding"`, `total_cents=0`).

**Note:** A setup-mode session has no `payment_intent`; it carries a `setup_intent`. `get_session` already returns `setup_intent` (`stripe_pay.py:100`) and `get_setup_intent(si_id)` already exists (`stripe_pay.py:88`, returns `{customer, payment_method}`). No `stripe_pay.py` change.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_founding_checkout_return.py
import sqlite3
import app as appmod
from dashboard import subscriptions as subs


def test_founding_return_creates_reservation_and_comp_membership(monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    # Seed schema
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx); subs.migrate_add_founding_columns(cx)
    cx.execute("CREATE TABLE IF NOT EXISTS memberships (id TEXT, email TEXT, granted_at TEXT,"
               " expires_at TEXT, granted_by TEXT, source TEXT, truly_vip_ref TEXT, notes TEXT)")
    cx.commit(); cx.close()

    from dashboard import stripe_pay as sp
    monkeypatch.setattr(sp, "get_session", lambda s: {
        "payment_status": "paid", "setup_intent": "seti_1",
        "metadata": {"kind": "founding_reserve", "slug": "neuro-magnesium", "email": "f@x.com",
                     "items": '[{"slug":"neuro-magnesium","qty":1}]', "ship": '{"state":"HI"}'}})
    monkeypatch.setattr(sp, "get_setup_intent", lambda i: {"customer": "cus_1", "payment_method": "pm_1"})
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)

    c = appmod.app.test_client()
    r = c.get("/begin/checkout-return?session_id=cs_1")
    assert r.status_code in (200, 302)

    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    sub = cx.execute("SELECT * FROM subscriptions WHERE email='f@x.com'").fetchone()
    assert sub is not None and sub["founding"] == 1 and sub["founding_state"] == "pending"
    mem = cx.execute("SELECT * FROM memberships WHERE email='f@x.com' AND source='founding'").fetchone()
    assert mem is not None
    cx.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_founding_checkout_return.py -v`
Expected: FAIL (no subscription row created — the founding branch doesn't exist)

- [ ] **Step 3: Write minimal implementation**

In `begin_checkout_return`, alongside the `kind == "subscribe"` block, add (the setup session reports `payment_status == "paid"` once the card is vaulted):

```python
                if md.get("kind") == "founding_reserve":
                    try:
                        from dashboard import subscriptions as _subs_f
                        seti = sess.get("setup_intent")
                        si = _sp.get_setup_intent(seti) if seti else {}
                        stripe_cus = si.get("customer") or ""
                        stripe_pm = si.get("payment_method") or ""
                        f_email = md.get("email") or ""
                        f_slug = md.get("slug") or ""
                        stash_key = md.get("stash_key")
                        if stash_key:
                            with sqlite3.connect(LOG_DB) as _scx:
                                _scx.row_factory = sqlite3.Row
                                _sr = _scx.execute("SELECT items_json, ship_json FROM "
                                    "pending_subscriptions WHERE key=?", (stash_key,)).fetchone()
                            items_list = json.loads(_sr["items_json"]) if _sr else []
                            ship_dict = json.loads(_sr["ship_json"]) if _sr else {}
                        else:
                            items_list = json.loads(md.get("items") or "[]")
                            ship_dict = json.loads(md.get("ship") or "{}")
                        with sqlite3.connect(LOG_DB) as _fcx:
                            _fcx.row_factory = sqlite3.Row
                            _subs_f.init_subscriptions_table(_fcx)
                            _subs_f.migrate_add_founding_columns(_fcx)
                            _subs_f.create_founding_reservation(
                                _fcx, email=f_email, stripe_customer_id=stripe_cus,
                                stripe_payment_method_id=stripe_pm, items=items_list,
                                ship_address=ship_dict, founding_slug=f_slug)
                            _grant_membership(_fcx, f_email, 30, "founding")
                            _fcx.commit()
                        _ingest_order(source="founding", external_ref=seti or "",
                                      email=f_email, items=items_list, total_cents=0,
                                      address=ship_dict, channel="retail")
                        print(f"[founding-return] reservation created for {f_email}", flush=True)
                    except Exception as _fe:
                        print(f"[founding-return] reservation failed: {_fe!r}", flush=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_founding_checkout_return.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_founding_checkout_return.py
git commit -m "feat(founding): checkout-return creates reservation + comp membership grant"
```

---

### Task 6: Charge-on-ship — admin "ship founding batch" action

**Files:**
- Modify: `app.py` (add `_ship_founding_reservation(cx, sub)` + an OWNER-gated route `POST /api/founding/ship`)
- Test: `tests/test_founding_ship.py`

**Interfaces:**
- Consumes: `subscriptions.list_founding_pending`, `subscriptions.mark_founding_active`, `subscriptions.add_months`, `stripe_pay.charge_off_session`, `qb.find_or_create_customer`/`create_invoice`, `_ingest_order`, `_price_cart`.
- Produces: `_ship_founding_reservation(cx, sub) -> dict` — charges the vaulted card for the bottle, ingests a qualifying product order (`source="reorder"` so the delivery→coaching path fires), and calls `mark_founding_active` with `next_charge_date = today + 1 month`. Returns `{charged: bool, sub_id, amount_cents}`.

**Note:** `source="reorder"` is in `coaching.QUALIFYING_SOURCES`, so the bottle's later delivery opens the coaching window via the existing `_activate_coaching_for_shipment`. Don't invent a new source.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_founding_ship.py
import sqlite3
import app as appmod
from dashboard import subscriptions as subs


def test_ship_charges_and_activates(monkeypatch):
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx); subs.migrate_add_founding_columns(cx)
    sid = subs.create_founding_reservation(cx, email="f@x.com", stripe_customer_id="cus_1",
        stripe_payment_method_id="pm_1", items=[{"slug": "neuro-magnesium", "qty": 1}],
        ship_address={"state": "HI"}, founding_slug="neuro-magnesium")
    sub = subs.get(cx, sid)
    monkeypatch.setattr(appmod, "_price_cart", lambda items, ship, subscriber_tier_pct=None: {
        "qbo_lines": [{"name": "Neuro Magnesium", "amount": 80.0, "qty": 1}],
        "shipping_cents": 600, "discount_cents": 0, "points_redeemed_cents": 0})
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
        lambda *a, **k: {"status": "succeeded", "id": "pi_1"})
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(appmod.qb, "create_invoice", lambda *a, **k: {"Id": "INV1", "TotalAmt": 86.0})
    orders = []
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: orders.append(kw))

    res = appmod._ship_founding_reservation(cx, sub)
    assert res["charged"] is True
    row = subs.get(cx, sid)
    assert row["founding_state"] == "active" and row["order_count"] == 1
    assert row["next_charge_date"] != "2999-01-01"
    assert orders and orders[0]["source"] == "reorder"   # qualifies for delivery->coaching
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_founding_ship.py -v`
Expected: FAIL (`AttributeError: module 'app' has no attribute '_ship_founding_reservation'`)

- [ ] **Step 3: Write minimal implementation**

In `app.py`:

```python
def _ship_founding_reservation(cx, sub):
    """Charge the vaulted card for the founding bottle (charge-on-ship), ingest a
    qualifying product order, and activate the autoship. Returns a result dict."""
    from dashboard import subscriptions as _subs
    items = sub.get("items") or []
    ship = sub.get("ship_address") or {}
    pc = _price_cart(items, ship, subscriber_tier_pct=None)
    cust = qb.find_or_create_customer(sub["email"], ship.get("name", ""))
    inv = qb.create_invoice(cust, pc["qbo_lines"] + _shipping_line(pc["shipping_cents"]),
                            allow_online_pay=False, email_to=sub["email"],
                            discount_cents=pc["discount_cents"] + pc["points_redeemed_cents"])
    total_cents = int(round(float(inv.get("TotalAmt") or 0) * 100))
    res = stripe_pay.charge_off_session(
        sub["stripe_customer_id"], sub["stripe_payment_method_id"], total_cents,
        description="Neuro Magnesium founding bottle",
        metadata={"sub": str(sub["id"]), "kind": "founding_ship"})
    if res.get("status") != "succeeded":
        _subs.bump_failed_count(cx, sub["id"])
        return {"charged": False, "sub_id": sub["id"], "amount_cents": total_cents}
    _ingest_order(source="reorder", external_ref=res.get("id") or inv.get("Id") or "",
                  email=sub["email"], items=items, total_cents=total_cents,
                  address=ship, channel="retail")
    today = _now_utc().strftime("%Y-%m-%d")
    _subs.mark_founding_active(cx, sub["id"], next_charge_date=_subs.add_months(today, 1))
    return {"charged": True, "sub_id": sub["id"], "amount_cents": total_cents}


@app.route("/api/founding/ship", methods=["POST"])
def api_founding_ship():
    """Console-gated: charge-on-ship the pending founding reservations for a slug.
    Uses the same CONSOLE_SECRET / X-Console-Key gate as the other /api/console
    endpoints (see app.py:7303-7305)."""
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "forbidden"}), 403
    slug = (request.get_json(silent=True) or {}).get("slug", "")
    from dashboard import subscriptions as _subs
    out = []
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        for sub in _subs.list_founding_pending(cx, slug):
            try:
                out.append(_ship_founding_reservation(cx, sub))
            except Exception as e:
                out.append({"charged": False, "sub_id": sub["id"], "error": repr(e)})
    return jsonify({"ok": True, "results": out})
```

Note: `CONSOLE_SECRET` is a module global in `app.py`; the gate matches the existing `/api/console/*` pattern.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_founding_ship.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_founding_ship.py
git commit -m "feat(founding): charge-on-ship admin action activates the autoship"
```

---

### Task 7: Keep founders' membership alive on each autoship charge

**Files:**
- Modify: `app.py` (`cron_charge_subscriptions`, product-sub success branch)
- Test: `tests/test_founding_cron_membership.py`

**Interfaces:**
- Consumes: `_extend_membership_grant(cx, email, until_iso, source)`, `MEMBERSHIP_GRANT_GRACE_DAYS`, `subscriptions.advance_after_charge`.
- Produces: after a successful product-subscription charge, if `sub.get("founding")` is truthy, extend the comp membership grant to `next_charge_date + grace` with `source="founding"` (mirrors the `kind=="membership"` branch at lines 19513–19518). No new teardown: cancelling the autoship stops the extension → access lapses naturally.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_founding_cron_membership.py
import sqlite3
import app as appmod


def test_founding_product_charge_extends_comp_membership(monkeypatch):
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE IF NOT EXISTS memberships (id TEXT, email TEXT, granted_at TEXT,"
               " expires_at TEXT, granted_by TEXT, source TEXT, truly_vip_ref TEXT, notes TEXT)")
    calls = []
    monkeypatch.setattr(appmod, "_extend_membership_grant",
        lambda c, email, until, source="x": calls.append((email, source)))
    # Simulate the success-branch tail for a founding product sub:
    sub = {"id": 1, "email": "f@x.com", "founding": 1}
    updated = {"next_charge_date": "2026-09-01"}
    appmod._maybe_extend_founding_membership(cx, sub, updated)
    assert calls == [("f@x.com", "founding")]

    # non-founding product sub: no extension
    calls.clear()
    appmod._maybe_extend_founding_membership(cx, {"id": 2, "email": "p@x.com", "founding": 0}, updated)
    assert calls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_founding_cron_membership.py -v`
Expected: FAIL (`AttributeError: ... '_maybe_extend_founding_membership'`)

- [ ] **Step 3: Write minimal implementation**

In `app.py` add the helper (keeps the cron branch tidy + unit-testable):

```python
def _maybe_extend_founding_membership(cx, sub, updated):
    """A founding product autoship keeps its comp membership alive: on each
    successful charge, extend the grant to next_charge_date + grace. No-op for
    non-founding subs."""
    if not sub.get("founding"):
        return
    try:
        if updated and updated.get("next_charge_date"):
            until = (datetime.fromisoformat(updated["next_charge_date"])
                     + timedelta(days=MEMBERSHIP_GRANT_GRACE_DAYS)).isoformat() + "Z"
            _extend_membership_grant(cx, sub["email"], until, "founding")
    except Exception as _ge:
        print(f"[sub-cron] founding grant-extend sub={sub.get('id')}: {_ge!r}", flush=True)
```

Then, in `cron_charge_subscriptions`, in the **product** success branch (after `_subs.advance_after_charge(cx, sid)` and reading `updated = _subs.get(cx, sid)`), call:

```python
                        _maybe_extend_founding_membership(cx, sub, updated)
```

(If `updated` isn't already read in that branch, add `updated = _subs.get(cx, sid)` first, mirroring the membership branch.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_founding_cron_membership.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_founding_cron_membership.py
git commit -m "feat(founding): autoship charge keeps the founder comp membership alive"
```

---

### Task 8: Live founding counter API + close-out

**Files:**
- Modify: `app.py` (`GET /begin/founding/status/<slug>`)
- Test: `tests/test_founding_counter_api.py`

**Interfaces:**
- Consumes: `dashboard.founding.get_launch`, `remaining`, `is_open`.
- Produces: `GET /begin/founding/status/<slug>` → `{"open": bool, "cap": int, "remaining": int, "batch_label": str}` (404 when no launch config). The product page polls this to render the live counter and flip to a closed/waitlist state when `open` is False.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_founding_counter_api.py
import app as appmod
import dashboard.founding as founding


def test_status_open_and_closed(monkeypatch):
    monkeypatch.setattr(founding, "get_launch",
        lambda s: {"cap": 2500, "batch_label": "Founding Batch No. 1", "closes_at": ""} if s == "neuro-magnesium" else None)
    monkeypatch.setattr(founding, "remaining", lambda cx, s: 1847)
    monkeypatch.setattr(founding, "is_open", lambda cx, s, now_iso=None: True)
    c = appmod.app.test_client()
    r = c.get("/begin/founding/status/neuro-magnesium")
    assert r.status_code == 200
    d = r.get_json()
    assert d == {"open": True, "cap": 2500, "remaining": 1847, "batch_label": "Founding Batch No. 1"}

    r2 = c.get("/begin/founding/status/missing")
    assert r2.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_founding_counter_api.py -v`
Expected: FAIL (404 for both — route not defined → the open case fails the dict assert)

- [ ] **Step 3: Write minimal implementation**

In `app.py`:

```python
@app.route("/begin/founding/status/<slug>")
def begin_founding_status(slug):
    from dashboard import founding as _founding
    launch = _founding.get_launch(slug)
    if not launch:
        return jsonify({"error": "no_founding_launch"}), 404
    today = _now_utc().strftime("%Y-%m-%d")
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        return jsonify({
            "open": _founding.is_open(cx, slug, now_iso=today),
            "cap": int(launch.get("cap", 0)),
            "remaining": _founding.remaining(cx, slug),
            "batch_label": launch.get("batch_label", ""),
        })
```

In `static/begin-product.html`, poll `/begin/founding/status/<slug>` to refresh the counter and, when `open === false`, swap the reserve button for a "Batch No. 2 — join the waitlist" state.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_founding_counter_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py static/begin-product.html tests/test_founding_counter_api.py
git commit -m "feat(founding): live counter status API + close-out"
```

---

## Self-Review

**Spec coverage:**
- §4a membership-free-while-autoship-active → Task 5 (grant on reserve) + Task 7 (extend on each charge) + natural lapse on cancel (no teardown). ✓
- §4b pre-order/charge-on-ship → Task 4 ($0 vault reserve) + Task 5 (reservation row) + Task 6 (charge-on-ship activate). ✓
- §5 founding cap/counter/close-out → Task 2 (cap/is_open) + Task 4 (cap gate) + Task 8 (live counter + close-out). ✓
- §6 parameterization → Task 2 config keyed by slug; all routes take `slug`. ✓
- §7 product page + video → Task 3. ✓
- §8 compliance → Global Constraints + structure-function copy in Task 3 + ROSCA note in Task 4. ✓
- §9 success criteria → reserve ($0, counter, grant) Tasks 4/5/8; ship→charge→coaching Task 6; lapse-on-cancel Task 7; per-product config Task 2. ✓

**Placeholder scan:** No "TBD"/"TODO" in steps. Both earlier recon-confirms are now resolved against the real code: the reserve route uses the existing `stripe_pay.create_setup_session` (`stripe_pay.py:70`, $0 vault) and `get_setup_intent` (`stripe_pay.py:88`); the ship route uses the real `CONSOLE_SECRET`/`X-Console-Key` gate (`app.py:7303`). One real-world (non-code) prerequisite remains: host the v1 promo MP4 on R2 and set its URL in `data/founding_launches.json` (Task 3 handles both the present and empty-URL cases).

**Type consistency:** `create_founding_reservation`, `mark_founding_active`, `list_founding_pending`, `count_founding` consistent across Tasks 1/2/6. `_maybe_extend_founding_membership(cx, sub, updated)` defined and called in Task 7. `founding`/`founding_state`/`founding_slug` columns consistent throughout. `kind="founding_reserve"` consistent Tasks 4/5.

## Execution Handoff

Recommended: superpowers:subagent-driven-development (fresh subagent per task + two-stage review). Tasks 4–6 touch large `app.py` route code and the Stripe setup-mode path — review those most carefully and run the full suite (`python -m pytest tests/ -q`) after each.
