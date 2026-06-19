# Spec 2a-3 — Review Gifts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a review hits 5 points, the AI suggests a catalog gift (tailored), Glen/Rae approve it in the console, and it auto-adds as a $0 line on the person's next in-house order.

**Architecture:** A gift catalog + `review_gifts` store; a `suggest_gift` scorer; the video worker records a suggestion at 5 points (within a monthly cap); dispatch-spine approve/reject actions + console display; the in-house order entry redeems approved gifts as $0 lines. All flag-gated behind `REVIEWS_GIFTS`.

**Tech Stack:** Python 3.11, Flask, SQLite (`chat_log.db`/`LOG_DB`), Anthropic haiku (`_cl`), APScheduler worker, pytest.

## Global Constraints

- **Trigger:** review total **== 5**; **cap: skip if the person has any non-rejected gift created in the prior 30 days**.
- **Catalog:** `data/review-gifts.json` (`[{sku,label,description}]`); AI picks one `sku` + a `reason`, tailored by the review + the person's recent order history. Fail-closed → no suggestion.
- **Delivery:** approved gift → `pending_for(email)`; in-house order entry adds a **$0 line** (`unit_cents:0, line_cents:0`, excluded from subtotal) + marks the gift **fulfilled** with the order id; customer lookup surfaces a pending-gift indicator.
- **Flag `REVIEWS_GIFTS`** (default off; requires `REVIEWS_VIDEO`). Fully inert when off.
- Gift suggestion/redemption wrapped so it never blocks scoring/credit/trim or order creation; idempotent (one suggestion per review; fulfilled gifts don't re-add).
- NO emoji; no em dashes in generated text (`_strip_dash`).
- **Test command (every task):** `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_reviews_spec2a3.py -v`
- Tests reload `app` via `importlib` with flags; mock the Anthropic client (`.messages.create(...) -> obj.content=[block(type='text',text=...)]`); idempotent action registration.

---

### Task 1: Catalog + `review_gifts` store

**Files:**
- Create: `data/review-gifts.json`, `dashboard/review_gifts.py`
- Test: `tests/test_reviews_spec2a3.py` (create)

**Interfaces:**
- Produces: `load_catalog(path=None) -> list`, `catalog_by_sku(path=None) -> dict`, `valid_sku(sku, path=None) -> bool`, `init_table(cx)`, `add_suggestion(cx, review_id, email, sku, label, reason) -> int`, `recent_active_gift(cx, email, days=30) -> bool`, `get_for_review(cx, review_id) -> dict|None`, `set_status(cx, gift_id, status, by="") -> None`, `swap_sku(cx, gift_id, sku, label) -> None`, `pending_for(cx, email) -> list[dict]`, `mark_fulfilled(cx, gift_id, order_id) -> None`, `suggested_queue(cx) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reviews_spec2a3.py`:

```python
import sqlite3
from dashboard import review_gifts as rg


def _cx():
    return sqlite3.connect(":memory:")


def test_catalog_loads_and_validates():
    cat = rg.load_catalog()
    skus = {g["sku"] for g in cat}
    assert {"gift-toothbrush", "gift-nightlight", "gift-tuningfork"} <= skus
    assert rg.valid_sku("gift-tuningfork") and not rg.valid_sku("gift-nope")


def test_add_and_get_for_review():
    cx = _cx()
    gid = rg.add_suggestion(cx, 7, "a@x.com", "gift-tuningfork", "Tuning fork", "into sound work")
    g = rg.get_for_review(cx, 7)
    assert g["id"] == gid and g["email"] == "a@x.com" and g["status"] == "suggested"
    assert g["gift_sku"] == "gift-tuningfork" and g["gift_label"] == "Tuning fork"


def test_recent_active_gift_cap():
    cx = _cx()
    rg.add_suggestion(cx, 1, "a@x.com", "gift-nightlight", "Night light", "r")
    assert rg.recent_active_gift(cx, "a@x.com", 30) is True          # within window, non-rejected
    assert rg.recent_active_gift(cx, "other@x.com", 30) is False
    # a rejected gift frees the slot
    cx2 = _cx()
    g = rg.add_suggestion(cx2, 1, "b@x.com", "gift-nightlight", "Night light", "r")
    rg.set_status(cx2, g, "rejected")
    assert rg.recent_active_gift(cx2, "b@x.com", 30) is False


def test_approve_swap_pending_and_fulfill():
    cx = _cx()
    gid = rg.add_suggestion(cx, 1, "a@x.com", "gift-nightlight", "Night light", "r")
    rg.swap_sku(cx, gid, "gift-toothbrush", "Bamboo toothbrush")
    rg.set_status(cx, gid, "approved", by="Glen")
    pend = rg.pending_for(cx, "a@x.com")
    assert len(pend) == 1 and pend[0]["gift_sku"] == "gift-toothbrush" and pend[0]["status"] == "approved"
    rg.mark_fulfilled(cx, gid, 555)
    assert rg.pending_for(cx, "a@x.com") == []                       # fulfilled -> not pending
    assert rg.get_for_review(cx, 1)["fulfilled_order_id"] == 555


def test_suggested_queue():
    cx = _cx()
    a = rg.add_suggestion(cx, 1, "a@x.com", "gift-nightlight", "Night light", "r")
    b = rg.add_suggestion(cx, 2, "b@x.com", "gift-toothbrush", "Toothbrush", "r")
    rg.set_status(cx, a, "approved")
    q = rg.suggested_queue(cx)
    assert [g["review_id"] for g in q] == [2]                        # only still-suggested
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (`ModuleNotFoundError: dashboard.review_gifts`).

- [ ] **Step 3a: Create `data/review-gifts.json`**

```json
[
  {"sku": "gift-toothbrush", "label": "Bamboo toothbrush", "description": "Eco bamboo toothbrush; pairs with daily-care routines."},
  {"sku": "gift-nightlight", "label": "Red nightlight", "description": "Low-blue red nightlight; supports healthy sleep and melatonin."},
  {"sku": "gift-tuningfork", "label": "Tuning fork", "description": "Therapeutic tuning fork; for sound, energy, and biofield work."}
]
```

- [ ] **Step 3b: Create `dashboard/review_gifts.py`**

```python
import json
import os
import datetime

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "review-gifts.json")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_catalog(path=None):
    try:
        with open(path or _DEFAULT_PATH) as fh:
            data = json.load(fh)
        return [g for g in data if g.get("sku")]
    except Exception:
        return []


def catalog_by_sku(path=None):
    return {g["sku"]: g for g in load_catalog(path)}


def valid_sku(sku, path=None):
    return sku in catalog_by_sku(path)


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS review_gifts ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, review_id INTEGER, email TEXT, "
        "gift_sku TEXT, gift_label TEXT, reason TEXT, status TEXT DEFAULT 'suggested', "
        "created_at TEXT, approved_by TEXT DEFAULT '', approved_at TEXT DEFAULT '', "
        "fulfilled_order_id INTEGER, fulfilled_at TEXT DEFAULT '')")
    cx.commit()


def _row(cx, where, args):
    cur = cx.cursor(); cur.row_factory = __import__("sqlite3").Row
    r = cur.execute(f"SELECT * FROM review_gifts WHERE {where}", args).fetchone()
    return dict(r) if r else None


def add_suggestion(cx, review_id, email, sku, label, reason):
    init_table(cx)
    cx.execute(
        "INSERT INTO review_gifts (review_id, email, gift_sku, gift_label, reason, status, created_at) "
        "VALUES (?,?,?,?,?, 'suggested', ?)",
        (review_id, (email or "").strip().lower(), sku, label, reason or "", _now()))
    cx.commit()
    return cx.execute("SELECT id FROM review_gifts WHERE review_id=? ORDER BY id DESC LIMIT 1",
                      (review_id,)).fetchone()[0]


def recent_active_gift(cx, email, days=30):
    init_table(cx)
    cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)).isoformat()
    return cx.execute(
        "SELECT 1 FROM review_gifts WHERE email=? AND status!='rejected' AND created_at>=? LIMIT 1",
        ((email or "").strip().lower(), cutoff)).fetchone() is not None


def get_for_review(cx, review_id):
    init_table(cx)
    return _row(cx, "review_id=? ORDER BY id DESC LIMIT 1", (review_id,))


def set_status(cx, gift_id, status, by=""):
    init_table(cx)
    if status == "approved":
        cx.execute("UPDATE review_gifts SET status=?, approved_by=?, approved_at=? WHERE id=?",
                   (status, by or "", _now(), gift_id))
    else:
        cx.execute("UPDATE review_gifts SET status=? WHERE id=?", (status, gift_id))
    cx.commit()


def swap_sku(cx, gift_id, sku, label):
    init_table(cx)
    cx.execute("UPDATE review_gifts SET gift_sku=?, gift_label=? WHERE id=?", (sku, label, gift_id))
    cx.commit()


def pending_for(cx, email):
    init_table(cx)
    cur = cx.cursor(); cur.row_factory = __import__("sqlite3").Row
    rows = cur.execute(
        "SELECT * FROM review_gifts WHERE email=? AND status='approved' AND fulfilled_order_id IS NULL "
        "ORDER BY id", ((email or "").strip().lower(),)).fetchall()
    return [dict(r) for r in rows]


def mark_fulfilled(cx, gift_id, order_id):
    init_table(cx)
    cx.execute("UPDATE review_gifts SET status='fulfilled', fulfilled_order_id=?, fulfilled_at=? WHERE id=?",
               (order_id, _now(), gift_id))
    cx.commit()


def suggested_queue(cx):
    init_table(cx)
    cur = cx.cursor(); cur.row_factory = __import__("sqlite3").Row
    rows = cur.execute(
        "SELECT * FROM review_gifts WHERE status='suggested' ORDER BY created_at DESC, id DESC").fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: Run to verify they pass**

Run the test command. Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add data/review-gifts.json dashboard/review_gifts.py tests/test_reviews_spec2a3.py
git commit -m "feat(reviews-2a3): gift catalog + review_gifts store (suggest/approve/fulfill, monthly cap)"
```

---

### Task 2: `suggest_gift` AI helper

**Files:**
- Modify: `dashboard/review_scoring.py`
- Test: `tests/test_reviews_spec2a3.py` (append)

**Interfaces:**
- Consumes: a `client` with `.messages.create(...)`.
- Produces: `build_gift_prompt(review_text, product, order_history, catalog) -> (system, user)` (pure); `suggest_gift(client, review_text, product, order_history, catalog, *, strip=lambda s: s) -> {"sku": str, "reason": str} | None` (returns None on parse error / empty catalog / invalid sku; never raises; reason dash-stripped).

- [ ] **Step 1: Write the failing tests**

Append:

```python
from dashboard import review_scoring as rs


class _Blk:
    def __init__(self, t): self.type = "text"; self.text = t


class _Msg:
    def __init__(self, t): self.content = [_Blk(t)]


class _FakeClient:
    def __init__(self, payload): self._p = payload
    @property
    def messages(self):
        outer = self
        class _M:
            def create(self, **kw): return _Msg(outer._p)
        return _M()


_CAT = [{"sku": "gift-tuningfork", "label": "Tuning fork", "description": "sound/energy"},
        {"sku": "gift-nightlight", "label": "Red nightlight", "description": "sleep"}]


def test_build_gift_prompt_includes_catalog_and_history():
    system, user = rs.build_gift_prompt("loved it, sleep better", {"name": "Longevity"},
                                        ["Magnesium", "Neuro"], _CAT)
    assert "gift-tuningfork" in user and "Longevity" in user and "Magnesium" in user


def test_suggest_gift_returns_valid_sku():
    c = _FakeClient('{"sku": "gift-nightlight", "reason": "they mentioned sleep"}')
    out = rs.suggest_gift(c, "sleep better", {"name": "X"}, [], _CAT)
    assert out == {"sku": "gift-nightlight", "reason": "they mentioned sleep"}


def test_suggest_gift_none_on_invalid_sku():
    c = _FakeClient('{"sku": "gift-nope", "reason": "x"}')
    assert rs.suggest_gift(c, "t", {"name": "X"}, [], _CAT) is None


def test_suggest_gift_none_on_empty_catalog():
    c = _FakeClient('{"sku": "gift-nightlight", "reason": "x"}')
    assert rs.suggest_gift(c, "t", {"name": "X"}, [], []) is None


def test_suggest_gift_none_on_bad_json():
    assert rs.suggest_gift(_FakeClient("not json"), "t", {"name": "X"}, [], _CAT) is None


def test_suggest_gift_strips_dashes():
    c = _FakeClient('{"sku": "gift-tuningfork", "reason": "good — fit"}')
    out = rs.suggest_gift(c, "t", {"name": "X"}, [], _CAT, strip=lambda s: s.replace("—", ","))
    assert "—" not in out["reason"]
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (`suggest_gift` missing).

- [ ] **Step 3: Implement in `dashboard/review_scoring.py`** (append)

```python
def build_gift_prompt(review_text, product, order_history, catalog):
    name = (product or {}).get("name", "")
    items = "\n".join(f"- {g['sku']}: {g.get('label','')} ({g.get('description','')})" for g in catalog)
    hist = ", ".join(order_history or []) or "(none)"
    system = ("You pick a thank-you gift for a customer who left an excellent video review of "
              "Dr. Glen Swartwout's supplements. Choose ONE gift from the catalog that best fits this "
              "person. Return ONLY a JSON object: {\"sku\": <one catalog sku>, \"reason\": <short why>}.")
    user = (f"Reviewed product: {name}\nTheir recent orders: {hist}\n\nReview:\n{review_text or '(none)'}\n\n"
            f"Gift catalog:\n{items}\n\nReturn only the JSON object.")
    return system, user


def suggest_gift(client, review_text, product, order_history, catalog, *, strip=lambda s: s):
    if not catalog:
        return None
    valid = {g["sku"] for g in catalog}
    system, user = build_gift_prompt(review_text, product, order_history, catalog)
    try:
        msg = client.messages.create(model=_MODEL, max_tokens=200, system=system,
                                      messages=[{"role": "user", "content": user}])
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text").strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            return None
        data = json.loads(text[start:end + 1])
        sku = data.get("sku")
        if sku not in valid:
            return None
        return {"sku": sku, "reason": strip(str(data.get("reason", "")))[:300]}
    except Exception:  # noqa: BLE001
        return None
```

- [ ] **Step 4: Run to verify they pass**

Run the test command. Expected: all Task-1 + Task-2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add dashboard/review_scoring.py tests/test_reviews_spec2a3.py
git commit -m "feat(reviews-2a3): suggest_gift (catalog pick, tailored, fail-closed)"
```

---

### Task 3: `REVIEWS_GIFTS` flag + worker gift-suggestion step

**Files:**
- Modify: `app.py` (flag; `_drain_review_videos` gift step)
- Test: `tests/test_reviews_spec2a3.py` (append)

**Interfaces:**
- Consumes: Task 1 `review_gifts.*`; Task 2 `review_scoring.suggest_gift`; `orders.list_orders_by_email`; worker locals `total`, `transcript`, `r`, `rid`, `prod`.
- Produces: `_REVIEWS_GIFTS` flag; the worker records a `suggested` gift when a review hits 5 points within the cap.

- [ ] **Step 1: Write the failing tests**

Append (a reload helper enabling all flags + seeding a scored 5-point review path):

```python
import importlib


def _reload_gift_app(monkeypatch, tmp_path, gifts="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REVIEWS_ENABLED", "true")
    monkeypatch.setenv("REVIEWS_VIDEO", "true")
    monkeypatch.setenv("REVIEWS_GIFTS", gifts)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _seed_scored_video(appmod, slug, email="b@x.com", video_points=5, written=0):
    import sqlite3
    from dashboard import product_reviews as pr, review_video_jobs as vj
    d = appmod._REVIEW_MEDIA_DIR / slug; d.mkdir(parents=True, exist_ok=True)
    (d / "v.webm").write_bytes(b"X")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr.upsert_review(cx, slug, email, "B", 5, video_kind="upload", video_ref="v.webm")
        pr.set_ai_result(cx, rid, written, "", 0); pr.set_points(cx, rid, written)
        vj.enqueue(cx, rid)
    return rid


def _patch_pipeline(monkeypatch, video_points=5, gift_sku="gift-nightlight"):
    import journal_blueprint
    monkeypatch.setattr(journal_blueprint, "_whisper_transcribe",
                        lambda p: {"text": "great review", "duration": 12.0, "words": []})
    from dashboard import review_scoring as rs
    monkeypatch.setattr(rs, "score_video", lambda *a, **k: {
        "video_points": video_points, "publish_risk": False, "risk_reasons": "", "recommend_publish": True})
    monkeypatch.setattr(rs, "suggest_gift", lambda *a, **k: ({"sku": gift_sku, "reason": "fits"} if gift_sku else None))


def test_worker_suggests_gift_at_5_points(monkeypatch, tmp_path):
    appmod = _reload_gift_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    rid = _seed_scored_video(appmod, slug)
    _patch_pipeline(monkeypatch, video_points=5)
    appmod._drain_review_videos()
    import sqlite3
    from dashboard import review_gifts as rg
    with sqlite3.connect(appmod.LOG_DB) as cx:
        g = rg.get_for_review(cx, rid)
    assert g is not None and g["status"] == "suggested" and g["gift_sku"] == "gift-nightlight"


def test_worker_no_gift_under_5(monkeypatch, tmp_path):
    appmod = _reload_gift_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    rid = _seed_scored_video(appmod, slug)
    _patch_pipeline(monkeypatch, video_points=3)        # total 3
    appmod._drain_review_videos()
    import sqlite3
    from dashboard import review_gifts as rg
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert rg.get_for_review(cx, rid) is None


def test_worker_monthly_cap(monkeypatch, tmp_path):
    appmod = _reload_gift_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    import sqlite3
    from dashboard import review_gifts as rg
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rg.add_suggestion(cx, 999, "b@x.com", "gift-toothbrush", "Toothbrush", "earlier")  # within 30d
    rid = _seed_scored_video(appmod, slug, email="b@x.com")
    _patch_pipeline(monkeypatch, video_points=5)
    appmod._drain_review_videos()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert rg.get_for_review(cx, rid) is None        # capped


def test_worker_gift_flag_off(monkeypatch, tmp_path):
    appmod = _reload_gift_app(monkeypatch, tmp_path, gifts="false")
    assert appmod._REVIEWS_GIFTS is False
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    rid = _seed_scored_video(appmod, slug)
    _patch_pipeline(monkeypatch, video_points=5)
    appmod._drain_review_videos()
    import sqlite3
    from dashboard import review_gifts as rg
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert rg.get_for_review(cx, rid) is None
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (`_REVIEWS_GIFTS` undefined / no gift).

- [ ] **Step 3: Implement in `app.py`**

Add the flag next to `_REVIEWS_VIDEO_TRIM`:

```python
_REVIEWS_GIFTS = os.environ.get("REVIEWS_GIFTS", "").strip().lower() in ("1", "true", "yes")
```

In `_drain_review_videos`, after the trim block (the `if _REVIEWS_VIDEO_TRIM:` section) and still inside the per-job `try`, add the gift step:

```python
            if _REVIEWS_GIFTS and total == 5:
                try:
                    from dashboard import review_gifts as _rg
                    from dashboard import orders as _o2
                    with sqlite3.connect(LOG_DB) as cx:
                        _has_gift = _rg.get_for_review(cx, rid) is not None
                        _capped = _rg.recent_active_gift(cx, r["email"], 30)
                    if not _has_gift and not _capped:
                        _cat = _rg.load_catalog()
                        with sqlite3.connect(LOG_DB) as cx:
                            cx.row_factory = sqlite3.Row
                            _hist = []
                            for _o in _o2.list_orders_by_email(cx, r["email"], limit=5):
                                for _it in (_o.get("items") or []):
                                    if _it.get("name"):
                                        _hist.append(_it["name"])
                        _sg = _rs.suggest_gift(_cl, transcript, prod, _hist[:10], _cat, strip=_strip_dash)
                        if _sg:
                            _lbl = _rg.catalog_by_sku().get(_sg["sku"], {}).get("label", _sg["sku"])
                            with sqlite3.connect(LOG_DB) as cx:
                                _rg.add_suggestion(cx, rid, r["email"], _sg["sku"], _lbl, _sg["reason"])
                except Exception as e:  # noqa: BLE001 - gift never blocks scoring/credit/trim
                    print(f"[reviews-video] gift suggest failed rid={rid}: {e}", flush=True)
```

(`total`, `transcript`, `prod`, `r`, `rid`, `_rs` are all in scope in the worker. `_rs` is the `review_scoring` module already imported at the top of `_drain_review_videos`.)

- [ ] **Step 4: Run to verify they pass**

Run the test command, then the sweep `pytest tests/ -k "reviews" -q`. Expected: all pass, no regressions.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_reviews_spec2a3.py
git commit -m "feat(reviews-2a3): REVIEWS_GIFTS flag + worker gift suggestion at 5 points (monthly cap)"
```

---

### Task 4: Console gift actions + display

**Files:**
- Modify: `dashboard/reviews_actions.py` (gift_approve / gift_reject), `app.py` (`/api/console/reviews` includes gift; `/api/console/gift-catalog`), `static/console-reviews.html`
- Test: `tests/test_reviews_spec2a3.py` (append)

**Interfaces:**
- Consumes: Task 1 `review_gifts.{get_for_review, set_status, swap_sku, valid_sku, load_catalog}`.
- Produces: actions `reviews.gift_approve {review_id, sku?}`, `reviews.gift_reject {review_id}` (LOW_WRITE, (OWNER,OPS)); `/api/console/reviews` rows carry `gift` (or None); `GET /api/console/gift-catalog`.

- [ ] **Step 1: Write the failing tests**

Append:

```python
from dashboard.rbac import Actor, OWNER


def test_gift_actions_approve_swap_reject(monkeypatch, tmp_path):
    appmod = _reload_gift_app(monkeypatch, tmp_path)
    import sqlite3
    from dashboard import product_reviews as pr, review_gifts as rg
    from dashboard import dispatch as d
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr.upsert_review(cx, "x", "a@x.com", "Ann", 5, video_kind="upload", video_ref="v.webm")
        gid = rg.add_suggestion(cx, rid, "a@x.com", "gift-nightlight", "Night light", "r")
        # approve with a swap sku
        res = d.dispatch_action(cx, "reviews.gift_approve", {"review_id": rid, "sku": "gift-tuningfork"},
                                Actor(role=OWNER, name="Glen"), source="panel")
        assert res["status"] == "done"
        g = rg.get_for_review(cx, rid)
        assert g["status"] == "approved" and g["gift_sku"] == "gift-tuningfork" and g["approved_by"] == "Glen"
        # reject path on a fresh review
        rid2 = pr.upsert_review(cx, "x", "b@x.com", "Bob", 5, video_kind="upload", video_ref="v.webm")
        g2 = rg.add_suggestion(cx, rid2, "b@x.com", "gift-nightlight", "Night light", "r")
        d.dispatch_action(cx, "reviews.gift_reject", {"review_id": rid2},
                          Actor(role=OWNER, name="Glen"), source="panel")
        assert rg.get_for_review(cx, rid2)["status"] == "rejected"


def test_console_reviews_includes_gift_and_catalog(monkeypatch, tmp_path):
    appmod = _reload_gift_app(monkeypatch, tmp_path)
    import dashboard as _d
    _d.CONSOLE_SECRET = ""
    import sqlite3
    from dashboard import product_reviews as pr, review_gifts as rg
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr.upsert_review(cx, "x", "a@x.com", "Ann", 5, video_kind="upload", video_ref="v.webm")
        rg.add_suggestion(cx, rid, "a@x.com", "gift-nightlight", "Night light", "fits sleep")
    c = appmod.app.test_client()
    rows = c.get("/api/console/reviews").get_json()["pending"]
    row = next(r for r in rows if r["email"] == "a@x.com")
    assert row["gift"]["gift_sku"] == "gift-nightlight" and row["gift"]["status"] == "suggested"
    cat = c.get("/api/console/gift-catalog").get_json()["catalog"]
    assert any(g["sku"] == "gift-tuningfork" for g in cat)
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (unknown action / no `gift` key / catalog 404).

- [ ] **Step 3: Implement**

In `dashboard/reviews_actions.py`, add executors + register them inside `register()` (after the existing three, before the function ends):

```python
def _exec_gift_approve(params, ctx):
    from dashboard import review_gifts as _rg
    rid = int(params.get("review_id") or 0)
    if not rid:
        raise ValueError("review_id required")
    g = _rg.get_for_review(ctx["cx"], rid)
    if not g:
        raise ValueError("no gift for review")
    sku = (params.get("sku") or "").strip()
    if sku and _rg.valid_sku(sku):
        _rg.swap_sku(ctx["cx"], g["id"], sku, _rg.catalog_by_sku().get(sku, {}).get("label", sku))
    _rg.set_status(ctx["cx"], g["id"], "approved", by=_name(ctx.get("actor")))
    return {"review_id": rid, "status": "approved"}


def _exec_gift_reject(params, ctx):
    from dashboard import review_gifts as _rg
    rid = int(params.get("review_id") or 0)
    if not rid:
        raise ValueError("review_id required")
    g = _rg.get_for_review(ctx["cx"], rid)
    if g:
        _rg.set_status(ctx["cx"], g["id"], "rejected", by=_name(ctx.get("actor")))
    return {"review_id": rid, "status": "rejected"}
```

Register both inside `register()` (guard already returns early if registered):

```python
    register_action(Action(key="reviews.gift_approve", module="reviews", title="Approve review gift",
        description="Approve the AI-suggested gift (optionally swap the item) for a 5-point review.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_gift_approve))
    register_action(Action(key="reviews.gift_reject", module="reviews", title="Reject review gift",
        description="Reject the AI-suggested gift for a review.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_gift_reject))
```

In `app.py` `api_console_reviews_list`, when `_REVIEWS_GIFTS`, attach each row's gift:

```python
    if _REVIEWS_GIFTS:
        from dashboard import review_gifts as _rg
        with sqlite3.connect(LOG_DB) as cx:
            for r in pending:
                r["gift"] = _rg.get_for_review(cx, r["id"])
```

Add the catalog endpoint after it:

```python
@app.route("/api/console/gift-catalog", methods=["GET"])
def api_console_gift_catalog():
    bad = _sales_console_ok()
    if bad:
        return bad
    from dashboard import review_gifts as _rg
    return jsonify({"ok": True, "catalog": _rg.load_catalog()})
```

Front-end (manual visual pass — note in report): `static/console-reviews.html` — when `rv.gift` exists, show "Suggested gift: <label> — <reason>" with Approve / Swap (a `<select>` of `/api/console/gift-catalog`) / Reject buttons POSTing to `/api/action/reviews.gift_approve` (with optional `sku`) and `/api/action/reviews.gift_reject`. NO emoji.

- [ ] **Step 4: Run to verify they pass**

Run the test command. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add dashboard/reviews_actions.py app.py static/console-reviews.html tests/test_reviews_spec2a3.py
git commit -m "feat(reviews-2a3): console gift approve/swap/reject + catalog endpoint + display"
```

---

### Task 5: Redemption — $0 gift line at in-house order entry + pending indicator

**Files:**
- Modify: `app.py` (`api_orders_manual`; `api_customers_search`)
- Test: `tests/test_reviews_spec2a3.py` (append)

**Interfaces:**
- Consumes: Task 1 `review_gifts.{pending_for, mark_fulfilled}`.
- Produces: an approved gift is auto-added as a $0 line on the next in-house order + marked fulfilled; `api_customers_search` people include `pending_gifts`.

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_order_entry_adds_gift_line_and_fulfills(monkeypatch, tmp_path):
    appmod = _reload_gift_app(monkeypatch, tmp_path)
    # the in-house order route is OWNER-gated; resolve_actor returns OWNER for the console key
    import dashboard as _d
    _d.CONSOLE_SECRET = "k"; appmod.CONSOLE_SECRET = "k"
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    p = appmod._get_product(slug)
    import sqlite3
    from dashboard import review_gifts as rg
    with sqlite3.connect(appmod.LOG_DB) as cx:
        gid = rg.add_suggestion(cx, 1, "buyer@x.com", "gift-nightlight", "Red nightlight", "r")
        rg.set_status(cx, gid, "approved", by="Glen")
    c = appmod.app.test_client()
    body = {"customer": {"email": "buyer@x.com", "name": "Buyer",
                         "address": {"street": "1 A St", "city": "Hilo", "state": "HI", "zip": "96720"}},
            "lines": [{"slug": slug, "qty": 1}]}
    r = c.post("/api/orders/manual", json=body, headers={"X-Console-Key": "k"}).get_json()
    assert r["ok"]
    oid = r["order_id"]
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import orders as o
        order = o.list_orders_by_email(cx, "buyer@x.com")[0]
        names = [it.get("name", "") for it in order["items"]]
        assert any("Red nightlight" in n for n in names)            # $0 gift line present
        gift_lines = [it for it in order["items"] if it.get("gift")]
        assert gift_lines and gift_lines[0]["unit_cents"] == 0
        assert rg.get_for_review(cx, 1)["fulfilled_order_id"] == oid  # marked fulfilled
        assert rg.pending_for(cx, "buyer@x.com") == []
    # a SECOND order does not re-add the gift
    r2 = c.post("/api/orders/manual", json=body, headers={"X-Console-Key": "k"}).get_json()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import orders as o
        order2 = [x for x in o.list_orders_by_email(cx, "buyer@x.com") if x["id"] == r2["order_id"]][0]
        assert not any(it.get("gift") for it in order2["items"])
```

(NOTE: the in-house order route validates the customer address + products. If the test 400s on a missing required field, the implementer adds the minimal required fields the handler needs — read `api_orders_manual` to see what `ship`/`addr_in` require. The asserted behavior is what matters: a $0 `gift` line is added on the first order and the gift is marked fulfilled, not re-added on the second.)

- [ ] **Step 2: Run to verify it fails**

Run the test command. Expected: FAIL (no gift line on the order).

- [ ] **Step 3: Implement in `app.py`**

In `api_orders_manual`, after `if not cart: return jsonify(... "no valid products"...)` and the `items_rec` is fully built, gather + append gift lines (guard on flag + email):

```python
    _gift_rows = []
    _gift_email = (customer.get("email") or "").strip().lower()
    if _REVIEWS_GIFTS and _gift_email:
        try:
            from dashboard import review_gifts as _rg
            _gcx = _sqlite3.connect(LOG_DB)
            try:
                _gift_rows = _rg.pending_for(_gcx, _gift_email)
            finally:
                _gcx.close()
            for _g in _gift_rows:
                items_rec.append({"slug": _g["gift_sku"], "name": _g["gift_label"] + " (gift)",
                                  "qty": 1, "unit_cents": 0, "line_cents": 0, "gift": True})
        except Exception as e:  # noqa: BLE001 - gift never blocks order creation
            print(f"[orders.manual] gift add failed: {e}", flush=True); _gift_rows = []
```

Then, inside the existing `cx` try-block, after `oid = _bos_orders.upsert_order(...)`, mark each fulfilled:

```python
        if _gift_rows and oid:
            from dashboard import review_gifts as _rg2
            for _g in _gift_rows:
                _rg2.mark_fulfilled(cx, _g["id"], oid)
```

In `api_customers_search`, when `_REVIEWS_GIFTS`, attach pending gifts to each returned person:

```python
    if _REVIEWS_GIFTS:
        from dashboard import review_gifts as _rg
        for _p in people:
            _pe = (_p.get("email") or "").strip().lower()
            _p["pending_gifts"] = [{"label": g["gift_label"]} for g in _rg.pending_for(cx, _pe)] if _pe else []
```

(`people` + the `cx` connection are in scope in `api_customers_search`.)

- [ ] **Step 4: Run to verify it passes**

Run the test command, then the full sweep `pytest tests/ -k "reviews or orders" -q`. Expected: all pass, no regressions.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_reviews_spec2a3.py
git commit -m "feat(reviews-2a3): redeem approved gift as $0 order line + pending-gift indicator"
```

---

## Self-Review (plan author)

- **Spec coverage:** catalog+store (T1) → spec Catalog/Store; suggest_gift (T2) → AI suggestion; worker trigger+flag+cap (T3) → Worker trigger; console actions+display+catalog (T4) → Console approval; redemption+indicator (T5) → Redemption. Flag `REVIEWS_GIFTS` woven through T3/T4/T5.
- **Decisions honored:** trigger total==5 (T3); monthly cap via `recent_active_gift` non-rejected/30d (T1/T3); tailored by order history (T2/T3); catalog-grounded with `valid_sku` (T1/T2/T4); $0 line excluded from subtotal + fulfilled idempotent (T5); pending indicator (T5); flag-off inert; no emoji / no em dashes.
- **Type consistency:** `review_gifts` fn signatures, `suggest_gift(...) -> {sku,reason}|None`, action keys `reviews.gift_approve|gift_reject`, the `gift` row key, `pending_gifts` — used identically across tasks.
- **Confirm-then-use flagged in-task:** the `_drain_review_videos` insertion point (after trim block, T3); the in-house order route's required address fields (T5 test note); `reviews_actions.register()` guard already idempotent (T4).
