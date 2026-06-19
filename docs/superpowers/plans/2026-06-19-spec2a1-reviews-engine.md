# Spec 2a-1 — Product Reviews Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let verified buyers rate/review products, earn AI-scored store-credit points automatically, and surface approved reviews on the sales page — built on the existing dispatch-spine + console pattern.

**Architecture:** A new `product_reviews` table + data module; an AI scoring/compliance module (haiku); a flag-gated submit API that verifies the buyer, scores the written review, and auto-credits points; a reorder star-gate; a console moderation queue (dispatch-spine actions) that gates publication; and a sales-page testimonials section. Video is captured (link/upload) but its AI scoring/trim is deferred to 2a-2.

**Tech Stack:** Python 3.11, Flask, SQLite (`chat_log.db` via `LOG_DB`), Anthropic SDK (`_cl`, `claude-haiku-4-5-20251001`), vanilla-JS static pages, pytest.

## Global Constraints

- **Reward = points, fully AI-automated.** 1 point = $1 = 100¢. Points capped at **5 total per review**. Written quality scores **0–2**. Points **auto-credit on the AI compliance-gate PASS**; humans gate **publication only**, never points.
- **Verified buyers only.** A buyer is verified for a slug when `list_orders_by_email` returns an order with `status` in `{paid, new, packed, shipped, done}` containing an item whose `_resolve_buy_slug(name) == slug`.
- **One review per (slug, email)** — `UNIQUE(product_slug, email)` is the done-state; capture never re-prompts.
- **Star rating is required to reorder** an unreviewed item.
- Ship behind flag **`REVIEWS_ENABLED`** (default off): capture/serve 404 and no testimonials section until on.
- Points credit is idempotent: `order_ref=f"review:{review_id}"`, `reason=f"review:{slug}"`.
- **NO emoji** (SVG/text glyphs only). **No em dashes** in generated text (`_strip_dash`).
- OUT of scope: deep AI video scoring/transcript + auto-trim (2a-2); AI gift suggestion (2a-3); referral coupon (2b).
- **Test command (every task):** `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_reviews_spec2a1.py -v`
- Tests reload `app` via `importlib`; `register()` must be idempotent; if a module injects deps via `configure`, add an autouse fixture clearing its `_DEPS` (mirror Phase-5 `_reset_spa_deps`). Mock the Anthropic client with a fake exposing `.messages.create(...) -> obj.content=[block(type='text', text=...)]`.

---

### Task 1: Reviews data layer

**Files:**
- Create: `dashboard/product_reviews.py`
- Test: `tests/test_reviews_spec2a1.py` (create)

**Interfaces:**
- Produces:
  - `init_table(cx)`
  - `has_reviewed(cx, slug, email) -> bool`
  - `upsert_review(cx, slug, email, name, rating, body="", video_kind="", video_ref="") -> int` (returns review id; UNIQUE(slug,email) upsert, preserves id on update, resets `status='pending'` and clears prior ai/points so a re-submit re-moderates)
  - `set_ai_result(cx, review_id, ai_score, ai_verdict, recommend_publish)`
  - `set_points(cx, review_id, points)`
  - `set_status(cx, review_id, status, by="")`
  - `set_featured(cx, review_id, on)`
  - `get_review(cx, review_id) -> dict|None`
  - `approved_for_slug(cx, slug) -> list[dict]` (status=approved, featured first then newest)
  - `aggregate(cx, slug) -> {"count": int, "avg": float}` (approved only; avg rounded to 1 dp, 0 when none)
  - `pending_queue(cx, limit=100) -> list[dict]` (status=pending newest first)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reviews_spec2a1.py`:

```python
import sqlite3
from dashboard import product_reviews as pr


def _cx():
    return sqlite3.connect(":memory:")


def test_upsert_and_done_state():
    cx = _cx()
    rid = pr.upsert_review(cx, "longevity", "a@x.com", "Ann", 5, body="great")
    assert isinstance(rid, int)
    assert pr.has_reviewed(cx, "longevity", "a@x.com") is True
    assert pr.has_reviewed(cx, "longevity", "b@x.com") is False
    # re-submit updates the same row (UNIQUE slug,email) and re-moderates
    rid2 = pr.upsert_review(cx, "longevity", "a@x.com", "Ann", 4, body="updated")
    assert rid2 == rid
    row = pr.get_review(cx, rid)
    assert row["rating"] == 4 and row["body"] == "updated" and row["status"] == "pending"


def test_ai_points_status_feature_setters():
    cx = _cx()
    rid = pr.upsert_review(cx, "x", "a@x.com", "Ann", 5, body="b")
    pr.set_ai_result(cx, rid, 2, "ok", 1)
    pr.set_points(cx, rid, 2)
    pr.set_status(cx, rid, "approved", by="Glen")
    pr.set_featured(cx, rid, True)
    r = pr.get_review(cx, rid)
    assert r["ai_score"] == 2 and r["points_awarded"] == 2
    assert r["status"] == "approved" and r["reviewed_by"] == "Glen" and r["featured"] == 1


def test_approved_for_slug_and_aggregate_exclude_unapproved():
    cx = _cx()
    a = pr.upsert_review(cx, "x", "a@x.com", "Ann", 4, body="a")
    b = pr.upsert_review(cx, "x", "b@x.com", "Bob", 2, body="b")
    c = pr.upsert_review(cx, "x", "c@x.com", "Cy", 5, body="c")
    pr.set_status(cx, a, "approved"); pr.set_status(cx, c, "approved")
    pr.set_status(cx, b, "rejected")
    pr.set_featured(cx, c, True)
    rows = pr.approved_for_slug(cx, "x")
    assert [r["email"] for r in rows] == ["c@x.com", "a@x.com"]   # featured first
    agg = pr.aggregate(cx, "x")
    assert agg["count"] == 2 and agg["avg"] == 4.5                 # (4+5)/2
    assert pr.aggregate(cx, "none") == {"count": 0, "avg": 0.0}


def test_pending_queue_only_pending():
    cx = _cx()
    a = pr.upsert_review(cx, "x", "a@x.com", "Ann", 4, body="a")
    b = pr.upsert_review(cx, "x", "b@x.com", "Bob", 5, body="b")
    pr.set_status(cx, a, "approved")
    q = pr.pending_queue(cx)
    assert [r["email"] for r in q] == ["b@x.com"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run the Global-Constraints test command.
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.product_reviews'`.

- [ ] **Step 3: Implement `dashboard/product_reviews.py`**

```python
import datetime


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS product_reviews ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, product_slug TEXT, email TEXT, name TEXT, "
        "rating INTEGER NOT NULL, body TEXT DEFAULT '', video_kind TEXT DEFAULT '', "
        "video_ref TEXT DEFAULT '', ai_score INTEGER DEFAULT 0, ai_verdict TEXT DEFAULT '', "
        "ai_recommend_publish INTEGER DEFAULT 0, points_awarded INTEGER DEFAULT 0, "
        "status TEXT DEFAULT 'pending', featured INTEGER DEFAULT 0, created_at TEXT, "
        "reviewed_at TEXT, reviewed_by TEXT, UNIQUE(product_slug, email))")
    cx.commit()


def _row(cx, where, args):
    cx.row_factory = __import__("sqlite3").Row
    r = cx.execute(f"SELECT * FROM product_reviews WHERE {where}", args).fetchone()
    return dict(r) if r else None


def has_reviewed(cx, slug, email):
    init_table(cx)
    e = (email or "").strip().lower()
    return cx.execute("SELECT 1 FROM product_reviews WHERE product_slug=? AND email=?",
                      (slug, e)).fetchone() is not None


def upsert_review(cx, slug, email, name, rating, body="", video_kind="", video_ref=""):
    init_table(cx)
    e = (email or "").strip().lower()
    now = _now()
    cx.execute(
        "INSERT INTO product_reviews (product_slug, email, name, rating, body, video_kind, "
        "video_ref, status, created_at) VALUES (?,?,?,?,?,?,?,'pending',?) "
        "ON CONFLICT(product_slug, email) DO UPDATE SET name=excluded.name, rating=excluded.rating, "
        "body=excluded.body, video_kind=excluded.video_kind, video_ref=excluded.video_ref, "
        "status='pending', ai_score=0, ai_verdict='', ai_recommend_publish=0, points_awarded=0, "
        "featured=0, reviewed_at='', reviewed_by=''",
        (slug, e, name or "", int(rating), body or "", video_kind or "", video_ref or "", now))
    cx.commit()
    return cx.execute("SELECT id FROM product_reviews WHERE product_slug=? AND email=?",
                      (slug, e)).fetchone()[0]


def set_ai_result(cx, review_id, ai_score, ai_verdict, recommend_publish):
    init_table(cx)
    cx.execute("UPDATE product_reviews SET ai_score=?, ai_verdict=?, ai_recommend_publish=? WHERE id=?",
               (int(ai_score), ai_verdict or "", 1 if recommend_publish else 0, review_id))
    cx.commit()


def set_points(cx, review_id, points):
    init_table(cx)
    cx.execute("UPDATE product_reviews SET points_awarded=? WHERE id=?", (int(points), review_id))
    cx.commit()


def set_status(cx, review_id, status, by=""):
    init_table(cx)
    cx.execute("UPDATE product_reviews SET status=?, reviewed_at=?, reviewed_by=? WHERE id=?",
               (status, _now(), by or "", review_id))
    cx.commit()


def set_featured(cx, review_id, on):
    init_table(cx)
    cx.execute("UPDATE product_reviews SET featured=? WHERE id=?", (1 if on else 0, review_id))
    cx.commit()


def get_review(cx, review_id):
    init_table(cx)
    return _row(cx, "id=?", (review_id,))


def approved_for_slug(cx, slug):
    init_table(cx)
    cx.row_factory = __import__("sqlite3").Row
    rows = cx.execute(
        "SELECT * FROM product_reviews WHERE product_slug=? AND status='approved' "
        "ORDER BY featured DESC, created_at DESC, id DESC", (slug,)).fetchall()
    return [dict(r) for r in rows]


def aggregate(cx, slug):
    init_table(cx)
    row = cx.execute(
        "SELECT COUNT(*), AVG(rating) FROM product_reviews WHERE product_slug=? AND status='approved'",
        (slug,)).fetchone()
    n = row[0] or 0
    return {"count": n, "avg": round(row[1], 1) if n else 0.0}


def pending_queue(cx, limit=100):
    init_table(cx)
    cx.row_factory = __import__("sqlite3").Row
    rows = cx.execute(
        "SELECT * FROM product_reviews WHERE status='pending' ORDER BY created_at DESC, id DESC LIMIT ?",
        (limit,)).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run the test command. Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/product_reviews.py tests/test_reviews_spec2a1.py
git commit -m "feat(reviews-2a1): product_reviews data layer (done-state, status, aggregate)"
```

---

### Task 2: AI scoring + compliance gate

**Files:**
- Create: `dashboard/review_scoring.py`
- Test: `tests/test_reviews_spec2a1.py` (append)

**Interfaces:**
- Consumes: a `client` with `.messages.create(model=, max_tokens=, system=, messages=)` returning `obj.content=[block(type='text', text=...)]`.
- Produces:
  - `build_review_prompt(product, body) -> (system, user)` (pure)
  - `score_review(client, product, body, *, strip=lambda s: s) -> {"compliance_ok": bool, "reasons": str, "quality_points": int(0..2), "recommend_publish": bool}` — parses the model's JSON; on any parse/compliance miss returns a safe default `{"compliance_ok": False, "reasons": "...", "quality_points": 0, "recommend_publish": False}`.

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


def test_build_prompt_mentions_product_and_rules():
    system, user = rs.build_review_prompt({"name": "Longevity"}, "helped my energy")
    assert "Longevity" in user and "helped my energy" in user
    assert "structure" in (system + user).lower() or "disease" in (system + user).lower()


def test_score_review_quality_points():
    c = _FakeClient('{"compliance_ok": true, "reasons": "specific", "quality_points": 2, "recommend_publish": true}')
    out = rs.score_review(c, {"name": "X"}, "Detailed, specific, authentic review of my experience.")
    assert out["compliance_ok"] is True and out["quality_points"] == 2 and out["recommend_publish"] is True


def test_score_review_gates_disease_claim():
    c = _FakeClient('{"compliance_ok": false, "reasons": "disease claim", "quality_points": 0, "recommend_publish": false}')
    out = rs.score_review(c, {"name": "X"}, "This cured my cancer in two weeks!")
    assert out["compliance_ok"] is False and out["quality_points"] == 0


def test_score_review_bad_json_safe_default():
    c = _FakeClient("not json at all")
    out = rs.score_review(c, {"name": "X"}, "whatever")
    assert out["compliance_ok"] is False and out["quality_points"] == 0


def test_score_review_strips_dashes_in_reasons():
    c = _FakeClient('{"compliance_ok": true, "reasons": "good — solid", "quality_points": 1, "recommend_publish": true}')
    out = rs.score_review(c, {"name": "X"}, "ok", strip=lambda s: s.replace("—", ","))
    assert "—" not in out["reasons"]
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.review_scoring'`.

- [ ] **Step 3: Implement `dashboard/review_scoring.py`**

```python
import json

_MODEL = "claude-haiku-4-5-20251001"

_COMPLIANCE = (
    "Reject (compliance_ok=false) any review that claims to diagnose, treat, cure, or prevent a "
    "disease, names a disease as cured/healed, contains personal contact info (PII), is spam, or is "
    "abusive. Otherwise compliance_ok=true. Use structure/function framing for what is acceptable."
)


def build_review_prompt(product, body):
    name = (product or {}).get("name", "")
    system = (
        "You score short customer product reviews for Dr. Glen Swartwout's supplements. "
        "Return ONLY a JSON object with keys: compliance_ok (bool), reasons (short string), "
        "quality_points (integer 0, 1, or 2), recommend_publish (bool). "
        "quality_points rewards specificity, authenticity, and usefulness to other shoppers, "
        "NOT length or keyword stuffing. " + _COMPLIANCE)
    user = (f"Product: {name}\n\nReview:\n{body or '(no written review)'}\n\n"
            "Return only the JSON object, no prose.")
    return system, user


def _safe_default(reasons):
    return {"compliance_ok": False, "reasons": reasons, "quality_points": 0, "recommend_publish": False}


def score_review(client, product, body, *, strip=lambda s: s):
    system, user = build_review_prompt(product, body)
    try:
        msg = client.messages.create(model=_MODEL, max_tokens=300, system=system,
                                      messages=[{"role": "user", "content": user}])
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
        text = text.strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            return _safe_default("unparseable scoring response")
        data = json.loads(text[start:end + 1])
        return {
            "compliance_ok": bool(data.get("compliance_ok")),
            "reasons": strip(str(data.get("reasons", "")))[:500],
            "quality_points": max(0, min(2, int(data.get("quality_points", 0)))),
            "recommend_publish": bool(data.get("recommend_publish")),
        }
    except Exception as e:  # noqa: BLE001
        return _safe_default(f"scoring error: {e}")
```

- [ ] **Step 4: Run to verify they pass**

Run the test command. Expected: all Task-1 + Task-2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add dashboard/review_scoring.py tests/test_reviews_spec2a1.py
git commit -m "feat(reviews-2a1): AI review scoring + compliance gate (haiku, safe defaults)"
```

---

### Task 3: Submit API + buyer-verify + points + upload

**Files:**
- Modify: `app.py` (flag, `POST /api/reviews`, upload serve route, startup configure)
- Test: `tests/test_reviews_spec2a1.py` (append)

**Interfaces:**
- Consumes: Task 1 `product_reviews.*`; Task 2 `review_scoring.score_review`; `points.credit/init_points_table/balance`; `orders.list_orders_by_email`; app globals `_resolve_buy_slug`, `_get_product`, `_cl`, `_strip_dash`, `get_authenticated_user`, `LOG_DB`, `DATA_DIR`.
- Produces:
  - flag `_REVIEWS_ENABLED = os.environ.get("REVIEWS_ENABLED","").strip().lower() in ("1","true","yes")`
  - `_review_verified_buyer(cx, email, slug) -> bool`
  - `POST /api/reviews` (form or JSON: `slug, rating, body?, video_kind?, video_url?` + optional file `video`); returns `{ok, review_id, points_awarded, status}`
  - `GET /review-media/<slug>/<filename>` (gated serve route)
  - module-level `_REVIEW_MEDIA_DIR = DATA_DIR / "review-media"`

- [ ] **Step 1: Write the failing tests**

Append (uses a reload helper + a fake `_cl`):

```python
import importlib


def _reload_reviews_app(monkeypatch, tmp_path, enabled="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REVIEWS_ENABLED", enabled)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _seed_paid_order(appmod, email, product_name):
    import sqlite3
    from dashboard import orders as o
    with sqlite3.connect(appmod.LOG_DB) as cx:
        o.upsert_order(cx, email=email, items=[{"name": product_name, "qty": 1}],
                       total_cents=7000, status="paid")


def test_submit_requires_verified_buyer(monkeypatch, tmp_path):
    appmod = _reload_reviews_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    c = appmod.app.test_client()
    r = c.post("/api/reviews", json={"slug": slug, "rating": 5, "body": "great",
                                     "email": "stranger@x.com"})
    assert r.status_code == 403


def test_submit_scores_and_credits_points(monkeypatch, tmp_path):
    appmod = _reload_reviews_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    name = appmod._get_product(slug)["name"]
    _seed_paid_order(appmod, "buyer@x.com", name)
    # fake the scorer to pass with 2 points
    from dashboard import review_scoring as rs
    monkeypatch.setattr(rs, "score_review", lambda *a, **k: {
        "compliance_ok": True, "reasons": "ok", "quality_points": 2, "recommend_publish": True})
    c = appmod.app.test_client()
    r = c.post("/api/reviews", json={"slug": slug, "rating": 5, "body": "specific and useful",
                                     "email": "buyer@x.com"}).get_json()
    assert r["ok"] and r["points_awarded"] == 2 and r["status"] == "pending"
    import sqlite3
    from dashboard import points
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "buyer@x.com") == 200   # 2 points * 100c
    # idempotent: re-submit does not double-credit
    c.post("/api/reviews", json={"slug": slug, "rating": 4, "body": "edit", "email": "buyer@x.com"})
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "buyer@x.com") == 200


def test_submit_gate_fail_no_points(monkeypatch, tmp_path):
    appmod = _reload_reviews_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    name = appmod._get_product(slug)["name"]
    _seed_paid_order(appmod, "buyer@x.com", name)
    from dashboard import review_scoring as rs
    monkeypatch.setattr(rs, "score_review", lambda *a, **k: {
        "compliance_ok": False, "reasons": "disease claim", "quality_points": 0, "recommend_publish": False})
    c = appmod.app.test_client()
    r = c.post("/api/reviews", json={"slug": slug, "rating": 5, "body": "cured my disease",
                                     "email": "buyer@x.com"}).get_json()
    assert r["ok"] and r["points_awarded"] == 0
    import sqlite3
    from dashboard import points
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "buyer@x.com") == 0


def test_submit_404_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_reviews_app(monkeypatch, tmp_path, enabled="false")
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    r = appmod.app.test_client().post("/api/reviews", json={"slug": slug, "rating": 5})
    assert r.status_code == 404
```

(If `orders.upsert_order`'s signature differs, the implementer adjusts `_seed_paid_order` to the real signature — confirm via `grep -n "def upsert_order" dashboard/orders.py`. The required outcome: an order row for that email with `status='paid'` and an item named `product_name`.)

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (route 404 / missing flag).

- [ ] **Step 3: Implement in `app.py`**

Add the flag near the other `_SALES_*` flags (search `_SALES_PAGES_ENABLED =`):

```python
_REVIEWS_ENABLED = os.environ.get("REVIEWS_ENABLED", "").strip().lower() in ("1", "true", "yes")
_REVIEW_MEDIA_DIR = DATA_DIR / "review-media"
```

Add the verify helper + routes (place after the `begin_product_image` route, search `def begin_product_image`):

```python
_REVIEW_PAID_STATUSES = ("paid", "new", "packed", "shipped", "done")


def _review_verified_buyer(cx, email, slug):
    from dashboard import orders as _o
    cx.row_factory = sqlite3.Row
    for o in _o.list_orders_by_email(cx, email):
        if (o.get("status") or "").lower() not in _REVIEW_PAID_STATUSES:
            continue
        for it in (o.get("items") or []):
            if _resolve_buy_slug((it.get("name") or "").strip()) == slug:
                return True
    return False


@app.route("/api/reviews", methods=["POST"])
def api_submit_review():
    if not _REVIEWS_ENABLED:
        return ("", 404)
    data = request.get_json(silent=True) or request.form
    slug = (data.get("slug") or "").strip()
    p = _get_product(slug)
    if not p:
        return jsonify({"ok": False, "error": "unknown product"}), 404
    au = get_authenticated_user(request) or {}
    email = ((data.get("email") or au.get("email") or "")).strip().lower()
    name = (data.get("name") or au.get("name") or "").strip()
    try:
        rating = int(data.get("rating") or 0)
    except (TypeError, ValueError):
        rating = 0
    if not email or rating < 1 or rating > 5:
        return jsonify({"ok": False, "error": "email and rating 1-5 required"}), 400
    body = (data.get("body") or "").strip()

    # optional video: link (video_url) or upload (file 'video')
    video_kind, video_ref = "", ""
    f = request.files.get("video") if request.files else None
    if f and f.filename:
        safe = re.sub(r"[^\w.\-]", "_", f.filename)[-80:]
        d = _REVIEW_MEDIA_DIR / slug
        d.mkdir(parents=True, exist_ok=True)
        f.save(str(d / safe))
        video_kind, video_ref = "upload", safe
    elif (data.get("video_url") or "").strip():
        video_kind, video_ref = "link", (data.get("video_url") or "").strip()[:500]

    from dashboard import product_reviews as _pr
    from dashboard import review_scoring as _rs
    from dashboard import points as _points
    with sqlite3.connect(LOG_DB) as cx:
        if not _review_verified_buyer(cx, email, slug):
            return jsonify({"ok": False, "error": "no verified purchase"}), 403
        rid = _pr.upsert_review(cx, slug, email, name, rating, body, video_kind, video_ref)
        score = _rs.score_review(_cl, p, body, strip=_strip_dash) if body else {
            "compliance_ok": True, "reasons": "", "quality_points": 0, "recommend_publish": False}
        _pr.set_ai_result(cx, rid, score["quality_points"], score["reasons"], score["recommend_publish"])
        pts = min(5, score["quality_points"]) if score["compliance_ok"] else 0
        if pts > 0:
            try:
                _points.init_points_table(cx)
                _points.credit(cx, email, value_cents=pts * 100,
                               reason=f"review:{slug}", order_ref=f"review:{rid}")
            except Exception as e:  # noqa: BLE001 - points never block the submit
                print(f"[reviews] points credit failed rid={rid}: {e}", flush=True)
                pts = 0
        _pr.set_points(cx, rid, pts)
        return jsonify({"ok": True, "review_id": rid, "points_awarded": pts, "status": "pending"})


@app.route("/review-media/<slug>/<filename>")
def review_media(slug, filename):
    if not re.match(r'^[\w.\-]+$', filename):
        return ("", 404)
    d = _REVIEW_MEDIA_DIR / slug
    if not (d / filename).exists():
        return ("", 404)
    return send_from_directory(str(d), filename)
```

- [ ] **Step 4: Run to verify they pass**

Run the test command. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_reviews_spec2a1.py
git commit -m "feat(reviews-2a1): submit API, buyer-verify, AI-scored points auto-credit, upload"
```

---

### Task 4: Reorder star-gate

**Files:**
- Modify: `app.py` (`api_reorder_items` annotation)
- Modify: `static/reorder.html` (star input + submit) — confirm the actual reorder page filename via `grep -n "reorder" app.py | grep send_from_directory`; if the page is inline, add the JS where the reorder items render.
- Test: `tests/test_reviews_spec2a1.py` (append)

**Interfaces:**
- Consumes: `product_reviews.has_reviewed`.
- Produces: each item in `GET /api/reorder/items` gains `"reviewed": bool` (true when the buyer already reviewed that slug, or when `REVIEWS_ENABLED` is off so the gate is inert).

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_reorder_items_annotated_with_reviewed(monkeypatch, tmp_path):
    appmod = _reload_reviews_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    name = appmod._get_product(slug)["name"]
    _seed_paid_order(appmod, "buyer@x.com", name)
    import sqlite3
    from dashboard import product_reviews as pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        pr.upsert_review(cx, slug, "buyer@x.com", "B", 5, body="done")
    c = appmod.app.test_client()
    c.set_cookie("rm_reorder_email", "buyer@x.com")  # confirm the real cookie name used by _reorder_email_from_cookie
    items = c.get("/api/reorder/items").get_json()["items"]
    target = next((it for it in items if it.get("slug") == slug), None)
    assert target is not None and target["reviewed"] is True
```

(The implementer MUST confirm the reorder identity cookie name by reading `_reorder_email_from_cookie` and use it in `set_cookie`.)

- [ ] **Step 2: Run to verify it fails**

Run the test command. Expected: FAIL (`reviewed` key missing).

- [ ] **Step 3: Implement**

In `api_reorder_items` (app.py ~9160), after each `items.append({...})`, annotate the appended dict. Replace the `items.append({...})` block so each item includes `reviewed`:

```python
            reviewed = True
            if _REVIEWS_ENABLED and (p and p.get("slug")):
                from dashboard import product_reviews as _pr
                reviewed = _pr.has_reviewed(cx, p["slug"], email)
            items.append({
                "name": nm,
                "slug": (p.get("slug") if p else None),
                "current_price_cents": (p.get("price_cents") if p else None),
                "current_price": (f"${p['price_cents'] / 100:.2f}" if p else None),
                "qty": int(it.get("qty") or 1),
                "last_ordered": (o.get("created_at") or "")[:10],
                "available": bool(p),
                "reviewed": reviewed,
            })
```

(The `cx` is open in `api_reorder_items` — confirm it stays in scope for the `has_reviewed` call; if the `with` block closed, move the annotation inside it.)

Front-end (manual visual pass — note in the report): in the reorder page template, when an item has `reviewed === false`, render a required 1–5 star input and POST `{slug, rating}` to `/api/reviews` before allowing that line into the cart. Keep it minimal; NO emoji (use `★`/`☆` text glyphs).

- [ ] **Step 4: Run to verify it passes**

Run the test command. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app.py static/ tests/test_reviews_spec2a1.py
git commit -m "feat(reviews-2a1): reorder star-gate (reviewed annotation + star input)"
```

---

### Task 5: Moderation actions + console

**Files:**
- Create: `dashboard/reviews_actions.py`
- Create: `static/console-reviews.html`
- Modify: `app.py` (register/configure at startup; `GET /api/console/reviews`; `/console/reviews` serve route)
- Modify: `static/op-nav.js` (BOS sub-tab after `sales`)
- Test: `tests/test_reviews_spec2a1.py` (append)

**Interfaces:**
- Consumes: `product_reviews.{set_status,set_featured,pending_queue,get_review}`; `actions.{Action,register_action,get_action,LOW_WRITE}`; `rbac.{OWNER,OPS}`; `dispatch.dispatch_action`; `_sales_console_ok`.
- Produces: actions `reviews.approve {id}`, `reviews.reject {id}`, `reviews.feature {id, on}` (LOW_WRITE, perm (OWNER,OPS)); `register()` idempotent; `GET /api/console/reviews -> {ok, pending:[...], recent:[...]}`.

- [ ] **Step 1: Write the failing tests**

Append:

```python
from dashboard.rbac import Actor, OWNER


def test_review_actions_approve_reject_feature(monkeypatch, tmp_path):
    appmod = _reload_reviews_app(monkeypatch, tmp_path)
    import sqlite3
    from dashboard import product_reviews as pr
    from dashboard import dispatch as d
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr.upsert_review(cx, "x", "a@x.com", "Ann", 5, body="b")
        res = d.dispatch_action(cx, "reviews.approve", {"id": rid},
                                Actor(role=OWNER, name="Glen"), source="panel")
        assert res["status"] == "done" and pr.get_review(cx, rid)["status"] == "approved"
        d.dispatch_action(cx, "reviews.feature", {"id": rid, "on": True},
                          Actor(role=OWNER, name="Glen"), source="panel")
        assert pr.get_review(cx, rid)["featured"] == 1


def test_console_reviews_endpoint(monkeypatch, tmp_path):
    appmod = _reload_reviews_app(monkeypatch, tmp_path)
    import dashboard as _d
    _d.CONSOLE_SECRET = ""  # pass-through when unset
    import sqlite3
    from dashboard import product_reviews as pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        pr.upsert_review(cx, "x", "a@x.com", "Ann", 5, body="b")
    body = appmod.app.test_client().get("/api/console/reviews").get_json()
    assert body["ok"] and any(r["email"] == "a@x.com" for r in body["pending"])
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (unknown action / route 404).

- [ ] **Step 3: Implement `dashboard/reviews_actions.py`** (mirror `sales_pages_actions.py`)

```python
"""Phase 2a-1 console actions for product-review moderation: approve / reject / feature."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import product_reviews as _pr


def _name(actor):
    return (getattr(actor, "name", "") or getattr(actor, "role", "") or "console")


def _exec_approve(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    _pr.set_status(ctx["cx"], rid, "approved", by=_name(ctx.get("actor")))
    return {"id": rid, "status": "approved"}


def _exec_reject(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    _pr.set_status(ctx["cx"], rid, "rejected", by=_name(ctx.get("actor")))
    return {"id": rid, "status": "rejected"}


def _exec_feature(params, ctx):
    rid = int(params.get("id") or 0)
    if not rid:
        raise ValueError("id required")
    _pr.set_featured(ctx["cx"], rid, bool(params.get("on")))
    return {"id": rid, "featured": bool(params.get("on"))}


def register():
    if get_action("reviews.approve"):
        return
    register_action(Action(key="reviews.approve", module="reviews", title="Approve review",
        description="Publish a product review on its sales page.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_approve))
    register_action(Action(key="reviews.reject", module="reviews", title="Reject review",
        description="Hide a product review from the sales page.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_reject))
    register_action(Action(key="reviews.feature", module="reviews", title="Feature review",
        description="Pin/unpin a product review at the top of its section.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_feature))
```

In `app.py`, after the Phase-5 `_spa.register()` block, add:

```python
from dashboard import reviews_actions as _ra
_ra.register()
```

Add the console routes after `console_sales_pages_page` (search `def console_sales_pages_page`):

```python
@app.route("/console/reviews")
def console_reviews_page():
    resp = send_from_directory(STATIC, "console-reviews.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/console/reviews", methods=["GET"])
def api_console_reviews_list():
    bad = _sales_console_ok()
    if bad:
        return bad
    from dashboard import product_reviews as _pr
    with sqlite3.connect(LOG_DB) as cx:
        pending = _pr.pending_queue(cx)
    for r in pending:
        r["product_name"] = (_get_product(r["product_slug"]) or {}).get("name", r["product_slug"])
        if r.get("video_kind") == "upload" and r.get("video_ref"):
            r["video_url"] = f"/review-media/{r['product_slug']}/{r['video_ref']}"
        elif r.get("video_kind") == "link":
            r["video_url"] = r.get("video_ref")
    return jsonify({"ok": True, "pending": pending, "recent": []})
```

Create `static/console-reviews.html` modeled on `static/console-sales-pages.html` (gate/`key()`/`api()` with `X-Console-Key`; `op-nav.js` `data-active="bos" data-sub="reviews"`). It lists the pending queue rows showing product, star rating, written body, video link (if any), the **AI publish recommendation + reasons** (`ai_recommend_publish`, `ai_verdict`), and **points already awarded**; each row has Approve / Reject / Feature buttons POSTing to `/api/action/reviews.<key>`. NO emoji (use `★`/text glyphs).

Add to `static/op-nav.js` after the `sales` BOS sub-tab entry: `{ id: "reviews", label: "Reviews", href: "/console/reviews" + qs },` and add `"reviews"` to the valid-`data-sub` comment.

- [ ] **Step 4: Run to verify they pass**

Run the test command. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add dashboard/reviews_actions.py static/console-reviews.html static/op-nav.js app.py tests/test_reviews_spec2a1.py
git commit -m "feat(reviews-2a1): moderation actions + /console/reviews queue + nav"
```

---

### Task 6: Display — testimonials section + UGC videos

**Files:**
- Modify: `app.py` (`begin_product_page_data` — add reviews section + UGC videos)
- Modify: `static/begin-product.html` (renderer)
- Test: `tests/test_reviews_spec2a1.py` (append)

**Interfaces:**
- Consumes: `product_reviews.{approved_for_slug, aggregate}`.
- Produces: a new section dict `{"id":"reviews","title":"What people are saying","default_open":False,"body":{...}}` inserted after `research`; body = `{"aggregate":{"count","avg"}, "reviews":[{"name","rating","body"}], "disclaimer":"Individual results vary."}`. Approved videos appended to the video section as `{src,title,provider,kind:"ugc"}`.

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_page_data_reviews_section_only_approved(monkeypatch, tmp_path):
    appmod = _reload_reviews_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    import sqlite3
    from dashboard import product_reviews as pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        a = pr.upsert_review(cx, slug, "a@x.com", "Ann", 5, body="approved one")
        pr.upsert_review(cx, slug, "b@x.com", "Bob", 1, body="pending one")
        pr.set_status(cx, a, "approved")
    data = appmod.app.test_client().get(f"/begin/product-page-data/{slug}").get_json()
    sec = next((s for s in data["sections"] if s["id"] == "reviews"), None)
    assert sec is not None
    bodies = [r["body"] for r in sec["body"]["reviews"]]
    assert "approved one" in bodies and "pending one" not in bodies
    assert sec["body"]["aggregate"]["count"] == 1 and sec["body"]["aggregate"]["avg"] == 5.0
    assert "Individual results vary" in sec["body"]["disclaimer"]
    # section order: reviews comes right after research
    ids = [s["id"] for s in data["sections"]]
    assert ids.index("reviews") == ids.index("research") + 1


def test_page_data_no_reviews_section_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_reviews_app(monkeypatch, tmp_path, enabled="false")
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    data = appmod.app.test_client().get(f"/begin/product-page-data/{slug}").get_json()
    assert all(s["id"] != "reviews" for s in data["sections"])
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (no `reviews` section).

- [ ] **Step 3: Implement in `begin_product_page_data`**

After the `sections = [...]` list is built (and before the `_SALES_AI_COPY_ENABLED` block), insert the reviews section after `research` when the flag is on:

```python
    if _REVIEWS_ENABLED:
        from dashboard import product_reviews as _pr
        try:
            with sqlite3.connect(LOG_DB) as _rcx:
                _agg = _pr.aggregate(_rcx, slug)
                _approved = _pr.approved_for_slug(_rcx, slug)
            _revs = [{"name": (r.get("name") or "A verified buyer"),
                      "rating": r.get("rating"), "body": r.get("body") or ""}
                     for r in _approved if (r.get("body") or "").strip()]
            _rsec = {"id": "reviews", "title": "What people are saying", "default_open": False,
                     "body": {"aggregate": _agg, "reviews": _revs,
                              "disclaimer": "Individual results vary."}}
            _ri = next((i for i, s in enumerate(sections) if s["id"] == "research"), len(sections) - 1)
            sections.insert(_ri + 1, _rsec)
            # approved UGC videos -> existing video section
            _ugc = [r for r in _approved if (r.get("video_kind") in ("link", "upload")) and r.get("video_ref")]
            if _ugc:
                _vsec = next((s for s in sections if s["id"] == "video"), None)
                if _vsec and isinstance(_vsec.get("body"), dict):
                    for r in _ugc:
                        _src = (r["video_ref"] if r["video_kind"] == "link"
                                else f"/review-media/{slug}/{r['video_ref']}")
                        _prov = "link" if r["video_kind"] == "link" else "mp4"
                        _vsec["body"].setdefault("videos", []).append(
                            {"src": _src, "title": f"Review from {r.get('name') or 'a verified buyer'}",
                             "provider": _prov, "kind": "ugc"})
        except Exception as _e:
            print(f"[reviews] page-data section skipped: {_e}", flush=True)
```

Front-end (`static/begin-product.html`): add a `renderReviewsBody(body)` that shows `aggregate.avg` as `★` glyphs + `(count)`, lists each review (name + `★` rating + escaped body), and the disclaimer; wire it in `buildSectionBody` for `sec.id === 'reviews'`. NO emoji. (Visual = manual pass; note it in the report.)

- [ ] **Step 4: Run to verify they pass**

Run the test command, then the full sales regression sweep:
`doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/ -k "reviews or sales" -q`
Expected: all pass, no regressions.

- [ ] **Step 5: Commit**

```bash
git add app.py static/begin-product.html tests/test_reviews_spec2a1.py
git commit -m "feat(reviews-2a1): testimonials section + UGC videos on the sales page"
```

---

### Task 7: Post-purchase email review link (thin)

**Files:**
- Modify: `app.py` (tokenized review link mint/resolve + send hook)
- Test: `tests/test_reviews_spec2a1.py` (append)

**Interfaces:**
- Consumes: the existing token pattern used by `/invoice/<token>` — confirm via `grep -n "invoice" app.py | grep -i token`. Reuse the same mint/verify helpers (e.g. itsdangerous signer or the existing token table) rather than inventing a new scheme.
- Produces: `_review_link(email, slug) -> str` (absolute URL with a signed token); `GET /review/<token>` resolves to the review form pre-bound to (email, slug); `_send_review_invite(email, name, slug)` sends via `dashboard/inbox.send_email` as Dr. Glen.

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_review_token_roundtrip(monkeypatch, tmp_path):
    appmod = _reload_reviews_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    tok = appmod._review_token_mint("buyer@x.com", slug)
    email, got_slug = appmod._review_token_verify(tok)
    assert email == "buyer@x.com" and got_slug == slug
    assert appmod._review_token_verify("garbage") == (None, None)
```

- [ ] **Step 2: Run to verify it fails**

Run the test command. Expected: FAIL (`_review_token_mint` undefined).

- [ ] **Step 3: Implement** mint/verify reusing the app's existing signer (confirm the signer in use — `grep -n "URLSafeSerializer\|itsdangerous\|SECRET_KEY\|serializer" app.py`). Example using `itsdangerous` (only if that's what the app already uses):

```python
def _review_token_mint(email, slug):
    from itsdangerous import URLSafeSerializer
    s = URLSafeSerializer(app.secret_key or os.environ.get("SECRET_KEY", "rm-reviews"))
    return s.dumps({"e": (email or "").strip().lower(), "s": slug}, salt="review-link")


def _review_token_verify(tok):
    from itsdangerous import URLSafeSerializer, BadData
    s = URLSafeSerializer(app.secret_key or os.environ.get("SECRET_KEY", "rm-reviews"))
    try:
        d = s.loads(tok, salt="review-link")
        return (d.get("e"), d.get("s"))
    except (BadData, Exception):
        return (None, None)


@app.route("/review/<token>")
def review_form_page(token):
    if not _REVIEWS_ENABLED:
        return ("", 404)
    email, slug = _review_token_verify(token)
    if not email or not _get_product(slug):
        return ("Invalid or expired link.", 404)
    # serve the same review form, pre-bound to (email, slug) via query params the form reads
    return redirect(f"/begin/product/{slug}?review=1&rt={token}")


def _send_review_invite(email, name, slug):
    from dashboard import inbox as _inbox
    p = _get_product(slug) or {}
    url = f"{PUBLIC_BASE_URL}/review/{_review_token_mint(email, slug)}"
    subject = f"How is your {p.get('name','order')}? Share a quick review"
    body = (f"Aloha {name or ''},\n\nIf you have a moment, I would love your honest review of "
            f"{p.get('name','your recent order')}. It helps others and earns you store credit:\n\n{url}\n\n"
            f"In wellness,\nDr. Glen & Rae")
    _inbox.send_email(email, subject, _strip_dash(body), from_name="Dr. Glen Swartwout")
```

(If the app does not use `itsdangerous`, use whatever signed-token mechanism `/invoice/<token>` uses — the test only requires `_review_token_mint`/`_review_token_verify` roundtrip + garbage→(None,None). The send hook is best-effort; wiring it into the post-purchase flow is a follow-on — leave a one-line note in the report. NO emoji, no em dashes.)

- [ ] **Step 4: Run to verify it passes**

Run the test command. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_reviews_spec2a1.py
git commit -m "feat(reviews-2a1): tokenized post-purchase review link + invite hook"
```

---

## Self-Review (plan author)

- **Spec coverage:** data layer (T1) → spec Data; AI scoring+gate (T2) → spec AI scoring; submit+points+buyer-verify+upload (T3) → spec Capture/Points; reorder star-gate (T4) → spec Capture(reorder); moderation+console (T5) → spec Moderation; display+UGC (T6) → spec Display; email entry (T7) → spec Capture(email). Flag `REVIEWS_ENABLED` woven through T3/T5/T6/T7. All spec sections covered.
- **Decisions honored:** points auto-credit on gate-pass, capped 5, 1pt=$1, written 0-2 (T3); humans gate publication only (T5 actions, points independent); verified-buyer = paid-or-later status (T3); done-state UNIQUE + no re-prompt (T1/T4); video captured not AI-scored (T3); no emoji / no em dashes.
- **Type consistency:** `product_reviews` function names + `score_review` shape + action keys `reviews.approve|reject|feature` + section id `reviews` used identically across tasks.
- **Known confirmations left to implementers (each flagged in-task):** real `orders.upsert_order` signature (T3 seed), reorder identity cookie name (T4), the app's signed-token mechanism (T7). These are confirm-then-use, not placeholders.
