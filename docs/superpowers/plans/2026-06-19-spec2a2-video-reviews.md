# Spec 2a-2 — Video Reviews Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record a review video in-browser, transcribe + AI-score it on a Render-side background worker, and credit video points (capped at 5 total) — points decoupled from a publish-risk compliance flag.

**Architecture:** Extend the merged 2a-1 reviews engine: additive `product_reviews` columns + a tiny video-job queue; a `score_video` scorer (Anthropic haiku); enqueue-on-submit + an APScheduler worker that transcribes via the existing Whisper helper, scores, and credits points; an in-browser recorder. No new external dependency.

**Tech Stack:** Python 3.11, Flask, SQLite (`chat_log.db`/`LOG_DB`), OpenAI Whisper (`journal_blueprint._whisper_transcribe`), Anthropic haiku (`_cl`, `claude-haiku-4-5-20251001`), APScheduler, `MediaRecorder` (vanilla JS), pytest.

## Global Constraints

- **Points decoupled from compliance.** `video_points` (0–5) reward genuine quality (spam/low-effort → 0); a disease-cure/PII claim does **NOT** zero points — it sets `publish_risk=True` + reasons for the human moderator. Review total = `min(5, ai_score + video_points)` where `ai_score` is the 2a-1 written points. Points auto-credit; humans gate publication only.
- **Idempotent video credit:** `order_ref=f"review:{id}:video"`, `reason=f"review:{slug}"`.
- **Length:** 90s default; 300s once the buyer has a prior successful video review (`video_points>0` AND `status='approved'`).
- **Flag:** new `REVIEWS_VIDEO` (default off), requires `REVIEWS_ENABLED`. Fully inert when off (no recorder, no enqueue, worker no-op).
- Only `video_kind='upload'` videos are scored (recordings arrive as uploads); link videos get no auto points.
- All Whisper/AI calls run in the background worker, never a web request. Each job wrapped so one failure never aborts the sweep.
- **NO emoji** (★/☆/text glyphs OK). **No em dashes** in generated text (`_strip_dash`).
- **Test command (every task):** `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_reviews_spec2a2.py -v`
- Tests reload `app` via `importlib`; the worker + scheduler registration must be idempotent/flag-guarded. Mock `_whisper_transcribe` and `score_video` in worker tests; mock the Anthropic client (`.messages.create(...) -> obj.content=[block(type='text',text=...)]`) in scorer tests.

---

### Task 1: Schema columns + data layer + video-job queue

**Files:**
- Modify: `dashboard/product_reviews.py`
- Create: `dashboard/review_video_jobs.py`
- Test: `tests/test_reviews_spec2a2.py` (create)

**Interfaces:**
- Consumes: existing `product_reviews.init_table`, `upsert_review`, `set_status`, `_now`.
- Produces:
  - `product_reviews.set_video_result(cx, review_id, video_points, transcript, status, publish_risk=0, video_verdict="")`
  - `product_reviews.has_successful_video(cx, email) -> bool`
  - `review_video_jobs.init_table(cx)`, `enqueue(cx, review_id)`, `claim_pending(cx, limit=3) -> list[int]`, `mark(cx, review_id, status)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reviews_spec2a2.py`:

```python
import sqlite3
from dashboard import product_reviews as pr
from dashboard import review_video_jobs as vj


def _cx():
    return sqlite3.connect(":memory:")


def test_video_columns_and_set_video_result():
    cx = _cx()
    rid = pr.upsert_review(cx, "x", "a@x.com", "Ann", 5, body="b", video_kind="upload", video_ref="v.webm")
    pr.set_video_result(cx, rid, 4, "great transcript", "scored", publish_risk=1, video_verdict="disease claim")
    r = pr.get_review(cx, rid)
    assert r["video_points"] == 4 and r["transcript"] == "great transcript"
    assert r["video_status"] == "scored" and r["publish_risk"] == 1 and r["video_verdict"] == "disease claim"


def test_has_successful_video():
    cx = _cx()
    rid = pr.upsert_review(cx, "x", "a@x.com", "Ann", 5, video_kind="upload", video_ref="v.webm")
    assert pr.has_successful_video(cx, "a@x.com") is False     # no points / not approved yet
    pr.set_video_result(cx, rid, 5, "t", "scored")
    assert pr.has_successful_video(cx, "a@x.com") is False     # scored but not approved
    pr.set_status(cx, rid, "approved")
    assert pr.has_successful_video(cx, "a@x.com") is True      # points + approved


def test_video_job_queue_roundtrip():
    cx = _cx()
    vj.enqueue(cx, 11)
    vj.enqueue(cx, 12)
    assert vj.claim_pending(cx) == [11, 12]
    vj.mark(cx, 11, "done")
    assert vj.claim_pending(cx) == [12]
    vj.enqueue(cx, 11)   # re-enqueue resets to pending
    assert set(vj.claim_pending(cx)) == {11, 12}
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (`set_video_result` / module `review_video_jobs` missing).

- [ ] **Step 3a: Extend `dashboard/product_reviews.py`**

In `init_table`, after the `CREATE TABLE`, add the additive columns (lazy ALTER) before `cx.commit()`:

```python
    for _col in ("video_points INTEGER DEFAULT 0", "transcript TEXT DEFAULT ''",
                 "video_status TEXT DEFAULT ''", "publish_risk INTEGER DEFAULT 0",
                 "video_verdict TEXT DEFAULT ''"):
        try:
            cx.execute(f"ALTER TABLE product_reviews ADD COLUMN {_col}")
        except sqlite3.OperationalError:
            pass
```

(Confirm `import sqlite3` is at the top of the module — it is, from the 2a-1 fix.) Append the two functions:

```python
def set_video_result(cx, review_id, video_points, transcript, status, publish_risk=0, video_verdict=""):
    init_table(cx)
    cx.execute(
        "UPDATE product_reviews SET video_points=?, transcript=?, video_status=?, "
        "publish_risk=?, video_verdict=? WHERE id=?",
        (int(video_points), transcript or "", status, 1 if publish_risk else 0,
         video_verdict or "", review_id))
    cx.commit()


def has_successful_video(cx, email):
    init_table(cx)
    e = (email or "").strip().lower()
    return cx.execute(
        "SELECT 1 FROM product_reviews WHERE email=? AND video_points>0 AND status='approved' LIMIT 1",
        (e,)).fetchone() is not None
```

- [ ] **Step 3b: Create `dashboard/review_video_jobs.py`**

```python
import datetime


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS review_video_jobs ("
        "review_id INTEGER PRIMARY KEY, status TEXT DEFAULT 'pending', "
        "enqueued_at TEXT, done_at TEXT)")
    cx.commit()


def enqueue(cx, review_id):
    init_table(cx)
    cx.execute(
        "INSERT INTO review_video_jobs (review_id, status, enqueued_at) VALUES (?,'pending',?) "
        "ON CONFLICT(review_id) DO UPDATE SET status='pending', enqueued_at=excluded.enqueued_at, done_at=NULL",
        (review_id, _now()))
    cx.commit()


def claim_pending(cx, limit=3):
    init_table(cx)
    rows = cx.execute(
        "SELECT review_id FROM review_video_jobs WHERE status='pending' "
        "ORDER BY enqueued_at LIMIT ?", (limit,)).fetchall()
    return [r[0] for r in rows]


def mark(cx, review_id, status):
    init_table(cx)
    cx.execute("UPDATE review_video_jobs SET status=?, done_at=? WHERE review_id=?",
               (status, _now(), review_id))
    cx.commit()
```

- [ ] **Step 4: Run to verify they pass**

Run the test command. Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/product_reviews.py dashboard/review_video_jobs.py tests/test_reviews_spec2a2.py
git commit -m "feat(reviews-2a2): video columns + set_video_result/has_successful_video + job queue"
```

---

### Task 2: `score_video` (AI video scoring + publish-risk flag)

**Files:**
- Modify: `dashboard/review_scoring.py`
- Test: `tests/test_reviews_spec2a2.py` (append)

**Interfaces:**
- Consumes: a `client` with `.messages.create(...) -> obj.content=[block(type='text', text=...)]`.
- Produces:
  - `build_video_prompt(product, transcript) -> (system, user)` (pure)
  - `score_video(client, product, transcript, *, strip=lambda s: s) -> {"video_points": int(0..5), "publish_risk": bool, "risk_reasons": str, "recommend_publish": bool}` — fail-closed safe default `{"video_points":0,"publish_risk":False,"risk_reasons":...,"recommend_publish":False}` on any error.

- [ ] **Step 1: Write the failing tests**

Append (reuse the `_FakeClient` pattern from the 2a-1 test file — redefine locally here):

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


def test_build_video_prompt_includes_transcript():
    system, user = rs.build_video_prompt({"name": "Longevity"}, "I felt more energy in two weeks")
    assert "Longevity" in user and "energy in two weeks" in user


def test_score_video_quality_points():
    c = _FakeClient('{"video_points": 5, "publish_risk": false, "risk_reasons": "", "recommend_publish": true}')
    out = rs.score_video(c, {"name": "X"}, "clear specific authentic spoken review")
    assert out["video_points"] == 5 and out["publish_risk"] is False and out["recommend_publish"] is True


def test_score_video_disease_claim_still_scores_but_flags_risk():
    c = _FakeClient('{"video_points": 4, "publish_risk": true, "risk_reasons": "claims to cure disease", "recommend_publish": false}')
    out = rs.score_video(c, {"name": "X"}, "this cured my illness completely")
    assert out["video_points"] == 4               # NOT zeroed
    assert out["publish_risk"] is True and "cure" in out["risk_reasons"].lower()


def test_score_video_spam_zero():
    c = _FakeClient('{"video_points": 0, "publish_risk": false, "risk_reasons": "low effort", "recommend_publish": false}')
    out = rs.score_video(c, {"name": "X"}, "uhh idk")
    assert out["video_points"] == 0


def test_score_video_bad_json_safe_default():
    out = rs.score_video(_FakeClient("not json"), {"name": "X"}, "x")
    assert out["video_points"] == 0 and out["publish_risk"] is False and out["recommend_publish"] is False


def test_score_video_strips_dashes():
    c = _FakeClient('{"video_points": 3, "publish_risk": true, "risk_reasons": "risk — here", "recommend_publish": false}')
    out = rs.score_video(c, {"name": "X"}, "x", strip=lambda s: s.replace("—", ","))
    assert "—" not in out["risk_reasons"]
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (`score_video` missing).

- [ ] **Step 3: Implement in `dashboard/review_scoring.py`** (append)

```python
_VIDEO_RISK = (
    "Set publish_risk=true with a short risk_reasons when the spoken review claims to diagnose, "
    "treat, cure, or prevent a disease, names a disease as cured, or contains personal contact info "
    "(PII). This flags a PUBLISHING risk for a human reviewer; it does NOT lower video_points."
)


def build_video_prompt(product, transcript):
    name = (product or {}).get("name", "")
    system = (
        "You score the transcript of a short spoken customer video review of Dr. Glen Swartwout's "
        "supplements. Return ONLY a JSON object with keys: video_points (integer 0..5), "
        "publish_risk (bool), risk_reasons (short string), recommend_publish (bool). "
        "video_points rewards a clear, specific, authentic spoken experience; low-effort, vague, "
        "spammy, or abusive transcripts score 0. " + _VIDEO_RISK)
    user = (f"Product: {name}\n\nVideo transcript:\n{transcript or '(empty)'}\n\n"
            "Return only the JSON object, no prose.")
    return system, user


def _safe_video_default(reasons):
    return {"video_points": 0, "publish_risk": False, "risk_reasons": reasons, "recommend_publish": False}


def score_video(client, product, transcript, *, strip=lambda s: s):
    system, user = build_video_prompt(product, transcript)
    try:
        msg = client.messages.create(model=_MODEL, max_tokens=300, system=system,
                                      messages=[{"role": "user", "content": user}])
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text").strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            return _safe_video_default("unparseable scoring response")
        import json as _json
        data = _json.loads(text[start:end + 1])
        return {
            "video_points": max(0, min(5, int(data.get("video_points", 0)))),
            "publish_risk": bool(data.get("publish_risk")),
            "risk_reasons": strip(str(data.get("risk_reasons", "")))[:500],
            "recommend_publish": bool(data.get("recommend_publish")),
        }
    except Exception as e:  # noqa: BLE001
        return _safe_video_default(f"scoring error: {e}")
```

(`_MODEL` and `import json` already exist at the top of the module from 2a-1; the inner `import json as _json` is belt-and-suspenders — if `json` is already module-imported, use it directly instead.)

- [ ] **Step 4: Run to verify they pass**

Run the test command. Expected: all Task-1 + Task-2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add dashboard/review_scoring.py tests/test_reviews_spec2a2.py
git commit -m "feat(reviews-2a2): score_video (quality points + publish-risk flag, points decoupled)"
```

---

### Task 3: `REVIEWS_VIDEO` flag + enqueue-on-submit + length limits

**Files:**
- Modify: `app.py` (flag; enqueue in `api_submit_review`; `_allowed_video_seconds`; `GET /api/reviews/limits`)
- Test: `tests/test_reviews_spec2a2.py` (append)

**Interfaces:**
- Consumes: Task 1 `review_video_jobs.enqueue`, `product_reviews.has_successful_video`; existing `api_submit_review`, `_REVIEWS_ENABLED`, `get_authenticated_user`, `LOG_DB`.
- Produces: `_REVIEWS_VIDEO` flag; review submit enqueues a video job + sets `video_status='pending'` for upload videos when the flag is on; `_allowed_video_seconds(email) -> int`; `GET /api/reviews/limits?email=` → `{"max_seconds": int}`.

- [ ] **Step 1: Write the failing tests**

Append (a reload helper that sets both flags):

```python
import importlib


def _reload_video_app(monkeypatch, tmp_path, video="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REVIEWS_ENABLED", "true")
    monkeypatch.setenv("REVIEWS_VIDEO", video)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _seed_paid_order(appmod, email, product_name):
    import sqlite3
    from dashboard import orders as o
    with sqlite3.connect(appmod.LOG_DB) as cx:
        o.upsert_order(cx, source="test", external_ref=f"t-{email}", email=email,
                       items=[{"name": product_name, "qty": 1}], total_cents=7000, status="paid")


def test_upload_review_enqueues_video_job(monkeypatch, tmp_path):
    appmod = _reload_video_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    name = appmod._get_product(slug)["name"]
    _seed_paid_order(appmod, "buyer@x.com", name)
    from dashboard import review_scoring as rs
    monkeypatch.setattr(rs, "score_review", lambda *a, **k: {
        "compliance_ok": True, "reasons": "", "quality_points": 1, "recommend_publish": True})
    c = appmod.app.test_client()
    r = c.post("/api/reviews", json={"slug": slug, "rating": 5, "body": "nice",
                                     "email": "buyer@x.com", "video_url": "",
                                     "video_kind": "upload", "video_ref": "v.webm"}).get_json()
    # NOTE: a real upload sets video_kind server-side; here we assert the enqueue path by checking the job table
    import sqlite3
    from dashboard import review_video_jobs as vj
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert r["review_id"] in vj.claim_pending(cx)
    assert r.get("video_status") == "pending"


def test_limits_endpoint_90_then_300(monkeypatch, tmp_path):
    appmod = _reload_video_app(monkeypatch, tmp_path)
    c = appmod.app.test_client()
    base = c.get("/api/reviews/limits?email=new@x.com").get_json()
    assert base["max_seconds"] == 90
    import sqlite3
    from dashboard import product_reviews as pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr.upsert_review(cx, "x", "vet@x.com", "V", 5, video_kind="upload", video_ref="v.webm")
        pr.set_video_result(cx, rid, 5, "t", "scored"); pr.set_status(cx, rid, "approved")
    up = c.get("/api/reviews/limits?email=vet@x.com").get_json()
    assert up["max_seconds"] == 300
```

(NOTE for the implementer: the cleanest way to assert the enqueue is to have `api_submit_review` accept an explicit `video_kind`/`video_ref` in JSON OR — preferred — to drive a real multipart upload. If JSON `video_kind`/`video_ref` is not honored by the existing handler, switch the test to a multipart upload with a small `.webm` file part named `video` so the server sets `video_kind='upload'` itself. Match whatever the 2a-1 handler actually reads.)

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (`/api/reviews/limits` 404; no enqueue).

- [ ] **Step 3: Implement in `app.py`**

Add the flag near `_REVIEWS_ENABLED`:

```python
_REVIEWS_VIDEO = os.environ.get("REVIEWS_VIDEO", "").strip().lower() in ("1", "true", "yes")
```

In `api_submit_review`, after the review row is created and written-scored (after `_pr.set_points(cx, rid, pts)`, before the `return jsonify(...)`), add the enqueue + status:

```python
        _video_status = ""
        if _REVIEWS_VIDEO and video_kind == "upload" and video_ref:
            from dashboard import review_video_jobs as _vj
            from dashboard import product_reviews as _pr2
            _pr2.set_video_result(cx, rid, 0, "", "pending")
            _vj.enqueue(cx, rid)
            _video_status = "pending"
        return jsonify({"ok": True, "review_id": rid, "points_awarded": pts,
                        "status": "pending", "video_status": _video_status})
```

(Replace the existing single-line `return jsonify({...})` with the block above so `video_status` is included.)

Add the limits helper + endpoint after the `review_media` route:

```python
def _allowed_video_seconds(email):
    from dashboard import product_reviews as _pr
    try:
        with sqlite3.connect(LOG_DB) as cx:
            return 300 if _pr.has_successful_video(cx, email) else 90
    except Exception:
        return 90


@app.route("/api/reviews/limits", methods=["GET"])
def api_review_limits():
    if not _REVIEWS_VIDEO:
        return jsonify({"max_seconds": 0})
    au = get_authenticated_user(request) or {}
    email = ((request.args.get("email") or au.get("email") or "")).strip().lower()
    return jsonify({"max_seconds": _allowed_video_seconds(email)})
```

- [ ] **Step 4: Run to verify they pass**

Run the test command. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_reviews_spec2a2.py
git commit -m "feat(reviews-2a2): REVIEWS_VIDEO flag, enqueue video job on upload, length-limits endpoint"
```

---

### Task 4: Background worker `_drain_review_videos` + points credit

**Files:**
- Modify: `app.py` (`_drain_review_videos`; register in `_start_scheduler`)
- Test: `tests/test_reviews_spec2a2.py` (append)

**Interfaces:**
- Consumes: Task 1 `review_video_jobs.{claim_pending,mark}`, `product_reviews.{get_review,set_video_result,set_points}`; Task 2 `review_scoring.score_video`; `journal_blueprint._whisper_transcribe`; `points.credit`; `_cl`, `_strip_dash`, `_get_product`, `_REVIEW_MEDIA_DIR`, `LOG_DB`.
- Produces: `_drain_review_videos()` (flag-guarded; no-op when off).

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_worker_scores_and_credits_delta(monkeypatch, tmp_path):
    appmod = _reload_video_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    import sqlite3
    from dashboard import product_reviews as pr, review_video_jobs as vj, points
    # seed a review whose written already credited 2, with a video file on disk
    d = appmod._REVIEW_MEDIA_DIR / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "v.webm").write_bytes(b"FAKEVIDEO")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr.upsert_review(cx, slug, "buyer@x.com", "B", 5, body="written",
                               video_kind="upload", video_ref="v.webm")
        pr.set_ai_result(cx, rid, 2, "ok", 1)          # written = 2
        pr.set_points(cx, rid, 2)
        points.init_points_table(cx); points.credit(cx, "buyer@x.com", value_cents=200,
                                                    reason=f"review:{slug}", order_ref=f"review:{rid}")
        vj.enqueue(cx, rid)
    import journal_blueprint
    monkeypatch.setattr(journal_blueprint, "_whisper_transcribe", lambda p: {"text": "great spoken review"})
    from dashboard import review_scoring as rs
    monkeypatch.setattr(rs, "score_video", lambda *a, **k: {
        "video_points": 5, "publish_risk": False, "risk_reasons": "", "recommend_publish": True})
    appmod._drain_review_videos()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        r = pr.get_review(cx, rid)
        assert r["video_points"] == 5 and r["video_status"] == "scored"
        assert r["points_awarded"] == 5                       # min(5, 2+5)
        assert points.balance(cx, "buyer@x.com") == 500        # 200 written + 300 video delta
        assert vj.claim_pending(cx) == []                      # job done
    # idempotent: re-running does not double-credit
    with sqlite3.connect(appmod.LOG_DB) as cx:
        vj.enqueue(cx, rid)
    appmod._drain_review_videos()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "buyer@x.com") == 500


def test_worker_disease_claim_still_credits_and_flags(monkeypatch, tmp_path):
    appmod = _reload_video_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    import sqlite3
    from dashboard import product_reviews as pr, review_video_jobs as vj, points
    d = appmod._REVIEW_MEDIA_DIR / slug; d.mkdir(parents=True, exist_ok=True)
    (d / "v.webm").write_bytes(b"X")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr.upsert_review(cx, slug, "b@x.com", "B", 5, video_kind="upload", video_ref="v.webm")
        pr.set_ai_result(cx, rid, 0, "", 0); pr.set_points(cx, rid, 0)
        points.init_points_table(cx); vj.enqueue(cx, rid)
    import journal_blueprint
    monkeypatch.setattr(journal_blueprint, "_whisper_transcribe", lambda p: {"text": "cured my disease"})
    from dashboard import review_scoring as rs
    monkeypatch.setattr(rs, "score_video", lambda *a, **k: {
        "video_points": 4, "publish_risk": True, "risk_reasons": "disease cure claim", "recommend_publish": False})
    appmod._drain_review_videos()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        r = pr.get_review(cx, rid)
        assert r["video_points"] == 4 and r["publish_risk"] == 1 and "disease" in r["video_verdict"].lower()
        assert points.balance(cx, "b@x.com") == 400            # credited despite risk


def test_worker_missing_file_marks_failed(monkeypatch, tmp_path):
    appmod = _reload_video_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    import sqlite3
    from dashboard import product_reviews as pr, review_video_jobs as vj
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr.upsert_review(cx, slug, "b@x.com", "B", 5, video_kind="upload", video_ref="gone.webm")
        vj.enqueue(cx, rid)
    appmod._drain_review_videos()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert pr.get_review(cx, rid)["video_status"] == "failed"


def test_worker_noop_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_video_app(monkeypatch, tmp_path, video="false")
    assert appmod._REVIEWS_VIDEO is False
    appmod._drain_review_videos()   # no raise
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (`_drain_review_videos` missing).

- [ ] **Step 3: Implement in `app.py`** (mirror `_drain_sales_image_queue`; place near it)

```python
def _drain_review_videos():
    """Scheduler job: transcribe + AI-score review videos, credit points (off web workers)."""
    if not _REVIEWS_VIDEO:
        return
    from dashboard import review_video_jobs as _vj, product_reviews as _pr
    from dashboard import review_scoring as _rs, points as _points
    import journal_blueprint as _jb
    try:
        with sqlite3.connect(LOG_DB) as cx:
            pending = _vj.claim_pending(cx, 3)
    except Exception as e:
        print(f"[reviews-video] queue read failed: {e}", flush=True); return
    for rid in pending:
        try:
            with sqlite3.connect(LOG_DB) as cx:
                r = _pr.get_review(cx, rid)
            if not r or r.get("video_kind") != "upload" or not r.get("video_ref"):
                with sqlite3.connect(LOG_DB) as cx:
                    _pr.set_video_result(cx, rid, 0, "", "failed"); _vj.mark(cx, rid, "failed")
                continue
            path = _REVIEW_MEDIA_DIR / r["product_slug"] / r["video_ref"]
            if not path.exists():
                with sqlite3.connect(LOG_DB) as cx:
                    _pr.set_video_result(cx, rid, 0, "", "failed"); _vj.mark(cx, rid, "failed")
                continue
            transcript = (_jb._whisper_transcribe(str(path)) or {}).get("text", "")
            prod = _get_product(r["product_slug"]) or {"name": r["product_slug"]}
            sc = _rs.score_video(_cl, prod, transcript, strip=_strip_dash)
            written = int(r.get("ai_score") or 0)
            total = min(5, written + sc["video_points"])
            delta = max(0, total - written)
            with sqlite3.connect(LOG_DB) as cx:
                if delta > 0:
                    try:
                        _points.init_points_table(cx)
                        _points.credit(cx, r["email"], value_cents=delta * 100,
                                       reason=f"review:{r['product_slug']}",
                                       order_ref=f"review:{rid}:video")
                    except Exception as e:  # noqa: BLE001 - points never block job completion
                        print(f"[reviews-video] credit failed rid={rid}: {e}", flush=True)
                _pr.set_points(cx, rid, total)
                _pr.set_video_result(cx, rid, sc["video_points"], transcript, "scored",
                                     publish_risk=1 if sc["publish_risk"] else 0,
                                     video_verdict=sc["risk_reasons"])
                _vj.mark(cx, rid, "done")
        except Exception as e:  # noqa: BLE001 - one job's failure never aborts the sweep
            print(f"[reviews-video] job {rid} failed: {e}", flush=True)
            try:
                with sqlite3.connect(LOG_DB) as cx:
                    _pr.set_video_result(cx, rid, 0, "", "failed"); _vj.mark(cx, rid, "failed")
            except Exception:
                pass
```

Register it in `_start_scheduler` (next to the sales-image jobs):

```python
        scheduler.add_job(_drain_review_videos, "interval", minutes=1, id="review_videos")
```

- [ ] **Step 4: Run to verify they pass**

Run the test command, then the broader sweep:
`doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/ -k "reviews" -q`
Expected: all pass, no regressions vs 2a-1.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_reviews_spec2a2.py
git commit -m "feat(reviews-2a2): _drain_review_videos worker (transcribe/score/credit) + scheduler"
```

---

### Task 5: Recording UI + console publish-risk display

**Files:**
- Modify: `static/begin-product.html` (review form recorder)
- Modify: `static/console-reviews.html` (show transcript + video score + publish-risk warning)
- Modify: `app.py` (`/api/console/reviews` exposes the new fields)
- Test: `tests/test_reviews_spec2a2.py` (append — console API field exposure)

**Interfaces:**
- Consumes: Task 1 columns via `product_reviews.pending_queue` (returns full rows including the new columns); `GET /api/reviews/limits`.
- Produces: `/api/console/reviews` rows include `video_points`, `transcript`, `publish_risk`, `video_verdict`, `video_status`.

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_console_reviews_exposes_video_fields(monkeypatch, tmp_path):
    appmod = _reload_video_app(monkeypatch, tmp_path)
    import dashboard as _d
    _d.CONSOLE_SECRET = ""
    import sqlite3
    from dashboard import product_reviews as pr
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr.upsert_review(cx, slug, "a@x.com", "Ann", 5, video_kind="upload", video_ref="v.webm")
        pr.set_video_result(cx, rid, 4, "spoken transcript", "scored", publish_risk=1, video_verdict="disease claim")
    rows = appmod.app.test_client().get("/api/console/reviews").get_json()["pending"]
    row = next(r for r in rows if r["email"] == "a@x.com")
    assert row["video_points"] == 4 and row["transcript"] == "spoken transcript"
    assert row["publish_risk"] == 1 and row["video_verdict"] == "disease claim"
```

- [ ] **Step 2: Run to verify it fails**

Run the test command. Expected: FAIL (fields absent — `pending_queue` returns full rows, so this likely passes already; if so, confirm the keys exist and move the assertion to what's missing). If `pending_queue` already returns `SELECT *`, the row includes these columns and the test passes at Step 2 — in that case this task's API change is a no-op and you only add the front-end. Verify by reading `pending_queue`.

- [ ] **Step 3: Implement**

If `/api/console/reviews` already surfaces full rows (it calls `pending_queue` which is `SELECT *`), no API change is needed beyond confirming the keys. Otherwise add the columns to the row dicts.

Front-end (manual visual pass — note in the report):
- `static/console-reviews.html`: for each pending row, when `video_status === 'scored'` show the transcript (escaped) + `video_points`; when `publish_risk` is truthy, show a prominent text warning line "PUBLISH RISK: " + escaped `video_verdict` (NO emoji; a text label or an SVG/`!` glyph).
- `static/begin-product.html` review form: add a `MediaRecorder` recorder (Record / Stop / Preview / Re-record) that first calls `GET /api/reviews/limits?email=<email>` to get `max_seconds`, auto-stops at that limit, and attaches the recorded blob as the `video` file part of the `/api/reviews` POST (filename `review.webm`). Keep link/upload options. NO emoji (use ★/☆/text).

- [ ] **Step 4: Run to verify it passes**

Run the test command, then `pytest tests/ -k "reviews" -q`. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app.py static/console-reviews.html static/begin-product.html tests/test_reviews_spec2a2.py
git commit -m "feat(reviews-2a2): in-browser recorder + console transcript/publish-risk display"
```

---

## Self-Review (plan author)

- **Spec coverage:** schema+queue (T1) → spec Schema/Queue; score_video (T2) → spec AI video scoring + decoupled compliance; flag+enqueue+limits (T3) → spec Enqueue/Length; worker+points (T4) → spec Worker/Points; recorder+console (T5) → spec Recording UI/Display. Flag `REVIEWS_VIDEO` woven through T3/T4 (+limits) and recorder.
- **Decisions honored:** points decoupled from compliance (disease-claim video credits + publish_risk flag, T2/T4 tests); total `min(5, written+video)` idempotent `review:{id}:video` (T4); 90→300 length earned by a successful video (T3); upload-only scoring; flag-off fully inert (T3/T4/T5); no emoji / no em dashes.
- **Type consistency:** `score_video` shape, `set_video_result(...)` signature, `review_video_jobs` fns, the `video_points/transcript/publish_risk/video_verdict/video_status` columns, and `_drain_review_videos` used identically across tasks.
- **Confirm-then-use flagged in-task:** the exact way `api_submit_review` reads `video_kind`/`video_ref` (T3 — prefer a real multipart upload in the test if JSON fields aren't honored); whether `pending_queue` already returns the new columns (T5 Step 2).
