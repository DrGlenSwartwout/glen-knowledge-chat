# Spec 2a-2b — Auto-trim Review Videos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-trim each scored review video to its first/last detected speech (Whisper word timestamps) via a bundled ffmpeg re-encode, gated behind `REVIEWS_VIDEO_TRIM`.

**Architecture:** A small `dashboard/video_trim.py` (window math + ffmpeg runner) + an additive `video_orig_ref` column with a `set_trimmed` setter; the existing `_drain_review_videos` worker keeps the Whisper `words`/`duration` and, after crediting points, trims and repoints `video_ref` to the trimmed copy. Non-destructive; flag-gated.

**Tech Stack:** Python 3.11, Flask, SQLite (`chat_log.db`/`LOG_DB`), `imageio-ffmpeg` (bundled static ffmpeg binary), subprocess, pytest.

## Global Constraints

- **Re-encode** (frame-accurate): `ffmpeg -y -ss <start> -to <end> -i <src> -c:v libx264 -c:a aac -movflags +faststart <dst>`. Output `.mp4`.
- **Padding:** LEAD 0.3s before first word, TRAIL 0.7s after last word, clamped to the clip.
- **Skip trim** (return `None`) when: no words, `end <= start`, or total removed `< 1.0s`.
- **Non-destructive:** keep the original file; trimmed copy is `<stem>-trim.mp4`; `video_ref` repointed to it; original preserved in `video_orig_ref` (set once).
- **Flag `REVIEWS_VIDEO_TRIM`** (default off; requires `REVIEWS_VIDEO`). Trim fully inert when off (no ffmpeg import/run).
- Trim runs only in the background worker (never a web request), wrapped so a failure leaves `video_ref` + credited points untouched and never aborts the sweep.
- `imageio_ffmpeg` imported **lazily** inside the trim helper (module import must not fail without it).
- NO emoji; no em dashes in generated text.
- **Test command (every task):** `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_reviews_spec2a2b.py -v` (the `deploy-chat311` venv does NOT have `imageio-ffmpeg`; tests must mock `_ffmpeg_exe` + the subprocess runner — never invoke real ffmpeg).

---

### Task 1: `dashboard/video_trim.py` — window math + ffmpeg runner

**Files:**
- Create: `dashboard/video_trim.py`
- Test: `tests/test_reviews_spec2a2b.py` (create)

**Interfaces:**
- Produces:
  - `compute_trim_window(words, duration, *, lead=0.3, trail=0.7, min_removed=1.0) -> tuple[float,float] | None`
  - `_ffmpeg_exe() -> str` (lazy `import imageio_ffmpeg`; tests monkeypatch this)
  - `trim_video(src_path, dst_path, start, end, *, runner=None) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reviews_spec2a2b.py`:

```python
from dashboard import video_trim as vt


def test_compute_trim_window_trims_with_padding():
    words = [{"word": "hi", "start": 2.0, "end": 2.4}, {"word": "there", "start": 2.4, "end": 9.0}]
    win = vt.compute_trim_window(words, 12.0)
    assert win == (2.0 - 0.3, 9.0 + 0.7)          # (1.7, 9.7), clamped within [0,12]


def test_compute_trim_window_clamps_to_clip():
    words = [{"word": "a", "start": 0.1, "end": 0.2}, {"word": "b", "start": 0.2, "end": 11.9}]
    win = vt.compute_trim_window(words, 12.0)
    assert win == (0.0, 12.0)                       # lead/trail clamped to clip bounds


def test_compute_trim_window_none_when_no_words():
    assert vt.compute_trim_window([], 12.0) is None


def test_compute_trim_window_none_when_negligible():
    # speech spans almost the whole clip -> removed < 1.0s -> skip
    words = [{"word": "a", "start": 0.2, "end": 0.3}, {"word": "b", "start": 0.3, "end": 11.5}]
    assert vt.compute_trim_window(words, 12.0) is None


def test_trim_video_builds_command_and_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(vt, "_ffmpeg_exe", lambda: "/fake/ffmpeg")
    src = tmp_path / "v.webm"; src.write_bytes(b"X")
    dst = tmp_path / "v-trim.mp4"
    seen = {}

    def fake_runner(cmd, **kw):
        seen["cmd"] = cmd
        dst.write_bytes(b"TRIMMED")            # simulate ffmpeg producing the output
        class R: returncode = 0
        return R()

    ok = vt.trim_video(str(src), str(dst), 1.7, 9.7, runner=fake_runner)
    assert ok is True and dst.exists()
    c = seen["cmd"]
    assert c[0] == "/fake/ffmpeg" and "-ss" in c and "1.7" in c and "-to" in c and "9.7" in c
    assert "libx264" in c and str(src) in c and str(dst) in c


def test_trim_video_false_on_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(vt, "_ffmpeg_exe", lambda: "/fake/ffmpeg")
    src = tmp_path / "v.webm"; src.write_bytes(b"X")
    dst = tmp_path / "v-trim.mp4"

    def fail_runner(cmd, **kw):
        class R: returncode = 1
        return R()

    assert vt.trim_video(str(src), str(dst), 1.0, 5.0, runner=fail_runner) is False


def test_trim_video_false_on_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(vt, "_ffmpeg_exe", lambda: "/fake/ffmpeg")
    def boom(cmd, **kw): raise OSError("no binary")
    assert vt.trim_video("a", "b", 1.0, 5.0, runner=boom) is False
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (`ModuleNotFoundError: dashboard.video_trim`).

- [ ] **Step 3: Implement `dashboard/video_trim.py`**

```python
import subprocess


def compute_trim_window(words, duration, *, lead=0.3, trail=0.7, min_removed=1.0):
    """Window (start, end) trimming dead air to first/last word +/- padding, or None to skip."""
    try:
        dur = float(duration or 0)
        if not words or dur <= 0:
            return None
        first = float(words[0].get("start", 0))
        last = float(words[-1].get("end", 0))
        start = max(0.0, first - lead)
        end = min(dur, last + trail)
        if end <= start:
            return None
        removed = start + (dur - end)
        if removed < min_removed:
            return None
        return (start, end)
    except (TypeError, ValueError, KeyError, IndexError):
        return None


def _ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def trim_video(src_path, dst_path, start, end, *, runner=None):
    """Re-encode a frame-accurate trimmed copy. Returns True on success, never raises."""
    run = runner or subprocess.run
    try:
        cmd = [_ffmpeg_exe(), "-y", "-ss", str(start), "-to", str(end), "-i", str(src_path),
               "-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", str(dst_path)]
        import os
        res = run(cmd, capture_output=True, timeout=300)
        return getattr(res, "returncode", 1) == 0 and os.path.exists(str(dst_path))
    except Exception as e:  # noqa: BLE001
        print(f"[video-trim] failed: {e}", flush=True)
        return False
```

- [ ] **Step 4: Run to verify they pass**

Run the test command. Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/video_trim.py tests/test_reviews_spec2a2b.py
git commit -m "feat(reviews-2a2b): video_trim window math + ffmpeg re-encode runner"
```

---

### Task 2: `video_orig_ref` column + `set_trimmed` + dependency

**Files:**
- Modify: `dashboard/product_reviews.py`
- Modify: `requirements.txt`
- Test: `tests/test_reviews_spec2a2b.py` (append)

**Interfaces:**
- Consumes: existing `product_reviews.init_table`, `upsert_review`, `get_review`.
- Produces: `product_reviews.set_trimmed(cx, review_id, trimmed_ref)` (sets `video_orig_ref` to the current `video_ref` once, then repoints `video_ref`).

- [ ] **Step 1: Write the failing tests**

Append:

```python
import sqlite3
from dashboard import product_reviews as pr


def test_set_trimmed_preserves_original_once():
    cx = sqlite3.connect(":memory:")
    rid = pr.upsert_review(cx, "x", "a@x.com", "Ann", 5, video_kind="upload", video_ref="orig.webm")
    pr.set_trimmed(cx, rid, "orig-trim.mp4")
    r = pr.get_review(cx, rid)
    assert r["video_ref"] == "orig-trim.mp4" and r["video_orig_ref"] == "orig.webm"
    # a second trim keeps the FIRST original, repoints video_ref again
    pr.set_trimmed(cx, rid, "orig-trim2.mp4")
    r = pr.get_review(cx, rid)
    assert r["video_ref"] == "orig-trim2.mp4" and r["video_orig_ref"] == "orig.webm"
```

- [ ] **Step 2: Run to verify it fails**

Run the test command. Expected: FAIL (`set_trimmed` missing / `video_orig_ref` column absent).

- [ ] **Step 3: Implement**

In `dashboard/product_reviews.py` `init_table`, add `"video_orig_ref TEXT DEFAULT ''"` to the additive-columns loop (the loop the 2a-2 task created — append this column to its tuple). Then append:

```python
def set_trimmed(cx, review_id, trimmed_ref):
    init_table(cx)
    cur = cx.cursor(); cur.row_factory = sqlite3.Row
    row = cur.execute("SELECT video_ref, video_orig_ref FROM product_reviews WHERE id=?",
                      (review_id,)).fetchone()
    if not row:
        return
    orig = row["video_orig_ref"] or row["video_ref"] or ""
    cx.execute("UPDATE product_reviews SET video_ref=?, video_orig_ref=? WHERE id=?",
               (trimmed_ref, orig, review_id))
    cx.commit()
```

In `requirements.txt`, add a line:

```
imageio-ffmpeg
```

- [ ] **Step 4: Run to verify it passes**

Run the test command. Expected: all Task-1 + Task-2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add dashboard/product_reviews.py requirements.txt tests/test_reviews_spec2a2b.py
git commit -m "feat(reviews-2a2b): video_orig_ref + set_trimmed; add imageio-ffmpeg dep"
```

---

### Task 3: Worker trim step + `REVIEWS_VIDEO_TRIM` flag

**Files:**
- Modify: `app.py` (flag; `_drain_review_videos` keeps words/duration + trims)
- Test: `tests/test_reviews_spec2a2b.py` (append)

**Interfaces:**
- Consumes: Task 1 `video_trim.{compute_trim_window, trim_video}`; Task 2 `product_reviews.set_trimmed`; existing worker `_drain_review_videos`, `_REVIEWS_VIDEO`, `_REVIEW_MEDIA_DIR`.
- Produces: `_REVIEWS_VIDEO_TRIM` flag; the worker repoints `video_ref` to a `<stem>-trim.mp4` when trimming succeeds.

- [ ] **Step 1: Write the failing tests**

Append (a reload helper that turns on all three flags):

```python
import importlib


def _reload_trim_app(monkeypatch, tmp_path, trim="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REVIEWS_ENABLED", "true")
    monkeypatch.setenv("REVIEWS_VIDEO", "true")
    monkeypatch.setenv("REVIEWS_VIDEO_TRIM", trim)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _seed_video_job(appmod, slug, ref="v.webm"):
    import sqlite3
    from dashboard import product_reviews as pr, review_video_jobs as vj
    d = appmod._REVIEW_MEDIA_DIR / slug; d.mkdir(parents=True, exist_ok=True)
    (d / ref).write_bytes(b"FAKEVIDEO")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr.upsert_review(cx, slug, "b@x.com", "B", 5, video_kind="upload", video_ref=ref)
        vj.enqueue(cx, rid)
    return rid


def _patch_transcribe_and_score(monkeypatch, words):
    import journal_blueprint
    monkeypatch.setattr(journal_blueprint, "_whisper_transcribe",
                        lambda p: {"text": "hello there", "duration": 12.0, "words": words})
    from dashboard import review_scoring as rs
    monkeypatch.setattr(rs, "score_video", lambda *a, **k: {
        "video_points": 4, "publish_risk": False, "risk_reasons": "", "recommend_publish": True})


def test_worker_trims_and_repoints(monkeypatch, tmp_path):
    appmod = _reload_trim_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    rid = _seed_video_job(appmod, slug)
    _patch_transcribe_and_score(monkeypatch, [{"word": "hi", "start": 2.0, "end": 2.4},
                                              {"word": "there", "start": 2.4, "end": 9.0}])
    from dashboard import video_trim as vt
    def fake_trim(src, dst, start, end, **kw):
        open(dst, "wb").write(b"TRIMMED"); return True
    monkeypatch.setattr(vt, "trim_video", fake_trim)
    appmod._drain_review_videos()
    import sqlite3
    from dashboard import product_reviews as pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        r = pr.get_review(cx, rid)
    assert r["video_ref"].endswith("-trim.mp4") and r["video_orig_ref"] == "v.webm"
    assert r["video_points"] == 4 and r["video_status"] == "scored"   # scoring unaffected


def test_worker_trim_failure_keeps_original(monkeypatch, tmp_path):
    appmod = _reload_trim_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    rid = _seed_video_job(appmod, slug)
    _patch_transcribe_and_score(monkeypatch, [{"word": "hi", "start": 2.0, "end": 2.4},
                                              {"word": "there", "start": 2.4, "end": 9.0}])
    from dashboard import video_trim as vt
    monkeypatch.setattr(vt, "trim_video", lambda *a, **k: False)
    appmod._drain_review_videos()
    import sqlite3
    from dashboard import product_reviews as pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        r = pr.get_review(cx, rid)
    assert r["video_ref"] == "v.webm" and (r["video_orig_ref"] or "") == ""   # unchanged
    assert r["video_points"] == 4                                             # points intact


def test_worker_no_trim_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_trim_app(monkeypatch, tmp_path, trim="false")
    assert appmod._REVIEWS_VIDEO_TRIM is False
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    rid = _seed_video_job(appmod, slug)
    _patch_transcribe_and_score(monkeypatch, [{"word": "hi", "start": 2.0, "end": 9.0}])
    from dashboard import video_trim as vt
    called = {"n": 0}
    monkeypatch.setattr(vt, "trim_video", lambda *a, **k: called.__setitem__("n", called["n"] + 1) or True)
    appmod._drain_review_videos()
    import sqlite3
    from dashboard import product_reviews as pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert pr.get_review(cx, rid)["video_ref"] == "v.webm"
    assert called["n"] == 0


def test_worker_no_trim_when_no_words(monkeypatch, tmp_path):
    appmod = _reload_trim_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    rid = _seed_video_job(appmod, slug)
    _patch_transcribe_and_score(monkeypatch, [])   # no words -> compute returns None
    from dashboard import video_trim as vt
    monkeypatch.setattr(vt, "trim_video", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not trim")))
    appmod._drain_review_videos()
    import sqlite3
    from dashboard import product_reviews as pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert pr.get_review(cx, rid)["video_ref"] == "v.webm"
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (`_REVIEWS_VIDEO_TRIM` undefined / no trim).

- [ ] **Step 3: Implement in `app.py`**

Add the flag next to `_REVIEWS_VIDEO`:

```python
_REVIEWS_VIDEO_TRIM = os.environ.get("REVIEWS_VIDEO_TRIM", "").strip().lower() in ("1", "true", "yes")
```

In `_drain_review_videos`, change the transcript line (currently `transcript = (_jb._whisper_transcribe(str(path)) or {}).get("text", "")`) to capture words + duration:

```python
            _whisper = _jb._whisper_transcribe(str(path)) or {}
            transcript = _whisper.get("text", "")
            _words = _whisper.get("words", []) or []
            _duration = _whisper.get("duration") or 0
```

Then, after the `with sqlite3.connect(LOG_DB) as cx:` block that credits points + marks the job done (after `_vj.mark(cx, rid, "done")`), add the trim step (still inside the per-job `try`):

```python
            if _REVIEWS_VIDEO_TRIM:
                try:
                    from dashboard import video_trim as _vt
                    win = _vt.compute_trim_window(_words, _duration)
                    if win:
                        _src = path
                        _dst = _REVIEW_MEDIA_DIR / r["product_slug"] / (_src.stem + "-trim.mp4")
                        if _vt.trim_video(str(_src), str(_dst), win[0], win[1]):
                            with sqlite3.connect(LOG_DB) as cx:
                                _pr.set_trimmed(cx, rid, _dst.name)
                except Exception as e:  # noqa: BLE001 - trim never undoes a credited score
                    print(f"[reviews-video] trim failed rid={rid}: {e}", flush=True)
```

(`path` is a `Path` already in scope from `path = _REVIEW_MEDIA_DIR / r["product_slug"] / r["video_ref"]`, so `path.stem` works.)

- [ ] **Step 4: Run to verify they pass**

Run the test command, then the broader sweep:
`doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/ -k "reviews" -q`
Expected: all pass, no regressions vs 2a-2.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_reviews_spec2a2b.py
git commit -m "feat(reviews-2a2b): REVIEWS_VIDEO_TRIM flag + worker trim step (repoint to trimmed clip)"
```

---

## Self-Review (plan author)

- **Spec coverage:** trim helper (T1) → spec `video_trim.py`; schema `video_orig_ref`/`set_trimmed` + dep (T2) → spec Schema + Dependency; worker step + flag (T3) → spec Worker extension + Flag. All spec sections covered.
- **Decisions honored:** re-encode libx264/aac (T1 cmd); 0.3/0.7 padding + 1.0 min-removed (T1 compute); separate `REVIEWS_VIDEO_TRIM` flag (T3); non-destructive `video_orig_ref` preserve (T2/T3); trim after credit so points are never undone (T3 ordering); lazy ffmpeg import + mocked-runner tests (T1); flag-off + no-words inert (T3 tests).
- **Type consistency:** `compute_trim_window(...) -> tuple|None`, `trim_video(...,runner=) -> bool`, `set_trimmed(cx, id, trimmed_ref)`, `video_orig_ref` column, `_REVIEWS_VIDEO_TRIM` flag — used identically across tasks.
- **Confirm-then-use flagged in-task:** the 2a-2 additive-column loop in `init_table` (T2 appends to it); `path.stem` availability in the worker (T3, in scope).
