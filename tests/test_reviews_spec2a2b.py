from dashboard import video_trim as vt


def test_compute_trim_window_trims_with_padding():
    words = [{"word": "hi", "start": 2.0, "end": 2.4}, {"word": "there", "start": 2.4, "end": 9.0}]
    win = vt.compute_trim_window(words, 12.0)
    assert win == (2.0 - 0.3, 9.0 + 0.7)          # (1.7, 9.7), clamped within [0,12]


def test_compute_trim_window_clamps_start_to_zero():
    # first word near 0: lead pushes start below 0 -> clamped to 0; real trailing trim remains
    words = [{"word": "a", "start": 0.1, "end": 0.2}, {"word": "b", "start": 0.2, "end": 8.0}]
    win = vt.compute_trim_window(words, 12.0)
    assert win == (0.0, 8.0 + 0.7)                  # start clamped to 0, end = last + trail


def test_compute_trim_window_none_when_speech_fills_clip():
    # speech nearly fills the clip -> after clamping, removed < 1.0s -> skip (no pointless re-encode)
    words = [{"word": "a", "start": 0.1, "end": 0.2}, {"word": "b", "start": 0.2, "end": 11.9}]
    assert vt.compute_trim_window(words, 12.0) is None


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
