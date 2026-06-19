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
