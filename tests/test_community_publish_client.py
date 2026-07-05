# tests/test_community_publish_client.py
from dashboard import community_catalog as _cat


def test_cut_outtakes_calls_trimmer_per_range(tmp_path):
    calls = []
    def fake_trim(src, dst, start, end, *, runner=None):
        calls.append((start, end)); open(dst, "wb").close()
    outs = _cat.cut_outtakes("/src.mp4",
        [{"start": 2.0, "end": 5.0, "title": "A", "interest_tags": ["x"]},
         {"start": 10.0, "end": 14.0, "title": "B", "interest_tags": []}],
        workdir=str(tmp_path), trimmer=fake_trim)
    assert calls == [(2.0, 5.0), (10.0, 14.0)]
    assert outs[0]["title"] == "A" and outs[0]["path"].endswith(".mp4")


def test_publish_session_uploads_then_posts(tmp_path):
    f1 = tmp_path / "o0.mp4"; f1.write_bytes(b"x")
    uploaded = []
    def fake_upload(data, filename, *, base_url, console_key, http_put=None):
        uploaded.append(filename); return f"/portal-asset/{filename}"
    posted = {}
    def fake_post(url, *, json=None, headers=None, timeout=None):
        posted["url"] = url; posted["body"] = json; posted["key"] = headers.get("X-Console-Key")
        class _R:
            status_code = 200
            def json(self): return {"ok": True, "content_id": 7, "outtakes": 1}
        return _R()
    out = _cat.publish_session(
        base_url="https://prod.example", console_key="SEKRET",
        full={"type": "coaching_replay", "title": "T", "description": "d",
              "video_ref": "https://rumble.com/v-1", "interest_tags": ["s"], "transcript": "t"},
        outtake_files=[{"title": "A", "interest_tags": ["s"], "path": str(f1)}],
        uploader=fake_upload, poster=fake_post)
    assert out["content_id"] == 7
    assert posted["url"].endswith("/api/console/community/publish")
    assert posted["key"] == "SEKRET"
    assert posted["body"]["outtakes"][0]["video_ref"] == "/portal-asset/o0.mp4"
    assert posted["body"]["video_ref"] == "https://rumble.com/v-1"
