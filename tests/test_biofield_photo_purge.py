import os, sqlite3, time
from dashboard import biofield_store as bs

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    bs.init_table(cx); return cx

def test_clear_photo(tmp_path):
    cx = _cx()
    p = tmp_path / "a.jpg"; p.write_bytes(b"x")
    bs.set_photo_on_file(cx, "p@x.com", str(p))
    assert bs.get(cx, "p@x.com")["photo_on_file"]
    bs.clear_photo(cx, "p@x.com")
    r = bs.get(cx, "p@x.com")
    assert not r["photo_on_file"] and not r["photo_path"]

def test_purge_expired_photos(tmp_path):
    cx = _cx()
    root = tmp_path / "biofield-photos"; root.mkdir()
    old = root / "old.jpg"; old.write_bytes(b"x")
    new = root / "new.jpg"; new.write_bytes(b"y")
    bs.set_photo_on_file(cx, "old@x.com", str(old))
    bs.set_photo_on_file(cx, "new@x.com", str(new))
    now = time.time()
    os.utime(old, (now - 40 * 86400, now - 40 * 86400))
    os.utime(new, (now - 2 * 86400, now - 2 * 86400))
    n = bs.purge_expired_photos(cx, photo_root=str(root), retention_days=30, now_ts=now)
    assert n == 1
    assert not old.exists() and new.exists()
    assert not bs.get(cx, "old@x.com")["photo_on_file"]
    assert bs.get(cx, "new@x.com")["photo_on_file"]

def test_purge_missing_root_is_noop(tmp_path):
    cx = _cx()
    assert bs.purge_expired_photos(cx, photo_root=str(tmp_path / "nope"),
                                   retention_days=30, now_ts=time.time()) == 0
