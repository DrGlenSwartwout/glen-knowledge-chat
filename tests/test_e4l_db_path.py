import os
from dashboard import biofield_e4l as be


def test_db_path_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("E4L_DB", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # no e4l.db in DATA_DIR yet -> local ~/AI-Training default
    assert be._db_path().endswith("AI-Training/e4l.db")
    # once a synced copy exists on the (prod) persistent disk -> use it
    (tmp_path / "e4l.db").write_bytes(b"SQLite format 3\x00")
    assert be._db_path() == str(tmp_path / "e4l.db")
    # an explicit E4L_DB env always wins
    monkeypatch.setenv("E4L_DB", "/custom/path/e4l.db")
    assert be._db_path() == "/custom/path/e4l.db"
    # explicit arg beats everything
    assert be._db_path("/arg/e4l.db") == "/arg/e4l.db"
