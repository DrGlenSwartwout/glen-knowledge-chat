import sqlite3, os
from dashboard import biofield_e4l

def _mk(tmp_path):
    p = tmp_path / "e4l.db"
    cx = sqlite3.connect(p)
    cx.execute("CREATE TABLE e4l_pattern_structures (code TEXT NOT NULL, "
               "structure TEXT NOT NULL, stype TEXT, is_primary INTEGER DEFAULT 0, "
               "source_phrase TEXT, PRIMARY KEY (code, structure))")
    cx.executemany("INSERT INTO e4l_pattern_structures(code,structure,stype) VALUES (?,?,?)",
                   [("ED11","Liver","organ"), ("ED11","Anger","emotion"),
                    ("ED12","Fear","emotion"), ("ED3","Cell","organ")])
    cx.commit(); cx.close()
    return str(p)

def test_returns_emotions_by_code(tmp_path):
    db = _mk(tmp_path)
    out = biofield_e4l.emotions_for_codes(["ED11","ED12","ED3","NOPE"], db_path=db)
    assert out["ED11"] == ["Anger"]
    assert out["ED12"] == ["Fear"]
    assert "ED3" not in out and "NOPE" not in out   # no emotion row

def test_missing_db_returns_empty(tmp_path):
    assert biofield_e4l.emotions_for_codes(["ED11"], db_path=str(tmp_path/"nope.db")) == {}

def test_blank_codes_never_raises():
    assert biofield_e4l.emotions_for_codes([], db_path=None) == {}
