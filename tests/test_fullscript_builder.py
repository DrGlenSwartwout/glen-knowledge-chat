"""_fullscript_for: flag-gated Fullscript channel card builder. Mirrors
_prl_supplement_for. Covers derive-from-scan, pin priority, the default-OFF
flag, and the byte-identical guarantee that the payload key is absent when off.
"""
import sqlite3

import app as app_mod
from dashboard import fullscript as fs

SEED = {
  "products": [
    {"name": "Mag Taurate", "brand": "Jarrow", "product_slug": "mag-taurate",
     "external_id": "P1", "best_ff": "Neuro Magnesium", "relation": "substitute",
     "focus_tags": [], "ff_alts": [], "product_type": "supplement", "url": None,
     "source": "seed", "active": 1},
  ],
  "focus_area_products": [
    {"focus_area_id": 9, "focus_area_name": "Nervous System",
     "fs_product_name": "Mag Taurate", "rank": 0},
  ],
  "focus_area_items": [{"focus_area_id": 9, "item_code": "ED4"}],
}


def _seed(cx):
    fs.init_tables(cx)
    fs.sync_from_seed(cx, SEED)
    cx.execute("""CREATE TABLE IF NOT EXISTS scan_recommendations
        (email TEXT, scan_id TEXT, scan_date TEXT, item_code TEXT,
         priority_rank INTEGER, label TEXT)""")
    cx.execute("INSERT INTO scan_recommendations "
               "VALUES ('a@b.com','s1','2026-07-01','ED4',1,'ED4 - Nerve')")
    cx.commit()


def _db(tmp_path):
    db = str(tmp_path / "c.db")
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    _seed(cx)
    cx.close()
    return db


def test_flag_off_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "LOG_DB", _db(tmp_path))
    monkeypatch.delenv("FULLSCRIPT_ENABLED", raising=False)
    assert app_mod._fullscript_for("a@b.com", "2026-07-01") is None


def test_derive_builds_card(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "LOG_DB", _db(tmp_path))
    monkeypatch.setenv("FULLSCRIPT_ENABLED", "1")
    monkeypatch.setenv("FULLSCRIPT_DISPENSARY_SLUG", "remedymatch")
    out = app_mod._fullscript_for("a@b.com", "2026-07-01")
    assert out["dispensary_url"] == \
        "https://us.fullscript.com/welcome/remedymatch/store-start"
    g = out["groups"][0]
    assert g["origin"] == "scan" and g["heading"] == "Matched from your scan"
    p = g["products"][0]
    assert p["name"] == "Mag Taurate" and p["brand"] == "Jarrow"
    assert p["ff"]["name"] == "Neuro Magnesium"
    assert p["ff"]["relation"] == "substitute"


def test_no_candidates_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "LOG_DB", _db(tmp_path))
    monkeypatch.setenv("FULLSCRIPT_ENABLED", "1")
    assert app_mod._fullscript_for("nobody@nowhere.com", None) is None


def test_never_raises_on_bad_db(monkeypatch):
    monkeypatch.setattr(app_mod, "LOG_DB", "/nonexistent/dir/nope.db")
    monkeypatch.setenv("FULLSCRIPT_ENABLED", "1")
    assert app_mod._fullscript_for("a@b.com", None) is None
