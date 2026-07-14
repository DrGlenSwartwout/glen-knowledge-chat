"""_prl_supplement_for: flag-gated PRL Supplement portal card builder.
Covers derive (item-code -> focus area -> product), mirror override, the
default-OFF flag, and the correction that ranking must not be sliced to 6
before dropping product-less (uncovered) focus areas -- otherwise a
higher-ranked uncovered focus area can starve the card of real entries.
"""
import sqlite3
import json

import app as app_mod
from dashboard import prl_supplement as prl


def _seed(cx):
    prl.init_tables(cx)
    prl.sync_from_seed(cx, {
      "products": [{"name": "NeuroVen", "url": "u", "best_ff": "Neuroprotect",
                    "relation": "substitute", "focus_tags": [], "ff_alts": [],
                    "external_id": "1", "product_type": "supplement"}],
      "focus_area_products": [{"focus_area_id": 9, "focus_area_name": "Nervous System",
                               "prl_product_name": "NeuroVen", "rank": 0}],
      "focus_area_items": [{"focus_area_id": 9, "item_code": "ED4"}],
    })
    cx.execute("""CREATE TABLE IF NOT EXISTS scan_recommendations
        (email TEXT, scan_id TEXT, scan_date TEXT, item_code TEXT, priority_rank INTEGER)""")
    cx.execute("INSERT INTO scan_recommendations VALUES ('a@b.com','s1','2026-07-01','ED4',1)")
    cx.commit()


def test_derive_builds_card(monkeypatch, tmp_path):
    db = str(tmp_path / "c.db")
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row; _seed(cx); cx.close()
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    monkeypatch.setenv("PRL_SUPPLEMENT_ENABLED", "1")
    out = app_mod._prl_supplement_for("a@b.com", "2026-07-01")
    assert out["source"] == "derived"
    assert out["prl_link"] == "https://truly.vip/prl"
    fa = out["focus_areas"][0]
    assert fa["name"] == "Nervous System"
    assert fa["products"][0]["name"] == "NeuroVen"
    assert fa["products"][0]["ff"]["name"] == "Neuroprotect"


def test_flag_off_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("PRL_SUPPLEMENT_ENABLED", "")
    assert app_mod._prl_supplement_for("a@b.com", "2026-07-01") is None


def test_mirror_overrides_derive(monkeypatch, tmp_path):
    db = str(tmp_path / "c.db")
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row; _seed(cx)
    cx.execute("INSERT INTO prl_scan_mirror VALUES ('s1', ?, '2026-07-13')", (json.dumps(
        {"patterns": [{"Name": "Stomach",
                       "PatternItems": [{"ScanItemName": "ED8 - Stomach"}],
                       "PRLProducts": [{"Name": "GastroVen"}]}]}),))
    cx.commit(); cx.close()
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    monkeypatch.setenv("PRL_SUPPLEMENT_ENABLED", "1")
    out = app_mod._prl_supplement_for("a@b.com", "2026-07-01")
    assert out["source"] == "mirror"
    assert out["focus_areas"][0]["name"] == "Stomach"
    assert out["focus_areas"][0]["products"][0]["name"] == "GastroVen"


def test_uncovered_high_rank_does_not_starve_card(monkeypatch, tmp_path):
    """focus_areas_for_items ranks ALL item-matched focus areas, including ones
    with no PRL products (uncovered). If 6+ uncovered focus areas out-rank a
    covered one by hit count, slicing to 6 BEFORE dropping the uncovered ones
    would throw the covered focus area away entirely. The builder must rank
    all, then keep only covered ones while walking the ranked list, so a
    lower-ranked but covered focus area still makes the card."""
    db = str(tmp_path / "c.db")
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    prl.init_tables(cx)
    # One covered focus area (has a product) matching a single scan item -> 1 hit.
    # Six uncovered focus areas (no product row at all) each matching BOTH scan
    # items -> 2 hits each, so all six out-rank the covered focus area.
    focus_area_items = [{"focus_area_id": 9, "item_code": "ED4"}]
    for fid in range(101, 107):
        focus_area_items.append({"focus_area_id": fid, "item_code": "ED4"})
        focus_area_items.append({"focus_area_id": fid, "item_code": "ED5"})
    prl.sync_from_seed(cx, {
        "products": [{"name": "NeuroVen", "url": "u", "best_ff": "Neuroprotect",
                      "relation": "substitute", "focus_tags": [], "ff_alts": [],
                      "external_id": "1", "product_type": "supplement"}],
        "focus_area_products": [{"focus_area_id": 9, "focus_area_name": "Nervous System",
                                 "prl_product_name": "NeuroVen", "rank": 0}],
        "focus_area_items": focus_area_items,
    })
    cx.execute("""CREATE TABLE IF NOT EXISTS scan_recommendations
        (email TEXT, scan_id TEXT, scan_date TEXT, item_code TEXT, priority_rank INTEGER)""")
    cx.execute("INSERT INTO scan_recommendations VALUES ('a@b.com','s1','2026-07-01','ED4',1)")
    cx.execute("INSERT INTO scan_recommendations VALUES ('a@b.com','s1','2026-07-01','ED5',2)")
    cx.commit(); cx.close()

    monkeypatch.setattr(app_mod, "LOG_DB", db)
    monkeypatch.setenv("PRL_SUPPLEMENT_ENABLED", "1")
    out = app_mod._prl_supplement_for("a@b.com", "2026-07-01")
    assert out is not None
    names = [fa["name"] for fa in out["focus_areas"]]
    assert "Nervous System" in names
