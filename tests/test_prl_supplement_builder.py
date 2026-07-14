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
        (email TEXT, scan_id TEXT, scan_date TEXT, item_code TEXT, priority_rank INTEGER, label TEXT)""")
    cx.execute("INSERT INTO scan_recommendations VALUES ('a@b.com','s1','2026-07-01','ED4',1,'ED4 - Nerve')")
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
    assert fa["items"] == ["ED4 - Nerve"]  # friendly label from scan_recommendations.label, not bare code "ED4"
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
        (email TEXT, scan_id TEXT, scan_date TEXT, item_code TEXT, priority_rank INTEGER, label TEXT)""")
    cx.execute("INSERT INTO scan_recommendations VALUES ('a@b.com','s1','2026-07-01','ED4',1,'ED4 - Nerve')")
    cx.execute("INSERT INTO scan_recommendations VALUES ('a@b.com','s1','2026-07-01','ED5',2,'ED5 - Something')")
    cx.commit(); cx.close()

    monkeypatch.setattr(app_mod, "LOG_DB", db)
    monkeypatch.setenv("PRL_SUPPLEMENT_ENABLED", "1")
    out = app_mod._prl_supplement_for("a@b.com", "2026-07-01")
    assert out is not None
    names = [fa["name"] for fa in out["focus_areas"]]
    assert "Nervous System" in names


def test_derive_uses_newest_scan_when_date_none(monkeypatch, tmp_path):
    """A normal portal visit passes scan_date=None (no date param) -- the builder
    must resolve the client's NEWEST scan, exactly like _scan_recommendations_for
    does, instead of matching zero rows and hiding the card (C1)."""
    db = str(tmp_path / "c.db")
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
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
        (email TEXT, scan_id TEXT, scan_date TEXT, item_code TEXT, priority_rank INTEGER, label TEXT)""")
    # Older scan matches nothing covered (a code with no focus-area mapping).
    cx.execute("INSERT INTO scan_recommendations VALUES ('a@b.com','s0','2026-06-01','ZZ9',1,'ZZ9 - Unmapped')")
    # Newest scan matches the covered "Nervous System" focus area via ED4.
    cx.execute("INSERT INTO scan_recommendations VALUES ('a@b.com','s1','2026-07-01','ED4',1,'ED4 - Nerve')")
    cx.commit(); cx.close()

    monkeypatch.setattr(app_mod, "LOG_DB", db)
    monkeypatch.setenv("PRL_SUPPLEMENT_ENABLED", "1")
    out = app_mod._prl_supplement_for("a@b.com", None)
    assert out is not None
    assert out["source"] == "derived"
    fa = out["focus_areas"][0]
    assert fa["name"] == "Nervous System"
    assert fa["items"] == ["ED4 - Nerve"]


def test_ff_slug_flows_through(monkeypatch, tmp_path):
    """The FF counterpart's slug must flow from _resolve_remedy_slug into the
    payload's ff.slug. Guards the _resolve_remedy_slug({"name": ...}) dict-arg
    contract -- the earlier bare-string call raised AttributeError (swallowed),
    silently always yielding slug=None. Slug resolution is forced deterministic
    so the test doesn't depend on the app's real title->slug index (loaded from
    prod data at import)."""
    db = str(tmp_path / "c.db")
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row; _seed(cx); cx.close()
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    monkeypatch.setenv("PRL_SUPPLEMENT_ENABLED", "1")
    # Resolver receives a dict {"name": <ff>} (the fixed contract); return a slug
    # only for that shape, so a regression to the bare-string call fails here.
    monkeypatch.setattr(app_mod, "_resolve_remedy_slug",
                        lambda x: "neuroprotect" if (x or {}).get("name") == "Neuroprotect" else None)
    out = app_mod._prl_supplement_for("a@b.com", "2026-07-01")
    ff = out["focus_areas"][0]["products"][0]["ff"]
    assert ff["name"] == "Neuroprotect"
    assert ff["slug"] == "neuroprotect"  # non-None slug flowed through _prl_ff_view
