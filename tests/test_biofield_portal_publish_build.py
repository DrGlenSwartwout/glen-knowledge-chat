import sqlite3
from dashboard import biofield_portal_publish as bpp
from dashboard.biofield_authoring import create_test, add_chain_row
from dashboard.biofield_narrative import save_narrative

CATALOG = {
    "vitality":       {"name": "Vitality"},
    "chelation":      {"name": "Chelation"},
    "nous-energy":    {"name": "Nous Energy"},
    "neuro-magnesium":{"name": "Neuro Magnesium"},
    "terrain-restore":{"name": "Terrain Restore"},
}

def _seed_karin(cx):
    tid = create_test(cx, "Karin Takahashi", "permanentlyyours777@hawaiiantel.net",
                      "2026-06-25")
    aid = f"a{tid}"
    add_chain_row(cx, aid, layer=1, head="ED3 Cell Driver", most_affected="Circulation",
                  remedy="Vitality", dosage="1 capsule", frequency="daily", timing="with food")
    add_chain_row(cx, aid, layer=2, head="EI6 Kidney pH", most_affected="Kidney",
                  remedy="Chelation", dosage="1 capsule", frequency="daily", timing="")
    add_chain_row(cx, aid, layer=2, head="EI6 Kidney pH", most_affected="Kidney",
                  remedy="Nous Energy", dosage="one a day", frequency="", timing="")
    add_chain_row(cx, aid, layer=3, head="EI10 Circulation", most_affected="Heart",
                  remedy="Focus, Neuromagnesium", dosage="two scoops", frequency="a day", timing="")
    add_chain_row(cx, aid, layer=4, head="Psychoemotional", most_affected="Psychoemotional",
                  remedy="Community Spirit Formula in Terrain Restore",
                  dosage="10 drops", frequency="3 times a day", timing="before meals")
    return aid

def test_build_maps_layers_dedups_and_prices(tmp_path):
    cx = sqlite3.connect(":memory:")
    aid = _seed_karin(cx)
    out = bpp.build_portal_content(cx, aid, special_price_cents=5000, catalog=CATALOG)

    assert out["email"] == "permanentlyyours777@hawaiiantel.net"
    assert out["name"] == "Karin Takahashi"
    assert out["scan_date"] == "2026-06-25"
    assert out["unresolved"] == []
    c = out["content"]
    assert c["biofield_status"] == "confirmed"
    # 5 chain rows -> 5 walkthrough layers
    assert len(c["layers"]) == 5
    l0 = c["layers"][0]
    assert l0["n"] == 1 and l0["title"] == "ED3 Cell Driver" and l0["remedy"] == "Vitality"
    assert l0["dosing"] == "1 capsule daily with food"
    # reorder deduped to 5 unique slugs (Focus,Neuromagnesium -> one neuro-magnesium line)
    slugs = [it["slug"] for it in c["reorder_items"]]
    assert sorted(slugs) == ["chelation", "neuro-magnesium", "nous-energy",
                             "terrain-restore", "vitality"]
    assert all(it["price_cents"] == 5000 and it["qty"] == 1 for it in c["reorder_items"])

def test_build_meaning_from_narrative_segments(tmp_path):
    cx = sqlite3.connect(":memory:")
    aid = _seed_karin(cx)
    save_narrative(cx, aid,
        "Aloha Karin. Vitality restores your surface energy. Chelation clears the burden. "
        "Nous Energy steadies you. Focus, Neuromagnesium sharpens you. "
        "Community Spirit Formula in Terrain Restore holds your heart.")
    out = bpp.build_portal_content(cx, aid, special_price_cents=5000, catalog=CATALOG)
    assert "Vitality" in out["content"]["layers"][0]["meaning"]
    assert out["content"]["greeting"].startswith("Aloha")

def test_build_unresolved_remedy_is_reported_not_published(tmp_path):
    cx = sqlite3.connect(":memory:")
    tid = create_test(cx, "Test One", "t@example.com", "2026-06-25")
    aid = f"a{tid}"
    add_chain_row(cx, aid, layer=1, head="X", most_affected="X",
                  remedy="Invented Remedy ZZZ", dosage="1", frequency="daily", timing="")
    out = bpp.build_portal_content(cx, aid, special_price_cents=5000, catalog=CATALOG)
    assert out["unresolved"] == ["Invented Remedy ZZZ"]
    assert out["content"]["reorder_items"] == []
