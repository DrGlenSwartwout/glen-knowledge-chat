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

def _make_fmp_products(cx, rows):
    """Seed the local product catalog remedy_dosing() reads standard doses from."""
    cx.execute("CREATE TABLE fmp_snap_products "
               "(product_name TEXT, dosage TEXT, dosage_freq TEXT, dosage_timing TEXT)")
    cx.executemany("INSERT INTO fmp_snap_products "
                   "(product_name,dosage,dosage_freq,dosage_timing) VALUES (?,?,?,?)", rows)
    cx.commit()


def test_build_fills_blank_dosing_from_catalog_standard():
    cx = sqlite3.connect(":memory:")
    tid = create_test(cx, "Desi Test", "desi@example.com", "2026-07-14")
    aid = f"a{tid}"
    # Layer 1: practitioner left the dose blank -> the standard must fill it in.
    add_chain_row(cx, aid, layer=1, head="ED6 Heart", most_affected="Heart",
                  remedy="Vitality", dosage="", frequency="", timing="")
    # Layer 2: authored dose -> a manual biofield test overrides the standard.
    add_chain_row(cx, aid, layer=2, head="EI6 Kidney", most_affected="Kidney",
                  remedy="Chelation", dosage="2 capsules", frequency="daily", timing="with lunch")
    _make_fmp_products(cx, [
        ("Vitality", "1 capsule", "daily", "between meals"),
        ("Chelation", "1 capsule", "daily", "on rising"),
    ])
    out = bpp.build_portal_content(cx, aid, special_price_cents=5000, catalog=CATALOG)
    layers = {L["n"]: L for L in out["content"]["layers"]}
    assert layers[1]["dosing"] == "1 capsule daily between meals"   # standard applied
    assert layers[2]["dosing"] == "2 capsules daily with lunch"     # authored wins


def test_build_carries_terrain_phase_and_location():
    from dashboard.biofield_authoring import update_terrain
    cx = sqlite3.connect(":memory:")
    tid = create_test(cx, "Terra Test", "terra@example.com", "2026-07-18")
    aid = f"a{tid}"
    add_chain_row(cx, aid, layer=1, head="ED6 Heart", most_affected="Heart",
                  remedy="Vitality", dosage="1 cap", frequency="daily", timing="")
    update_terrain(cx, aid, phase=4, location="Toxicity")
    out = bpp.build_portal_content(cx, aid, special_price_cents=5000, catalog=CATALOG)
    assert out["content"]["phase"] == 4
    assert out["content"]["location"] == "Toxicity"


def test_build_omits_terrain_when_no_bsi():
    cx = sqlite3.connect(":memory:")
    tid = create_test(cx, "No Terra", "noterra@example.com", "2026-07-18")
    aid = f"a{tid}"
    add_chain_row(cx, aid, layer=1, head="ED6 Heart", most_affected="Heart",
                  remedy="Vitality", dosage="1 cap", frequency="daily", timing="")
    out = bpp.build_portal_content(cx, aid, special_price_cents=5000, catalog=CATALOG)
    assert out["content"]["phase"] is None
    assert out["content"]["location"] == ""


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

def test_build_populates_findings_from_scan(tmp_path):
    cx = sqlite3.connect(":memory:")
    aid = _seed_karin(cx)
    captured = {}
    def fake_provider(email, scan_date):
        captured["args"] = (email, scan_date)
        return [
            {"code": "ED3", "name": "Cell Driver", "rank": 1,
             "description": "The Cell Driver supports cellular energy.",
             "category": "ED", "group": "infoceutical"},
            {"code": "ER9", "name": "Environmental Load", "rank": 2,
             "description": "", "category": "ER", "group": "stress"},
        ]
    out = bpp.build_portal_content(cx, aid, special_price_cents=5000,
                                   catalog=CATALOG, findings_provider=fake_provider)
    f = out["content"]["findings"]
    assert len(f) == 2
    # trimmed to exactly the four portal-consumed fields; description preserved
    assert f[0] == {"code": "ED3", "name": "Cell Driver",
                    "description": "The Cell Driver supports cellular energy.", "rank": 1}
    # blank-description finding kept, description == ""
    assert f[1] == {"code": "ER9", "name": "Environmental Load",
                    "description": "", "rank": 2}
    # aligned to the published scan_date
    assert captured["args"] == ("permanentlyyours777@hawaiiantel.net", "2026-06-25")


def test_build_findings_empty_when_provider_raises(tmp_path):
    cx = sqlite3.connect(":memory:")
    aid = _seed_karin(cx)
    def boom(email, scan_date):
        raise RuntimeError("e4l.db unreadable")
    out = bpp.build_portal_content(cx, aid, special_price_cents=5000,
                                   catalog=CATALOG, findings_provider=boom)
    assert out["content"]["findings"] == []


def test_build_includes_time_of_day_schedule():
    cx = sqlite3.connect(":memory:")
    aid = _seed_karin(cx)
    c = bpp.build_portal_content(cx, aid, special_price_cents=5000, catalog=CATALOG)["content"]
    sched = c.get("schedule") or {}
    assert sched.get("slots") and sched.get("entries")          # time-of-day schedule present
    names = {e.get("name") for e in sched["entries"]}
    assert "Vitality" in names                                   # a dosed remedy is scheduled


def test_build_layers_carry_a_stresses_list():
    cx = sqlite3.connect(":memory:")
    aid = _seed_karin(cx)
    c = bpp.build_portal_content(cx, aid, special_price_cents=5000, catalog=CATALOG)["content"]
    assert all(isinstance(L.get("stresses"), list) for L in c["layers"])   # per-layer stresses field present
