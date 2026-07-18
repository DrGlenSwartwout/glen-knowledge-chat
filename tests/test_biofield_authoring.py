"""Increment 4a: writable authoring store. Authored tests render through the same
report shape as the FMP snapshot, so schedule/narrative/audio reuse unchanged."""
import sqlite3
from dashboard.biofield_authoring import (
    init_auth_tables, create_test, add_chain_row, update_chain_row,
    delete_chain_row, update_header, list_authored, authored_report,
    delete_test, confirm_row, resolve_remedy_name, resolve_stress_name,
    remedy_catalog, remedy_dosing, _title_case_name)


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "chat_log.db"))
    init_auth_tables(cx)
    return cx


def test_create_author_and_render(tmp_path):
    cx = _cx(tmp_path)
    tid = create_test(cx, "Jane Doe", "Jane@x.com", "2026-06-23")
    assert tid.startswith("a")
    add_chain_row(cx, tid, 2, "Acid", "Liver", "Sterol Max", "3 caps", "daily", "with food")
    add_chain_row(cx, tid, 1, "Night", "Night", "TMG", "1 scoop", "daily", "at night")
    rep = authored_report(cx, tid)
    assert rep["client"] == {"name": "Jane Doe", "email": "jane@x.com"}
    assert rep["date"] == "2026-06-23"
    assert [(l["layer"], l["head"], l["remedy"]) for l in rep["layers"]] == [
        (1, "Night", "TMG"), (2, "Acid", "Sterol Max")]
    slots = {e["name"]: e["slots"] for e in rep["schedule"]["entries"]}
    assert slots["TMG"] == ["Bedtime"] and slots["Sterol Max"] == ["Breakfast"]


def test_list_authored(tmp_path):
    cx = _cx(tmp_path)
    tid = create_test(cx, "Jane Doe", "jane@x.com", "2026-06-23")
    add_chain_row(cx, tid, 1, "Night", "Night", "TMG", "1 scoop", "daily", "at night")
    lst = list_authored(cx)
    assert len(lst) == 1
    assert lst[0]["test_id"] == tid and lst[0]["name"] == "Jane Doe"
    assert lst[0]["layer_count"] == 1 and lst[0]["authored"] is True


def test_update_and_delete_row(tmp_path):
    cx = _cx(tmp_path)
    tid = create_test(cx, "J", "j@x.com", "2026-06-23")
    rid = add_chain_row(cx, tid, 1, "Night", "Night", "TMG", "1 scoop", "daily", "at night")
    update_chain_row(cx, rid, layer=3, remedy="TMG Powder")
    rep = authored_report(cx, tid)
    # display layer is always renumbered 1..k by ordered_chain; verify update took + remedy
    assert rep["layers"][0]["layer"] == 1 and rep["layers"][0]["remedy"] == "TMG Powder"
    # confirm stored layer value was actually updated in the DB
    stored = cx.execute("SELECT layer FROM biofield_auth_chain WHERE id=?", (rid,)).fetchone()[0]
    assert stored == 3
    delete_chain_row(cx, rid)
    assert authored_report(cx, tid)["layers"] == []


def test_authored_report_depth_match(tmp_path):
    cx = _cx(tmp_path)
    from dashboard.biofield_dimensions import seed_dimensions, tag, DEPTH_KEY
    seed_dimensions(cx)
    tid = create_test(cx, "J", "j@x.com", "2026-06-23")
    rid = add_chain_row(cx, tid, 1, "Mercury", "Brain", "Binder", "1 cap", "daily", "with food")
    tag(cx, "auth_stress", rid, DEPTH_KEY, 5)   # stress acts at the nucleus
    tag(cx, "auth_remedy", rid, DEPTH_KEY, 1)   # remedy only reaches the gut
    layer = authored_report(cx, tid)["layers"][0]
    assert layer["stress_depth"] == 5 and layer["remedy_depth"] == 1
    assert layer["depth_status"] == "shallow"
    assert layer["depth_need"].lower().startswith("nucle")


def test_delete_test_removes_it(tmp_path):
    cx = _cx(tmp_path)
    tid = create_test(cx, "J", "j@x.com", "2026-06-23")
    add_chain_row(cx, tid, 1, "Acid", "Liver", "Sterol Max")
    delete_test(cx, tid)
    assert list_authored(cx) == []
    assert authored_report(cx, tid)["layers"] == []


def test_confirmed_flag_default_and_voice_and_confirm(tmp_path):
    cx = _cx(tmp_path)
    tid = create_test(cx, "J", "j@x.com", "2026-06-23")
    add_chain_row(cx, tid, 1, "Acid", "Liver", "Sterol Max")              # manual -> confirmed
    rid = add_chain_row(cx, tid, 2, "Tox", "Tox", "TMG", confirmed=0)     # voice -> unconfirmed
    byname = {l["remedy"]: l["confirmed"] for l in authored_report(cx, tid)["layers"]}
    assert byname["Sterol Max"] == 1 and byname["TMG"] == 0
    confirm_row(cx, rid)
    assert {l["remedy"]: l["confirmed"] for l in authored_report(cx, tid)["layers"]}["TMG"] == 1


def test_resolve_remedy_name_autocorrects(tmp_path):
    cx = _cx(tmp_path)
    cx.execute("CREATE TABLE fmp_snap_products(id_pk TEXT,product_name TEXT,dosage TEXT,dosage_freq TEXT,dosage_timing TEXT)")
    cx.executemany("INSERT INTO fmp_snap_products VALUES(?,?,?,?,?)",
                   [("1", "Perelandra Rose Essence", "", "", ""), ("2", "Microbiome", "", "", "")])
    cx.commit()
    assert resolve_remedy_name(cx, "Perlandra Rose Essence") == "Perelandra Rose Essence"
    assert resolve_remedy_name(cx, "Microbiome") == "Microbiome"
    assert resolve_remedy_name(cx, "Zzzqxw Nonsense") == "Zzzqxw Nonsense"   # no close match
    assert resolve_remedy_name(cx, "Perlandra Rose Essence in Terrain Restore") == \
        "Perelandra Rose Essence in Terrain Restore"                         # suffix preserved


def test_resolve_remedy_name_is_case_insensitive(tmp_path):
    # The mangle Glen saw: ASR lowercases, so a case-sensitive match missed
    # "Sulfur Syntropy". Lowercasing both sides recovers it.
    cx = _cx(tmp_path)
    cx.execute("CREATE TABLE fmp_snap_products(id_pk TEXT,product_name TEXT,dosage TEXT,dosage_freq TEXT,dosage_timing TEXT)")
    cx.executemany("INSERT INTO fmp_snap_products VALUES(?,?,?,?,?)",
                   [("1", "Sulfur Syntropy", "", "", ""), ("2", "Sobopla", "", "", "")])
    cx.commit()
    assert resolve_remedy_name(cx, "sulfur centropy") == "Sulfur Syntropy"   # close + case-insensitive
    assert resolve_remedy_name(cx, "sobopla") == "Sobopla"                   # casing corrected to canonical


def test_resolve_remedy_name_matches_distinctive_token_in_longer_name(tmp_path):
    # 'Sobopla' is a distinctive token inside a long catalog name; whole-string
    # fuzzy can't catch it, but a unique token match should.
    cx = _cx(tmp_path)
    cx.execute("CREATE TABLE fmp_snap_products(id_pk TEXT,product_name TEXT,dosage TEXT,dosage_freq TEXT,dosage_timing TEXT)")
    cx.executemany("INSERT INTO fmp_snap_products VALUES(?,?,?,?,?)", [
        ("1", "Perelandra Nature Program Essence Sobopla in Terrain Restore", "", "", ""),
        ("2", "Perelandra Rose Essence in Terrain Restore", "", "", ""),
        ("3", "Microbiome", "", "", "")])
    cx.commit()
    full = "Perelandra Nature Program Essence Sobopla in Terrain Restore"
    assert resolve_remedy_name(cx, "sobopla") == full
    # spoken with the Terrain Restore suffix must not double the suffix
    assert resolve_remedy_name(cx, "Sobopla in Terrain Restore") == full
    # a common token shared by several products is ambiguous -> no token match
    assert resolve_remedy_name(cx, "essence") == "Essence"


def test_discontinue_asterisk_stripped_but_still_listed(tmp_path):
    # A trailing '*' on an FMP product name is Glen's "intending to discontinue"
    # marker; the product is still active and sellable. The picker must list it
    # under the CLEAN name (so it matches the sellable catalog + the stress
    # coverage map, both of which store the clean name) while surfacing the intent
    # as a flag. Regression: "Vitamin P Polyphenols*" was dropped from invoices and
    # from the balancing panel because its name never matched "Vitamin P Polyphenols".
    cx = _cx(tmp_path)
    cx.executescript(
        "CREATE TABLE fmp_snap_products(id_pk TEXT,product_name TEXT,dosage TEXT,dosage_freq TEXT,dosage_timing TEXT);"
        "CREATE TABLE fmp_snap_products_phases(id_fk_product TEXT,text TEXT);"
        "CREATE TABLE fmp_snap_products_systems(id_fk_product TEXT,text TEXT);")
    cx.execute("INSERT INTO fmp_snap_products VALUES('1','Vitamin P Polyphenols*','1 capsule','daily','with food')")
    cx.commit()
    picks = remedy_catalog(cx, "Vitamin P")
    assert [p["name"] for p in picks] == ["Vitamin P Polyphenols"]     # clean name, listed
    assert picks[0]["discontinue_intent"] is True                      # intent surfaced, not hidden
    # dosing resolves despite the stored asterisk, keyed on the clean name
    assert remedy_dosing(cx, "Vitamin P Polyphenols") == {
        "dosage": "1 capsule", "frequency": "daily", "timing": "with food"}
    # fuzzy resolve of an ASR-mangled spoken name returns the clean canonical name
    assert resolve_remedy_name(cx, "vitamin p polyphenals") == "Vitamin P Polyphenols"


def test_resolve_remedy_name_title_cases_when_unmatched(tmp_path):
    # No catalog / no close match -> at least clean up the casing.
    cx = _cx(tmp_path)  # no fmp_snap_products table at all
    assert resolve_remedy_name(cx, "reverse age") == "Reverse Age"
    assert resolve_remedy_name(cx, "neuro-magnesium") == "Neuro-Magnesium"
    assert resolve_remedy_name(cx, "MB5") == "MB5"                            # code preserved
    assert resolve_remedy_name(cx, "perelandra essence in terrain restore") == \
        "Perelandra Essence in Terrain Restore"                              # small word + suffix


def test_resolve_stress_name_matches_vocab_and_title_cases(tmp_path):
    cx = _cx(tmp_path)
    cx.execute("CREATE TABLE fmp_snap_client_active_main_stress(id_pk TEXT, main_stress TEXT)")
    cx.executemany("INSERT INTO fmp_snap_client_active_main_stress VALUES(?,?)",
                   [("1", "Large Intestine Meridian"), ("2", "Toxicity")])
    cx.commit()
    assert resolve_stress_name(cx, "large intestine meridian") == "Large Intestine Meridian"  # canonical
    assert resolve_stress_name(cx, "toxisity") == "Toxicity"                  # fuzzy + case
    assert resolve_stress_name(cx, "reverse age") == "Reverse Age"            # unmatched -> title case


def test_title_case_name_edge_cases():
    assert _title_case_name("reverse age") == "Reverse Age"
    assert _title_case_name("head and tail of the chain") == "Head and Tail of the Chain"
    assert _title_case_name("MB5") == "MB5"
    assert _title_case_name("vitamin B12") == "Vitamin B12"
    assert _title_case_name("") == ""


def test_update_header(tmp_path):
    cx = _cx(tmp_path)
    tid = create_test(cx, "J", "j@x.com", "2026-06-23")
    update_header(cx, tid, name="Jane Q", date="2026-07-01")
    rep = authored_report(cx, tid)
    assert rep["client"]["name"] == "Jane Q" and rep["date"] == "2026-07-01"


def test_terrain_phase_persists_and_renders(tmp_path):
    from dashboard.biofield_authoring import update_terrain
    cx = _cx(tmp_path)
    tid = create_test(cx, "J", "j@x.com", "2026-06-23")
    # A brand-new test has no terrain reading yet.
    rep = authored_report(cx, tid)
    assert rep["phase"] is None and rep["location"] == ""
    update_terrain(cx, tid, phase=4, location="Toxicity")
    rep = authored_report(cx, tid)
    assert rep["phase"] == 4 and rep["location"] == "Toxicity"


def test_update_terrain_is_per_field(tmp_path):
    # Re-interpreting with no phase spoken must not wipe a phase already read.
    from dashboard.biofield_authoring import update_terrain
    cx = _cx(tmp_path)
    tid = create_test(cx, "J", "j@x.com", "2026-06-23")
    update_terrain(cx, tid, phase=2, location="Kidney")
    update_terrain(cx, tid, phase=None, location="")  # nothing spoken this pass
    rep = authored_report(cx, tid)
    assert rep["phase"] == 2 and rep["location"] == "Kidney"
