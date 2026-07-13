import json, os
SEED = os.path.join(os.path.dirname(__file__), "..", "data", "condition_programs_seed.json")

def _seed():
    with open(SEED) as f: return json.load(f)

def _slugs(prog): return [i["slug"] for i in prog["items"]]

def test_dry_amd_has_crocin_ocuheal_ocuflow_and_no_lipids_in_base():
    p = _seed()["condition_programs"]["dry-amd"]
    s = _slugs(p)
    assert "macular-wellness-crocin" in s
    assert "ocuheal-eye-drops" in s and "ocuflow-bedtime" in s
    assert "lipid-zyme" not in s and "lipid-cleanse" not in s
    wh = [m for m in p["modifiers"] if m["when"] == "drusen"][0]
    assert {i["slug"] for i in wh["items"]} == {"lipid-cleanse", "lipid-zyme"}
    ar = [m for m in p["modifiers"] if m["when"] == "on_areds2"][0]
    assert ar["action"] == "remove" and ar["source"] == "client-reported"

def test_wet_amd_moves_angiogenx_and_scar_to_modifiers():
    p = _seed()["condition_programs"]["wet-amd"]
    s = _slugs(p)
    assert "angiogenx" not in s and "scar-solve" not in s
    assert "macular-wellness-crocin" in s and p["consult_recommended"] is True
    whens = {m["when"] for m in p["modifiers"]}
    assert {"drusen", "on_areds2", "leakage", "scar"} <= whens

def test_dr_proliferative_modifier_and_ocuflow():
    p = _seed()["condition_programs"]["diabetic-retinopathy"]
    assert "ocuflow-bedtime" in _slugs(p)
    prolif = [m for m in p["modifiers"] if m["when"] == "proliferative"][0]
    assert prolif["source"] == "clinician-measured" and prolif["client_default"] is False
    assert p["consult_recommended"] is False  # Glen: DR stays one-click orderable

def test_ocuheal_in_every_program():
    progs = _seed()["condition_programs"]
    for key, p in progs.items():
        slugs = set(_slugs(p))
        for it in p["items"]:
            slugs.update(a["slug"] for a in it.get("alts", []))
        assert "ocuheal-eye-drops" in slugs, f"{key} missing OcuHeal"

def test_name_typos_fixed():
    progs = _seed()["condition_programs"]
    names = [i["name"] for p in progs.values() for i in p["items"]]
    assert "Clear Lens Eyedrops" not in names
    assert "Lens-Zyme Brunescense Buster" not in names
