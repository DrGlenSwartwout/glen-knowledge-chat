import sys
from pathlib import Path
repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))


def test_elemental_mg_for_mineral_chelate():
    from dashboard import ingredient_content as IC
    # Zinc Glycinate is 20% Zn in the FMP lookup -> 25mg compound = 5mg Zn
    assert round(IC.elemental_mg("Zinc Glycinate", 25), 1) == 5.0
    # Magnesium L-Threonate 7% -> 57mg = ~4mg Mg
    assert round(IC.elemental_mg("Magnesium L-Threonate", 57), 1) == 4.0
    # non-mineral / unknown -> None
    assert IC.elemental_mg("Tetrahydrocurcumin", 10) is None


def test_dv_and_rda_percent_glen_convention():
    from dashboard import ingredient_content as IC
    assert IC.dv_for("Vitamin B12") == (2.4, "mcg")
    assert IC.dv_for("Niacin") == (16, "mg")
    assert IC.dv_for("Chromium") == (35, "mcg")
    assert IC.dv_for("Vitamin D3")[1] == "IU"
    # %RDA on Glen's DVs
    assert round(IC.rda_percent("Vitamin B12", 60, "mcg")) == 2500
    assert round(IC.rda_percent("Niacin", 8, "mg")) == 50
    assert round(IC.rda_percent("Selenium", 100, "mcg")) == 182
    assert round(IC.rda_percent("Chromium", 200, "mcg")) == 571
    assert round(IC.rda_percent("Zinc", 5, "mg")) == 45
    # unit conversion (active in mg, DV in mcg)
    assert round(IC.rda_percent("Chromium", 0.2, "mg")) == 571
    # no DV -> None
    assert IC.rda_percent("Trans-Resveratrol", 50, "mg") is None


def test_iu_vitamins():
    from dashboard import ingredient_content as IC
    assert IC.is_iu_vitamin("Vitamin D3 (Cholecalciferol)")
    assert IC.is_iu_vitamin("Vitamin E (Mixed Tocopherols)")
    assert IC.is_iu_vitamin("Vitamin A (Retinyl Acetate)")
    assert not IC.is_iu_vitamin("Vitamin C (Ascorbic Acid)")


def test_label_dose():
    from dashboard import ingredient_content as IC
    # mineral chelate -> elemental + %RDA
    d = IC.label_dose("Zinc Glycinate", 25, "mg")
    assert d["amount"] == 5 and d["unit"] == "mg" and round(d["rda_percent"]) == 45
    # botanical -> raw amount, no RDA
    d2 = IC.label_dose("Astaxanthin 10% (Haematococcus pluvialis)", 60, "mg")
    assert d2["amount"] == 60 and d2["rda_percent"] is None


def test_bisglycinate_alias():
    from dashboard import ingredient_content as IC
    # "Zinc Bisglycinate" (preferred label term) resolves to the FMP "Zinc Glycinate" 20% record
    assert round(IC.elemental_mg("Zinc Bisglycinate", 25), 1) == 5.0
    assert IC.label_dose("Zinc Bisglycinate", 55, "mg")["amount"] == 11
