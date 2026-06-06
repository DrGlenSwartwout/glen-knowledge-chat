# Wire FMP ingredient-content into the enrichment pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** A reusable `dashboard/ingredient_content.py` that uses the ingested FMP Ingredients-tab data (`data/fmp-ingredient-content.json`) + a DV table to compute label-ready doses automatically: elemental content for mineral chelates, active content for nutrients, IU for the fat-soluble vitamins (A/D/E), %RDA on Glen's DV convention, and raw amount for botanicals. Then wire it into `scripts/enrich_products.py` so future enrichment carries these label doses without a manual pass.

**Builds on:** the merged FMP ingestion (PR #45). New branch `sess/ec0e1f15`, worktree `/tmp/wt-deploy-chat-ec0e1f15`.

**Glen's DV convention (older US DVs — verified against his own formulas):** B12 60mcg=1000% (DV 6mcg), Niacin 8mg=40% (DV 20mg), Selenium 100mcg=143% (DV 70mcg), Chromium 200mcg=167% (DV 120mcg), Zinc 5mg=33% (DV 15mg), Manganese 2mg=100% (DV 2mg), Vit C 60mg=100% (DV 60mg), D3 1000IU=250% (DV 400IU), Vit E 30IU=100% (DV 30IU), Magnesium 8.2mg=2% (DV 400mg), Calcium 10mg=1% (DV 1000mg). **A/D/E are in IU.**

---

## File Structure
- `dashboard/ingredient_content.py` (new): the lookup + DV table + `elemental_mg`, `dv_for`, `rda_percent`, `label_dose`.
- `tests/test_ingredient_content.py` (new): unit tests (verified against Glen's examples).
- `scripts/enrich_products.py` (modify): attach a `label` (amount/unit/rda) to each ingredient using the module.

---

## Task 1: `dashboard/ingredient_content.py` + tests

- [ ] **Step 1: Write the failing tests** `tests/test_ingredient_content.py`:

```python
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
    assert IC.dv_for("Vitamin B12") == (6, "mcg")
    assert IC.dv_for("Niacin") == (20, "mg")
    assert IC.dv_for("Chromium") == (120, "mcg")
    assert IC.dv_for("Vitamin D3")[1] == "IU"
    # %RDA on Glen's DVs
    assert round(IC.rda_percent("Vitamin B12", 60, "mcg")) == 1000
    assert round(IC.rda_percent("Niacin", 8, "mg")) == 40
    assert round(IC.rda_percent("Selenium", 100, "mcg")) == 143
    assert round(IC.rda_percent("Chromium", 200, "mcg")) == 167
    assert round(IC.rda_percent("Zinc", 5, "mg")) == 33
    # unit conversion (active in mg, DV in mcg)
    assert round(IC.rda_percent("Chromium", 0.2, "mg")) == 167
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
    assert d["amount"] == 5 and d["unit"] == "mg" and round(d["rda_percent"]) == 33
    # botanical -> raw amount, no RDA
    d2 = IC.label_dose("Astaxanthin 10% (Haematococcus pluvialis)", 60, "mg")
    assert d2["amount"] == 60 and d2["rda_percent"] is None
```

- [ ] **Step 2: Run to verify fail.** `python3 -m pytest tests/test_ingredient_content.py -q`

- [ ] **Step 3: Implement `dashboard/ingredient_content.py`:**

```python
"""Compute label-ready ingredient doses from the ingested FMP Ingredients-tab data
(data/fmp-ingredient-content.json) + Glen's DV convention. Minerals -> elemental
content; A/D/E -> IU; nutrients -> %RDA on the older US DVs Glen's FMP uses;
botanicals -> raw amount."""
import json
import os
import re

_REPO_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_CONTENT = None

# Glen's DV convention (older US Daily Values; verified against his formulas).
DV = {
    "vitamin a": (5000, "IU"), "vitamin c": (60, "mg"), "vitamin d": (400, "IU"),
    "vitamin d3": (400, "IU"), "vitamin e": (30, "IU"), "vitamin k": (80, "mcg"),
    "thiamin": (1.5, "mg"), "vitamin b1": (1.5, "mg"), "riboflavin": (1.7, "mg"),
    "vitamin b2": (1.7, "mg"), "niacin": (20, "mg"), "vitamin b3": (20, "mg"),
    "vitamin b5": (10, "mg"), "pantothenic acid": (10, "mg"), "vitamin b6": (2, "mg"),
    "folate": (400, "mcg"), "vitamin b9": (400, "mcg"), "5-mthf": (400, "mcg"),
    "vitamin b12": (6, "mcg"), "biotin": (300, "mcg"), "calcium": (1000, "mg"),
    "iron": (18, "mg"), "magnesium": (400, "mg"), "zinc": (15, "mg"),
    "selenium": (70, "mcg"), "copper": (2, "mg"), "manganese": (2, "mg"),
    "chromium": (120, "mcg"), "molybdenum": (75, "mcg"), "iodine": (150, "mcg"),
    "potassium": (3500, "mg"), "phosphorus": (1000, "mg"),
}
_UNIT_MG = {"mg": 1.0, "mcg": 0.001, "g": 1000.0}


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _content():
    global _CONTENT
    if _CONTENT is None:
        p = os.path.join(_REPO_DATA, "fmp-ingredient-content.json")
        try:
            _CONTENT = json.load(open(p)).get("ingredients", {})
        except Exception:
            _CONTENT = {}
    return _CONTENT


def get(name):
    """FMP content record for an ingredient (exact-normalized, else substring)."""
    d = _content()
    k = _norm(name)
    if k in d:
        return d[k]
    for kk, v in d.items():
        if kk and (kk in k or k in kk) and len(kk) > 4:
            return v
    return None


def elemental_mg(name, compound_mg):
    """Elemental mineral content = compound_mg x percent. None if no % / not applicable."""
    r = get(name)
    if r and r.get("percent"):
        try:
            return float(compound_mg) * float(r["percent"]) / 100.0
        except (TypeError, ValueError):
            return None
    return None


def _dv_key(nutrient):
    n = _norm(nutrient)
    if n in DV:
        return n
    for k in DV:
        if k in n:
            return k
    return None


def dv_for(nutrient):
    k = _dv_key(nutrient)
    return DV[k] if k else None


def rda_percent(nutrient, amount, unit):
    """%RDA = active amount / DV (unit-aware). None if no DV."""
    dv = dv_for(nutrient)
    if not dv:
        return None
    dv_val, dv_unit = dv
    if dv_unit == "IU":
        return (float(amount) / dv_val) * 100 if unit == "IU" else None
    amt_mg = float(amount) * _UNIT_MG.get(unit, 1.0)
    dv_mg = dv_val * _UNIT_MG.get(dv_unit, 1.0)
    return (amt_mg / dv_mg) * 100 if dv_mg else None


def is_iu_vitamin(name):
    n = _norm(name)
    return bool(re.search(r"\bvitamin (a|d|d3|e)\b", n))


def label_dose(name, amount, unit):
    """Label-ready dose: elemental for mineral chelates, raw otherwise; + %RDA.
    Returns {amount, unit, rda_percent}."""
    el = elemental_mg(name, float(amount) * _UNIT_MG.get(unit, 1.0)) if unit in _UNIT_MG else None
    if el is not None:
        amt, u = (round(el * 1000), "mcg") if el < 1 else (round(el, 2), "mg")
        return {"amount": amt, "unit": u, "rda_percent": rda_percent(name, amt, u)}
    return {"amount": amount, "unit": unit, "rda_percent": rda_percent(name, amount, unit)}
```

- [ ] **Step 4: Run to pass.** `python3 -m pytest tests/test_ingredient_content.py -q` -> all pass.
- [ ] **Step 5: Commit.** `git add dashboard/ingredient_content.py tests/test_ingredient_content.py && git commit -m "feat(bos): ingredient_content module (elemental + IU + %RDA from FMP lookup + DV table)"`

---

## Task 2: Wire into `scripts/enrich_products.py`

- [ ] **Step 1:** Near where each ingredient dict is built in `scripts/enrich_products.py` (the `{name, qty, unit, raw}` records), import the module and attach a `label`:

```python
try:
    from dashboard import ingredient_content as _ic
except Exception:
    _ic = None
...
# when building an ingredient record with a numeric qty + unit:
if _ic and qty and unit in ("mg", "mcg", "g"):
    try:
        ing["label"] = _ic.label_dose(name, qty, unit)
    except Exception:
        pass
```

(Adapt to the actual ingredient-building code; the goal: each candidate ingredient gains a `label` = {amount, unit, rda_percent} computed via the module, so future enrichment runs carry elemental/IU/%RDA automatically.)

- [ ] **Step 2: Verify** `python3 -m py_compile scripts/enrich_products.py dashboard/ingredient_content.py`. Run `python3 -c "from dashboard import ingredient_content as IC; print(IC.label_dose('Zinc Glycinate',25,'mg'), IC.label_dose('Chromium Polynicotinate',1.6,'mg'))"` -> Zinc 5mg 33%, Chromium ~200mcg 167%.
- [ ] **Step 3: Commit.** `git add scripts/enrich_products.py && git commit -m "feat(bos): enrichment attaches label_dose (elemental/IU/%RDA) per ingredient"`

---

## Self-Review
**Spec:** reusable elemental/IU/%RDA computation from the FMP lookup + Glen's DVs, wired so the enrichment produces label-ready doses automatically. DV table verified against Glen's own formula examples. A/D/E in IU.
**Placeholder scan:** none. **Type consistency:** `elemental_mg`/`dv_for`/`rda_percent`/`is_iu_vitamin`/`label_dose` consistent across tasks; `label_dose` returns `{amount, unit, rda_percent}`.
