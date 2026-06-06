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
    "iron": (18, "mg"), "magnesium": (400, "mg"), "zinc": (11, "mg"),
    "selenium": (70, "mcg"), "copper": (2, "mg"), "manganese": (2.3, "mg"),
    "chromium": (35, "mcg"), "molybdenum": (75, "mcg"), "iodine": (150, "mcg"),
    "potassium": (3500, "mg"), "phosphorus": (1000, "mg"),
}
_UNIT_MG = {"mg": 1.0, "mcg": 0.001, "g": 1000.0}

# Mineral chelates in the FMP lookup carry an elemental-symbol percent in their
# label_form, e.g. "Zinc Bisglycinate (20% Zn)", "Magnesium L-Threonate (7% Mg)",
# "GTF Chromium (Chromium Polynicotinate) (12.4% Cr)". Botanical/nutrient
# standardizations use a bare percent without an element symbol ("Astaxanthin 10%",
# "Withanolides 20%"). Only the former should be reduced to elemental content.
_MINERAL_PCT_RE = re.compile(
    r"\(\s*[\d.]+\s*%\s*"
    r"(?:Na|Mg|Al|Si|Ca|Cr|Mn|Fe|Co|Ni|Cu|Zn|Se|Mo|Sr|Li|Va|V|Bo|Sn|Sb|Rb|Cs|Cd|Ba|Ge|"
    r"K|P|I|B)\s*\)"
)


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
    # "bisglycinate" is the preferred label term; the FMP lookup stores "glycinate"
    alt = k.replace("bisglycinate", "glycinate")
    if alt != k and alt in d:
        return d[alt]
    for kk, v in d.items():
        if kk and (kk in k or k in kk) and len(kk) > 4:
            return v
    return None


def _is_mineral_chelate(record):
    """True only for elemental-mineral records (label_form has a `(N% Symbol)` tag)."""
    if not record:
        return False
    return bool(_MINERAL_PCT_RE.search(record.get("label_form") or ""))


def elemental_mg(name, compound_mg):
    """Elemental mineral content = compound_mg x percent. None if not a mineral
    chelate / no % / not applicable (botanical standardizations are NOT reduced)."""
    r = get(name)
    if r and r.get("percent") and _is_mineral_chelate(r):
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
