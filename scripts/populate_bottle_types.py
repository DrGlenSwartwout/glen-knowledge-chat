"""Populate each storefront product's bottle_type from the FileMaker packaging
export + family rules. Re-runnable; never overwrites an existing assignment.
Dry-run by default; --write patches data/products.json (committed baseline).

Recovery steps (in priority order after exact FMP join):
  1. Roll-on family rule (name/description contains roll-on/rollon/roll on → 30roll)
  2. Synergy↔Syntropy alias (both normalise to 'synergy')
  3. Suffix-stripped FMP index keys (powder, powders, tablets, capsules)
  4. Conservative difflib fuzzy fallback (cutoff ≥ 0.92, single match only)
"""
from __future__ import annotations
import argparse, csv, difflib, json, os, re, sys

FMP_EXPORT = os.environ.get("FMP_PRODUCTS_CSV", "/tmp/fmp-export/newapp/products.csv")
_INFO_RE = re.compile(r'^(ei|ed|es|et|mb)\s*\d', re.I)  # MR not sold as infoceuticals
# The same code can sit in parentheses instead of leading the name:
# "Microbes/Liver Meridian (EI-8)", "Emotional Stress Release (MB5)".
_INFO_PAREN_RE = re.compile(r'\((ei|ed|es|et|mb)\s*-?\s*\d+\)', re.I)
# Trailing packaging words to strip from FMP keys (and storefront names) for the suffix-strip step.
# "oil", "spray", "lotion", "drops" are intentionally excluded (part of product identity).
_SUFFIX_WORDS = re.compile(r'\s+(powder|powders|tablets|capsules)$')


def _norm(s):
    """Normalise to lowercase alphanumeric + spaces, with synergy/syntropy alias."""
    t = re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).strip()
    # Alias: treat 'syntropy' and 'synergy' as the same token so both naming
    # systems resolve to the same FMP index key.
    t = t.replace('syntropy', 'synergy')
    return t


def family_rule(slug, product):
    name = product.get("name", "")
    src = product.get("source", "")
    # Glen 2026-07-09: ALL Infoceuticals are a 30 ml dropper. The code-prefix regex only
    # caught EI-8 / ES3 style names; the named ones (Youth, Sleep, EMF Infoceutical) fell
    # through. Match the NAME only — a description merely mentioning "infoceutical" must
    # not capture an unrelated product.
    if (src == "infoceutical-catalog" or _INFO_RE.match(name.strip())
            or _INFO_PAREN_RE.search(name) or "infoceutical" in name.lower()):
        return "30ml"
    # Glen 2026-07-09: the whole "... in Terrain Restore" line ships in a 50 ml dropper,
    # whatever FMP's packaging column says. Matched on the NAME (a description could merely
    # mention the line) and requires the leading "in " so the base product is not captured.
    if "in terrain restore" in name.lower():
        return "Dropper 50 mL"
    text = f"{name} {product.get('description','')}".lower()
    if "eye drop" in text or "eyedrop" in text:
        return "Dropper 5 mL"
    # Roll-on family rule: any of the common spelling variants → 30roll
    if "roll-on" in text or "rollon" in text or "roll on" in text:
        return "30roll"
    return None


def classify_from_fmp(row):
    disp = (row.get("zc_sold_display") or "").lower().replace(" ", "")
    meas = (row.get("sold_measurement") or "").lower().strip()
    ftype = (row.get("type") or "").strip()
    # Glen 2026-07-09: an Essence is a 50 ml dropper, full stop. The TYPE is authoritative,
    # so a bad packaging cell cannot misclassify one: FMP has exactly one Essence row whose
    # zc_sold_display reads '50mg' (HRMNY Flower Essence) against 395 reading '50ml' — a
    # FileMaker typo. Without this, the mg guard below would simply decline, sending the
    # product to the review pile.
    if ftype.lower() == "essence":
        return "Dropper 50 mL"
    mml = re.match(r'^(\d+(?:\.\d+)?)ml$', disp)
    if mml or meas == "ml":
        ml = float(mml.group(1)) if mml else None
        return {5.0: "Dropper 5 mL", 15.0: "15ml", 50.0: "Dropper 50 mL", 100.0: "100ml"}.get(ml)  # 30/bulk -> None
    if any(x in disp for x in ("pullulan", "enteric", "vegicap", "gelcap", "capsule")) \
       or meas in ("pullulan", "enteric", "vegicaps", "gelcaps", "00 capsules"):
        mc = re.match(r'^(\d+)', disp)
        n = int(mc.group(1)) if mc else None
        if n is None:
            return None
        if n <= 40:
            return "30 Caps"
        if n <= 140:
            return "120 caps"
        return None
    # `disp` has had spaces stripped, so "500 mg" -> "500mg". Milligrams are a DOSE, not a
    # container size: a naive endswith("g") sent a 50mg flower essence to a 30 g jar.
    if (disp.endswith("g") and not disp.endswith("mg")) or meas == "g":
        return "120 caps" if ftype == "Pure Powders" else "30 g"
    return None


def _build_fmp_index(rows):
    """Build the fmp_by_name dict from an iterable of FMP row dicts.

    Each row is stored under:
      • its full normalised name  (e.g. 'msm synergy powder')
      • its suffix-stripped name  (e.g. 'msm synergy')

    The full name wins on collision (setdefault keeps first).  The stripped
    key is only added when it differs from the full key.
    """
    by_name: dict[str, dict] = {}
    for r in rows:
        full_key = _norm(r.get("product_name"))
        by_name.setdefault(full_key, r)
        stripped_key = _SUFFIX_WORDS.sub('', full_key)
        if stripped_key != full_key:
            by_name.setdefault(stripped_key, r)
    return by_name


def build_assignments(products, fmp_by_name):
    assignments, review = {}, []
    # Pre-compute fuzzy key list once (all keys in the FMP index).
    fmp_keys = list(fmp_by_name.keys())

    for slug, p in products.items():
        if p.get("bottle_type"):
            continue
        key = family_rule(slug, p)
        if not key:
            norm_name = _norm(p.get("name"))
            row = fmp_by_name.get(norm_name)
            if row is None:
                # Suffix-strip the storefront name and retry
                stripped = _SUFFIX_WORDS.sub('', norm_name)
                if stripped != norm_name:
                    row = fmp_by_name.get(stripped)
            if row is None:
                # Conservative fuzzy fallback (difflib, cutoff ≥ 0.92)
                matches = difflib.get_close_matches(norm_name, fmp_keys, n=1, cutoff=0.92)
                if matches:
                    row = fmp_by_name[matches[0]]
            key = classify_from_fmp(row) if row else None
        if key:
            assignments[slug] = key
        else:
            assignments_reason = "no family rule + no FMP packaging match"
            review.append({"slug": slug, "name": p.get("name", ""),
                           "reason": assignments_reason})
    return {"assignments": assignments, "review": review}


def _load_fmp(path):
    if not os.path.exists(path):
        return {}
    return _build_fmp_index(csv.DictReader(open(path)))


def _products_path():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.environ.get("DATA_DIR")
    if d and os.path.exists(os.path.join(d, "products.json")):
        return os.path.join(d, "products.json")
    return os.path.join(here, "data", "products.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    path = _products_path()
    with open(path) as f:
        doc = json.load(f)
    products = doc.get("products", {})
    fmp = _load_fmp(FMP_EXPORT)
    if not fmp:
        print(f"WARNING: no FMP export at {FMP_EXPORT} — only family rules will apply.")
    m = build_assignments(products, fmp)
    print(f"{len(m['assignments'])} products assigned; {len(m['review'])} need review.")
    for r in m["review"]:
        print(f"  REVIEW {r['slug']}: {r['name']!r} ({r['reason']})")
    if args.write:
        for slug, key in m["assignments"].items():
            products[slug]["bottle_type"] = key
        with open(path, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(m['assignments'])} assignments to {path}")
    else:
        print("(dry run — pass --write to patch products.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
