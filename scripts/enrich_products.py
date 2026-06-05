#!/usr/bin/env python3
"""
Stage A: Product-catalog enrichment — offline matching core.

Reconciles products.json slugs with authoritative ingredients from:
  1. FMP new (products.csv + products_items.csv + ingredients.csv) — primary
  2. T33_FORMULAS.csv — fallback for unmatched

Writes:
  data/products-enrich-candidate.json  — enrichment per slug (review before applying)
  data/products-enrich-report.md       — match quality summary + review lists

Does NOT touch data/products.json.

Usage:
    python3 scripts/enrich_products.py
"""

import csv
import json
import re
from difflib import get_close_matches, SequenceMatcher
from pathlib import Path

# ── Paths (override via env vars or edit here) ────────────────────────────────
WORKTREE = Path(__file__).parent.parent
PRODUCTS_JSON = WORKTREE / "data" / "products.json"
CANDIDATE_JSON = WORKTREE / "data" / "products-enrich-candidate.json"
REPORT_MD = WORKTREE / "data" / "products-enrich-report.md"

FMP_BASE = Path("/Users/remedymatch/AI-Training/00 System/fmp-extracts/2026-05-23")
FMP_PRODUCTS_CSV = FMP_BASE / "products.csv"
FMP_ITEMS_CSV = FMP_BASE / "products_items.csv"
FMP_INGREDIENTS_CSV = FMP_BASE / "ingredients.csv"

T33_CSV = Path(
    "/Users/remedymatch/AI-Training/00 System/fmp-extracts/2026-05-24/T33_FORMULAS.csv"
)

# ── Fuzzy-match thresholds ────────────────────────────────────────────────────
FUZZY_HIGH = 0.85   # high confidence fuzzy
FUZZY_LOW = 0.72    # low confidence fuzzy (still reported, flagged for review)

# ── Suffixes stripped during normalization ────────────────────────────────────
# Applied repeatedly until stable (handles stacked suffixes)
_SUFFIXES = [
    " nootropic",
    " complex",
    " formula",
    " formulation",
    " supplement",
    " syntropy",
    " synergy",
    " (request referral)",
    " homeoenergetic drops",
    " homeoenergetic",
    " terrain restore",
    " terrain support",
    " drops",
    " powder",
    " gelcaps",
    " capsules",
    " caps",
    " pellets",
    " drink mix",
    " drink",
]


def normalize(s: str) -> str:
    """Lowercase, strip HTML entities, strip punctuation, strip common suffixes."""
    if not s:
        return ""
    s = s.lower().strip()
    # Expand compound medical words before suffix stripping so both spellings match
    # e.g. "ACES Eyedrops" and "ACES Eye Drops" both normalize to "aces eye"
    s = re.sub(r"eyedrops", "eye drops", s)
    # Remove HTML entities / ampersand variants
    s = re.sub(r"&quot;", "", s)
    s = re.sub(r"&amp;", "and", s)
    s = re.sub(r"&lt;", "", s)
    s = re.sub(r"&gt;", "", s)
    s = re.sub(r"&[a-z]+;", "", s)
    s = s.replace("&", "and")
    # Strip leading asterisk used for draft products in FMP
    s = s.lstrip("*").strip()
    # Iteratively strip known suffixes
    changed = True
    while changed:
        changed = False
        for suffix in _SUFFIXES:
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
                changed = True
    # Collapse non-alphanumeric runs to a single space
    out: list[str] = []
    last_space = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            last_space = False
        elif not last_space:
            out.append(" ")
            last_space = True
    return "".join(out).strip()


def fuzzy_score(a: str, b: str) -> float:
    """SequenceMatcher ratio with substring boost.

    Boost rules (to avoid false positives from short alias tokens like "B" or "methyl"):
    - Boost if a is a substring of b (product name is contained within the alias).
    - Boost if b is a substring of a AND |b| >= |a| * 0.75
      (alias is not merely a short prefix of a longer product name like
      "methyl" in "methylselenocysteine").
    """
    score = SequenceMatcher(None, a, b).ratio()
    if a and b:
        if a in b:
            # Product name fully contained within the FMP/T33 alias — strong signal.
            score = max(score, 0.87)
        elif b in a and len(b) >= len(a) * 0.75:
            # Alias is a substantial substring of the product name — still valid.
            score = max(score, 0.87)
    return score


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_products_json() -> dict:
    """Return {slug: {name, pinecone_title, ...}}."""
    with open(PRODUCTS_JSON) as f:
        data = json.load(f)
    return data["products"]


def load_fmp(
) -> tuple[dict, dict, dict]:
    """
    Returns:
        fmp_prods:  {id_pk: row_dict}
        fmp_items:  {id_fk_product: [item_row, ...]}
        fmp_ings:   {id_pk: row_dict}
    """
    csv.field_size_limit(10 * 1024 * 1024)

    with open(FMP_PRODUCTS_CSV) as f:
        fmp_prods = {r["id_pk"]: r for r in csv.DictReader(f)}

    fmp_items: dict[str, list] = {}
    with open(FMP_ITEMS_CSV) as f:
        for r in csv.DictReader(f):
            fk = r["id_fk_product"]
            fmp_items.setdefault(fk, []).append(r)

    with open(FMP_INGREDIENTS_CSV) as f:
        fmp_ings = {r["id_pk"]: r for r in csv.DictReader(f)}

    return fmp_prods, fmp_items, fmp_ings


def load_t33() -> list[dict]:
    csv.field_size_limit(10 * 1024 * 1024)
    with open(T33_CSV) as f:
        return list(csv.DictReader(f))


# ── FMP matching ──────────────────────────────────────────────────────────────

def build_fmp_lookup(fmp_prods: dict) -> tuple[dict, list]:
    """
    Returns:
        by_norm: {normalized_name: [id_pk, ...]}
        norm_list: [str, ...] — all unique norms for fuzzy search
    """
    by_norm: dict[str, list] = {}
    for pid, row in fmp_prods.items():
        n = normalize(row["product_name"])
        if n:
            by_norm.setdefault(n, []).append(pid)
    return by_norm, list(by_norm.keys())


def match_fmp(
    match_key: str,
    fmp_by_norm: dict,
    fmp_norms: list,
) -> tuple[str | None, str, float]:
    """
    Returns (id_pk, confidence, score).
    confidence: 'high' | 'medium' | 'low' | 'none'
    score: 0.0–1.0
    """
    n = normalize(match_key)
    if not n:
        return None, "none", 0.0

    # Exact normalized match → high
    if n in fmp_by_norm:
        pids = fmp_by_norm[n]
        # Prefer Functional Formulation type if multiple
        return pids[0], "high", 1.0

    # Fuzzy search across all FMP products
    best_pid = None
    best_score = 0.0
    for norm_key, pids in fmp_by_norm.items():
        s = fuzzy_score(n, norm_key)
        if s > best_score:
            best_score = s
            best_pid = pids[0]

    if best_score >= FUZZY_HIGH:
        return best_pid, "medium", best_score
    elif best_score >= FUZZY_LOW:
        return best_pid, "low", best_score
    return None, "none", best_score


# ── Ingredient resolution ─────────────────────────────────────────────────────

def best_ingredient_name(ing: dict, zc_raw: str) -> str:
    """
    Resolve the best human-readable name for an ingredient row.
    Priority: name_common (first line) > name_favorite > name_raw > parse zc_raw_display.
    """
    nc = ing.get("name_common", "").strip()
    if nc:
        return nc.split("\n")[0].strip()
    nf = ing.get("name_favorite", "").strip()
    if nf:
        return nf
    nr = ing.get("name_raw", "").strip()
    if nr:
        return nr
    # Parse from "QTYunit - IngredientName"
    m = re.match(
        r"^[\d.]+\s*(?:mg|mcg|g|iu|ug|ml|ea\.?|%)\s*-\s*(.+)$",
        zc_raw.strip(),
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return zc_raw.strip()


def resolve_ingredients(
    fmp_id: str,
    fmp_items: dict,
    fmp_ings: dict,
) -> list[dict]:
    """Build ingredient list for a matched FMP product."""
    items = fmp_items.get(fmp_id, [])
    result = []
    for it in items:
        raw_id = it.get("id_fk_raw", "").strip()
        ing = fmp_ings.get(raw_id, {}) if raw_id else {}
        zc_raw = it.get("zc_raw_display", "").strip()
        name = best_ingredient_name(ing, zc_raw)
        qty_raw = it.get("qty", "").strip()
        qty = float(qty_raw) if qty_raw else None
        result.append(
            {
                "name": name,
                "qty": qty,
                "unit": it.get("unit_measurement", "").strip() or None,
                "raw": zc_raw or None,
            }
        )
    return result


# ── T33 matching + parsing ────────────────────────────────────────────────────

def build_t33_lookup(t33_rows: list[dict]) -> tuple[dict, list]:
    """
    Returns:
        by_norm: {normalized_alias: [row_index, ...]}
        norm_list: [str, ...]
    """
    by_norm: dict[str, list] = {}
    for i, row in enumerate(t33_rows):
        names_block = row.get("Name", "").strip()
        for alias in names_block.split("\n"):
            alias = alias.strip()
            if alias:
                n = normalize(alias)
                if n:
                    by_norm.setdefault(n, []).append(i)
    return by_norm, list(by_norm.keys())


def match_t33(
    match_key: str,
    t33_by_norm: dict,
    t33_norms: list,
) -> tuple[int | None, str, float]:
    """
    Returns (row_index, confidence, score).
    """
    n = normalize(match_key)
    if not n:
        return None, "none", 0.0

    if n in t33_by_norm:
        return t33_by_norm[n][0], "high", 1.0

    best_idx = None
    best_score = 0.0
    for norm_key, idxs in t33_by_norm.items():
        s = fuzzy_score(n, norm_key)
        if s > best_score:
            best_score = s
            best_idx = idxs[0]

    if best_score >= FUZZY_HIGH:
        return best_idx, "medium", best_score
    elif best_score >= FUZZY_LOW:
        return best_idx, "low", best_score
    return None, "none", best_score


# Pattern: version separator in T33 Key Ingredients.
# Versions are separated by blank lines OR by lines like "Name #N:" or "Formula N:"
_T33_VERSION_SEP = re.compile(
    r"\n\s*\n"  # blank line
    r"|"
    r"\n[A-Za-z][^\n]*#\d+\s*:"  # "Name #2:"
    r"|"
    r"\nFormula\s+\d+",  # "Formula 2"
)

# Ingredient-like line heuristic:
# Must contain a quantity token (number + unit or ratio like N:N)
_ING_QTY = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|iu|ug|ml|%|:1)\b",
    re.IGNORECASE,
)
_ING_SKIP = re.compile(
    r"^\s*(?:formula|batch|note|tray|calculation|version|date|\d+\/\d+\/\d+|#\d|\*\*)",
    re.IGNORECASE,
)


def parse_t33_top_block(formula_text: str) -> tuple[str, list[dict]]:
    """
    Extracts the TOP (current) version block and parses ingredient lines.
    Returns (top_block_text, [{"name": ..., "qty": ..., "unit": ..., "raw": ...}])
    """
    if not formula_text:
        return "", []

    # Take the first version block (before first blank-line-or-version separator)
    parts = _T33_VERSION_SEP.split(formula_text, maxsplit=1)
    top_block = parts[0].strip() if parts else formula_text.strip()

    ingredients = []
    for line in top_block.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _ING_SKIP.match(line):
            continue
        if not _ING_QTY.search(line):
            continue
        # Parse qty + unit from line
        m = re.search(
            r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|iu|ug|ml|%)",
            line,
            re.IGNORECASE,
        )
        qty = float(m.group(1)) if m else None
        unit = m.group(2).lower() if m else None
        # Clean ingredient name: strip leading dash/bullet/number
        name = re.sub(r"^[\d\.\-\*\•]+\s*", "", line).strip()
        # Remove bracketed inventory notes [30g] etc.
        name = re.sub(r"\s*\[.*?\]", "", name).strip()
        ingredients.append({"name": name, "qty": qty, "unit": unit, "raw": line})

    return top_block, ingredients


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data files...")
    products = load_products_json()
    fmp_prods, fmp_items, fmp_ings = load_fmp()
    t33_rows = load_t33()

    print(
        f"  products.json: {len(products)} slugs"
        f"  |  FMP products: {len(fmp_prods)}"
        f"  |  FMP items: {sum(len(v) for v in fmp_items.values())}"
        f"  |  FMP ingredients: {len(fmp_ings)}"
        f"  |  T33 rows: {len(t33_rows)}"
    )

    # Count FMP product types for reporting
    ff_total = sum(
        1 for r in fmp_prods.values() if r.get("type") == "Functional Formulation"
    )
    print(f"  FMP Functional Formulations: {ff_total}")

    fmp_by_norm, fmp_norms = build_fmp_lookup(fmp_prods)
    t33_by_norm, t33_norms = build_t33_lookup(t33_rows)

    # ── Per-slug matching ──────────────────────────────────────────────────────
    candidate: dict[str, dict] = {}

    stats = {
        "fmp_new_high": 0,
        "fmp_new_medium": 0,
        "fmp_new_low": 0,
        "fmp_old_t33_high": 0,
        "fmp_old_t33_medium": 0,
        "fmp_old_t33_low": 0,
        "none": 0,
    }
    # Track FMP type breakdown for matched slugs
    ff_slug_matches = 0

    # For report: low-confidence and unmatched lists
    low_conf_items: list[dict] = []
    unmatched_items: list[dict] = []

    for slug, pdata in products.items():
        match_key = pdata.get("pinecone_title") or pdata.get("name", "")
        slug_name = pdata.get("name", "")
        pinecone_title = pdata.get("pinecone_title", "")

        # ── FMP match ──────────────────────────────────────────────────────────
        fmp_id, fmp_conf, fmp_score = match_fmp(match_key, fmp_by_norm, fmp_norms)

        if fmp_id is not None:
            fmp_row = fmp_prods[fmp_id]
            ingredients = resolve_ingredients(fmp_id, fmp_items, fmp_ings)
            entry: dict = {
                "source": "fmp_new",
                "confidence": fmp_conf,
                "fmp_score": round(fmp_score, 3),
                "fmp_id": fmp_id,
                "fmp_name": fmp_row["product_name"],
                "fmp_type": fmp_row["type"],
                "ingredients": ingredients,
            }
            stats[f"fmp_new_{fmp_conf}"] += 1
            if fmp_row["type"] == "Functional Formulation":
                ff_slug_matches += 1
            if fmp_conf == "low":
                low_conf_items.append(
                    {
                        "slug": slug,
                        "name": slug_name,
                        "pinecone_title": pinecone_title,
                        "fmp_name": fmp_row["product_name"],
                        "score": fmp_score,
                        "source": "fmp_new",
                    }
                )
            candidate[slug] = entry
            continue

        # ── T33 fallback ───────────────────────────────────────────────────────
        t33_idx, t33_conf, t33_score = match_t33(match_key, t33_by_norm, t33_norms)

        if t33_idx is not None:
            t33_row = t33_rows[t33_idx]
            formula_text = t33_row.get("Key Ingredients for Formula", "").strip()
            top_block, ingredients = parse_t33_top_block(formula_text)
            entry = {
                "source": "fmp_old_t33",
                "confidence": t33_conf,
                "t33_score": round(t33_score, 3),
                "t33_name": t33_row.get("Name", "").replace("\n", " | "),
                "formula_text": top_block,
                "ingredients": ingredients,
            }
            stats[f"fmp_old_t33_{t33_conf}"] += 1
            if t33_conf == "low":
                low_conf_items.append(
                    {
                        "slug": slug,
                        "name": slug_name,
                        "pinecone_title": pinecone_title,
                        "fmp_name": t33_row.get("Name", "").split("\n")[0],
                        "score": t33_score,
                        "source": "fmp_old_t33",
                    }
                )
            candidate[slug] = entry
            continue

        # ── No match ───────────────────────────────────────────────────────────
        stats["none"] += 1
        # Capture best guesses for report
        best_fmp_guess = ""
        if fmp_norms:
            close = get_close_matches(
                normalize(match_key), fmp_norms, n=1, cutoff=0.0
            )
            if close:
                guess_pids = fmp_by_norm[close[0]]
                best_fmp_guess = (
                    f"{fmp_prods[guess_pids[0]]['product_name']} ({fmp_score:.2f})"
                )
        unmatched_items.append(
            {
                "slug": slug,
                "name": slug_name,
                "pinecone_title": pinecone_title,
                "best_fmp_guess": best_fmp_guess,
            }
        )
        candidate[slug] = {
            "source": "none",
            "confidence": "none",
            "ingredients": [],
        }

    # ── Write candidate JSON ───────────────────────────────────────────────────
    CANDIDATE_JSON.write_text(
        json.dumps(candidate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote: {CANDIDATE_JSON}")

    # ── Write report MD ────────────────────────────────────────────────────────
    total = len(products)
    fmp_new_total = stats["fmp_new_high"] + stats["fmp_new_medium"] + stats["fmp_new_low"]
    t33_total = (
        stats["fmp_old_t33_high"]
        + stats["fmp_old_t33_medium"]
        + stats["fmp_old_t33_low"]
    )

    report_lines = [
        "# Products Enrichment — Stage A Match Report",
        "",
        f"Generated: 2026-06-05  |  Source: `scripts/enrich_products.py`",
        "",
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total slugs | {total} |",
        f"| FMP-new matched (high conf) | {stats['fmp_new_high']} |",
        f"| FMP-new matched (medium conf) | {stats['fmp_new_medium']} |",
        f"| FMP-new matched (low conf) | {stats['fmp_new_low']} |",
        f"| **FMP-new total** | **{fmp_new_total}** |",
        f"| T33 fallback (high conf) | {stats['fmp_old_t33_high']} |",
        f"| T33 fallback (medium conf) | {stats['fmp_old_t33_medium']} |",
        f"| T33 fallback (low conf) | {stats['fmp_old_t33_low']} |",
        f"| **T33 fallback total** | **{t33_total}** |",
        f"| No match | {stats['none']} |",
        f"| Slugs matching a Functional Formulation FMP type | {ff_slug_matches} |",
        f"| FMP Functional Formulation products available | {ff_total} |",
        "",
        "## Low-Confidence Matches (needs Glen review)",
        "",
        "These matched but with a fuzzy score below 0.85. Verify the FMP/T33 name is "
        "actually the same product before applying.",
        "",
        "| Slug | Name | Pinecone Title | Best FMP/T33 Match | Score | Source |",
        "|------|------|----------------|-------------------|-------|--------|",
    ]

    for item in sorted(low_conf_items, key=lambda x: x["score"]):
        report_lines.append(
            f"| {item['slug']} | {item['name']} | {item['pinecone_title']} "
            f"| {item['fmp_name']} | {item['score']:.2f} | {item['source']} |"
        )

    report_lines += [
        "",
        "## Unmatched Slugs (no FMP or T33 match)",
        "",
        "These produced no match above the 0.72 cutoff. They may be Essences, "
        "Infoceuticals, or products not yet in FMP/T33. Ingredients will need "
        "manual entry.",
        "",
        "| Slug | Name | Pinecone Title | Best FMP Guess |",
        "|------|------|----------------|----------------|",
    ]

    for item in unmatched_items:
        report_lines.append(
            f"| {item['slug']} | {item['name']} | {item['pinecone_title']} "
            f"| {item['best_fmp_guess']} |"
        )

    REPORT_MD.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote: {REPORT_MD}")

    # ── stdout summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("ENRICHMENT MATCHING SUMMARY")
    print("=" * 60)
    print(f"  Total slugs:                   {total}")
    print(f"  FMP-new high confidence:        {stats['fmp_new_high']}")
    print(f"  FMP-new medium confidence:      {stats['fmp_new_medium']}")
    print(f"  FMP-new low confidence:         {stats['fmp_new_low']}")
    print(f"  FMP-new TOTAL:                  {fmp_new_total}")
    print(f"  T33 fallback high:              {stats['fmp_old_t33_high']}")
    print(f"  T33 fallback medium:            {stats['fmp_old_t33_medium']}")
    print(f"  T33 fallback low:               {stats['fmp_old_t33_low']}")
    print(f"  T33 fallback TOTAL:             {t33_total}")
    print(f"  Unmatched (source=none):        {stats['none']}")
    print(f"  Slugs matched to FF type:       {ff_slug_matches}")
    print(f"  FMP FF products available:      {ff_total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
