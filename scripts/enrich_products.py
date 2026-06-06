#!/usr/bin/env python3
"""
Product-catalog enrichment — matching core + optional Pinecone (GK) descriptions.

Reconciles products.json slugs with authoritative ingredients from:
  1. FMP new (products.csv + products_items.csv + ingredients.csv) — primary
  2. T33_FORMULAS.csv — fallback for unmatched OR FMP-matched-but-empty (Stage A.1)
  3. GrooveKart page copy from Pinecone (Stage B, --with-gk) — last-resort ingredient
     source + descriptions + stale-GK diff.

Ingredient source priority (Glen's rule):
  FMP-new (non-empty) -> T33 (non-empty) -> GK-parsed.
A product that matches an FMP record but has ZERO products_items rows is treated the
SAME as no FMP match: fall through to T33 (source "fmp_new_empty_t33").

Writes:
  data/products-enrich-candidate.json  — enrichment per slug (review before applying)
  data/products-enrich-report.md       — match quality summary + review lists
  data/products-stale-gk-report.md     — (Stage B only) stale-GK diff + GK-only flags

Does NOT touch data/products.json.

Usage:
    # offline (no network)
    python3 scripts/enrich_products.py

    # Stage B: add GK descriptions + stale-GK diff (needs PINECONE/OPENAI keys)
    doppler run -p remedy-match -c prd -- python3 scripts/enrich_products.py --with-gk
    doppler run -p remedy-match -c prd -- python3 scripts/enrich_products.py --with-gk --limit 15
"""

import argparse
import csv
import json
import os
import re
from difflib import get_close_matches, SequenceMatcher
from pathlib import Path

# Optional: label-ready dose computation (elemental/IU/%RDA from the FMP lookup +
# Glen's DV table). Imported defensively so this offline script never breaks if the
# dashboard package or its data file is unavailable.
try:
    from dashboard import ingredient_content as _ic
except Exception:
    _ic = None


def _attach_label(ing: dict) -> dict:
    """Attach ing['label'] = {amount, unit, rda_percent} for numeric mg/mcg/g
    ingredients via the ingredient_content module. No-op on any failure."""
    if not _ic:
        return ing
    qty = ing.get("qty")
    unit = ing.get("unit")
    if qty and unit in ("mg", "mcg", "g"):
        try:
            ing["label"] = _ic.label_dose(ing.get("name", ""), qty, unit)
        except Exception:
            pass
    return ing

# ── Paths (override via env vars or edit here) ────────────────────────────────
WORKTREE = Path(__file__).parent.parent
PRODUCTS_JSON = WORKTREE / "data" / "products.json"
CANDIDATE_JSON = WORKTREE / "data" / "products-enrich-candidate.json"
REPORT_MD = WORKTREE / "data" / "products-enrich-report.md"
STALE_GK_MD = WORKTREE / "data" / "products-stale-gk-report.md"

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
            _attach_label(
                {
                    "name": name,
                    "qty": qty,
                    "unit": it.get("unit_measurement", "").strip() or None,
                    "raw": zc_raw or None,
                }
            )
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
        ingredients.append(
            _attach_label({"name": name, "qty": qty, "unit": unit, "raw": line})
        )

    return top_block, ingredients


# ── Stage B: Pinecone (GrooveKart page copy) retrieval + parsing ──────────────
# Client construction mirrors app.py exactly to avoid version skew:
#   _oa  = OpenAI(api_key=OPENAI_API_KEY)
#   _pc  = Pinecone(api_key=PINECONE_API_KEY); _idx = _pc.Index("remedy-match-llc")
#   embed(text) -> _oa.embeddings.create(input=[text],
#                      model="text-embedding-ada-002").data[0].embedding
# _page_text() mirrors dashboard/product_content.py._page_text exactly:
#   top_k=30, namespace "specific-formulations", filter title==exact, sort chunk_index.

PINECONE_INDEX = "remedy-match-llc"
SPECIFIC_NS = "specific-formulations"
_EMBED_MODEL = "text-embedding-ada-002"


def build_gk_clients():
    """Construct OpenAI + Pinecone clients the way app.py does. Returns
    (idx, embed_fn). Raises on missing keys / import failure so the caller can
    STOP and report (per task: if keys missing, stop)."""
    from openai import OpenAI
    from pinecone import Pinecone

    oa_key = os.environ.get("OPENAI_API_KEY", "")
    pc_key = os.environ.get("PINECONE_API_KEY", "")
    if not oa_key:
        raise RuntimeError("OPENAI_API_KEY not set (need doppler env for --with-gk)")
    if not pc_key:
        raise RuntimeError("PINECONE_API_KEY not set (need doppler env for --with-gk)")

    _oa = OpenAI(api_key=oa_key)
    _pc = Pinecone(api_key=pc_key)
    _idx = _pc.Index(PINECONE_INDEX)

    def embed(text: str):
        return (
            _oa.embeddings.create(input=[text], model=_EMBED_MODEL)
            .data[0]
            .embedding
        )

    return _idx, embed


def gk_page_text(idx, embed, product: dict) -> dict | None:
    """Concatenated remedymatch.com page copy for a product, by EXACT title filter.
    Mirrors dashboard/product_content.py._page_text. Returns {text, url, n_chunks}
    or None. Resilient: returns None on any query failure (caller logs + continues)."""
    title = product.get("pinecone_title") or product.get("name")
    if not title:
        return None
    try:
        vec = embed(title)
        res = idx.query(
            vector=vec,
            top_k=30,
            namespace=SPECIFIC_NS,
            filter={"title": {"$eq": title}},
            include_metadata=True,
        )
        matches = res.matches if hasattr(res, "matches") else res.get("matches", [])
    except Exception as e:
        print(f"  [gk] query failed for {title!r}: {e}", flush=True)
        return None
    if not matches:
        return None
    matches = sorted(matches, key=lambda m: (m.metadata or {}).get("chunk_index", 0))
    text = "\n".join((m.metadata or {}).get("text", "") for m in matches).strip()
    md0 = matches[0].metadata or {}
    return {"text": text, "url": md0.get("url", ""), "n_chunks": len(matches)}


# GK "supplement facts" panels are flattened inline in the Pinecone page copy:
#   "...supplies: Mineral Factors: Calcium (...) 55 mg 6% Magnesium ... 24 mg 6% Zinc..."
# So we (1) locate the panel start, (2) split it into per-dosage segments, and
# (3) take the substance name that precedes each dosage token.

# Marks the start of the ingredient/supplement panel.
_GK_PANEL_START = re.compile(
    r"(?:each\s+(?:capsule|cap|vegicap|chlorophyll\s+\w+|serving|dropper|drop|scoop)[^:]*"
    r"supplies\s*:|supplies\s*:|"
    r"\bcontents?\s*:|\bingredients?\s*:|supplement\s+facts|other\s+ingredients?\s*:)",
    re.IGNORECASE,
)
# A dosage token: number + unit (mg/mcg/g/IU/ml) optionally followed by a %DV.
_GK_DOSE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mg|mcg|µg|ug|g|iu|ml)\b(?:\s*supplies\s*\d+(?:\.\d+)?\s*(?:iu|mg|mcg))?"
    r"(?:\s*[\d,]+\s*%|\s*\*)?",
    re.IGNORECASE,
)
# Category labels inside a panel (not ingredients themselves).
_GK_CATEGORY = re.compile(
    r"\b(?:mineral|vitamin|amine|amino\s*acid|botanical|endocrine|coenzyme|enzyme|"
    r"nutritional|other|active|herbal|antioxidant|fatty\s*acid|probiotic|mushroom|"
    r"adaptogen|nootropic|trace\s*element)\s+(?:factors?|complex|ingredients?|blend)\s*:",
    re.IGNORECASE,
)
# Boilerplate / prose markers that, if present in a candidate name, reject it.
_GK_PROSE = re.compile(
    r"\b(?:source|study|trial|review|meta-?analysis|randomized|risk|disease|patients?|"
    r"compared|comparing|increase|decrease|journal|et al|research|shows?|found|"
    r"according|et\b|www\.|http|price|retail|your price|add to cart)\b",
    re.IGNORECASE,
)


def _clean_ing_name(name: str) -> str:
    """Reduce a captured substance phrase to a clean ingredient name."""
    name = name.strip()
    # Remove bracketed inventory / form notes "[36g]" first (can be leading)
    name = re.sub(r"\s*\[[^\]]*\]", "", name).strip()
    # Drop any leading "X Factors:" / "Minerals:" / "Amines:" / "coenzyme form:"
    # category label and everything before its colon (panels are flattened, so a
    # category label can prefix the first ingredient in its group).
    name = _GK_CATEGORY.sub("", name).strip()
    name = re.sub(
        r"^.*?(?:factors?|complex|coenzymes?|minerals?|amines?|amino\s*acids?|"
        r"botanicals?|vitamins?|enzymes?|coenzyme\s*form|active\s*\w+\s*form)\s*:",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()
    # Drop leading bullets / list numbers / separators / stray close-paren
    name = re.sub(r"^[\-\*•‣◦⁃∙\d\.\)\]\s:;,]+", "", name).strip()
    # Keep parenthetical source descriptors but trim a trailing dangling "("
    name = re.sub(r"\s*\($", "", name).strip()
    # Tidy spaces inside parens: "( Bambusa spp .)" -> "(Bambusa spp.)"
    name = re.sub(r"\(\s+", "(", name)
    name = re.sub(r"\s+\)", ")", name)
    name = re.sub(r"\s+\.", ".", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip(" .,;:-")
    return name


def parse_gk_ingredients(page_text: str) -> list[str]:
    """Extract ingredient NAMES from a GK page's (flattened) supplement panel.

    Strategy: locate the panel start ('supplies:'/'Contents:'/'Ingredients:'/
    'Supplement Facts'), then walk every dosage token within the panel region and
    take the substance phrase immediately preceding it (bounded by the previous
    dosage token). De-duplicated, order-preserving cleaned names.
    """
    if not page_text:
        return []

    # Flatten newlines so an inline panel that spans chunk boundaries stays whole.
    flat = re.sub(r"\s+", " ", page_text)

    starts = [m.start() for m in _GK_PANEL_START.finditer(flat)]
    if not starts:
        return []

    names: list[str] = []
    seen: set[str] = set()

    for si, start in enumerate(starts):
        # Panel region runs to the next panel start (or end of text).
        end = starts[si + 1] if si + 1 < len(starts) else len(flat)
        region = flat[start:end]

        # Walk dosage tokens; name = text between previous token end and this token.
        prev_end = 0
        # Skip the panel-start marker itself.
        hdr = _GK_PANEL_START.match(region)
        if hdr:
            prev_end = hdr.end()

        for m in _GK_DOSE.finditer(region):
            seg = region[prev_end:m.start()]
            prev_end = m.end()
            # The substance name is the LAST clause in the preceding segment
            # (earlier clauses belong to the prior ingredient's trailing prose).
            # Cut at sentence/citation boundaries to drop research prose.
            seg = re.split(r"(?<=[.!?])\s+(?=[A-Z])", seg)[-1]
            nm = _clean_ing_name(seg)
            if not nm or len(nm) < 2 or len(nm) > 70:
                continue
            if not re.search(r"[A-Za-z]", nm):
                continue
            if _GK_PROSE.search(nm):
                continue
            # A real ingredient name is short (<= ~7 words).
            if len(nm.split()) > 8:
                # keep only the trailing few words (likely the substance)
                nm = " ".join(nm.split()[-6:])
            key = normalize(nm)
            if not key or len(key) < 2 or key in seen:
                continue
            seen.add(key)
            names.append(nm)

    return names


def clean_description(page_text: str, max_chars: int = 1200) -> str:
    """Trim + collapse GK page copy into a description blob (no ingredient parsing)."""
    if not page_text:
        return ""
    txt = re.sub(r"[ \t]+", " ", page_text)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    if len(txt) > max_chars:
        txt = txt[:max_chars].rstrip() + "…"
    return txt


def ingredient_name_set(ingredients: list) -> set[str]:
    """Normalized name set from a list of ingredient dicts or strings."""
    out: set[str] = set()
    for ing in ingredients:
        if isinstance(ing, dict):
            nm = ing.get("name", "")
        else:
            nm = ing
        n = normalize(nm)
        if n and len(n) >= 2:
            out.add(n)
    return out


def gk_name_set(gk_names: list[str]) -> set[str]:
    out: set[str] = set()
    for nm in gk_names:
        n = normalize(nm)
        if n and len(n) >= 2:
            out.add(n)
    return out


def _soft_in(name: str, other: set[str]) -> bool:
    """True if `name` is present in `other` allowing substring containment either
    way (e.g. 'vitamin c' matches 'vitamin c ascorbic acid'). Reduces false
    'stale' flags from differing phrasings of the same substance."""
    if name in other:
        return True
    for o in other:
        if name in o or o in name:
            # Require the shorter to be a meaningful chunk (>=4 chars) to avoid
            # 'b' matching 'biotin'.
            shorter = name if len(name) <= len(o) else o
            if len(shorter) >= 4:
                return True
    return False


def _set_diff(a: set[str], b: set[str]) -> set[str]:
    """Names in `a` that have no soft match in `b`."""
    return {x for x in a if not _soft_in(x, b)}


def pdata_name(products: dict, slug: str) -> str:
    p = products.get(slug, {})
    return p.get("name", "") or p.get("pinecone_title", "") or slug


# ── Main ──────────────────────────────────────────────────────────────────────

def _t33_lookup_entry(
    match_key: str,
    t33_rows: list,
    t33_by_norm: dict,
    t33_norms: list,
) -> tuple[dict | None, dict]:
    """Match a key against T33 and parse the top-block ingredients.

    Returns (low_conf_review_item_or_None, partial_entry_fields) where
    partial_entry_fields has keys: t33_score, t33_name, formula_text, ingredients,
    t33_confidence. If no match, ingredients == [] and t33_confidence == "none".
    """
    t33_idx, t33_conf, t33_score = match_t33(match_key, t33_by_norm, t33_norms)
    if t33_idx is None:
        return None, {
            "t33_confidence": "none",
            "t33_score": round(t33_score, 3),
            "ingredients": [],
        }
    t33_row = t33_rows[t33_idx]
    formula_text = t33_row.get("Key Ingredients for Formula", "").strip()
    top_block, ingredients = parse_t33_top_block(formula_text)
    fields = {
        "t33_confidence": t33_conf,
        "t33_score": round(t33_score, 3),
        "t33_name": t33_row.get("Name", "").replace("\n", " | "),
        "formula_text": top_block,
        "ingredients": ingredients,
    }
    review = None
    if t33_conf == "low":
        review = {
            "fmp_name": t33_row.get("Name", "").split("\n")[0],
            "score": t33_score,
        }
    return review, fields


def main() -> None:
    ap = argparse.ArgumentParser(description="Product-catalog enrichment")
    ap.add_argument(
        "--with-gk",
        "--stage-b",
        dest="with_gk",
        action="store_true",
        help="Stage B: pull GrooveKart descriptions from Pinecone, parse GK "
        "ingredients, resolve final ingredient priority, write stale-GK report.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stage B only: limit Pinecone queries to the first N slugs (sampling).",
    )
    args = ap.parse_args()

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
        "fmp_new_empty_t33_high": 0,
        "fmp_new_empty_t33_medium": 0,
        "fmp_new_empty_t33_low": 0,
        # FMP matched but empty AND T33 also had nothing -> stays fmp_new (empty)
        "fmp_new_empty_nofallback": 0,
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
            fmp_ingredients = resolve_ingredients(fmp_id, fmp_items, fmp_ings)

            if fmp_ingredients:
                # ── FMP has ingredients: authoritative new-FMP source ───────────
                entry: dict = {
                    "source": "fmp_new",
                    "confidence": fmp_conf,
                    "fmp_score": round(fmp_score, 3),
                    "fmp_id": fmp_id,
                    "fmp_name": fmp_row["product_name"],
                    "fmp_type": fmp_row["type"],
                    "ingredients": fmp_ingredients,
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

            # ── Stage A.1: FMP matched but ZERO products_items rows ─────────────
            # Per Glen's rule: treat the same as no-FMP-match -> fall through to
            # T33 for ingredients, but KEEP the FMP identity/confidence.
            t33_review, t33_fields = _t33_lookup_entry(
                match_key, t33_rows, t33_by_norm, t33_norms
            )
            entry = {
                "source": "fmp_new_empty_t33",
                "confidence": fmp_conf,  # keep the FMP identity confidence
                "fmp_score": round(fmp_score, 3),
                "fmp_id": fmp_id,
                "fmp_name": fmp_row["product_name"],
                "fmp_type": fmp_row["type"],
                "fmp_empty": True,
                "t33_confidence": t33_fields["t33_confidence"],
                "t33_score": t33_fields["t33_score"],
                "t33_name": t33_fields.get("t33_name", ""),
                "formula_text": t33_fields.get("formula_text", ""),
                "ingredients": t33_fields["ingredients"],
            }
            if fmp_row["type"] == "Functional Formulation":
                ff_slug_matches += 1
            if t33_fields["ingredients"]:
                stats[f"fmp_new_empty_t33_{fmp_conf}"] += 1
            else:
                # FMP empty AND T33 found nothing -> still source none of ingredients
                entry["source"] = "fmp_new_empty_nofallback"
                stats["fmp_new_empty_nofallback"] += 1
            if t33_review is not None and t33_fields["ingredients"]:
                low_conf_items.append(
                    {
                        "slug": slug,
                        "name": slug_name,
                        "pinecone_title": pinecone_title,
                        "fmp_name": f"(T33) {t33_review['fmp_name']}",
                        "score": t33_review["score"],
                        "source": "fmp_new_empty_t33",
                    }
                )
            candidate[slug] = entry
            continue

        # ── T33 fallback (no FMP identity match at all) ──────────────────────────
        t33_review, t33_fields = _t33_lookup_entry(
            match_key, t33_rows, t33_by_norm, t33_norms
        )

        if t33_fields["ingredients"] or t33_fields["t33_confidence"] != "none":
            t33_conf = t33_fields["t33_confidence"]
            entry = {
                "source": "fmp_old_t33",
                "confidence": t33_conf,
                "t33_score": t33_fields["t33_score"],
                "t33_name": t33_fields.get("t33_name", ""),
                "formula_text": t33_fields.get("formula_text", ""),
                "ingredients": t33_fields["ingredients"],
            }
            stats[f"fmp_old_t33_{t33_conf}"] += 1
            if t33_review is not None:
                low_conf_items.append(
                    {
                        "slug": slug,
                        "name": slug_name,
                        "pinecone_title": pinecone_title,
                        "fmp_name": t33_review["fmp_name"],
                        "score": t33_review["score"],
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

    # ── Resolve authoritative ingredient source (offline priority) ─────────────
    # Priority so far (offline): FMP-new(non-empty) -> T33(non-empty). GK fills in
    # Stage B. Record ingredients_source on every entry now; Stage B may upgrade
    # "none" -> "gk".
    for slug, entry in candidate.items():
        src = entry.get("source")
        if src == "fmp_new":
            entry["ingredients_source"] = "fmp_new"
        elif src in ("fmp_new_empty_t33", "fmp_old_t33") and entry.get("ingredients"):
            entry["ingredients_source"] = "t33"
        else:
            entry["ingredients_source"] = "none"

    # ── Stage B: Pinecone GK descriptions + GK ingredient parse + priority + diff
    stale_items: list[dict] = []      # authoritative vs GK differ materially
    gk_only_items: list[dict] = []    # ingredients_source == gk (unverified)
    no_ing_items: list[dict] = []     # no ingredients anywhere
    gk_stats = {
        "queried": 0,
        "page_found": 0,
        "page_missing": 0,
        "query_failed": 0,
        "gk_ingredients_parsed": 0,
    }

    if args.with_gk:
        print("\nStage B: connecting to Pinecone for GK descriptions...")
        try:
            idx, embed = build_gk_clients()
        except Exception as e:
            print(f"\nSTOP: cannot run Stage B — {e}")
            print("Offline candidate/report were NOT written (use no flag for offline).")
            raise SystemExit(2)

        slugs = list(products.keys())
        if args.limit and args.limit > 0:
            slugs = slugs[: args.limit]
            print(f"  --limit {args.limit}: sampling first {len(slugs)} slugs")

        for i, slug in enumerate(slugs, 1):
            pdata = products[slug]
            entry = candidate[slug]
            gk_stats["queried"] += 1
            page = gk_page_text(idx, embed, pdata)
            if page is None:
                gk_stats["page_missing"] += 1
                entry["description"] = ""
                entry["gk_ingredients"] = []
                continue
            gk_stats["page_found"] += 1
            entry["description"] = clean_description(page.get("text", ""))
            gk_names = parse_gk_ingredients(page.get("text", ""))
            entry["gk_ingredients"] = gk_names
            if gk_names:
                gk_stats["gk_ingredients_parsed"] += 1

            if i % 25 == 0:
                print(f"  ...{i}/{len(slugs)} slugs queried")

        print(
            f"  GK pages: {gk_stats['page_found']} found, "
            f"{gk_stats['page_missing']} missing of {gk_stats['queried']} queried"
        )

    # ── Final ingredient-source resolution + diff (runs whenever GK present) ────
    # FMP-new(non-empty) -> T33(non-empty) -> GK-parsed -> none.
    for slug, entry in candidate.items():
        auth_ings = entry.get("ingredients") or []
        auth_src = entry.get("ingredients_source", "none")
        gk_names = entry.get("gk_ingredients", []) if args.with_gk else []

        if auth_src in ("fmp_new", "t33") and auth_ings:
            # Authoritative source exists. If GK page exists, diff it.
            if args.with_gk and gk_names:
                a_set = ingredient_name_set(auth_ings)
                g_set = gk_name_set(gk_names)
                # Material difference: anything on GK not in auth, or vice versa.
                # Use a soft containment check to reduce phrasing noise.
                added = sorted(_set_diff(g_set, a_set))    # on GK, not authoritative
                removed = sorted(_set_diff(a_set, g_set))  # authoritative, not on GK
                if added or removed:
                    stale_items.append(
                        {
                            "slug": slug,
                            "name": pdata_name(products, slug),
                            "auth_source": auth_src,
                            "added_on_gk": added,
                            "removed_from_gk": removed,
                            "auth_count": len(a_set),
                            "gk_count": len(g_set),
                        }
                    )
        elif args.with_gk and gk_names:
            # No authoritative ingredients -> GK is the source (unverified).
            entry["ingredients"] = [{"name": n, "qty": None, "unit": None, "raw": n}
                                    for n in gk_names]
            entry["ingredients_source"] = "gk"
            gk_only_items.append(
                {
                    "slug": slug,
                    "name": pdata_name(products, slug),
                    "gk_ingredients": gk_names,
                }
            )

        # Recompute "no ingredients anywhere" after GK fill.
        if not (entry.get("ingredients") or []):
            entry["ingredients_source"] = "none"
            no_ing_items.append(
                {
                    "slug": slug,
                    "name": pdata_name(products, slug),
                    "source": entry.get("source"),
                }
            )

    # ── Write candidate JSON ───────────────────────────────────────────────────
    CANDIDATE_JSON.write_text(
        json.dumps(candidate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote: {CANDIDATE_JSON}")

    # ── Write report MD ────────────────────────────────────────────────────────
    total = len(products)
    fmp_new_total = stats["fmp_new_high"] + stats["fmp_new_medium"] + stats["fmp_new_low"]
    fmp_empty_t33_total = (
        stats["fmp_new_empty_t33_high"]
        + stats["fmp_new_empty_t33_medium"]
        + stats["fmp_new_empty_t33_low"]
    )
    t33_total = (
        stats["fmp_old_t33_high"]
        + stats["fmp_old_t33_medium"]
        + stats["fmp_old_t33_low"]
    )

    # ingredients_source breakdown across the full candidate set
    isrc = {"fmp_new": 0, "t33": 0, "gk": 0, "none": 0}
    for e in candidate.values():
        isrc[e.get("ingredients_source", "none")] = (
            isrc.get(e.get("ingredients_source", "none"), 0) + 1
        )

    report_lines = [
        "# Products Enrichment — Match Report (Stage A + A.1)",
        "",
        f"Generated: 2026-06-05  |  Source: `scripts/enrich_products.py`"
        + ("  |  Stage B (GK) run" if args.with_gk else "  |  offline run"),
        "",
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total slugs | {total} |",
        f"| FMP-new matched (high conf) | {stats['fmp_new_high']} |",
        f"| FMP-new matched (medium conf) | {stats['fmp_new_medium']} |",
        f"| FMP-new matched (low conf) | {stats['fmp_new_low']} |",
        f"| **FMP-new total (non-empty)** | **{fmp_new_total}** |",
        f"| FMP-matched-but-empty -> T33 ingredients (high) | {stats['fmp_new_empty_t33_high']} |",
        f"| FMP-matched-but-empty -> T33 ingredients (medium) | {stats['fmp_new_empty_t33_medium']} |",
        f"| FMP-matched-but-empty -> T33 ingredients (low) | {stats['fmp_new_empty_t33_low']} |",
        f"| **FMP-empty -> T33 total (source=fmp_new_empty_t33)** | **{fmp_empty_t33_total}** |",
        f"| FMP-empty AND T33 also empty (no ingredients) | {stats['fmp_new_empty_nofallback']} |",
        f"| T33 fallback, no FMP match (high conf) | {stats['fmp_old_t33_high']} |",
        f"| T33 fallback, no FMP match (medium conf) | {stats['fmp_old_t33_medium']} |",
        f"| T33 fallback, no FMP match (low conf) | {stats['fmp_old_t33_low']} |",
        f"| **T33-only fallback total** | **{t33_total}** |",
        f"| No match (no FMP, no T33) | {stats['none']} |",
        f"| Slugs matching a Functional Formulation FMP type | {ff_slug_matches} |",
        f"| FMP Functional Formulation products available | {ff_total} |",
        "",
        "## Final ingredients_source breakdown",
        "",
        f"| ingredients_source | Count |",
        f"|--------------------|-------|",
        f"| fmp_new | {isrc.get('fmp_new', 0)} |",
        f"| t33 | {isrc.get('t33', 0)} |",
        f"| gk | {isrc.get('gk', 0)} |",
        f"| none | {isrc.get('none', 0)} |",
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

    # ── Write stale-GK report (Stage B only) ───────────────────────────────────
    if args.with_gk:
        queried_n = gk_stats["queried"]
        sg = [
            "# Products — Stale GrooveKart Report (Stage B)",
            "",
            f"Generated: 2026-06-05  |  Source: `scripts/enrich_products.py --with-gk`"
            + (f" --limit {args.limit}" if args.limit else ""),
            "",
            "Ingredient source priority: **FMP-new (non-empty) -> T33 (non-empty) -> "
            "GK-parsed**. Where an authoritative set (FMP or T33) AND a GK page both "
            "exist, the GK ingredient names are diffed against the authoritative set; "
            "a material difference means the GrooveKart sales page is STALE (FMP/T33 "
            "wins, the page needs updating).",
            "",
            "## Counts",
            "",
            f"| Metric | Count |",
            f"|--------|-------|",
            f"| Slugs queried against Pinecone | {queried_n} |",
            f"| GK page found | {gk_stats['page_found']} |",
            f"| GK page missing | {gk_stats['page_missing']} |",
            f"| GK page had parseable ingredients | {gk_stats['gk_ingredients_parsed']} |",
            f"| **STALE GK pages (authoritative vs GK differ)** | **{len(stale_items)}** |",
            f"| **GK-only ingredients (unverified, no FMP/T33)** | **{len(gk_only_items)}** |",
            f"| **No ingredients anywhere** | **{len(no_ing_items)}** |",
            "",
            "## (a) Stale GK pages — authoritative (FMP/T33) vs GrooveKart differ",
            "",
            "`+ on GK` = on the GK page but NOT in the authoritative set (likely an "
            "outdated/extra page ingredient). `- missing on GK` = in the authoritative "
            "set but NOT mentioned on the GK page (page omits a current ingredient).",
            "",
            "| Slug | Name | Auth | Auth# | GK# | + on GK | - missing on GK |",
            "|------|------|------|-------|-----|---------|------------------|",
        ]
        for it in sorted(stale_items, key=lambda x: -(len(x["added_on_gk"]) + len(x["removed_from_gk"]))):
            sg.append(
                f"| {it['slug']} | {it['name']} | {it['auth_source']} "
                f"| {it['auth_count']} | {it['gk_count']} "
                f"| {', '.join(it['added_on_gk']) or '—'} "
                f"| {', '.join(it['removed_from_gk']) or '—'} |"
            )

        sg += [
            "",
            "## (b) GK-only products (ingredients_source = gk, UNVERIFIED)",
            "",
            "These have NO FMP and NO T33 record. Ingredients were taken from the "
            "GrooveKart page only and are NOT cross-checked against an authoritative "
            "source. Verify before publishing.",
            "",
            "| Slug | Name | GK ingredients (parsed) |",
            "|------|------|--------------------------|",
        ]
        for it in gk_only_items:
            ings = ", ".join(it["gk_ingredients"][:20])
            if len(it["gk_ingredients"]) > 20:
                ings += f", …(+{len(it['gk_ingredients']) - 20})"
            sg.append(f"| {it['slug']} | {it['name']} | {ings} |")

        sg += [
            "",
            "## (c) Products with NO ingredients anywhere (FMP empty, no T33, no GK)",
            "",
            "Need manual ingredient entry. (If `--limit` was used, slugs not yet "
            "queried against GK appear here too.)",
            "",
            "| Slug | Name | Source |",
            "|------|------|--------|",
        ]
        for it in no_ing_items:
            sg.append(f"| {it['slug']} | {it['name']} | {it['source']} |")

        STALE_GK_MD.write_text("\n".join(sg) + "\n", encoding="utf-8")
        print(f"Wrote: {STALE_GK_MD}")

    # ── stdout summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("ENRICHMENT MATCHING SUMMARY")
    print("=" * 64)
    print(f"  Total slugs:                      {total}")
    print(f"  FMP-new (non-empty) high/med/low: "
          f"{stats['fmp_new_high']}/{stats['fmp_new_medium']}/{stats['fmp_new_low']}"
          f"  = {fmp_new_total}")
    print(f"  FMP-empty -> T33  high/med/low:   "
          f"{stats['fmp_new_empty_t33_high']}/{stats['fmp_new_empty_t33_medium']}/"
          f"{stats['fmp_new_empty_t33_low']}  = {fmp_empty_t33_total}")
    print(f"  FMP-empty AND T33-empty:          {stats['fmp_new_empty_nofallback']}")
    print(f"  T33-only fallback high/med/low:   "
          f"{stats['fmp_old_t33_high']}/{stats['fmp_old_t33_medium']}/"
          f"{stats['fmp_old_t33_low']}  = {t33_total}")
    print(f"  No match (source=none):           {stats['none']}")
    print(f"  Slugs matched to FF type:         {ff_slug_matches}")
    print(f"  FMP FF products available:        {ff_total}")
    print("-" * 64)
    print("  ingredients_source breakdown:")
    print(f"    fmp_new: {isrc.get('fmp_new', 0)}   t33: {isrc.get('t33', 0)}   "
          f"gk: {isrc.get('gk', 0)}   none: {isrc.get('none', 0)}")
    if args.with_gk:
        print("-" * 64)
        print(f"  STALE GK pages:        {len(stale_items)}")
        print(f"  GK-only (unverified):  {len(gk_only_items)}")
        print(f"  No ingredients at all: {len(no_ing_items)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
