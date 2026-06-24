"""Ingredient resolver for the ingredient page. Maps a URL slug to an ingredient
name + its FMP record, the formulations that use it, and its research studies."""
import json
import re
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_FMP = _ROOT / "data" / "fmp-ingredient-content.json"
_PRODUCTS = _ROOT / "data" / "products.json"


def slugify(name):
    s = (name or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-")[:40]


@lru_cache(maxsize=1)
def _fmp_records():
    """Return the ingredients sub-dict from fmp-ingredient-content.json.
    The file has top-level keys _source and ingredients; we want the latter."""
    try:
        raw = json.loads(_FMP.read_text())
        if isinstance(raw, dict):
            return raw.get("ingredients", {}) or {}
        return {}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _name_index():
    """{slug: canonical_name} over all known ingredient names (FMP + products)."""
    idx = {}
    for rec in _fmp_records().values():
        if not isinstance(rec, dict):
            continue
        nm = (rec.get("name") or "").strip().replace("\n", " ")
        if nm:
            idx.setdefault(slugify(nm), nm)
    try:
        prods = json.loads(_PRODUCTS.read_text()).get("products", {})
    except Exception:
        prods = {}
    for p in prods.values():
        for ing in (p.get("ingredients") or []):
            nm = (ing.get("name") if isinstance(ing, dict) else ing) or ""
            nm = nm.strip()
            if nm:
                idx.setdefault(slugify(nm), nm)
    return idx


def _fmp_for(name):
    try:
        from dashboard import ingredient_content
        return ingredient_content.get(name) or {}
    except Exception:
        return {}


def resolve(slug):
    name = _name_index().get(slug)
    if not name:
        return None
    return {"slug": slug, "name": name, "fmp": _fmp_for(name)}


def formulations_with(name):
    target = slugify(name)
    out = []
    try:
        prods = json.loads(_PRODUCTS.read_text()).get("products", {})
    except Exception:
        return out
    for pslug, p in prods.items():
        for ing in (p.get("ingredients") or []):
            nm = (ing.get("name") if isinstance(ing, dict) else ing) or ""
            if slugify(nm) == target:
                out.append({"slug": pslug, "name": p.get("name", pslug)})
                break
    return out


def research_studies(name, k=12):
    try:
        from dashboard import product_content
        return product_content._research_sources(name, k=k) or []
    except Exception:
        return []
