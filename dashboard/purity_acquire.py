"""Excipient acquisition orchestrator (purity Phase 2b).

acquire(product) resolves a product's Other Ingredients through the source
cascade and returns a uniform {raw, parsed, source, ok} shape. Phase 2b ships
Source A only (the Fullscript public product page). DB-free by contract: it
makes the slow network + model calls and MUST run outside _db_lock; the caller
(the console route) does the DB write.

Failure is never fatal and never optimistic: any miss returns parsed=None, and
the caller screens None -> 'unrated' (never green). The fabrication guard lives
in document_extract.extract_other_ingredients; this module only orchestrates
and splits.
"""
import re

from dashboard import fullscript_ingredients as _fi
from dashboard import document_extract as _dx

_MISS = {"raw": "", "parsed": None, "source": "fullscript", "ok": False}

_LABEL_RE = re.compile(r"^\s*(other|non[- ]?medicinal)\s+ingredients?\s*:\s*", re.I)


def split_other_ingredients(line):
    """Split an Other Ingredients line into individual items on commas,
    semicolons, and the word 'and'. Strips a leading label and a trailing
    period; drops empties. Parenthetical descriptors are kept -- the screen's
    _normalize handles them."""
    s = _LABEL_RE.sub("", line or "").strip().rstrip(".")
    if not s:
        return []
    parts = re.split(r"\s*,\s*|\s*;\s*|\s+and\s+", s)
    return [p.strip() for p in parts if p.strip()]


def acquire(product, *, fetch=None, call_model=None):
    """Acquire Other Ingredients for `product` (dict: product_slug, name,
    brand, optional sku). Returns {raw, parsed, source, ok}. DB-free; holds no
    lock; never raises."""
    try:
        slug = (product or {}).get("product_slug")
        text = _fi.fetch_page_text(slug, fetch=fetch)
        if not text:
            return dict(_MISS)
        line = _dx.extract_other_ingredients(
            text, name=product.get("name") or "", brand=product.get("brand") or "",
            sku=product.get("sku") or "", slug=slug or "", call_model=call_model)
        if line is None:                     # not found / unverifiable
            return dict(_MISS)               # -> parsed None -> unrated, never green
        if line == "":                       # verifiably lists NO other ingredients
            return {"raw": "None (no other ingredients listed)",
                    "parsed": [], "source": "fullscript", "ok": True}  # -> green
        parsed = split_other_ingredients(line)
        if not parsed:                       # verified line parsed to zero items
            return dict(_MISS)               # -> parsed None -> unrated, never green
        return {"raw": line, "parsed": parsed, "source": "fullscript", "ok": True}
    except Exception:
        return dict(_MISS)
