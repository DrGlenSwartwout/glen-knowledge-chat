"""Publish an authored Biofield Intake report to the illtowell.com client portal.

Pure / none-raising builder + an injectable prod POST. PHI stays local; only the
finished portal payload crosses to prod via the existing /admin/portal/upsert.
"""
import re

from dashboard.practitioner_portal import name_to_slug
from dashboard import wholesale_pricing as _pricing

# Protocol wordings that differ from the catalog. Keyed by alphanumeric-only,
# lowercased remedy text so "Focus, Neuromagnesium" and "Focus Neuro-Magnesium"
# collapse to the same key.
ALIAS_SLUGS = {
    "focusneuromagnesium": "neuro-magnesium",
    "communityspiritformulainterrainrestore": "terrain-restore",
}


def _norm_key(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def load_catalog():
    """The slug-keyed products map (data/products.json 'products')."""
    return _pricing._load_catalog()


def resolve_remedy_slug(name, catalog):
    """Resolve a protocol remedy name to a catalog slug: alias override first,
    then the in-repo fuzzy resolver. None when genuinely unresolvable."""
    if not (name or "").strip():
        return None
    alias = ALIAS_SLUGS.get(_norm_key(name))
    if alias:
        return alias
    return name_to_slug(name, catalog)


def _dosing(layer):
    parts = [(layer.get("dosage") or "").strip(),
             (layer.get("frequency") or "").strip(),
             (layer.get("timing") or "").strip()]
    return " ".join(p for p in parts if p)


def _cue_candidates(layer):
    """Ordered phrases to locate this layer in the narrative blob."""
    rem = (layer.get("remedy") or "").strip()
    out = []
    if rem:
        out.append(rem)
        first = rem.split(",")[0].strip()      # "Focus, Neuromagnesium" -> "Focus"
        if first and first != rem:
            out.append(first)
    head = (layer.get("head") or "").strip()
    if head:
        out.append(head)
    return out


def segment_narrative(narrative, layers):
    """Split the single narrative blob into one segment per layer, by locating
    each layer's cue (remedy, else its first word, else head) in increasing
    order. Returns a list aligned to ``layers``; ``[]`` when it cannot align."""
    text = narrative or ""
    if not text or not layers:
        return []
    low = text.lower()
    positions = []
    cursor = 0
    for layer in layers:
        found = -1
        for cue in _cue_candidates(layer):
            idx = low.find(cue.lower(), cursor)
            if idx != -1:
                found = idx
                break
        if found == -1:
            return []                          # a layer has no cue -> fall back
        positions.append(found)
        cursor = found + 1
    # positions are strictly increasing by construction (each search starts past
    # the previous hit). Slice between consecutive cue starts.
    segs = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        segs.append(text[start:end].strip())
    return segs
