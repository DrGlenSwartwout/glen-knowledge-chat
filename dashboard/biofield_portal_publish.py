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
