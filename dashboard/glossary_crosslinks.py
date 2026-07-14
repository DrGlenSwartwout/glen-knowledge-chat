"""Phase 3 cross-links between the two glossaries: clinical Organs entries <->
E4L stress patterns, joined on organ name. E4L pattern pages already tag each
pattern with its organs (e4l_pattern_structures, stype='organ'); those names are
matched to the clinical Organs catalogue by a conservative canonical form
(normalise + drop a trailing "Gland" + singularise). No fuzzy matching."""
import re

from dashboard import pattern_glossary as _pg


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


# Curated aliases: E4L organ canonical form -> clinical organ canonical form, for
# spelling/naming differences the conservative canon() can't bridge. Add confident
# pairs only (a wrong join surfaces the wrong patterns).
_ORGAN_ALIASES = {
    "gallbladder": "gall bladder",       # E4L "Gallbladder" -> clinical "Gall Bladder"
    "ovarie": "ovary",                    # E4L "Ovaries" -> clinical "Ovary"
    "colon": "mucosa of the colon",       # E4L "Colon" -> clinical "Mucosa of the Colon"
}


def canon(name):
    """Conservative canonical organ name for matching. Falls back to the plain
    normalised form if the descriptor-stripping would empty it; then applies a
    small curated alias map for spelling/naming variants."""
    n = _norm(name)
    c = re.sub(r"\bglands?\b", "", n)
    c = re.sub(r"s\b", "", c)          # crude singularise (Ducts -> Duct)
    c = re.sub(r"\s+", " ", c).strip()
    c = c or n
    return _ORGAN_ALIASES.get(c, c)


def organ_to_patterns(cx):
    """{canonical organ name: [{slug, name}]} — for each organ tagged on an E4L
    pattern, the patterns (that have a glossary page) involving it. Deduped by
    pattern slug, sorted by name."""
    try:
        rows = cx.execute(
            "SELECT s.structure AS organ, i.code AS code, i.name AS name, i.full_name AS full_name "
            "FROM e4l_pattern_structures s JOIN e4l_items i ON i.code = s.code "
            "WHERE s.stype = 'organ'").fetchall()
    except Exception:
        return {}
    acc = {}
    for r in rows:
        code = (r["code"] or "").strip()
        if not code:
            continue
        slug = _pg.slug_for(code)
        name = (r["full_name"] or r["name"] or code).strip()
        acc.setdefault(canon(r["organ"]), {})[slug] = {"slug": slug, "name": name}
    return {k: sorted(v.values(), key=lambda p: p["name"].lower()) for k, v in acc.items()}


def clinical_organ_index(catalog):
    """{canonical organ name: clinical organ entry slug} from the catalogue's
    Organs dimension (first wins)."""
    idx = {}
    for d in (catalog or {}).get("dimensions", []):
        if d.get("key") != "organs":
            continue
        for e in d.get("entries", []):
            c = canon(e.get("name", ""))
            if c:
                idx.setdefault(c, e.get("slug"))
    return idx
