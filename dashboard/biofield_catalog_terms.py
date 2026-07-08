"""Curated catalog vocabulary for Biofield Intake transcription accuracy.

Two consumers, one curated list:
  * Deepgram live transcription -- fed as `keyterm` boosts so coined remedy /
    formulation names come through the ASR clean instead of mangled.
  * The LLM causal-chain interpreter -- fed as a normalization glossary so a
    clearly-misheard remedy can be mapped to its catalog spelling.

Why curate: `fmp_snap_products` holds ~1200 rows, most of them equipment and
packaging ("100 mL wide mouth bottle", "172 Hz Tuning Fork", "Cookware Set"),
and Deepgram nova-3 caps keyterms near 100. Dumping the table would spend the
whole boost budget on tea balls. So we prioritize:
  1. fixed clinical/method vocabulary (Glen's framework terms),
  2. remedies he has ACTUALLY prescribed (biofield_auth_chain),
  3. remaining formulation names, with equipment/packaging filtered out,
deduped case-insensitively, highest priority first, then truncated to the cap.
"""
import re
from urllib.parse import quote

# Deepgram nova-3 keyterm practical ceiling.
KEYTERM_CAP = 100
# The LLM has no such limit; give it a wider glossary for normalization.
GLOSSARY_CAP = 300

# Framework / method terms Glen speaks that the ASR does not know and that do
# NOT live in the product table. Highest priority -- always boosted.
CLINICAL_VOCAB = [
    "Terrain Restore", "Terrain Support", "infoceutical", "biofield",
    "causal chain", "gemmotherapy", "gem elixir", "flower essence", "ORMUS",
    "Perelandra", "syntropy", "homeopathic", "tincture", "regenerative peptide",
    "Energetic Driver", "Energetic Star", "meridian", "bio-energetic", "BSI",
    "epigenetic", "most affected", "head and tail", "muscle testing",
]

# A name is equipment/packaging (skip it) if it starts with a digit
# ("100 mL ...", "172 Hz ...", "23 Piece ...") or contains one of these tokens.
_EQUIPMENT_TOKENS = (
    "machine", "tuning fork", "quartz bowl", "quartz frosted", "tea ball",
    "cookware", "mesh", "cable", "hammer", "wide mouth bottle", "bottle for",
    "spooky scalar", "capsule filling", "cm ", " ml ",
)

_TERRAIN_RE = re.compile(r"\s+in\s+terrain\s+restore\s*$", re.I)


def _has(cx, table):
    return cx.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _strip_terrain(name):
    """'Foxglove Flower Essence in Terrain Restore' -> 'Foxglove Flower Essence'.
    The delivery base is boosted separately via CLINICAL_VOCAB."""
    return _TERRAIN_RE.sub("", (name or "").strip()).strip()


def _is_equipment(name):
    n = (name or "").strip().lower()
    if not n:
        return True
    if n[0].isdigit():
        return True
    return any(tok in n for tok in _EQUIPMENT_TOKENS)


def build_terms(cx, cap=KEYTERM_CAP):
    """Curated, deduped, priority-ordered term list (<= cap). cx is a sqlite conn.

    Priority: clinical vocab -> prescribed remedies (by usage) -> filtered
    formulation names. Clinical vocab and prescribed remedies are added first so
    they always survive the cap."""
    seen, out = set(), []

    def add(term):
        term = (term or "").strip()
        if not term:
            return
        key = term.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(term)

    for t in CLINICAL_VOCAB:
        add(t)

    if _has(cx, "biofield_auth_chain"):
        rows = cx.execute(
            "SELECT remedy, COUNT(*) c FROM biofield_auth_chain "
            "WHERE TRIM(COALESCE(remedy,''))<>'' "
            "GROUP BY LOWER(remedy) ORDER BY c DESC"
        ).fetchall()
        for (remedy, _c) in rows:
            add(_strip_terrain(remedy))

    if _has(cx, "fmp_snap_products") and len(out) < cap:
        rows = cx.execute(
            "SELECT DISTINCT product_name FROM fmp_snap_products "
            "WHERE TRIM(COALESCE(product_name,''))<>''"
        ).fetchall()
        for (name,) in rows:
            if len(out) >= cap:
                break
            if _is_equipment(name):
                continue
            add(_strip_terrain(name))

    return out[:cap]


def keyterm_query(terms):
    """Deepgram nova-3 query fragment: '&keyterm=<t1>&keyterm=<t2>...' (url-encoded)."""
    return "".join("&keyterm=" + quote(t) for t in terms if t)


def glossary_text(terms):
    """Comma-joined glossary for the LLM interpret prompt."""
    return ", ".join(t for t in terms if t)
