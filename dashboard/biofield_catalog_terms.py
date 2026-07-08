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

# Deepgram caps keyterm prompting at 500 TOKENS across all keyterms (not a count,
# not a URL length) -- exceeding it fails the socket with a bare 400 Bad Request:
#   "Keyterm limit exceeded. The maximum number of tokens across all keyterms is 500."
# Glen's terms are rare medical/botanical words, so they tokenize densely: 100 of
# them blew past 500. We budget tokens with headroom, and Deepgram's own guidance is
# to "focus on the most important 20-50 terms".
KEYTERM_TOKEN_BUDGET = 400
# Belt-and-braces count cap; the token budget is what actually binds.
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

# A name is equipment/packaging/merch (skip it) if it starts with a digit
# ("100 mL ...", "172 Hz ...", "23 Piece ...") or contains one of these tokens.
# Boost budget is scarce -- it must not be spent on t-shirts and tuning forks.
_EQUIPMENT_TOKENS = (
    # equipment & packaging
    "machine", "tuning fork", "quartz bowl", "quartz frosted", "tea ball",
    "cookware", "mesh", "cable", "hammer", "wide mouth bottle", "bottle for",
    "spooky scalar", "capsule filling", "cm ", " ml ",
    # devices & materials
    "oscillator", "shielding", "millimeter wave", "ribbon", "vinyl", "oilcloth",
    # apparel / merch
    "shirt", "boxer", "underwear", "briefs", "sleeve", "sock",
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


def estimate_tokens(term):
    """Conservative upper bound on how many tokens a keyterm costs Deepgram.

    Deepgram does not publish its tokenizer, so we over-estimate rather than risk a
    400. Glen's vocabulary is rare words ('gemmotherapy', 'Perelandra') that split
    into many subword tokens; ~3 chars/token matched the observed 500-token cutoff.
    Each term also carries at least one token per whitespace-separated word."""
    term = (term or "").strip()
    if not term:
        return 0
    words = len(term.split())
    return max(words, -(-len(term) // 3))  # ceil(len/3), never fewer than word count


def build_terms(cx, cap=KEYTERM_CAP, token_budget=None):
    """Curated, deduped, priority-ordered term list. cx is a sqlite conn.

    Priority: clinical vocab -> prescribed remedies (by usage) -> filtered
    formulation names, so the most valuable terms survive truncation.

    Bounded by `cap` (count) and, when `token_budget` is given, by the estimated
    total token cost -- Deepgram's real limit. A term that would overflow the
    budget is skipped, but cheaper later terms may still fit."""
    seen, out = set(), []
    spent = [0]

    def add(term):
        term = (term or "").strip()
        if not term or len(out) >= cap:
            return
        key = term.lower()
        if key in seen:
            return
        if token_budget is not None:
            cost = estimate_tokens(term)
            if spent[0] + cost > token_budget:
                return  # skip this one; a shorter later term may still fit
            spent[0] += cost
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

    if _has(cx, "fmp_snap_products"):
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

    return out


def build_keyterms(cx):
    """The list actually sent to Deepgram: count-capped AND token-budgeted."""
    return build_terms(cx, cap=KEYTERM_CAP, token_budget=KEYTERM_TOKEN_BUDGET)


def keyterm_query(terms):
    """Deepgram nova-3 query fragment: '&keyterm=<t1>&keyterm=<t2>...' (url-encoded)."""
    return "".join("&keyterm=" + quote(t) for t in terms if t)


def glossary_text(terms):
    """Comma-joined glossary for the LLM interpret prompt."""
    return ", ".join(t for t in terms if t)
