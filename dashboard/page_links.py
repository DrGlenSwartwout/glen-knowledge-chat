"""Pure deterministic matcher: surface already-published pages as chat link cards.

No Flask, no DB, no network. build_index() turns a list of approved-page records
into a phrase lookup; match_page_links() finds whole-word phrase hits in text
(longest-first) and returns ready-to-render card dicts. Never raises on normal input.
"""
import json
import re

_SUB_BY_KIND = {
    "topic": "Read the guide",
    "ingredient": "See the ingredient",
    "product": "View product",
}


def load_aliases(path):
    """Read a {phrase: slug} JSON map. Missing/unreadable/invalid -> {}."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return {str(k).strip().lower(): str(v).strip()
                    for k, v in data.items() if str(k).strip() and str(v).strip()}
    except Exception:  # noqa: BLE001 - missing/invalid alias file must never break the caller
        pass
    return {}


def _slug_words(slug):
    return re.sub(r"[-_]+", " ", str(slug or "")).strip().lower()


def build_index(pages, *, alias_map=None):
    """Map phrase(lower) -> {title, href, kind, gated}. First page wins on collision."""
    index = {}
    by_slug = {}
    for p in (pages or []):
        slug = str(p.get("slug") or "").strip()
        if not slug:
            continue
        rec = {
            "title": p.get("name") or slug,
            "href": p.get("href") or "",
            "kind": p.get("kind") or "topic",
            "gated": bool(p.get("gated")),
        }
        by_slug[slug] = rec
        for phrase in (str(p.get("name") or "").strip().lower(), _slug_words(slug)):
            if phrase and phrase not in index:
                index[phrase] = rec
    # aliases point at a slug; only add if the slug is a real page and the phrase is free
    for phrase, slug in (alias_map or {}).items():
        ph = str(phrase or "").strip().lower()
        rec = by_slug.get(str(slug or "").strip())
        if ph and rec and ph not in index:
            index[ph] = rec
    return index


def match_page_links(text, index, *, limit=2):
    """Return up to `limit` deduped link cards for phrases present in text (longest first)."""
    if not text or not index:
        return []
    low = " " + re.sub(r"\s+", " ", str(text).lower()) + " "
    # longest phrases first so "magnesium glycinate" claims its span before "magnesium"
    phrases = sorted(index.keys(), key=len, reverse=True)
    cards = []
    seen_hrefs = set()
    claimed = []  # list of (start, end) spans already consumed by a longer phrase

    def _overlaps(s, e):
        return any(not (e <= cs or s >= ce) for cs, ce in claimed)

    for phrase in phrases:
        if not phrase:
            continue
        # word-boundary search: phrase must sit between non-word chars
        pat = r"(?<![\w])" + re.escape(phrase) + r"(?![\w])"
        for m in re.finditer(pat, low):
            s, e = m.start(), m.end()
            if _overlaps(s, e):
                continue
            claimed.append((s, e))
            rec = index[phrase]
            href = rec.get("href")
            if not href or href in seen_hrefs:
                continue
            seen_hrefs.add(href)
            cards.append({
                "key": f"{rec.get('kind', 'topic')}:" + href.rstrip("/").rsplit("/", 1)[-1],
                "title": rec.get("title") or phrase,
                "sub": _SUB_BY_KIND.get(rec.get("kind"), "Read the guide"),
                "href": href,
            })
            break  # one card per phrase
        if len(cards) >= limit:
            break
    return cards[:limit]
