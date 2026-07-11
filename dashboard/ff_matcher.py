"""Scan -> ranked Functional Formulation product matches. Pure + injected deps
(Pinecone query, slug resolver, destination) so it is unit-testable offline.
Names + meanings only — NEVER dosing; dosing is a clinical instruction added at
Glen's review."""


def _query_text(scan_items):
    parts = []
    for it in scan_items or []:
        label = (it.get("label") or "").strip()
        if label:
            parts.append(label)
    return "; ".join(parts)


def generate_ff_matches(scan_items, *, query_matches, resolve_slug, destination, top_k=5):
    text = _query_text(scan_items)
    if not text:
        return []
    try:
        candidates = query_matches(text, top_k) or []
    except Exception:
        return []
    out, seen = [], set()
    for c in candidates:
        name = ((c.get("metadata") or {}).get("name") or "").strip()
        if not name:
            continue
        slug = resolve_slug(name)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        out.append({
            "name": name,
            "slug": slug,
            "url": destination(slug),
            "meaning": ((c.get("metadata") or {}).get("meaning") or "").strip(),
            "score": float(c.get("score") or 0.0),
        })
    out.sort(key=lambda m: (-m["score"], m["name"]))
    return out[:top_k]
