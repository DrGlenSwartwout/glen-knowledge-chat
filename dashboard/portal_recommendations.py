"""Pure transform: a client's product_sources + prefs -> portal sections. One collapsible
section per source category; a product appears in each source it has an event for, ranked by
that source's count (desc) then recency (desc); top_n shown + a remainder count. Hidden
products excluded. Sections ordered by the registry."""
from dashboard.recommendation_sources import RECOMMENDATION_SOURCES


def build_sections(product_sources, notes, section_state, resolve_product, *, top_n=5):
    by_source = {}
    for p in product_sources:
        if p.get("hidden"):
            continue
        pk = p["product_key"]
        n = notes.get(pk, {})
        prod = resolve_product(pk) or {}
        icons = [{"source": s["source"], "count": s["count"],
                  "icon": (RECOMMENDATION_SOURCES.get(s["source"]) or {}).get("icon", "•"),
                  "first_touch": s.get("first_touch", "")} for s in p["sources"]]
        for s in p["sources"]:
            by_source.setdefault(s["source"], []).append({
                "product_key": pk, "name": prod.get("name") or pk, "url": prod.get("url") or "",
                "icons": icons,
                "operator_note": n.get("operator_note", ""), "client_note": n.get("client_note", ""),
                "_count": s["count"], "_recent": s.get("last_touch", "") or ""})
    out = []
    for key, meta in RECOMMENDATION_SOURCES.items():
        prods = by_source.get(key)
        if not prods:
            continue
        prods.sort(key=lambda e: (e["_count"], e["_recent"]), reverse=True)   # count desc, recency desc
        shown = prods[:top_n]
        out.append({
            "source": key, "label": meta["label"], "icon": meta["icon"],
            "collapsed": bool(section_state.get(key, False)),
            "total": len(prods), "shown": len(shown),
            "products": [{k: v for k, v in e.items() if not k.startswith("_")} for e in shown]})
    return out
