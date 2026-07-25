"""Role-aware excipient screen. Screens ONLY the Other Ingredients against the
avoid-list; the actives list is accepted for interface symmetry but is never
consulted (a substance's role decides -- silica as a nutrient is not a filler).
Pure: no Flask, no app import.

Contract for `other_ingredients`:
  None -> no excipient data was obtained -> color 'unrated' (NEVER green).
  []   -> the product is known to list no other ingredients -> 'green'.
  list -> screened item by item.
"""


def _normalize(name):
    """Lowercase and strip common descriptors so aliases match real labels."""
    s = (name or "").lower()
    for cut in ("(vegetable source)", "(vegetable)", "(as a flow agent)", "(from rice)"):
        s = s.replace(cut, "")
    return " ".join(s.split()).strip()


def _hits(normalized_item, entries):
    for e in entries:
        for alias in e["aliases"]:
            if alias in normalized_item:
                return True
    return False


def screen_label(actives, other_ingredients, avoidlist):
    version = avoidlist.get("version", "")
    if other_ingredients is None:                       # no data -> unrated, never green
        return {"color": "unrated", "red_hits": [], "yellow_hits": [],
                "avoidlist_version": version}
    red_hits, yellow_hits = [], []
    for raw in other_ingredients:
        norm = _normalize(raw)
        if _hits(norm, avoidlist["red"]):
            red_hits.append(raw)
        elif _hits(norm, avoidlist["yellow"]):
            yellow_hits.append(raw)
    if red_hits:
        color = "red"
    elif yellow_hits:
        color = "yellow"
    else:
        color = "green"
    return {"color": color, "red_hits": red_hits, "yellow_hits": yellow_hits,
            "avoidlist_version": version}
