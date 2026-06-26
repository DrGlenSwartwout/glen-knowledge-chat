"""Mine a client's consolidated /api/people profile into discrete stress labels
for the local Biofield Intake balancing loop (B3a). Pure: the free-text extractor
is injected so this is testable offline."""

_DISCRETE = ("tags", "conditions", "terrain_concerns", "body_systems")
_FREETEXT = ("challenges", "goals", "notes")


def _items(v):
    """A profile field may be a list or a comma/semicolon-separated string."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        parts = v
    else:
        parts = str(v).replace(";", ",").split(",")
    return [p.strip() for p in (str(x).strip() for x in parts) if p.strip()]


def mine_profile_stresses(profile, extract):
    """profile dict + extract(text)->[labels] -> deduped stress labels."""
    profile = profile or {}
    labels = []
    for field in _DISCRETE:
        labels.extend(_items(profile.get(field)))
    free = "\n".join(str(profile.get(f) or "").strip() for f in _FREETEXT if profile.get(f))
    if free.strip():
        labels.extend(extract(free) or [])
    out, seen = [], set()
    for label in labels:
        label = (label or "").strip()
        k = label.lower()
        if label and k not in seen:
            seen.add(k)
            out.append(label)
    return out
