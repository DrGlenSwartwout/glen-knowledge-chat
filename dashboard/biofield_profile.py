"""Mine a client's consolidated /api/people profile into discrete stress labels
for the local Biofield Intake balancing loop (B3a). Pure: the free-text extractor
is injected so this is testable offline."""

_DISCRETE = ("tags", "conditions", "terrain_concerns", "body_systems")
_FREETEXT = ("challenges", "goals", "notes")

# Only the `tags` field is a mixed CRM/marketing bucket; conditions/terrain/systems
# are already clinical. A tag is "operational" (pipeline/lifecycle/marketing state,
# NOT health status) if its namespace or wording matches these. Practice Better (pb:)
# tags are mostly clinical conditions, so pb: is kept UNLESS its value is a program/
# certification marker caught by the keyword list below.
_OP_NAMESPACES = {"type", "consent", "state", "source", "pract", "practitioner",
                  "reengagement", "stage", "pipeline", "status", "lifecycle", "utm",
                  "campaign", "email", "e4l", "nes", "list", "segment", "funnel", "journey"}
_OP_KEYWORDS = ("certification", "masterclass", "replay", "webinar", "onboard",
                "unsubscribe", "opted-in", "opt-in", "concierge", "account", "engage",
                "cert", "course", "affiliate", "referral", "coupon", "checkout")
_OP_BARE = {"begin", "concierge", "nes client", "client", "practitioner", "member"}


def is_operational_tag(tag):
    """True if a CRM tag describes pipeline/marketing state rather than health status."""
    t = (tag or "").strip().lower().strip('[]"\'')
    if not t:
        return True
    if ":" in t and t.split(":", 1)[0].strip() in _OP_NAMESPACES:
        return True
    if t in _OP_BARE:
        return True
    return any(kw in t for kw in _OP_KEYWORDS)


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
        items = _items(profile.get(field))
        if field == "tags":
            items = [t for t in items if not is_operational_tag(t)]
        labels.extend(items)
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
