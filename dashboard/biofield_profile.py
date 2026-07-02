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
                  "campaign", "email", "e4l", "nes", "list", "segment", "funnel", "journey",
                  "household"}
_OP_KEYWORDS = ("certification", "masterclass", "replay", "webinar", "onboard",
                "unsubscribe", "opted-in", "opt-in", "opted", "concierge", "account",
                "engage", "cert", "course", "affiliate", "referral", "coupon", "checkout",
                # observed CRM/marketing/lifecycle vocabulary in people.tags
                "chatbot", "aweber", "fireside", "aerai", "paramedic", "membership",
                "bounce", "invite", "licensed", "subscriber", "e4l.db")
# Startswith markers for tag-structured operational values. 'beta-personal' (not
# 'beta-') so health terms like beta-carotene / beta-glucan are NOT dropped.
_OP_PREFIXES = ("topic-", "budget", "beta-personal", "e4l-", "portal-",
                "practitioner-", "household:", "close_leads")
_OP_BARE = {"begin", "concierge", "nes client", "client", "practitioner", "member",
            "email", "fireside", "chatbot"}


def is_operational_tag(tag):
    """True if a CRM tag describes pipeline/marketing state rather than health status.
    Denylist (namespaces + keywords + startswith markers + bare tags) tuned to the
    real people.tags vocabulary; free-text health terms (Inflammation, Heavy metals,
    beta-carotene, lead toxicity) are deliberately NOT matched."""
    t = (tag or "").strip().lower().strip('[]"\'')
    if not t:
        return True
    if ":" in t and t.split(":", 1)[0].strip() in _OP_NAMESPACES:
        return True
    if t in _OP_BARE:
        return True
    if t.startswith(_OP_PREFIXES):
        return True
    if t.startswith("pb:") and t[3:].strip() in _PB_NON_CONDITION:
        return True   # pb: roles/tools/admin (od, rn, zyto, cert, member, ...) aren't conditions
    return any(kw in t for kw in _OP_KEYWORDS)


# The `people.tags` field is a broad CRM bucket (marketing, lifecycle, chat topics,
# roles). Real clinical conditions live in the Practice Better `pb:` namespace, so a
# denylist can't keep up — an allowlist of pb: conditions is the reliable signal.
# These pb: values are NOT conditions (roles, tools, admin) and are excluded.
_PB_NON_CONDITION = {"od", "rn", "np", "do", "md", "dc", "lac", "nd", "zyto", "cert",
                     "member", "membership", "account", "ash", "staff", "vip"}


def is_health_tag(tag):
    """True if a CRM tag is a clinical/health condition worth showing (e.g. in a
    biofield reveal). Keeps Practice Better `pb:<condition>` tags; drops everything
    else — marketing/lifecycle/topic/role tags and non-condition pb: values."""
    t = (tag or "").strip().lower().strip('[]"\'')
    if not t.startswith("pb:"):
        return False
    val = t[3:].strip()
    return bool(val) and val not in _PB_NON_CONDITION and not is_operational_tag(t)


def clean_health_tag(tag):
    """Display form of a health tag: strip the `pb:` prefix (e.g. 'pb:wet-amd' -> 'wet-amd')."""
    s = (tag or "").strip().strip('[]"\'')
    return s[3:] if s.lower().startswith("pb:") else s


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
