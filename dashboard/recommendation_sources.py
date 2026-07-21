"""Extensible registry of recommendation sources. Each source: a display label,
an icon (shown in the portal with a count at its center), and a kind:
  clinical   -> the recommendation being generated is the counted action
  engagement -> a client action (click/add/order) is the counted action
Adding a source later is a dict entry, not a schema change."""

RECOMMENDATION_SOURCES = {
    "biofield":   {"label": "Biofield",   "icon": "📡", "kind": "clinical"},
    "intake":     {"label": "Intake",     "icon": "📝", "kind": "clinical"},
    "scan":       {"label": "Scan",       "icon": "🔬", "kind": "engagement"},
    "chat":       {"label": "Chat",       "icon": "💬", "kind": "engagement"},
    "self":       {"label": "Self",       "icon": "🛒", "kind": "engagement"},
    "email":      {"label": "Email",      "icon": "✉️", "kind": "engagement"},
    "newsletter": {"label": "Newsletter", "icon": "📰", "kind": "engagement"},
    "ads":        {"label": "Ads",        "icon": "📣", "kind": "engagement"},
    "social":     {"label": "Social",     "icon": "📱", "kind": "engagement"},
    "purchased":  {"label": "Purchased",  "icon": "✅", "kind": "engagement"},
}


def get_source(key):
    return RECOMMENDATION_SOURCES.get(key)


def known_source(key):
    return key in RECOMMENDATION_SOURCES
