"""Practitioner-authored storefront profile: the bridge from a public slug to
the self-authored fields shown on /p/<slug>.

Provenance rule (spec 2026-07-20-practitioner-storefront-editor.md): the
`practitioners` table already holds bio/photo/specialties/city/state, but that
data is SCRAPED. The storefront publishes a field only when the practitioner has
self-authored/confirmed it, tracked by practitioners.profile_self_authored_at.
Scraped rows (timestamp null) return {} and the storefront shows name +
disclosure only.

This module owns the sqlite (affiliate_signups) -> Postgres (practitioners) hop
and the provenance gate, so public_surface.py stays a thin caller. Any failure
in the read path degrades to {} — a public page must never 500 on a profile read.
"""

import re

MAX_BIO = 600
MAX_SERVICES = 12
MAX_SERVICE_LEN = 60
MAX_LOC_LEN = 80

_TAG_RE = re.compile(r"<[^>]+>")

PROFILE_PUBLIC_FIELDS = frozenset({
    "bio", "photo_url", "logo_url", "services", "location", "accepting_clients",
})


def _norm(s):
    """Strip HTML tags and collapse whitespace."""
    return " ".join(_TAG_RE.sub("", s or "").split()).strip()


def sanitize_bio(text):
    """Strip HTML, collapse whitespace. Raise ValueError if >600 chars after
    cleaning. Does NOT strip URLs/emails/phones — a practitioner may include
    their own contact detail in their own bio, and over-stripping prose is a
    known failure mode."""
    clean = _norm(text)
    if len(clean) > MAX_BIO:
        raise ValueError(f"bio exceeds {MAX_BIO} characters")
    return clean


def clean_services(items):
    """Strip HTML per item, drop empties, cap 12 items x 60 chars."""
    out = []
    for it in (items or []):
        v = _norm(str(it))[:MAX_SERVICE_LEN].strip()
        if v:
            out.append(v)
        if len(out) >= MAX_SERVICES:
            break
    return out


def format_location(city, state):
    city = _norm(city)[:MAX_LOC_LEN]
    state = _norm(state)[:MAX_LOC_LEN]
    if city and state:
        return f"{city}, {state}"
    return city or ""
