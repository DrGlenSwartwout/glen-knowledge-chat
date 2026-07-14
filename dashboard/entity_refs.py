"""Resolve a clinical entity (stress pattern, remedy, function, ingredient) to a
small {name, info, href} record used by the shared entity-ref hover component.

All functions are pure/wrapped and never raise into callers: on any miss they
return info="" and/or href=None so the frontend degrades to plain text. Gating
(a remedy is only resolved for an unblurred report) is the CALLER's job — this
module has no notion of payment state."""
import json
import re

from dashboard import biofield_meanings as _bm
from dashboard import biofield_authoring as _ba
from dashboard import topic_pages as _tp
from dashboard import ingredient_pages as _ip
from dashboard.ingredients import slugify as _slugify


def clip(text, sentences=2, cap=280):
    text = (text or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = " ".join(parts[:sentences]).strip()
    if len(out) > cap:
        out = out[:cap].rstrip() + "…"
    return out


def _first_text(obj):
    """First meaningful string in a page's content, shape-robust. Prefers named
    summary-ish keys, then any nested string."""
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, dict):
        for k in ("summary", "intro", "overview", "what", "what_it_is",
                  "description", "body", "text"):
            t = _first_text(obj.get(k))
            if t:
                return t
        for v in obj.values():
            t = _first_text(v)
            if t:
                return t
    if isinstance(obj, list):
        for v in obj:
            t = _first_text(v)
            if t:
                return t
    return ""


def _page_content(page):
    """Return the parsed content object from a page dict. ingredient_pages and
    topic_pages both expose the parsed content under the key "content"; accept a
    raw "content_json" (dict or JSON string) as a fallback."""
    if not isinstance(page, dict):
        return {}
    raw = page.get("content")
    if raw is None:
        raw = page.get("content_json")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {"text": raw}
    return raw or {}


def pattern_ref(name, description, href=""):
    """Stress pattern: pop-up + optional link-out to its glossary page
    (`/learn/pattern/<slug>`). Defaults to pop-up only when no href is given."""
    return {"name": (name or "").strip(), "info": clip(description, sentences=3),
            "href": href or None}


def remedy_ref(cx, spoken, product_exists=None, slug=None):
    """Remedy -> product page + curated meaning. When `slug` is given (the caller
    already knows the product slug), the meaning/href use it directly; otherwise the
    spoken name is fuzzy-resolved to a catalog slug. href only when that slug passes
    product_exists() — required to emit any href (None => pop-up only, the safe
    default)."""
    name = (spoken or "").strip()
    use_slug = (slug or "").strip()
    if not name and not use_slug:
        return {"name": name, "info": "", "href": None}
    if not use_slug:
        try:
            resolved = _ba.resolve_remedy_name(cx, name) or name
        except Exception:
            resolved = name
        use_slug = _slugify(resolved)
    try:
        meaning = _bm.get_map(cx).get(use_slug, "")
    except Exception:
        meaning = ""
    href = None
    if use_slug and callable(product_exists):
        try:
            if product_exists(use_slug):
                href = f"/begin/product/{use_slug}"
        except Exception:
            href = None
    return {"name": name or use_slug, "info": clip(meaning, sentences=2), "href": href}


def function_ref(cx, title):
    """Function/structure title -> /learn topic page. href + info only when an
    APPROVED topic page of kind 'function' exists for the slug; else plain."""
    name = (title or "").strip()
    if not name:
        return {"name": name, "info": "", "href": None}
    slug = _slugify(name)
    try:
        page = _tp.get_page(cx, slug)
    except Exception:
        page = None
    if not page or (page.get("kind") or "") != "function" or (page.get("state") or "") != "approved":
        return {"name": name, "info": "", "href": None}
    info = clip(_first_text(_page_content(page)), sentences=2)
    return {"name": name, "info": info, "href": f"/learn/{slug}"}


def ingredient_ref(cx, name, slug, page_getter=None):
    """Ingredient -> its ingredient page + a short 'what it is' summary. href +
    info only when an ingredient page exists for the slug; else plain."""
    name = (name or "").strip()
    slug = (slug or "").strip()
    if not slug:
        return {"name": name, "info": "", "href": None}
    getter = page_getter if callable(page_getter) else (lambda s: _safe_ing_page(cx, s))
    try:
        page = getter(slug)
    except Exception:
        page = None
    if not page:
        return {"name": name, "info": "", "href": None}
    info = clip(_first_text(_page_content(page)), sentences=2)
    return {"name": name, "info": info, "href": f"/begin/ingredient/{slug}"}


def _safe_ing_page(cx, slug):
    try:
        return _ip.get_page(cx, slug)
    except Exception:
        return None
