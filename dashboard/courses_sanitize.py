"""Sanitizer for lesson HTML bodies (Stage 1.5 content model).

SECURITY-CRITICAL: lesson bodies now come from Practice Better as raw HTML
and are rendered directly to the browser (see courses_blueprint.lesson_page).
This module is the only thing standing between that stored HTML and an XSS
against every visitor to /learn/<course>/<module>/<lesson>. Allow-list only:
anything not explicitly permitted is stripped.
"""

from __future__ import annotations

from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString

# Tags that pass through untouched (modulo attribute filtering below).
ALLOWED_TAGS = {
    "p", "br", "h1", "h2", "h3", "h4", "span", "strong", "em", "b", "i", "u",
    "a", "ul", "ol", "li", "table", "thead", "tbody", "tr", "th", "td", "hr",
    "blockquote", "img", "div", "iframe",
}

# Per-tag attribute allow-list. Any tag not listed here that survives the
# tag allow-list keeps NO attributes at all (id/class/style/data-*/on* all
# stripped).
ALLOWED_ATTRS = {
    "a": {"href"},
    "iframe": {"src", "width", "height", "frameborder", "allow", "allowfullscreen"},
    "img": {"src", "alt", "width", "height"},
    "td": {"colspan", "rowspan"},
    "th": {"colspan", "rowspan"},
}

# Tags whose contents are always destroyed along with the tag itself, even
# before the general allow-list pass runs (so nested instances anywhere in
# the tree are caught, not just top-level ones).
ALWAYS_STRIP_WITH_CONTENTS = ("script", "style")

IFRAME_ALLOWED_HOSTS = {
    "youtube.com", "www.youtube.com",
    "youtube-nocookie.com", "www.youtube-nocookie.com",
    "rumble.com", "www.rumble.com",
}

_SAFE_URL_SCHEMES = ("http", "https")


def _safe_url(value: str) -> bool:
    value = (value or "").strip()
    if not value:
        return False
    if value.startswith("//"):
        return True
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme.lower() in _SAFE_URL_SCHEMES


def _hostname(value: str) -> str:
    try:
        return (urlparse((value or "").strip()).hostname or "").lower()
    except ValueError:
        return ""


def _is_empty_p(tag) -> bool:
    """A <p> that renders nothing but whitespace/<br> — collapse fodder."""
    if tag.get_text(strip=True):
        return False
    for descendant in tag.find_all(True):
        if descendant.name != "br":
            return False
    return True


def _collapse_empty_runs(soup: BeautifulSoup) -> None:
    """Collapse consecutive empty <p> or <br> siblings down to a single one.
    Whitespace-only text nodes between them don't break a run."""
    containers = [soup] + soup.find_all(True)
    for parent in containers:
        run_kind = None
        for child in list(parent.contents):
            if isinstance(child, NavigableString):
                if child.strip() == "":
                    continue  # whitespace is neutral, doesn't break a run
                run_kind = None
                continue
            kind = None
            if getattr(child, "name", None) == "br":
                kind = "br"
            elif getattr(child, "name", None) == "p" and _is_empty_p(child):
                kind = "p"
            if kind is not None and kind == run_kind:
                child.decompose()
            else:
                run_kind = kind


def sanitize_html(html: str) -> str:
    """Return an allow-listed, safe-to-render subset of `html`."""
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # 1. Nuke script/style tags and their contents, anywhere in the tree.
    for tag in soup.find_all(ALWAYS_STRIP_WITH_CONTENTS):
        tag.decompose()

    # 2. Any tag not on the allow-list gets unwrapped (its safe text/child
    #    content is kept, the tag itself is discarded).
    for tag in list(soup.find_all(True)):
        if tag.name not in ALLOWED_TAGS:
            tag.unwrap()

    # 3. iframes: enforce the host allow-list before we even look at their
    #    other attributes. Anything not youtube/youtube-nocookie/rumble is
    #    dropped whole, not just neutered.
    for tag in list(soup.find_all("iframe")):
        host = _hostname(tag.get("src", ""))
        if host not in IFRAME_ALLOWED_HOSTS:
            tag.decompose()

    # 4. Attribute filtering for everything that's left: allow-list per tag,
    #    plus scheme validation for the URL-bearing attributes.
    for tag in list(soup.find_all(True)):
        allowed = ALLOWED_ATTRS.get(tag.name, set())
        for attr in list(tag.attrs.keys()):
            if attr not in allowed:
                del tag.attrs[attr]
        if tag.name == "a" and "href" in tag.attrs and not _safe_url(tag.attrs["href"]):
            del tag.attrs["href"]
        if tag.name == "img" and "src" in tag.attrs and not _safe_url(tag.attrs["src"]):
            del tag.attrs["src"]

    # 5. Tame bloated content: collapse runs of empty <p>/<br>.
    _collapse_empty_runs(soup)

    return str(soup)
