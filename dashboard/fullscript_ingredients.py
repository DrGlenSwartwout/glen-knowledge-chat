"""Fetch a Fullscript public catalog product page and return its text.

Source A of the purity-acquisition cascade (spec Section 2). The page at
https://fullscript.com/catalog/products/<slug> is unauthenticated and carries
the manufacturer's Other Ingredients line verbatim; a plain GET with a browser
User-Agent returns it (Fullscript is behind Cloudflare, which 403s default
python UAs -- see reference_cloudflare_ua_ban).

This module ONLY fetches and cleans. It does not parse ingredients and makes no
model call -- extraction + the fabrication guard live in document_extract, and
orchestration in purity_acquire.
"""
import re

PRODUCT_URL = "https://fullscript.com/catalog/products/{slug}"

# A real browser UA; mirrors dashboard/ghl_email.py. Default python-requests /
# urllib UAs are 403'd by Cloudflare.
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

_TIMEOUT = 15


def _default_fetch(url, headers):
    import requests
    return requests.get(url, headers=headers, timeout=_TIMEOUT)


def _clean(text):
    """Decode \\uXXXX escapes, strip HTML tags, collapse whitespace.

    The ingredient text sits inside a JS/RSC string in the page payload, so it
    arrives with \\u003c-style escapes and surrounding tags. Decoding + tag
    stripping yields readable text in which the Other Ingredients line is a
    single contiguous run -- which is what the extractor quotes and
    verify_quotes checks against. We decode ONLY the \\uXXXX form (not a full
    unicode_escape pass, which would corrupt real multibyte characters)."""
    text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)
    text = re.sub(r"<[^>]+>", " ", text)      # drop HTML tags
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def fetch_page_text(slug, *, fetch=None):
    """Cleaned text of the public product page for `slug`, or None on any
    failure (blank slug, non-200, network/parse exception). Never raises."""
    s = (slug or "").strip()
    if not s:
        return None
    fetch = fetch or _default_fetch
    url = PRODUCT_URL.format(slug=s)
    try:
        resp = fetch(url, {"User-Agent": _UA})
        if getattr(resp, "status_code", None) != 200:
            return None
        return _clean(resp.text or "")
    except Exception:
        return None
