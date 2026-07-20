"""The /sample demo portal is only worth building if something points at it.

These assert the funnel and practitioner-recruitment entry points exist. Pure
file reads — no app import, so no doppler guard needed.

Note the coupling: /sample is gated by PUBLIC_SURFACE_ENABLED. If that flag is
ever turned off, these links become dead ends on live pages. Turning the
feature off means removing these links in the same change.
"""

from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "static"


def _read(name):
    return (STATIC / name).read_text(encoding="utf-8")


def test_practitioner_page_links_to_the_sample_portal():
    """The practitioner page describes the patient portal in prose. Showing it
    is the whole point of building a demo."""
    html = _read("practitioner.html")
    assert 'href="/sample"' in html


def test_begin_funnel_links_to_the_sample_portal():
    html = _read("begin.html")
    assert 'href="/sample"' in html


def test_sample_links_are_not_slug_attributed():
    """These are Glen's own marketing surfaces, not a holder's share link — they
    must not carry someone's referral slug and mis-attribute a signup."""
    for name in ("practitioner.html", "begin.html"):
        html = _read(name)
        assert 'href="/sample/' not in html, f"{name} links to a slug-attributed sample"
