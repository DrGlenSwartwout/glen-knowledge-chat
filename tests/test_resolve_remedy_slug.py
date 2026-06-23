"""Regression: in-catalog remedies were dropped when the product's `name`
differs from its `pinecone_title`, or the title carried HTML entities. The
resolver must match by name (HTML-unescaped) too, not only pinecone_title.
Runs against the real data/products.json (stable catalog entries)."""
import app

r = app._resolve_remedy_slug


def test_resolves_when_pinecone_title_differs_from_name():
    assert r({"name": "Brain Boost"}) == "brain-boost"                 # title 'Brain Boost Nootropic'
    assert r({"name": "Lens-Zyme Brunescense Buster"}) == "lens-zyme"  # title 'Lens-Zyme'
    assert r({"name": "GI Repair Helicobacter Pylori Terrain Support"}) == "gi-repair"


def test_resolves_through_html_entities_in_title():
    assert r({"name": "Free & Easy"}) == "free-and-easy"   # title 'Free &amp; Easy'
    assert r({"name": "Rise & Shine"}) == "rise--shine"    # title 'Rise &amp; Shine'


def test_existing_resolution_preserved():
    assert r({"name": "MR2 Calm Mind"}) == "mr2-calm-mind"   # exact pinecone_title
    assert r({"slug": "brain-boost"}) == "brain-boost"       # valid slug passes through
    assert r({"name": "ei8"}) is not None                    # code fallback still works


def test_unknown_name_still_drops():
    assert r({"name": "Definitely Not A Product 9000"}) is None
    assert r({"name": ""}) is None
