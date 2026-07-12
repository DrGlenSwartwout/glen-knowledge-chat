import os
from scripts.harvest_related_products import parse_related


def test_parse_related_extracts_urls():
    html = open(os.path.join(os.path.dirname(__file__), "fixtures", "remedymatch-related.html")).read()
    urls = parse_related(html)
    assert "https://remedymatch.com/remedies/syntropy/56-immune-modulation" in urls
    assert "https://remedymatch.com/resources/50-healing-glaucoma-book" in urls
    # each product is linked twice on the real page (image + title); dedup
    # must collapse both to a single entry per product.
    assert len(urls) == 2
    # a bare category/breadcrumb link (no trailing <id>-<slug>) must never
    # be mistaken for a product-id link.
    assert "https://remedymatch.com/remedies/syntropy/" not in urls
