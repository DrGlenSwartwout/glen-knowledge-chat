import sys
from pathlib import Path

import pytest


def _mod():
    r = str(Path(__file__).resolve().parent.parent)
    if r not in sys.path:
        sys.path.insert(0, r)
    try:
        from dashboard import topic_render
        return topic_render
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"topic_render not importable: {e}")


def _approved_page():
    return {
        "slug": "low-energy", "kind": "symptom", "name": "Low Energy", "state": "approved",
        "content": {"overview": "People often notice tiredness.",
                    "contributing_factors": "Sleep and minerals.",
                    "what_people_explore": "Many explore nutrition."},
        "links": {"ingredients": [{"slug": "folate", "name": "Folate"}],
                  "products": [{"slug": "neuro-magnesium", "name": "Neuro Magnesium"}],
                  "topics": [{"slug": "detox", "name": "Detox"}]},
        "seo": {"title": "Low Energy — wellness overview",
                "meta_description": "An educational look at low energy."},
    }


def test_is_public_only_for_approved():
    tr = _mod()
    assert tr.is_public(_approved_page()) is True
    draft = dict(_approved_page(), state="draft")
    assert tr.is_public(draft) is False
    assert tr.is_public(None) is False


def test_approved_render_has_seo_and_sections_and_links():
    tr = _mod()
    html = tr.render_page_html(_approved_page(), base_url="https://x.test")
    assert "People often notice tiredness." in html
    assert '<meta name="description" content="An educational look at low energy.' in html
    assert "application/ld+json" in html
    assert "/begin/ingredient/folate" in html
    assert "/begin/product/neuro-magnesium" in html
    assert "/learn/detox" in html
    assert "/begin" in html  # CTA


def test_pending_render_has_no_section_text_and_a_request_form():
    tr = _mod()
    html = tr.render_pending_html("low-energy", "Low Energy")
    assert "People often notice tiredness." not in html
    assert 'action="/learn/low-energy/request"' in html


def test_render_escapes_section_text():
    tr = _mod()
    page = _approved_page()
    page["content"]["overview"] = "5 < 10 & <script>alert(1)</script>"
    html = tr.render_page_html(page, base_url="https://x.test")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_approved_render_has_site_chrome():
    """Styled to match the site: brandbar, house fonts, gold CTA button, footer disclaimer."""
    tr = _mod()
    html = tr.render_page_html(_approved_page(), base_url="https://x.test")
    assert "brand-name" in html          # brandbar present
    assert "Open+Sans" in html           # house font link
    assert "cta-btn" in html             # styled gold CTA
    assert "not a substitute for diagnosis" in html  # footer disclaimer


def test_pending_render_has_site_chrome_and_noindex():
    tr = _mod()
    html = tr.render_pending_html("low-energy", "Low Energy")
    assert "brand-name" in html
    assert 'name="robots" content="noindex"' in html
    assert "not a substitute for diagnosis" in html


def test_index_render_styles_and_links_topics():
    tr = _mod()
    html = tr.render_index_html([{"slug": "low-energy", "name": "Low Energy"},
                                 {"slug": "detox", "name": "Detox"}])
    assert "brand-name" in html              # styled chrome
    assert "Open+Sans" in html
    assert '/learn/low-energy' in html
    assert "Low Energy" in html
    assert "Detox" in html


def test_index_render_escapes_topic_name():
    tr = _mod()
    html = tr.render_index_html([{"slug": "x", "name": "<script>alert(1)</script>"}])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_jsonld_neutralizes_script_breakout():
    tr = _mod()
    page = _approved_page()
    page["content"]["overview"] = "danger </script><script>alert(1)</script> end"
    html = tr.render_page_html(page, base_url="https://x.test")
    # the JSON-LD block must not contain a raw closing </script> from the payload
    jsonld_start = html.index("application/ld+json")
    jsonld_end = html.index("</script>", jsonld_start)  # this is the REAL closing tag
    jsonld_block = html[jsonld_start:jsonld_end]
    assert "</script>" not in jsonld_block       # payload's </script> was neutralized
    assert "<\\/script>" in jsonld_block          # hardened form present


def test_suggest_form_posts_and_is_noindex():
    tr = _mod()
    html = tr.render_suggest_html("magnesium-deficiency", "Magnesium Deficiency")
    assert 'action="/learn/suggest/magnesium-deficiency"' in html
    assert 'name="robots" content="noindex"' in html
    assert "Magnesium Deficiency" in html
    assert "brand-name" in html  # site chrome


def test_suggest_submitted_shows_confirmation_no_form():
    tr = _mod()
    html = tr.render_suggest_html("dry-skin", "Dry Skin", submitted=True)
    assert "<form" not in html
    assert "email you" in html.lower()


def test_suggest_escapes_name():
    tr = _mod()
    html = tr.render_suggest_html("x", "<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_sitemap_xml_includes_loc_and_lastmod():
    tr = _mod()
    rows = [
        {"slug": "low-energy", "updated_at": "2026-06-22T01:02:03+00:00", "approved_at": ""},
        {"slug": "detox", "updated_at": "", "approved_at": "2026-06-20T09:00:00+00:00"},
        {"slug": "no-dates"},
    ]
    xml = tr.render_sitemap_xml(rows, "https://illtowell.com")
    assert xml.startswith('<?xml')
    assert "<loc>https://illtowell.com/learn/low-energy</loc>" in xml
    assert "<lastmod>2026-06-22</lastmod>" in xml          # from updated_at
    assert "<lastmod>2026-06-20</lastmod>" in xml          # falls back to approved_at
    # a row with no timestamps still emits a <url> but no <lastmod>
    assert "<loc>https://illtowell.com/learn/no-dates</loc></url>" in xml
