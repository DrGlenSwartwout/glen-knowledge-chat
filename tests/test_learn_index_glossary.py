from dashboard import topic_render as tr


def test_learn_index_surfaces_glossaries():
    html = tr.render_index_html([{"slug": "sleep", "name": "Sleep"}])
    assert 'href="/learn/glossary"' in html      # clinical glossary
    assert 'href="/learn/patterns"' in html      # E4L stress-pattern glossary
    assert 'href="/learn/sleep"' in html          # topic links still present
    assert "Reference glossaries" in html
