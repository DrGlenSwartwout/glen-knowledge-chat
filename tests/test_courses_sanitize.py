from dashboard.courses_sanitize import sanitize_html


# --- Script / style are stripped entirely (tag + contents) -----------------

def test_script_tag_and_contents_removed():
    out = sanitize_html("<p>hi</p><script>alert(1)</script><p>bye</p>")
    assert "<script" not in out
    assert "alert(1)" not in out
    assert "hi" in out and "bye" in out


def test_script_nested_inside_div_removed():
    out = sanitize_html('<div><script>document.cookie</script><p>ok</p></div>')
    assert "<script" not in out
    assert "document.cookie" not in out
    assert "ok" in out


def test_style_tag_and_contents_removed():
    out = sanitize_html("<style>body{background:url(javascript:alert(1))}</style><p>text</p>")
    assert "<style" not in out
    assert "background" not in out
    assert "text" in out


# --- Event-handler attributes stripped --------------------------------------

def test_onclick_attribute_stripped():
    out = sanitize_html('<p onclick="alert(1)">click me</p>')
    assert "onclick" not in out
    assert "click me" in out


def test_onerror_attribute_stripped_from_img():
    out = sanitize_html('<img src="https://example.com/x.png" onerror="alert(1)">')
    assert "onerror" not in out
    assert 'src="https://example.com/x.png"' in out


def test_style_attribute_stripped():
    out = sanitize_html('<p style="background:red" class="foo" data-x="1">hi</p>')
    assert "style=" not in out
    assert "class=" not in out
    assert "data-x" not in out


# --- javascript: / data: URLs dropped ---------------------------------------

def test_javascript_href_dropped():
    out = sanitize_html('<a href="javascript:alert(1)">click</a>')
    assert "javascript:" not in out
    assert "click" in out


def test_data_href_dropped():
    out = sanitize_html('<a href="data:text/html,<script>alert(1)</script>">x</a>')
    assert "data:" not in out
    assert "<script" not in out


def test_http_href_kept():
    out = sanitize_html('<a href="https://example.com/page">link</a>')
    assert 'href="https://example.com/page"' in out


def test_protocol_relative_href_kept():
    out = sanitize_html('<a href="//example.com/page">link</a>')
    assert 'href="//example.com/page"' in out


def test_javascript_img_src_dropped():
    out = sanitize_html('<img src="javascript:alert(1)" alt="x">')
    assert "javascript:" not in out


def test_data_img_src_dropped():
    out = sanitize_html('<img src="data:image/png;base64,AAAA" alt="x">')
    assert "data:" not in out


# --- iframe host allow-list --------------------------------------------------

def test_youtube_iframe_kept():
    out = sanitize_html('<iframe src="https://www.youtube.com/embed/abc123" width="560" height="315"></iframe>')
    assert "<iframe" in out
    assert "youtube.com/embed/abc123" in out


def test_youtube_nocookie_iframe_kept():
    out = sanitize_html('<iframe src="https://www.youtube-nocookie.com/embed/abc123"></iframe>')
    assert "<iframe" in out
    assert "youtube-nocookie.com" in out


def test_rumble_iframe_kept():
    out = sanitize_html('<iframe src="https://rumble.com/embed/v1abcd/" allowfullscreen></iframe>')
    assert "<iframe" in out
    assert "rumble.com/embed/v1abcd" in out


def test_evil_iframe_dropped():
    out = sanitize_html('<p>before</p><iframe src="https://evil.com/steal"></iframe><p>after</p>')
    assert "<iframe" not in out
    assert "evil.com" not in out
    assert "before" in out and "after" in out


def test_iframe_missing_src_dropped():
    out = sanitize_html('<iframe width="100"></iframe>')
    assert "<iframe" not in out


def test_iframe_javascript_src_dropped():
    out = sanitize_html('<iframe src="javascript:alert(1)"></iframe>')
    assert "<iframe" not in out
    assert "javascript:" not in out


def test_iframe_onload_attribute_stripped():
    out = sanitize_html('<iframe src="https://rumble.com/embed/v1abcd/" onload="alert(1)"></iframe>')
    assert "onload" not in out
    assert "<iframe" in out


# --- allowed formatting survives ---------------------------------------------

def test_headings_and_strong_survive():
    out = sanitize_html("<h2>Section</h2><p>Some <strong>bold</strong> and <em>italic</em> text.</p>")
    assert "<h2>Section</h2>" in out
    assert "<strong>bold</strong>" in out
    assert "<em>italic</em>" in out


def test_table_cells_keep_colspan_rowspan_strip_other_attrs():
    out = sanitize_html('<table><tr><td colspan="2" rowspan="1" style="color:red" class="x">cell</td></tr></table>')
    assert 'colspan="2"' in out
    assert 'rowspan="1"' in out
    assert "style=" not in out
    assert "class=" not in out


def test_unknown_tag_unwrapped_not_dropping_text():
    out = sanitize_html("<marquee>scrolling text</marquee>")
    assert "<marquee" not in out
    assert "scrolling text" in out


# --- empty-paragraph / <br> run collapsing -----------------------------------

def test_collapses_long_run_of_empty_paragraphs():
    out = sanitize_html("<p>start</p>" + "<p><br></p>" * 5 + "<p>end</p>")
    assert out.count("<p><br") == 1 or out.count("<p><br/></p>") <= 1
    assert "start" in out and "end" in out


def test_collapses_long_run_of_br_tags():
    out = sanitize_html("<p>start</p>" + "<br>" * 6 + "<p>end</p>")
    assert out.count("<br") == 1


def test_does_not_collapse_isolated_br():
    out = sanitize_html("<p>line one<br>line two</p>")
    assert out.count("<br") == 1
    assert "line one" in out and "line two" in out


# --- misc / defensive ---------------------------------------------------------

def test_empty_input_returns_empty_string():
    assert sanitize_html("") == ""
    assert sanitize_html(None) == ""


def test_plain_text_passes_through():
    out = sanitize_html("just some text")
    assert "just some text" in out
