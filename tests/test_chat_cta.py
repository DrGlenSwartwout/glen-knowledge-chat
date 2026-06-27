from dashboard.chat_cta import parse_cta, stream_visible, SENTINEL

def test_parse_email_directive():
    ans = "Here is your brief answer.\n\n⟦CTA⟧ email |  | Send my full report"
    text, cta = parse_cta(ans)
    assert text == "Here is your brief answer."
    assert cta == {"type": "email", "target": "", "label": "Send my full report"}

def test_parse_page_directive_with_url():
    ans = "Body.\n⟦CTA⟧ page | https://x.com/p | Read the full breakdown"
    text, cta = parse_cta(ans)
    assert cta["type"] == "page" and cta["target"] == "https://x.com/p"
    assert SENTINEL not in text

def test_no_directive_returns_none():
    text, cta = parse_cta("Just an answer, no directive.")
    assert cta is None and text == "Just an answer, no directive."

def test_unknown_type_stripped_and_none():
    text, cta = parse_cta("Body.\n⟦CTA⟧ bogus | | x")
    assert cta is None and SENTINEL not in text  # never leak the sentinel

def test_inline_type():
    _, cta = parse_cta("Body.\n⟦CTA⟧ inline | | ")
    assert cta == {"type": "inline", "target": "", "label": ""}


# --- stream_visible tests ---

def _join(tokens):
    return "".join(stream_visible(tokens))


def test_stream_visible_no_sentinel():
    result = _join(["Hello ", "world", " today"])
    assert result == "Hello world today"


def test_stream_visible_split_sentinel_each_char():
    # The bug: sentinel arrives as separate tokens
    tokens = ["Body text.", "⟦", "CTA", "⟧", " email |  | x"]
    result = _join(tokens)
    assert result == "Body text."
    assert "⟦" not in result


def test_stream_visible_whole_sentinel_one_token():
    tokens = ["Body.", "⟦CTA⟧ page | u | l"]
    result = _join(tokens)
    assert result == "Body."
    assert "⟦" not in result


def test_stream_visible_sentinel_split_ctabracket():
    # Sentinel split as ⟦CTA | ⟧
    tokens = ["A.", "⟦CTA", "⟧ inline | | "]
    result = _join(tokens)
    assert result == "A."
    assert "⟦" not in result


def test_stream_visible_short_tail_flushed():
    # Tail shorter than hold must still be flushed at stream end
    result = _join(["Hi"])
    assert result == "Hi"
