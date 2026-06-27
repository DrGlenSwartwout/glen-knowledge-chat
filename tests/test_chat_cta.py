from dashboard.chat_cta import parse_cta, SENTINEL

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
