from dashboard.chat_cta import parse_cta, stream_visible, SENTINEL, parse_chips, CHIPS_SENTINEL

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


def test_stream_visible_drains_input_so_cta_payload_survives():
    # Regression: stream_visible must DRAIN its input so the caller's `full_answer`
    # accumulator keeps the directive's argument tokens (which come AFTER the
    # sentinel). Previously the generator was abandoned at the sentinel and
    # parse_cta saw an empty directive -> None (CTA never rendered).
    full = []
    tokens = ["Body.", "⟦CTA⟧", " email", " | ", " | ", "Send report"]
    def gen():
        for t in tokens:
            full.append(t)
            yield t
    visible = "".join(stream_visible(gen()))
    assert SENTINEL not in visible
    clean, cta = parse_cta("".join(full))
    assert cta is not None and cta["type"] == "email"   # payload survives the stream


# --- parse_chips tests ---

def test_parse_chips_basic():
    text, chips = parse_chips("Are you hyper or hypo?\n⟦CHIPS⟧ Overactive | Underactive | Not sure")
    assert text == "Are you hyper or hypo?"
    assert chips == ["Overactive", "Underactive", "Not sure"]

def test_parse_chips_absent():
    text, chips = parse_chips("What is your main concern?")
    assert chips == [] and text == "What is your main concern?"

def test_parse_chips_caps_at_4_and_trims_empties():
    _, chips = parse_chips("Q\n⟦CHIPS⟧ a | b | c | d | e |  ")
    assert chips == ["a", "b", "c", "d"]      # cap 4, drop empties

def test_stream_visible_param_sentinel_hides_chips():
    out = "".join(stream_visible(["Pick one.", "⟦", "CHIPS", "⟧", " a | b"], sentinel=CHIPS_SENTINEL))
    assert out == "Pick one." and "⟦" not in out      # split-token safe

def test_stream_visible_default_still_hides_cta():
    out = "".join(stream_visible(["Body.", "⟦CTA⟧ email |  | x"]))
    assert out == "Body." and SENTINEL not in out

# --- End-to-end: the directive PAYLOAD (after the sentinel) must survive in the
# caller's `full` accumulator. stream_visible must DRAIN its input generator so
# the per-token side effect (full.append) runs for the post-sentinel tokens.
# Regression for the Critical: parse(full) returned empty because the option
# tokens after the sentinel were never pulled. (CTA path had the same defect.)

def test_stream_visible_drains_input_so_chips_payload_survives():
    full = []
    tokens = ["Pick one.", "\n", "⟦CHIPS⟧", " Yes", " |", " No", " |", " Maybe"]
    def gen():
        for t in tokens:
            full.append(t)
            yield t
    visible = "".join(stream_visible(gen(), sentinel=CHIPS_SENTINEL))
    assert visible == "Pick one.\n"                 # directive hidden from stream
    clean, chips = parse_chips("".join(full))        # full kept the WHOLE answer
    assert chips == ["Yes", "No", "Maybe"]           # payload survives (the fix)
    assert clean == "Pick one."
    # (the CTA-payload-survives regression lives above; not duplicated here)
