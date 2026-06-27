"""Structural eval for the 5-beat brief instruction.

Calls Haiku with the real brief instruction and asserts invariants:
- A CTA directive is emitted (cta is not None)
- cta type is in VALID_TYPES
- The sentinel ⟦CTA⟧ is NOT visible in the cleaned text
- Word count <= 240
- The string "Hook" does not appear in the cleaned text

Skips if ANTHROPIC_API_KEY is absent.
The rung CHOICE (page/email/action/inline) is model judgment — not asserted.
"""
import os
import re
import importlib
import sys
import pytest
from pathlib import Path

CASES = [
    "What foods are best for macular degeneration?",                          # expect page
    "I have wet AMD; changed my diet since Nov but bloodwork is the same. Try Gundry's lectin protocol?",  # expect email
]


def _app(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as a
        importlib.reload(a)
        return a
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="needs API key"
)
def test_brief_emits_valid_cta_and_is_bounded(monkeypatch, tmp_path):
    a = _app(monkeypatch, tmp_path)
    from dashboard.chat_cta import parse_cta, VALID_TYPES
    import anthropic

    cl = anthropic.Anthropic()
    instr = a._brief_synth_instruction()

    for q in CASES:
        r = cl.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=a.SYSTEM_PROMPT + "\n" + instr,
            messages=[{"role": "user", "content": q}],
        )
        raw = r.content[0].text
        clean, cta = parse_cta(raw)

        assert cta is not None, f"no CTA directive emitted for: {q!r}\nRaw output:\n{raw}"
        assert cta["type"] in VALID_TYPES, (
            f"CTA type {cta['type']!r} not in {VALID_TYPES} for: {q!r}"
        )
        assert "⟦CTA⟧" not in clean, (
            f"Sentinel leaked into visible text for: {q!r}\nClean:\n{clean}"
        )
        word_count = len(clean.split())
        assert word_count <= 240, (
            f"Response too long ({word_count} words > 240) for: {q!r}\nClean:\n{clean}"
        )
        assert "Hook" not in clean, (
            f"Label 'Hook' leaked into visible text for: {q!r}\nClean:\n{clean}"
        )
