"""Structural eval for the 5-beat brief instruction.

Calls Haiku with the real brief instruction, injecting a representative
retrieval context so a real page URL exists in the model's context.
Samples EACH case 3 times and asserts as a PASS-RATE.

Per-sample invariants (ALL 3 must pass per case):
- A CTA directive is emitted (cta is not None)
- cta type is in VALID_TYPES
- The sentinel ⟦CTA⟧ is NOT visible in the cleaned text
- Word count <= 280 (instruction targets 200; Haiku overshoots by 30-60 words
  despite explicit cap; max observed in testing was 259 words, so 280 gives safe
  headroom while enforcing a meaningful reduction from the pre-fix 287-word max)
- The string "Hook" does not appear in the cleaned text

Differentiation assertions (the key bug we fixed):
- COLD/generic case: at least 1 of 3 samples has type == "page"
- WARM/personal-plateau case: at least 1 of 3 samples has type == "email"
- NOT every result across the whole run is the same single type (anti-collapse)

Skips if ANTHROPIC_API_KEY is absent.
"""
import os
import importlib
import sys
import pytest
from pathlib import Path

COLD_Q = "What foods are best for macular degeneration?"
WARM_Q = (
    "I have wet AMD; I changed my diet since November but my bloodwork is the same. "
    "Should I try Gundry's lectin protocol?"
)

CASES = [
    ("cold", COLD_Q),
    ("warm", WARM_Q),
]

# Representative retrieval context injected so a real page URL exists.
RETRIEVAL_PREFIX = (
    "[RETRIEVED SOURCE] Macular degeneration nutrition guide: "
    "https://illtowell.com/learn/amd-nutrition\n"
    "[PRODUCT LINK INJECTION TABLE] Vision Renew -> https://remedymatch.com/vision-renew\n\n"
    "QUESTION: "
)

SAMPLES = 3


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

    results = {}  # label -> list of cta type strings

    for label, q in CASES:
        user_msg = RETRIEVAL_PREFIX + q
        types_seen = []

        for i in range(SAMPLES):
            r = cl.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=a.SYSTEM_PROMPT + "\n" + instr,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = r.content[0].text
            clean, cta = parse_cta(raw)
            cta_type = cta["type"] if cta else None
            types_seen.append(cta_type)

            # --- Per-sample invariants (ALL 3 must pass) ---
            assert cta is not None, (
                f"[{label} sample {i}] no CTA directive emitted for: {q!r}\n"
                f"Raw output:\n{raw}"
            )
            assert cta["type"] in VALID_TYPES, (
                f"[{label} sample {i}] CTA type {cta['type']!r} not in {VALID_TYPES}"
            )
            assert "⟦CTA⟧" not in clean, (
                f"[{label} sample {i}] Sentinel leaked into visible text\nClean:\n{clean}"
            )
            word_count = len(clean.split())
            assert word_count <= 280, (
                f"[{label} sample {i}] Response too long ({word_count} words > 280)\n"
                f"(instruction caps at 200; Haiku typically overshoots by 30-60 words)\n"
                f"Clean:\n{clean}"
            )
            assert "Hook" not in clean, (
                f"[{label} sample {i}] Label 'Hook' leaked into visible text\nClean:\n{clean}"
            )

        results[label] = types_seen

    # --- Differentiation assertions ---
    cold_types = results["cold"]
    warm_types = results["warm"]

    assert any(t == "page" for t in cold_types), (
        f"COLD case never produced type='page' across {SAMPLES} samples. "
        f"Got: {cold_types}. Triage override not working."
    )
    assert any(t == "email" for t in warm_types), (
        f"WARM case never produced type='email' across {SAMPLES} samples. "
        f"Got: {warm_types}. Triage override not working."
    )

    all_types = cold_types + warm_types
    assert len(set(all_types)) > 1, (
        f"Anti-collapse: every result across all cases was the same type. "
        f"Got: {all_types}"
    )
