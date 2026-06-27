"""Structural eval for the 5-beat brief instruction.

Calls Haiku with the real brief instruction, injecting a representative
retrieval context so a real page URL exists in the model's context.
Samples EACH case 5 times and asserts as PASS-RATES (LLM output is stochastic;
a single rogue sample must not fail the gate).

Structural invariants (>= SAMPLES-1 of SAMPLES per case):
- A CTA directive is emitted (cta is not None) with a valid type
- The sentinel ⟦CTA⟧ is NOT visible in the cleaned text
- Word count <= WORD_CEIL (a "bounded brief" check, not the ~200 aspiration)
- The string "Hook" does not appear in the cleaned text

Differentiation (the key bug we fixed: every case collapsed to 'action'):
- COLD/generic case: >= 2 of 5 samples have type == "page"
- WARM/personal-plateau case: >= 2 of 5 samples have type == "email"

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

SAMPLES = 5


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

    # LLM output is stochastic, so assert as RATES over SAMPLES, not all-must-pass
    # (a single rogue sample must not fail the gate). Structural invariants must
    # hold for at least SAMPLES-1 of SAMPLES; differentiation needs a minority
    # (>=2) so it stays stable while still catching a full collapse-to-one-type.
    WORD_CEIL = 280            # "is it a bounded brief", not the ~200 aspiration (max observed ~242)
    MIN_STRUCT_OK = SAMPLES - 1
    MIN_DIFFERENTIATED = 2

    results = {}   # label -> list of cta type strings
    word_stats = {}

    for label, q in CASES:
        user_msg = RETRIEVAL_PREFIX + q
        types_seen, struct_ok, words = [], 0, []

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
            wc = len(clean.split())
            words.append(wc)
            ok = (
                cta is not None
                and cta["type"] in VALID_TYPES
                and "⟦CTA⟧" not in clean
                and wc <= WORD_CEIL
                and "Hook" not in clean
            )
            if ok:
                struct_ok += 1

        results[label] = types_seen
        word_stats[label] = words
        print(f"[{label}] types={types_seen} words={words} struct_ok={struct_ok}/{SAMPLES}")

        assert struct_ok >= MIN_STRUCT_OK, (
            f"[{label}] only {struct_ok}/{SAMPLES} samples met the structural "
            f"invariants (directive present, valid type, sentinel stripped, "
            f"<= {WORD_CEIL} words, no 'Hook'). types={types_seen} words={words}"
        )

    # --- Differentiation (the bug we fixed: everything collapsed to 'action') ---
    cold_pages = sum(1 for t in results["cold"] if t == "page")
    warm_emails = sum(1 for t in results["warm"] if t == "email")
    print(f"differentiation: cold page={cold_pages}/{SAMPLES} warm email={warm_emails}/{SAMPLES}")

    assert cold_pages >= MIN_DIFFERENTIATED, (
        f"COLD/generic case produced type='page' only {cold_pages}/{SAMPLES} times "
        f"(need >= {MIN_DIFFERENTIATED}). Got {results['cold']}. Triage not working."
    )
    assert warm_emails >= MIN_DIFFERENTIATED, (
        f"WARM/personal case produced type='email' only {warm_emails}/{SAMPLES} times "
        f"(need >= {MIN_DIFFERENTIATED}). Got {results['warm']}. Triage not working."
    )
