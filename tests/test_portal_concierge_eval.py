"""Pass-rate eval: portal concierge prompt grounds answers in client data.

Skipped when ANTHROPIC_API_KEY is absent (CI / unit-only runs).
Run under Doppler:
  S=/private/tmp/claude-501/-Users-remedymatch-AI-Training/.../scratchpad/t3pcz
  mkdir -p "$S"
  doppler run -p remedy-match -c prd -- env DATA_DIR="$S" \
    python3 -m pytest tests/test_portal_concierge_eval.py -q -p no:cacheprovider
"""

import os
import sys
import pytest
import anthropic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import dashboard.portal_concierge as pc

# --- tuneable constants ---
SAMPLES = 5
MIN_GROUNDED = 3   # samples that reference seeded data (owned remedy OR finding)
MIN_STYLE = 4      # samples with no em dash and no "Hook:"

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 400

QUESTION = "What's going on with my stress, and what do I already have that helps?"

# Seeded distinctive data
_ORDERS = [{"items": [{"name": "Terrain Restore", "qty": 1}]}]
_CONTENT = {
    "findings": [{"code": "EI8", "name": "stress"}],
    "layers": [{"n": 3, "title": "Liver terrain", "remedy": "Terrain Restore"}],
}

# Terms that prove grounding
_GROUNDING_TERMS = ["terrain restore", "stress", "liver terrain", "ei8"]


@pytest.fixture(scope="module")
def api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


@pytest.fixture(scope="module")
def system_text():
    ctx = pc.build_context(_CONTENT, _ORDERS)
    return pc.system_prompt(ctx)


def _contains_grounding(text):
    """True if text references any seeded data term (case-insensitive)."""
    lower = text.lower()
    return any(term in lower for term in _GROUNDING_TERMS)


def _has_style_violation(text):
    """True if text has an em dash or a 'Hook:' prefix."""
    return "—" in text or "Hook:" in text


def test_grounding_and_style_pass_rate(api_key, system_text):
    client = anthropic.Anthropic(api_key=api_key)

    grounded_count = 0
    style_ok_count = 0
    snippets = []

    for i in range(SAMPLES):
        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_text,
            messages=[{"role": "user", "content": QUESTION}],
        )
        text = msg.content[0].text if msg.content else ""
        grounded = _contains_grounding(text)
        style_ok = not _has_style_violation(text)

        if grounded:
            grounded_count += 1
        if style_ok:
            style_ok_count += 1

        matched = [t for t in _GROUNDING_TERMS if t in text.lower()]
        snippets.append({
            "i": i + 1,
            "grounded": grounded,
            "style_ok": style_ok,
            "matched_terms": matched,
            "snippet": text[:200].replace("\n", " "),
        })

    print("\n--- Portal Concierge Grounding Eval ---")
    print(f"Model: {MODEL}  |  Samples: {SAMPLES}")
    print(f"System prompt (first 300 chars): {system_text[:300]!r}")
    print()
    for s in snippets:
        mark_g = "GROUNDED" if s["grounded"] else "UNGROUNDED"
        mark_s = "STYLE-OK" if s["style_ok"] else "STYLE-FAIL"
        print(f"  [{s['i']}] {mark_g} ({s['matched_terms']}) | {mark_s}")
        print(f"       {s['snippet']!r}")
    print()
    print(f"Grounding rate: {grounded_count}/{SAMPLES}  (need >={MIN_GROUNDED})")
    print(f"Style rate:     {style_ok_count}/{SAMPLES}  (need >={MIN_STYLE})")

    assert grounded_count >= MIN_GROUNDED, (
        f"Grounding too low: {grounded_count}/{SAMPLES} samples referenced seeded data "
        f"(need >={MIN_GROUNDED}). Check that build_context/system_prompt includes owned "
        f"remedies and findings in the prompt."
    )
    assert style_ok_count >= MIN_STYLE, (
        f"Style violations too frequent: only {style_ok_count}/{SAMPLES} clean "
        f"(need >={MIN_STYLE}). Model emitted em dash or 'Hook:' prefix."
    )
