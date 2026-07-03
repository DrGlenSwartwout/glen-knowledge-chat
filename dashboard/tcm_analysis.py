"""Shared TCM/emotional voice-analysis engine.

Extracted from journal_blueprint.py so both the standalone voice Journal and the
member portal chat can run the same Haiku element analysis. Pure functions +
constants only; no Flask, no store, no DB. Callers pass in transcript + lexical.
"""
import json
import os

import requests

ANTHROPIC_MESSAGES    = "https://api.anthropic.com/v1/messages"
HAIKU_MODEL           = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Haiku 4.5 — structured TCM analysis
# ---------------------------------------------------------------------------

# 48-emotion vocabulary inherited from Hume's Expression Measurement taxonomy.
# Locked here so future Haiku model upgrades produce comparable score vectors,
# AND so swapping in a Hume-successor (if one emerges) is a single function.
HUME_48_EMOTIONS = [
    "Admiration", "Adoration", "Aesthetic Appreciation", "Amusement", "Anger",
    "Anxiety", "Awe", "Awkwardness", "Boredom", "Calmness",
    "Concentration", "Confusion", "Contemplation", "Contempt", "Contentment",
    "Craving", "Desire", "Determination", "Disappointment", "Disgust",
    "Distress", "Doubt", "Ecstasy", "Embarrassment", "Empathic Pain",
    "Entrancement", "Envy", "Excitement", "Fear", "Guilt",
    "Horror", "Interest", "Joy", "Love", "Nostalgia",
    "Pain", "Pride", "Realization", "Relief", "Romance",
    "Sadness", "Satisfaction", "Shame", "Surprise (negative)",
    "Surprise (positive)", "Sympathy", "Tiredness", "Triumph",
]

HAIKU_SYSTEM_PROMPT = f"""You are a clinical analysis engine for daily voice-journal entries. Your output is ingested by a Traditional Chinese Medicine (TCM) practitioner's longitudinal-tracking dashboard.

You read the entry through FIVE LAYERS:

1. EMOTION VECTOR — score each of the 48 emotion dimensions on 0.0–1.0 (continuous, not categorical). Most should be near 0; only the genuinely-present emotions register.
   The 48 dimensions: {", ".join(HUME_48_EMOTIONS)}.

2. FIVE ELEMENTS (horizontal axis — emotional/organ resonance).
   • Wood / Liver–Gallbladder — anger, frustration, irritation, determination, drive
   • Fire / Heart–Small Intestine–Pericardium — joy, mania, ecstasy, romance, anxiety-as-overstimulation
   • Earth / Spleen–Stomach — worry, contemplation, rumination, sympathy
   • Metal / Lung–Large Intestine — sadness, grief, regret, nostalgia, disappointment
   • Water / Kidney–Bladder — fear, dread, distress, shame, deep depletion
   Score 0–100 each, sum to 100.

3. THREE TREASURES (vertical axis — constitutional depth).
   • Jing (Essence) — Kidney-rooted hereditary substance. Signals: fundamental (not situational) fatigue, depletion language ("running on fumes for years"), bone/teeth/hair/sexual-vitality complaints, generational/family-line themes, voice low-pitched/monotone/breathy.
   • Qi (Vital Force) — current functional vitality. Signals: situational energy, breath, digestion, immunity, current-stress level. Voice: speech rate, breath support (sentence-length-before-breath), volume.
   • Shen (Spirit) — presence, awareness, joy, meaning, eyes-lit-up. Signals: clarity, awe, contemplative depth, capacity for joy, prosodic vibrancy.
   Score 0–100 each, sum to 100.
   Confidence note: Jing is constitutional and slow-moving. From a single 30s–5min entry, Jing confidence is typically 0.30–0.60 unless multiple deep-fatigue indicators stack. Qi and Shen are confidently scored from one entry (typical 0.70–0.90).

4. POLYVAGAL STATE (autonomic).
   • ventral_vagal — safe, connected, social-engaged, regulated
   • sympathetic — mobilized, activated, fight/flight, urgent
   • dorsal_vagal — collapsed, shut-down, freeze, dissociated
   Score 0–100 each, sum to 100.
   Read both content cues ("I feel safe" / "I'm shutting down" / "I'm wired") AND lexical-pace cues from the metrics provided (high wpm + low pause_density → sympathetic; very low wpm + high pause_density + low type_token_ratio → dorsal; balanced metrics → ventral).

5. INTERNAL CONGRUENCE (transcript-internal).
   Does the speaker contradict themselves within this entry — saying one thing then immediately undermining it? Are stated feelings congruent with the felt-sense the language carries?
   Output: {{"score": 0.0–1.0, "self_contradictions": [...quoted phrases...], "notes": "short clinical observation"}}.

OUTPUT STRICTLY AS A SINGLE JSON OBJECT, NO PROSE BEFORE OR AFTER, with this exact shape:

{{
  "emotions": {{ "<name>": <0–1>, ... all 48 ... }},
  "elements": {{ "Wood": <0–100>, "Fire": <0–100>, "Earth": <0–100>, "Metal": <0–100>, "Water": <0–100> }},
  "treasures": {{ "Jing": <0–100>, "Qi": <0–100>, "Shen": <0–100> }},
  "treasure_confidence": {{ "Jing": <0–1>, "Qi": <0–1>, "Shen": <0–1> }},
  "polyvagal_state": {{ "ventral_vagal": <0–100>, "sympathetic": <0–100>, "dorsal_vagal": <0–100> }},
  "congruence": {{ "score": <0–1>, "self_contradictions": [...], "notes": "..." }},
  "top_themes": [<3–6 short strings>]
}}

Be precise, not poetic. The dashboard renders these numbers directly.
"""


# Forced-tool schema mirroring HAIKU_SYSTEM_PROMPT's output shape. Kept loose
# (open score maps) so the model fills the named keys per the prompt while the
# API still guarantees the result is valid, parsed JSON — which is the whole
# point: free-text fields can't break the structure anymore.
_NUM_MAP = {"type": "object", "additionalProperties": {"type": "number"}}
ANALYSIS_TOOL = {
    "name": "emit_analysis",
    "description": "Return the structured TCM/emotional analysis for the journal entry.",
    "input_schema": {
        "type": "object",
        "properties": {
            "emotions": _NUM_MAP,
            "elements": _NUM_MAP,
            "treasures": _NUM_MAP,
            "treasure_confidence": _NUM_MAP,
            "polyvagal_state": _NUM_MAP,
            "congruence": {
                "type": "object",
                "properties": {
                    "score": {"type": "number"},
                    "self_contradictions": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
            },
            "top_themes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["emotions", "elements", "treasures"],
    },
}


def _haiku_analyze(transcript: str, lexical: dict) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    user_message = (
        f"Lexical metrics from Whisper word timestamps:\n"
        f"  wpm: {lexical.get('wpm')}\n"
        f"  pause_density: {lexical.get('pause_density')}\n"
        f"  pause_count: {lexical.get('pause_count')}\n"
        f"  breath_proxy: {lexical.get('breath_proxy')}  (avg words per breath-bounded run)\n"
        f"  type_token_ratio: {lexical.get('type_token_ratio')}\n"
        f"  word_count: {lexical.get('word_count')}\n"
        f"  median_word_dur_ms: {lexical.get('median_word_dur_ms')}\n\n"
        f"Transcript:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
        f"Produce the JSON now."
    )

    payload = {
        "model": HAIKU_MODEL,
        "max_tokens": 4096,
        "system": [
            {
                "type": "text",
                "text": HAIKU_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {"role": "user", "content": user_message}
        ],
        # Force structured output. Returning the analysis as a tool input makes the
        # API hand back already-valid structured data, so free-text fields like
        # `self_contradictions` (which Haiku phrases with literal quotes) can no
        # longer corrupt the JSON the way model-emitted text JSON did.
        "tools": [ANALYSIS_TOOL],
        "tool_choice": {"type": "tool", "name": "emit_analysis"},
    }

    resp = requests.post(
        ANTHROPIC_MESSAGES,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(f"Haiku {resp.status_code}: {resp.text[:300]}")

    body = resp.json()
    # Primary path: the forced tool call carries the analysis as a parsed dict.
    for b in body.get("content", []):
        if b.get("type") == "tool_use" and b.get("name") == "emit_analysis":
            inp = b.get("input")
            if isinstance(inp, dict) and inp:
                return inp
    # Defensive fallback: older/text responses → tolerant JSON extraction.
    text_blocks = [b.get("text", "") for b in body.get("content", []) if b.get("type") == "text"]
    raw = "".join(text_blocks).strip()
    parsed = _extract_json(raw)
    if parsed is None:
        raise RuntimeError(f"Haiku returned non-JSON: {raw[:300]}")
    return parsed


def _extract_json(text: str):
    """Tolerant JSON extractor — handles fenced blocks and stray prose."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None
