"""Increment 4c Phase 2: turn a spoken testing transcript into structured causal-chain
rows, using Dr. Glen Swartwout's mapped narration grammar (see
00 System/fmp-biofield-4c-narration-sample-2026-06-23.md).

Best-effort: the LLM proposes rows; Glen/Rae review them in the editor and the full
transcript stays the fallback. `complete(system, user) -> str` is injected so this is
testable without a live API call.
"""
import json
import re

_SYSTEM = (
    "You convert a clinician's spoken biofield-testing transcript (Dr. Glen Swartwout) into a "
    "structured causal chain. Return STRICT JSON ONLY, no prose:\n"
    '{"header": str, "layers": [{"layer": int, "head": str, "most_affected": str, "remedy": str}]}\n'
    "Glen's grammar:\n"
    "- 'BSI N times A to B times C, phase P' and 'the location of the N is X' -> summarize into `header` "
    "(a short string); do NOT turn the BSI/phase/location into a layer.\n"
    "- 'X is the head and tail of the [first/second/...] causal chain' OR 'the [Nth] layer is X, head and "
    "tail' -> a layer at that number with head = most_affected = X ('head and tail' means the stress is both "
    "the head and the most-affected end of that layer).\n"
    "- 'balanced by [REMEDY]' or 'balances with [REMEDY]' -> that layer's `remedy`; keep the remedy as spoken "
    "(e.g. 'Microbiome', 'Neuro-Magnesium').\n"
    "- Layer numbers come from 'first'/'second'/'third' or 'first causal chain'/'second layer' -> 1, 2, 3...\n"
    "- 'that also balances Y' is a cross-link for context -> do NOT make it its own layer.\n"
    "- Emit a layer only when it has BOTH a stress (head) and a remedy. Order layers by number. "
    "If nothing is parseable, return an empty layers array."
)


def build_interpret_prompt(transcript):
    return {"system": _SYSTEM, "user": "TRANSCRIPT:\n" + (transcript or "")}


def _parse_json(text):
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except Exception:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except Exception:
            return {}
    return {}


def interpret_transcript(transcript, complete):
    """transcript + complete(system,user) -> {header, layers:[{layer,head,most_affected,remedy}]}."""
    if not (transcript or "").strip():
        return {"header": "", "layers": []}
    p = build_interpret_prompt(transcript)
    data = _parse_json(complete(p["system"], p["user"]))
    layers = []
    for l in (data.get("layers") or []):
        if not isinstance(l, dict):
            continue
        head = (l.get("head") or "").strip()
        remedy = (l.get("remedy") or "").strip()
        if not (head and remedy):
            continue
        try:
            layer = int(l.get("layer"))
        except (TypeError, ValueError):
            layer = None
        layers.append({"layer": layer, "head": head,
                       "most_affected": (l.get("most_affected") or head).strip(),
                       "remedy": remedy})
    layers.sort(key=lambda x: (x["layer"] is None, x["layer"] or 0))
    return {"header": (data.get("header") or "").strip(), "layers": layers}
