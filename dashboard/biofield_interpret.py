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
    '{"header": str, "layers": [{"layer": int, "head": str, "most_affected": str, '
    '"remedy": str, "dosage": str, "frequency": str, "timing": str}]}\n'
    "A single causal layer can need MORE THAN ONE remedy to balance. When a layer is "
    "balanced by several remedies, list them all in a `remedies` array on that layer "
    '(each item {"remedy": str, "dosage": str, "frequency": str, "timing": str}); use the '
    "plain `remedy` string only when a layer has exactly one. NEVER keep just the first "
    "remedy and drop the rest.\n"
    "Glen's grammar:\n"
    "- 'BSI N times A to B times C, phase P' and 'the location of the N is X' -> summarize into `header` "
    "(a short string); do NOT turn the BSI/phase/location into a layer.\n"
    "- 'X is the head and tail of the [first/second/...] causal chain' OR 'the [Nth] layer is X, head and "
    "tail' -> a layer at that number with head = most_affected = X ('head and tail' means the stress is both "
    "the head and the most-affected end of that layer).\n"
    "- 'balanced by [REMEDY]' or 'balances with [REMEDY]' -> that layer's `remedy`; keep the remedy as spoken. "
    "'balanced by [REMEDY A] and [REMEDY B]' (or 'and also [REMEDY C]') -> that layer needs ALL of them: put "
    "each in the layer's `remedies` array.\n"
    "- If a DOSE is spoken with the remedy (e.g. 'neuromagnesium one scoop twice a day', or '10 drops "
    "three times a day before food'), put ONLY the product name in `remedy` ('neuromagnesium'), the amount in "
    "`dosage` ('one scoop' / '10 drops'), the rate in `frequency` ('twice a day'), and any with/before-food "
    "note in `timing`. If no dose is spoken, leave dosage/frequency/timing as empty strings.\n"
    "- TERRAIN RESTORE: if the remedy is an essence, homeopathic, tincture, gemmotherapy, "
    "regenerative peptide, or liquid ORMUS remedy, it is delivered in the Terrain Restore liquid base "
    "-> set `remedy` to '[name] in Terrain Restore' (e.g. 'Perelandra essence in Terrain Restore'). "
    "Capsules, powders, and tablets are NOT in Terrain Restore.\n"
    "- Layer numbers come from 'first'/'second'/'third'/'fourth' or 'first causal chain'/'second layer' -> "
    "1, 2, 3, 4...\n"
    "- CAPTURE EVERY numbered layer the clinician mentions, INCLUDING the last one, even if its stress is "
    "vague (e.g. 'the essence') or the remedy is unusual or imperfectly transcribed. Do not skip a layer "
    "just because it is short or oddly worded.\n"
    "- 'that also balances Y' is a cross-link for context -> do NOT make it its own layer.\n"
    "- Emit a layer when it has a stress (head) and a remedy. Order layers by number. "
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
        if not head:
            continue
        try:
            layer = int(l.get("layer"))
        except (TypeError, ValueError):
            layer = None
        most_affected = (l.get("most_affected") or head).strip()
        # A layer can carry several remedies. Accept both forms: a `remedies` array
        # (items are strings or {remedy,dosage,frequency,timing} dicts) and the plain
        # single `remedy`. Emit one chain entry per remedy, sharing layer/head.
        for spec in _layer_remedies(l):
            remedy = (spec.get("remedy") or "").strip()
            if not remedy:
                continue
            layers.append({"layer": layer, "head": head, "most_affected": most_affected,
                           "remedy": remedy,
                           "dosage": (spec.get("dosage") or "").strip(),
                           "frequency": (spec.get("frequency") or "").strip(),
                           "timing": (spec.get("timing") or "").strip()})
    # Stable sort by layer keeps multiple remedies for the same layer in spoken order.
    layers.sort(key=lambda x: (x["layer"] is None, x["layer"] or 0))
    return {"header": (data.get("header") or "").strip(), "layers": layers}


_STRESS_SYSTEM = (
    "You read a clinician's spoken biofield-testing transcript (Dr. Glen Swartwout) and extract "
    "ONLY the distinct stress / issue / weakness names they name as present — NOT remedies, layers, "
    'or doses. Return STRICT JSON ONLY, no prose: {"stresses": [str, ...]}.\n'
    "- 'the stress is X', 'I'm seeing X', 'also X', 'there's X here' -> X is a stress.\n"
    "- If a remedy is named (e.g. 'balanced by Neuro Magnesium'), do NOT include the remedy; you MAY "
    "include the stress it balances if that stress is named.\n"
    "- Deduplicate. If nothing is parseable, return an empty list."
)


def build_stress_prompt(transcript):
    return {"system": _STRESS_SYSTEM, "user": "TRANSCRIPT:\n" + (transcript or "")}


def interpret_stresses(transcript, complete):
    """transcript + complete(system,user) -> [stress label, ...] (Phase 1, stresses only)."""
    if not (transcript or "").strip():
        return []
    p = build_stress_prompt(transcript)
    data = _parse_json(complete(p["system"], p["user"]))
    out, seen = [], set()
    for s in (data.get("stresses") or []):
        label = (s if isinstance(s, str) else (s.get("name") if isinstance(s, dict) else "")) or ""
        label = label.strip()
        k = label.lower()
        if label and k not in seen:
            seen.add(k)
            out.append(label)
    return out


def _layer_remedies(l):
    """Normalize a layer dict's remedy/remedies into a list of dose-bearing dicts."""
    out = []
    raw = l.get("remedies")
    if isinstance(raw, list) and raw:
        for it in raw:
            if isinstance(it, str):
                out.append({"remedy": it})
            elif isinstance(it, dict):
                out.append({"remedy": it.get("remedy") or it.get("name") or "",
                            "dosage": it.get("dosage"), "frequency": it.get("frequency"),
                            "timing": it.get("timing")})
        return out
    # Single-remedy fallback: dose lives on the layer itself.
    return [{"remedy": l.get("remedy") or "", "dosage": l.get("dosage"),
             "frequency": l.get("frequency"), "timing": l.get("timing")}]
