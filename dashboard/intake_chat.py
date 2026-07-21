"""Conversational intake: prompt construction + reply parsing for the chat mode.

The LLM (site model + voice) walks a client through INTAKE_FORM one section at a
time in Glen's voice. Each reply carries a short spoken line plus the structured
field `updates` that fill the live form in real time. Pure helpers only — the
anthropic call + streaming live in the route.
"""
import json


def form_outline():
    """A compact, model-facing description of every field to collect: id, label,
    type, and (for scales) what each number means."""
    from dashboard import intake as _intake
    out = []
    for sec in _intake.INTAKE_FORM["sections"]:
        out.append(f"\n## {sec['title']}")
        for f in sec["fields"]:
            fid, ftype = f["id"], f["type"]
            if ftype == "scale":
                opts = "; ".join(f"{o['value']}={o['label']}" for o in f["options"])
                out.append(f"- {fid} (scale, record the number 1-5 or 1-10): {f['label']} [{opts}]")
            elif ftype == "table":
                cols = ", ".join(c["id"] for c in f["columns"])
                out.append(f"- {fid} (list of rows, each {{{cols}}}): {f['label']}")
            elif ftype == "single_choice":
                out.append(f"- {fid} (choose one of {f['options']}): {f['label']}")
            elif ftype == "consent":
                out.append(f"- {fid} (consent — leave for the final review/sign step): {f['label']}")
            else:
                out.append(f"- {fid} ({ftype}): {f['label']}")
    return "\n".join(out)


# Fields already known from the opt-in step — the guide must NOT re-ask these.
PREFILLED = ("first_name", "last_name", "email")


def build_system(base_voice, name):
    """Intake-guide system prompt. `base_voice` is the site's own voice/system
    prompt (injected by the route so the guide sounds like the site bot)."""
    who = name or "there"
    return (
        (base_voice or "").strip()
        + "\n\n---\n\n"
        "You are now acting as Dr. Glen's warm clinical intake guide. You are gently "
        f"walking {who} through their intake form by conversation, in the same voice.\n\n"
        "RULES:\n"
        "- Cover ONE section at a time, in order. Ask in plain, caring language — never "
        "dump the whole form at once.\n"
        f"- Their name and email are already on file; do NOT ask for {', '.join(PREFILLED)}.\n"
        "- For scale questions, explain the choices simply and record the single number "
        "the person lands on.\n"
        "- Record ONLY what they actually tell you. If they decline or have none, skip the "
        "field — never invent an answer.\n"
        "- Leave the consent/terms for the final review-and-sign step; do not collect it here.\n\n"
        "OUTPUT: respond with ONE JSON object and nothing else:\n"
        '{"say": "<your short warm reply: acknowledge what they said, then ask the next '
        'question>", "updates": {"<field_id>": <value>, ...}, "done": <true only once every '
        'required field is gathered and you have confirmed the person is finished>}\n'
        "`updates` carries only the fields learned from their LAST message (empty object if "
        "none). Table fields take a list of row objects. Never put first_name/last_name/email "
        "in updates."
    )


def parse_reply(text):
    """Extract (say, updates, done) from a model reply. Tolerant of fenced code
    blocks or leading prose; falls back to treating the whole text as `say`."""
    raw = (text or "").strip()
    obj = _extract_json(raw)
    if not isinstance(obj, dict):
        return raw, {}, False
    say = str(obj.get("say") or "").strip()
    updates = obj.get("updates") if isinstance(obj.get("updates"), dict) else {}
    done = bool(obj.get("done"))
    return say, updates, done


def _extract_json(raw):
    # strip a ```json ... ``` fence if present
    if "```" in raw:
        seg = raw.split("```", 2)
        if len(seg) >= 2:
            body = seg[1]
            if body.lstrip().lower().startswith("json"):
                body = body.lstrip()[4:]
            raw = body.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    # last resort: first {...} span
    i, j = raw.find("{"), raw.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(raw[i:j + 1])
        except Exception:
            return None
    return None
