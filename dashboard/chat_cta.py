"""Pure parser for the trailing CTA directive a brief answer emits.
Grammar: a final line  ⟦CTA⟧ <type> | <target> | <label>
type ∈ page|email|action|inline. Flask-free; unit-testable in isolation."""

SENTINEL = "⟦CTA⟧"
VALID_TYPES = ("page", "email", "action", "inline")

def parse_cta(answer: str):
    text = answer or ""
    idx = text.rfind(SENTINEL)
    if idx == -1:
        return (text.strip(), None)
    directive = text[idx + len(SENTINEL):].strip()
    clean = text[:idx].rstrip()
    parts = [p.strip() for p in directive.split("|")]
    ctype = parts[0].lower() if parts else ""
    if ctype not in VALID_TYPES:
        return (clean, None)          # strip the sentinel, no cta
    target = parts[1] if len(parts) > 1 else ""
    label = parts[2] if len(parts) > 2 else ""
    return (clean, {"type": ctype, "target": target, "label": label})
