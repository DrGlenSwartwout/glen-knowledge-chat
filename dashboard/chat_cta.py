"""Pure parser for the trailing CTA directive a brief answer emits.
Grammar: a final line  ⟦CTA⟧ <type> | <target> | <label>
type ∈ page|email|action|inline. Flask-free; unit-testable in isolation."""

SENTINEL = "⟦CTA⟧"
VALID_TYPES = ("page", "email", "action", "inline")


def stream_visible(tokens):
    """Yield the visible portion of a token stream, never emitting the CTA
    directive (the sentinel onward) and never flashing a PARTIAL sentinel.
    Holds back the trailing chars that could be the start of SENTINEL until
    they are confirmed safe, flushing the legitimate remainder at the end.
    `tokens` is any iterable of str chunks. Yields str deltas to send to the client."""
    acc = ""
    emitted = 0
    hold = len(SENTINEL) - 1
    for tok in tokens:
        acc += (tok or "")
        cut = acc.find(SENTINEL)
        if cut != -1:
            # full sentinel present: emit up to it, then stop forever
            if cut > emitted:
                yield acc[emitted:cut]
            emitted = cut
            return
        # no full sentinel yet: safe to emit everything except a possible
        # partial sentinel at the tail (the last `hold` chars)
        safe_end = max(emitted, len(acc) - hold)
        if safe_end > emitted:
            yield acc[emitted:safe_end]
            emitted = safe_end
    # stream ended with no sentinel: flush whatever remains (it was a real tail)
    if len(acc) > emitted:
        yield acc[emitted:]

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
