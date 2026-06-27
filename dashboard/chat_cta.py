"""Pure parser for the trailing CTA directive a brief answer emits.
Grammar: a final line  ⟦CTA⟧ <type> | <target> | <label>
type ∈ page|email|action|inline. Flask-free; unit-testable in isolation."""

SENTINEL = "⟦CTA⟧"
CHIPS_SENTINEL = "⟦CHIPS⟧"
VALID_TYPES = ("page", "email", "action", "inline")


def stream_visible(tokens, sentinel=SENTINEL):
    """Yield the visible portion of a token stream, never emitting the directive
    (the sentinel onward) and never flashing a PARTIAL sentinel.
    Holds back the trailing chars that could be the start of sentinel until
    they are confirmed safe, flushing the legitimate remainder at the end.
    `tokens` is any iterable of str chunks. Yields str deltas to send to the client.

    IMPORTANT: when the sentinel is found we stop YIELDING, but we still fully
    drain the input iterator. Callers feed a generator whose iteration has a side
    effect (appending every token to a `full`/`full_answer` accumulator that is
    later re-parsed for the directive's payload). If we abandoned the generator at
    the sentinel, the directive's own option/argument tokens (which come AFTER the
    sentinel) would never be pulled, never accumulated, and the re-parse would see
    an empty directive. So drain the rest before returning."""
    it = iter(tokens)
    acc = ""
    emitted = 0
    hold = len(sentinel) - 1
    for tok in it:
        acc += (tok or "")
        cut = acc.find(sentinel)
        if cut != -1:
            # full sentinel present: emit up to it, stop yielding, but DRAIN the
            # rest of the input so the caller's per-token side effects all run.
            if cut > emitted:
                yield acc[emitted:cut]
            emitted = cut
            for _ in it:
                pass
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

def parse_chips(answer: str):
    """Extract a trailing ⟦CHIPS⟧ a | b | c directive. Returns (clean_text, chips[<=4])."""
    text = answer or ""
    idx = text.rfind(CHIPS_SENTINEL)
    if idx == -1:
        return (text.strip(), [])
    directive = text[idx + len(CHIPS_SENTINEL):]
    clean = text[:idx].rstrip()
    chips = [c.strip() for c in directive.split("|")]
    chips = [c for c in chips if c][:4]
    return (clean, chips)
