"""Portal concierge context + prompt assembly. Pure / Flask-free / unit-testable.
Grounds the ongoing-concierge chat in the client's biofield findings + owned remedies."""

def build_context(content, orders):
    content = content or {}
    layers = [l for l in (content.get("layers") or []) if isinstance(l, dict)]
    findings = [f for f in (content.get("findings") or []) if isinstance(f, dict)]
    owned = []
    for o in (orders or []):
        for it in (o.get("items") or []):
            nm = (it.get("name") or "").strip()
            if nm and nm not in owned:
                owned.append(nm)
    return {"layers": layers, "findings": findings, "owned": owned,
            "has_data": bool(layers or findings or owned)}

_BASE = (
    "You are Dr. Glen Swartwout's warm, ongoing health concierge (naturopathic physician, "
    "Hilo Hawai'i) inside this client's private portal. They are a known client; help them "
    "with their scan findings, their remedies and protocol (what to take when), reorders, and "
    "well-matched complements. Calm, consultative, never pushy: they are served and in control.\n"
    "- Ground every answer in THEIR data below; reference their actual findings/remedies by name.\n"
    "- Ask ONE gentle question at a time when you need more. Functional Formulations first.\n"
    "- When it fits, suggest ONE complementary remedy at a time with a short plain reason.\n"
    "- Keep replies short and warm. Do not invent prices or URLs. No em dashes, no ALL CAPS, "
    "never prefix anything with 'Hook:'. Sign off as Dr. Glen only when concluding."
)

def system_prompt(ctx):
    ctx = ctx or {}
    parts = [_BASE, "\n\nTHIS CLIENT'S DATA:"]
    if ctx.get("owned"):
        parts.append("Remedies they already own: " + ", ".join(ctx["owned"]) + ".")
    fnd = [f.get("name") or f.get("code") for f in (ctx.get("findings") or []) if (f.get("name") or f.get("code"))]
    if fnd:
        parts.append("Scan findings: " + ", ".join(str(x) for x in fnd) + ".")
    for l in (ctx.get("layers") or []):
        seg = f"Layer {l.get('n','?')}: {l.get('title','')}".strip()
        if l.get("meaning"): seg += f" ({l['meaning']})"
        if l.get("remedy"): seg += f" - remedy: {l['remedy']}"
        parts.append(seg)
    if not ctx.get("has_data"):
        parts.append("(No scan or order data on file yet - answer generally and invite them to share.)")
    return "\n".join(parts)
