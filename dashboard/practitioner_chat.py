"""Self-contained product-selection chat scoped to a single practitioner's catalog.
Recommends ONLY from the supplied catalog; never emits external/RM links or prices."""
import json, re

_SYSTEM = (
    "You are a product-selection assistant for a natural-health practitioner's own "
    "dispensary. You may ONLY discuss and recommend products from the CATALOG provided "
    "below. Never mention, link to, or price any other store, website, or 'online' option. "
    "Never invent products not in the catalog. Keep replies short and supportive. "
    "Return JSON: {\"reply\": str, \"suggested_slugs\": [slugs from the catalog]}.\n\nCATALOG:\n"
)
_URL_RE = re.compile(r"https?://\S+", re.I)
_BANNED = re.compile(r"\b(truly\.vip|truly\.so|remedymatch|illtowell)\S*", re.I)


def _llm_json(system, messages):
    """One Haiku call returning parsed JSON. Monkeypatched in tests."""
    import app as _app
    r = _app._cl.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=400,
        system=system, messages=messages)
    txt = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0)) if m else {"reply": txt.strip(), "suggested_slugs": []}


def _scrub(text):
    text = _URL_RE.sub("", text or "")
    text = _BANNED.sub("our selection", text)
    return text.strip()


def scoped_reply(message, history, catalog):
    """Return {reply, suggested_slugs} grounded only in `catalog`
    ([{slug,name,description}]). Suggested slugs are validated against the catalog;
    the reply is scrubbed of any external links/store mentions."""
    valid = {c["slug"] for c in (catalog or [])}
    cat_txt = "\n".join(f"- {c['slug']}: {c.get('name','')} — {c.get('description','')}"
                        for c in (catalog or []))
    msgs = list(history or []) + [{"role": "user", "content": str(message or "")}]
    try:
        out = _llm_json(_SYSTEM + cat_txt, msgs)
    except Exception:
        return {"reply": "Sorry, I had trouble — please pick from the list.", "suggested_slugs": []}
    slugs = [s for s in (out.get("suggested_slugs") or []) if s in valid]
    return {"reply": _scrub(out.get("reply", "")), "suggested_slugs": slugs}
