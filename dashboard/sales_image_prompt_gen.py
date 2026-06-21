import os, json

_MODEL = os.environ.get("IMAGE_PROMPT_GEN_MODEL", "claude-haiku-4-5-20251001")

def _default_llm(prompt):
    import anthropic
    cli = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    resp = cli.messages.create(model=_MODEL, max_tokens=1500,
                               messages=[{"role": "user", "content": prompt}])
    return "".join(getattr(b, "text", "") for b in resp.content
                   if getattr(b, "type", "") == "text").strip()

def _parse_json_array(text):
    if not text:
        return []
    s = text.find("["); e = text.rfind("]")
    if s == -1 or e == -1 or e < s:
        return []
    try:
        data = json.loads(text[s:e + 1])
    except Exception:
        return []
    return data if isinstance(data, list) else []

def _winners_for_kind(cx, kind, limit=3):
    from dashboard import sales_image_leaderboard as _lb
    try:
        ids = {r[0] for r in cx.execute(
            "SELECT id FROM sales_prompt_variations WHERE kind=?", (kind,)).fetchall()}
        rows = _lb.leaderboard(cx, min_volume=0)["variations"]
        return [r["label"] for r in rows if r["key"] in ids][:limit]
    except Exception:
        return []

def _build_prompt(kind, n, existing, winners):
    from dashboard import sales_image_prompts as _sip
    body = _sip._BODY.get(kind, "")
    ex = "\n".join(f"- {t}" for t in existing) or "(none)"
    win = ", ".join(winners) if winners else "(no data yet)"
    return (
        f"You write image-generation SCENE prompts for a product's '{kind}' imagery.\n"
        f"Scene family: {body}\n"
        f"Currently winning variations (lean toward what works): {win}\n"
        f"Existing prompts — make NEW ones VISIBLY DISTINCT from all of these:\n{ex}\n\n"
        f"Write {n} new, distinct scene prompt_templates in the same family. Rules: NO product or brand "
        "names; NO instructions about text/letters/labels in the image (added later); each one a single "
        "vivid scene description. Return a STRICT JSON array ONLY (no prose, no code fences): "
        '[{"label": "short-kebab-label", "prompt_template": "the full scene description."}]'
    )

def generate_candidates(cx, kind, n=2, *, llm=None):
    from dashboard import sales_prompt_variations as _pv
    llm = llm or _default_llm
    existing = [v["prompt_template"] for v in (
        _pv.active_variations(cx, kind) + _pv.candidate_variations(cx, kind) + _pv.review_variations(cx, kind))]
    winners = _winners_for_kind(cx, kind)
    items = _parse_json_array(llm(_build_prompt(kind, n, existing, winners)))
    seen = set(existing)
    inserted = []
    for it in items:
        if not isinstance(it, dict):
            continue
        tmpl = (it.get("prompt_template") or "").strip()
        label = (it.get("label") or "ai-candidate").strip()
        if not tmpl or tmpl in seen:
            continue
        vid = _pv.insert_variation(cx, kind, label, tmpl, "review")
        seen.add(tmpl)
        inserted.append({"id": vid, "kind": kind, "label": label, "prompt_template": tmpl})
    return inserted
