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

def review_action(cx, variation_id, decision, prompt_template=None):
    from dashboard import sales_prompt_variations as _pv
    row = cx.execute("SELECT id FROM sales_prompt_variations WHERE id=?", (variation_id,)).fetchone()
    if not row:
        return {"ok": False, "error": "not found"}
    if prompt_template is not None and decision in ("approve", "edit"):
        cx.execute("UPDATE sales_prompt_variations SET prompt_template=? WHERE id=?",
                   (prompt_template, variation_id))
        cx.commit()
    if decision == "approve":
        _pv.set_state(cx, variation_id, "candidate")
    elif decision == "reject":
        _pv.set_state(cx, variation_id, "retired")
    elif decision == "edit":
        pass
    else:
        return {"ok": False, "error": "bad decision"}
    return {"ok": True}

def topup(cx, *, threshold=2, generate=None):
    from dashboard import sales_prompt_variations as _pv, sales_image_prompts as _sip
    generate = generate or generate_candidates
    done = {}
    for kind in _sip.IMAGE_KINDS:
        bench = len(_pv.candidate_variations(cx, kind)) + len(_pv.review_variations(cx, kind))
        if bench < threshold:
            generate(cx, kind, threshold - bench)
            done[kind] = True
    return done

def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))

def review_console_html(cx):
    from dashboard import sales_prompt_variations as _pv, sales_image_prompts as _sip
    parts = ["<h2>Prompt candidates (review)</h2>"]
    for kind in _sip.IMAGE_KINDS:
        parts.append(f"<h3>{_esc(kind)} "
                     f"<button onclick=\"pg('generate',{{kind:'{_esc(kind)}',n:2}})\">Generate 2</button></h3>")
        revs = _pv.review_variations(cx, kind)
        if not revs:
            parts.append("<p>(none in review)</p>")
        for v in revs:
            tid = f"pg{v['id']}"
            parts.append(
                f"<div class='pg-rev'><textarea id='{tid}' rows='2' cols='80'>{_esc(v['prompt_template'])}</textarea><br>"
                f"<button onclick=\"pg('review',{{id:{v['id']},decision:'approve',prompt_template:document.getElementById('{tid}').value}})\">Approve</button> "
                f"<button onclick=\"pg('review',{{id:{v['id']},decision:'edit',prompt_template:document.getElementById('{tid}').value}})\">Save edit</button> "
                f"<button onclick=\"pg('review',{{id:{v['id']},decision:'reject'}})\">Reject</button></div>")
    parts.append("<script>function pg(op,body){fetch('/console/image-prompts/'+op,{method:'POST',"
                 "headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})"
                 ".then(function(){location.reload();});}</script>")
    return "".join(parts)
