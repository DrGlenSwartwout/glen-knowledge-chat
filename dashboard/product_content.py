"""Product content layer for the RemedyMatch funnel.

Source of truth for ingredients + page copy is Pinecone `specific-formulations`
(index remedy-match-llc) — each product is stored as chunks whose metadata.text
holds the remedymatch.com page (Contents/ingredients panel + benefit copy). We
fetch ALL chunks for a product by an EXACT title filter (deterministic — no fuzzy
embedding match, which mis-maps infoceuticals), order by chunk_index, and concat.

(The structured FMP ingredient tables live in a separate Supabase project that is
currently paused, so it is not a reliable runtime dependency. Pinecone carries the
same product copy and is always available.)

Two generated, cached content types per product:
  - 'card'       -> {description, ingredients[], benefits[]}  (the product page)
  - 'learn_more' -> {markdown, sources[]}                     (the research page)

Generation is grounded ONLY in retrieved text (page copy + per-ingredient research
studies from the `ingredients` namespace, which carry real study URLs), so the
model extracts/cites rather than invents. Results are cached in chat_log.db keyed
by (product_slug, content_type); regenerate via the console-gated refresh endpoint.
"""
import os
import re
import json
import sqlite3
import datetime as _dt
from pathlib import Path

SPECIFIC_NS = "specific-formulations"
RESEARCH_NS = "ingredients"
_MODEL = "claude-haiku-4-5-20251001"

LOG_DB = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))) / "chat_log.db"


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _dd(s):
    """Enforce Glen's no-em-dash rule (models slip despite the prompt)."""
    return s.replace("—", ", ") if isinstance(s, str) else s


def _clients():
    """Lazily reuse app.py's already-initialized clients (runtime import avoids the
    circular import at module load, since app imports this module)."""
    from app import _idx, _cl, embed
    return _idx, _cl, embed


# ── Cache table ───────────────────────────────────────────────────────────────
def init_product_content_table(cx):
    cx.executescript("""
        CREATE TABLE IF NOT EXISTS generated_product_content (
          id            INTEGER PRIMARY KEY AUTOINCREMENT,
          product_slug  TEXT NOT NULL,
          content_type  TEXT NOT NULL,
          content_json  TEXT NOT NULL,
          sources_json  TEXT,
          model_name    TEXT,
          generated_at  TEXT NOT NULL,
          UNIQUE (product_slug, content_type)
        );
        CREATE INDEX IF NOT EXISTS idx_gpc_slug ON generated_product_content(product_slug, content_type);
    """)


def purge_refusal_cache(cx):
    """Delete cached rows that captured a model refusal (the pre-fix Longevity bug:
    empty page copy made the model reply 'I'm unable to proceed... sections are
    empty', which got cached and served). Cleared rows regenerate, now grounded in
    products.json, on next view. Idempotent; safe to run at every startup."""
    try:
        cur = cx.execute(
            "DELETE FROM generated_product_content WHERE "
            "lower(content_json) LIKE '%unable to proceed%' "
            "OR lower(content_json) LIKE '%sections are empty%'")
        if cur.rowcount:
            print(f"[product_content] purged {cur.rowcount} cached refusal row(s)", flush=True)
        return cur.rowcount
    except Exception as e:
        print(f"[product_content] purge_refusal_cache failed: {e}", flush=True)
        return 0


def _cache_get(slug, ctype):
    try:
        with sqlite3.connect(LOG_DB) as cx:
            init_product_content_table(cx)
            row = cx.execute(
                "SELECT content_json, sources_json, generated_at FROM generated_product_content "
                "WHERE product_slug=? AND content_type=?", (slug, ctype)).fetchone()
        if not row:
            return None
        return {"content": json.loads(row[0]),
                "sources": json.loads(row[1]) if row[1] else [],
                "generated_at": row[2], "cached": True}
    except Exception as e:
        print(f"[product_content] cache_get failed {slug}/{ctype}: {e}", flush=True)
        return None


def _cache_put(slug, ctype, content, sources):
    try:
        with sqlite3.connect(LOG_DB) as cx:
            init_product_content_table(cx)
            cx.execute(
                "INSERT INTO generated_product_content "
                "(product_slug, content_type, content_json, sources_json, model_name, generated_at) "
                "VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(product_slug, content_type) DO UPDATE SET "
                "content_json=excluded.content_json, sources_json=excluded.sources_json, "
                "model_name=excluded.model_name, generated_at=excluded.generated_at",
                (slug, ctype, json.dumps(content), json.dumps(sources), _MODEL, _now()))
    except Exception as e:
        print(f"[product_content] cache_put failed {slug}/{ctype}: {e}", flush=True)


# ── Retrieval ─────────────────────────────────────────────────────────────────
def _page_text(product):
    """Concatenated remedymatch.com page copy for a product, by EXACT title filter.
    Returns {text, url, price, n_chunks} or None."""
    title = product.get("pinecone_title") or product.get("name")
    if not title:
        return None
    idx, _cl, embed = _clients()
    try:
        vec = embed(title)
        res = idx.query(vector=vec, top_k=30, namespace=SPECIFIC_NS,
                        filter={"title": {"$eq": title}}, include_metadata=True)
        matches = res.matches if hasattr(res, "matches") else res.get("matches", [])
    except Exception as e:
        print(f"[product_content] page_text query failed {title}: {e}", flush=True)
        return None
    if not matches:
        return None
    matches = sorted(matches, key=lambda m: (m.metadata or {}).get("chunk_index", 0))
    text = "\n".join((m.metadata or {}).get("text", "") for m in matches).strip()
    md0 = matches[0].metadata or {}
    return {"text": text, "url": md0.get("url", ""), "price": md0.get("price", ""),
            "n_chunks": len(matches)}


def _page_text_from_product(product):
    """Fallback page copy synthesized from products.json (description + ingredient
    list) for products whose panel is missing from Pinecone (truncated scrape) or
    when embeddings/Pinecone are unavailable. Returns {text, url, price, n_chunks}
    or None when there is no manual data to ground on."""
    desc = (product.get("description") or "").strip()
    lines = []
    for it in (product.get("ingredients") or []):
        if isinstance(it, dict):
            nm = (it.get("name") or "").strip()
            dose = (it.get("dose") or "").strip()
            if nm:
                lines.append(f"{nm} {dose}".strip())
        elif isinstance(it, str) and it.strip():
            lines.append(it.strip())
    if not desc and not lines:
        return None
    parts = []
    if desc:
        parts.append(desc)
    if lines:
        parts.append("Ingredients:\n" + "\n".join(lines))
    return {"text": "\n\n".join(parts).strip(), "url": product.get("url", ""),
            "price": "", "n_chunks": 0}


def _norm_ing(s):
    """Lowercase, strip punctuation to spaces, collapse whitespace — for matching
    a research source's ingredient against the product's own ingredient names."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (s or "").lower())).strip()


def _ingredient_in_product(source_ing, product_norms):
    """True if a research source's ingredient belongs to THIS product. Matched by
    bidirectional substring on normalized names so the legacy 'Magnesium L-Threonate'
    study is dropped for an ATA-Mg product, while short/long form variants of a real
    ingredient (e.g. 'Pyridoxal-5-Phosphate' vs 'Vitamin B6 (Pyridoxal-5-Phosphate)')
    still match. An empty/unknown source ingredient cannot be verified, so it drops."""
    s = _norm_ing(source_ing)
    if not s:
        return False
    return any(s in p or p in s for p in product_norms if p)


def _research_sources(name, k=8, ingredients=None):
    """Per-ingredient research studies from the `ingredients` namespace. Each carries
    a real study URL — the only URLs the learn-more copy is allowed to cite.

    When `ingredients` (the product's own ingredient list) is supplied, sources are
    filtered to those whose ingredient belongs to this product, so a semantically
    similar but wrong-formula study (e.g. Magnesium L-Threonate for an ATA-Mg product)
    never reaches the learn_more prompt. With no ingredient list, behavior is unchanged."""
    idx, _cl, embed = _clients()
    try:
        vec = embed(f"{name} mechanism clinical research evidence")
        res = idx.query(vector=vec, top_k=k, namespace=RESEARCH_NS, include_metadata=True)
        matches = res.matches if hasattr(res, "matches") else res.get("matches", [])
    except Exception as e:
        print(f"[product_content] research query failed {name}: {e}", flush=True)
        return []
    product_norms = []
    for it in (ingredients or []):
        nm = it.get("name") if isinstance(it, dict) else it
        n = _norm_ing(nm)
        if n:
            product_norms.append(n)
    out, seen = [], set()
    for m in matches:
        md = m.metadata or {}
        url = (md.get("url") or "").strip()
        if not url or url in seen:
            continue
        if product_norms and not _ingredient_in_product(md.get("ingredient", ""), product_norms):
            continue
        seen.add(url)
        out.append({
            "ingredient": md.get("ingredient", ""),
            "study_title": md.get("study_title", ""),
            "publication": md.get("publication", ""),
            "year": md.get("year", ""),
            "url": url,
            "text": (md.get("text", "") or "")[:600],
        })
    return out


# ── Voice ─────────────────────────────────────────────────────────────────────
# Mandatory structure-function guardrail. Appended to every generator's system
# prompt via _VOICE so card/how-it-works/learn-more cannot emit disease claims,
# even when the supplied page copy or research sources mention conditions.
_COMPLIANCE = (
    "COMPLIANCE (mandatory; this is dietary-supplement copy): use STRUCTURE-FUNCTION "
    "language only. Describe how the formula supports the normal structure and function "
    "of the body's tissues (for example 'supports the vitreous body', 'supports "
    "connective-tissue resilience', 'supports the body's own antioxidant defenses'). "
    "NEVER state or imply that the product diagnoses, treats, cures, prevents, reverses, "
    "addresses, stops, fixes, slows, or reduces any disease, condition, or symptom. Do NOT "
    "name a disease, condition, or symptom as something the product acts on (for example "
    "floaters, vitreous or retinal detachment, macular degeneration, glaucoma, cataract). "
    "Do NOT describe the product as halting a disease process (for example 'keeps the "
    "vitreous from shrinking or detaching'). You may describe the tissue or structure being "
    "supported, but not a condition being fixed. Prefer the verbs supports, nourishes, helps "
    "maintain; avoid treats, prevents, cures, reverses, addresses, stops, slows, eliminates. "
    "Keep it clinically honest and never overclaim."
)

# Ingredient-fidelity guardrail. Appended to every generator's system prompt via
# _VOICE. Stops cross-product bleed: two formulas can share a name (the ATA-Mg
# "Neuro Magnesium" vs the legacy Magnesium-L-Threonate "Focus Neuro-Magnesium"),
# and the model would otherwise carry the wrong form's mechanism over from research.
_FIDELITY = (
    "INGREDIENT FIDELITY (mandatory): describe ONLY the ingredients shown for THIS product "
    "in the supplied page copy / ingredient list, in exactly the forms listed. Do NOT mention, "
    "compare to, substitute, or describe any ingredient or alternative form that is not listed "
    "for this product. In particular, if the formula lists one form of a mineral, do NOT mention "
    "any other form of it (for example, if it lists Magnesium N-Acetyl-Taurate or ATA Mg, do NOT "
    "mention Magnesium L-Threonate, Magtein, or any other magnesium form). Never carry over "
    "content, mechanism, or studies from a different product that happens to share a similar name."
)

_VOICE = (
    "Write in Dr. Glen Swartwout's voice: calm, consultative, clinically grounded with a "
    "light spiritual register. Lead with validation of the reader's lived experience. "
    "Do NOT use em dashes (use commas, colons, or periods). Do NOT use ALL CAPS for emphasis "
    "(acronyms are fine). Never prefix anything with the word 'Hook:'. Never invent ingredients, "
    "prices, claims, or URLs that are not present in the supplied material.\n\n"
    + _COMPLIANCE + "\n\n" + _FIDELITY
)


# ── Generation: product card (description + ingredients + benefits) ───────────
_CARD_SYSTEM = (
    "You produce concise product-page content for a Functional Formulation. " + _VOICE + "\n\n"
    "From the SUPPLIED PAGE COPY only, return JSON: "
    "{\"description\": \"1-2 plain sentences on what this is and who it helps\", "
    "\"ingredients\": [\"verbatim ingredient lines from the page copy (with amounts where shown)\"], "
    "\"benefits\": [\"4-6 short functions/benefits bullets, each a phrase not a paragraph\"]}. "
    "Extract ingredients ONLY from the page copy; if the copy does not list ingredients, return an "
    "empty ingredients array (do not guess). Output ONLY the JSON, no prose, no code fences."
)


# ── Post-generation compliance gate ───────────────────────────────────────────
# The prompt guardrail (_COMPLIANCE in _VOICE) reduces but does not guarantee
# compliant output, because the research sources are condition-heavy and the
# model can drift on long-form copy. This deterministic gate is the backstop:
# scan generated text for disease/condition names + treatment-claim verbs, retry
# once with explicit feedback, and if it still violates, the caller degrades that
# section (empty rather than a non-compliant claim on a live page).
_DENY_RE = re.compile(
    r"\b(floaters?|detachment|macular degeneration|glaucoma|cataracts?|"
    r"treatments?|treats?|prevents?|cures?|reverses?|addresses)\b", re.I)


def _deny_hits(text: str) -> list:
    return sorted({m.group(0).lower() for m in _DENY_RE.finditer(text or "")})


def _gen_compliant(cl, system, user, max_tokens, check):
    """Generate text; if check(raw) contains denied terms, retry once with
    feedback. Returns (raw_text, ok); ok is False if it still violates."""
    def _run(u):
        msg = cl.messages.create(model=_MODEL, max_tokens=max_tokens, system=system,
                                 messages=[{"role": "user", "content": u}])
        return (msg.content[0].text if msg.content else "").strip()

    raw = _run(user)
    if not _deny_hits(check(raw)):
        return raw, True
    hits = _deny_hits(check(raw))
    fb = (user + "\n\nYOUR PREVIOUS DRAFT VIOLATED COMPLIANCE by using these forbidden words: "
          + ", ".join(hits) + ". Rewrite the ENTIRE response in pure structure-function language with "
          "ZERO disease/condition/symptom names (no 'floaters', no 'detachment', no disease names) and "
          "ZERO treatment-claim verbs (no treats/prevents/cures/reverses/addresses/treatment). Describe "
          "only how the formula supports the normal structure and function of the body's tissues.")
    raw2 = _run(fb)
    return raw2, not _deny_hits(check(raw2))


def _generate_card(product, page):
    idx, cl, embed = _clients()
    name = product.get("name", "")
    page_text = (page or {}).get("text", "")
    if not page_text:
        return {"description": "", "ingredients": [], "benefits": []}
    user = f"PRODUCT: {name}\n\nPAGE COPY:\n{page_text[:14000]}"
    try:
        raw, ok = _gen_compliant(cl, _CARD_SYSTEM, user, 1200, lambda r: r)
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        # If still non-compliant after retry, drop the generated prose (keep the
        # factual, verbatim ingredient list); benefits/description can be pinned.
        desc = _dd((data.get("description") or "").strip())
        bens = [_dd(s.strip()) for s in (data.get("benefits") or []) if s.strip()]
        if not ok:
            desc, bens = "", []
            print(f"[product_content] card still non-compliant after retry, dropped prose: {name}", flush=True)
        return {"description": desc,
                "ingredients": [s.strip() for s in (data.get("ingredients") or []) if s.strip()],
                "benefits": bens}
    except Exception as e:
        print(f"[product_content] card gen failed {name}: {e}", flush=True)
        return {"description": "", "ingredients": [], "benefits": []}


# ── Generation: learn-more research expansion ─────────────────────────────────
_LEARN_SYSTEM = (
    "You write a 'Learn more' research deep-dive for a Functional Formulation. " + _VOICE + "\n\n"
    "Use the PAGE COPY for what the formula is and does, and the RESEARCH SOURCES for mechanism "
    "and evidence. Write 450-750 words of warm, readable markdown: open by validating the reader's "
    "concern, explain how the key ingredients work and what the research shows, and close with an "
    "invitation to begin. You MAY reference a study inline by its publication and year. End with a "
    "'## Sources' section listing only studies from the supplied RESEARCH SOURCES, each as a markdown "
    "link using its exact url. Never cite a url that is not in the supplied sources. Output markdown only."
)


def _generate_learn_more(product, page, sources):
    idx, cl, embed = _clients()
    name = product.get("name", "")
    page_text = (page or {}).get("text", "")
    if not page_text and not sources:
        # No grounding material at all: never send an empty prompt (the model would
        # return a refusal, which used to get cached and served on the public page).
        return {"markdown": ""}
    src_block = "\n".join(
        f"- [{s['ingredient']}] {s['study_title']} ({s['publication']} {s['year']}) {s['url']}\n  {s['text']}"
        for s in sources) or "(no research sources retrieved)"
    user = (f"PRODUCT: {name}\n\nPAGE COPY:\n{page_text[:9000]}\n\n"
            f"RESEARCH SOURCES (cite only these urls):\n{src_block[:8000]}")
    try:
        raw, ok = _gen_compliant(cl, _LEARN_SYSTEM, user, 2400, lambda r: r)
        if not ok:
            print(f"[product_content] learn_more still non-compliant after retry, degraded to empty: {name}", flush=True)
            return {"markdown": ""}
        markdown = _dd(raw)
    except Exception as e:
        print(f"[product_content] learn_more gen failed {name}: {e}", flush=True)
        markdown = ""
    return {"markdown": markdown}


# ── Generation: "How it works" (short mechanism explainer) ────────────────────
_HOW_SYSTEM = (
    "You write a short 'How it works' explainer for a Functional Formulation. " + _VOICE + "\n\n"
    "From the SUPPLIED PAGE COPY, explain in plain language HOW the formula works in the body, the "
    "mechanism, what the key ingredients do and why they were chosen to work together. 90-150 words, "
    "2-3 short paragraphs, warm and clear. This is the mechanism, not a citation list and not a "
    "benefits list. Output plain text only (no markdown headers, no citations)."
)


def _generate_how_it_works(product, page):
    idx, cl, embed = _clients()
    name = product.get("name", "")
    page_text = (page or {}).get("text", "")
    if not page_text:
        return {"text": ""}
    user = f"PRODUCT: {name}\n\nPAGE COPY:\n{page_text[:12000]}"
    try:
        raw, ok = _gen_compliant(cl, _HOW_SYSTEM, user, 600, lambda r: r)
        if not ok:
            print(f"[product_content] how_it_works still non-compliant after retry, degraded to empty: {name}", flush=True)
            return {"text": ""}
        return {"text": _dd(raw)}
    except Exception as e:
        print(f"[product_content] how_it_works gen failed {name}: {e}", flush=True)
        return {"text": ""}


# ── Public API ────────────────────────────────────────────────────────────────
def get_or_generate(product, content_type, force=False):
    """Return cached content for (product, content_type), generating + caching on miss.
    content_type in {'card', 'how_it_works', 'learn_more'}. `product` is _get_product(slug)."""
    slug = product.get("slug")
    if not force:
        hit = _cache_get(slug, content_type)
        if hit:
            return hit

    page = _page_text(product) or _page_text_from_product(product)
    if content_type == "card":
        content = _generate_card(product, page)
        sources = []
        if page and page.get("url"):
            sources = [{"label": "Product page", "url": page["url"]}]
    elif content_type == "how_it_works":
        content = _generate_how_it_works(product, page)
        sources = []
    elif content_type == "learn_more":
        research = _research_sources(product.get("name", ""), ingredients=product.get("ingredients"))
        content = _generate_learn_more(product, page, research)
        sources = [{"label": f"{s['study_title']} ({s['publication']} {s['year']})".strip(),
                    "url": s["url"]} for s in research]
        if page and page.get("url"):
            sources.append({"label": "Product page", "url": page["url"]})
    else:
        return None

    _cache_put(slug, content_type, content, sources)
    return {"content": content, "sources": sources, "generated_at": _now(), "cached": False}
