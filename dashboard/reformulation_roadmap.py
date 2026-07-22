"""Reformulation roadmap from the free-product-review data.

Two layers:
  - frequency(): deterministic — which submitted products/brands recur most (pure SQL).
  - generate(): an LLM pass over the analyzer's reviews that clusters submissions into
    product categories and ranks a "what to formulate / reformulate next" roadmap:
    high demand + products that fall short = an opportunity for Glen's own line.

Roadmaps are cached in `reformulation_roadmap` (latest = highest id) so the console
reads instantly and regenerates on demand. Reads `supplement_reviews`."""
import datetime
import json
import re

_MODEL = "claude-haiku-4-5-20251001"


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS reformulation_roadmap (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            generated_at TEXT,
            n_reviews    INTEGER,
            roadmap_json TEXT
        )
    """)
    cx.commit()


def frequency(cx):
    """Deterministic demand signal: submission count per product+brand, most first.
    Works with any review status (a submission is a signal even before it's reviewed)."""
    try:
        rows = cx.execute(
            "SELECT product_name, product_brand, COUNT(*) AS n FROM supplement_reviews "
            "GROUP BY product_key ORDER BY n DESC, product_name LIMIT 200").fetchall()
    except Exception:
        return []
    return [{"product_name": r[0], "product_brand": r[1], "count": r[2]} for r in rows]


def corpus(cx, limit=300):
    """Reviews carrying the analyzer's assessment (ai_draft or confirmed) — the raw
    material for the roadmap. Requested-only rows have no critique yet, so skip them."""
    try:
        rows = cx.execute(
            "SELECT product_name, product_brand, review_text FROM supplement_reviews "
            "WHERE status IN ('ai_draft','confirmed') AND review_text IS NOT NULL "
            "AND TRIM(review_text) != '' ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    except Exception:
        return []
    return [{"product_name": r[0], "product_brand": r[1], "review_text": r[2]} for r in rows]


def _build_prompt(items):
    system = (
        "You are a product-development strategist for Dr. Glen Swartwout's supplement line "
        "(Functional Formulations). You are given supplements that prospective customers "
        "currently take, each with Glen's clinical review noting what the product gets right "
        "and where it falls short. Cluster them into product CATEGORIES (e.g. magnesium, "
        "fish oil, multivitamin) and produce a ranked REFORMULATION / NEW-PRODUCT ROADMAP: "
        "where demand is high AND the products people take fall short, that is an opportunity "
        "for Glen to formulate or reformulate. Return ONLY a JSON object: "
        '{"roadmap":[{"category":str,"submission_count":int,"common_weaknesses":[str],'
        '"reformulation_opportunity":str,"priority":int}]} where priority is 1-5 (5=highest). '
        "Sort by priority desc, then submission_count desc. No prose outside the JSON.")
    lines = []
    for it in items:
        nm = (it.get("product_name") or "").strip()
        br = (it.get("product_brand") or "").strip()
        rv = re.sub(r"\s+", " ", (it.get("review_text") or "")).strip()[:600]
        lines.append(f"- {nm}" + (f" ({br})" if br else "") + f": {rv}")
    user = "Submissions:\n" + "\n".join(lines) + "\n\nReturn only the JSON object."
    return system, user


def generate(cx, client, model=_MODEL):
    """Run the LLM synthesis over the review corpus, cache and return the roadmap.
    Empty corpus -> empty roadmap (no LLM call). None-raising."""
    init_table(cx)
    items = corpus(cx)
    if not items:
        return {"roadmap": [], "n_reviews": 0, "generated_at": _now()}
    system, user = _build_prompt(items)
    roadmap = []
    try:
        msg = client.messages.create(model=model, max_tokens=1500, system=system,
                                     messages=[{"role": "user", "content": user}])
        text = "".join(getattr(b, "text", "") for b in msg.content
                       if getattr(b, "type", "") == "text").strip()
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end >= 0:
            roadmap = json.loads(text[start:end + 1]).get("roadmap", []) or []
    except Exception:
        roadmap = []  # a bad LLM response yields an empty roadmap, never a 500
    gen_at = _now()
    cx.execute("INSERT INTO reformulation_roadmap (generated_at, n_reviews, roadmap_json) VALUES (?,?,?)",
               (gen_at, len(items), json.dumps(roadmap)))
    cx.commit()
    return {"roadmap": roadmap, "n_reviews": len(items), "generated_at": gen_at}


def latest(cx):
    """The most recently generated roadmap, or an empty shell if none yet."""
    init_table(cx)
    try:
        row = cx.execute(
            "SELECT generated_at, n_reviews, roadmap_json FROM reformulation_roadmap "
            "ORDER BY id DESC LIMIT 1").fetchone()
    except Exception:
        row = None
    if not row:
        return {"roadmap": [], "n_reviews": 0, "generated_at": None}
    try:
        roadmap = json.loads(row[2] or "[]")
    except Exception:
        roadmap = []
    return {"roadmap": roadmap, "n_reviews": row[1], "generated_at": row[0]}
