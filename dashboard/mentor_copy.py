"""AI copy generation and page assembly for mentor pages.

Two build paths, both in build_page():
  1. Seed path  - if dashboard/mentor_seed has authoritative content for the slug,
     write it verbatim. Ships vetted pages about real people with no generation.
  2. Grounded path - otherwise, retrieve context from the Pinecone `mentors`
     namespace (via an injected retriever) and write each section strictly from
     that source. No retriever or no context means no invented biography: the
     page stays unbuilt rather than hallucinated.

Voice follows Glen's copy rules: no em dashes, no ALL CAPS, warm and specific.
"""
import json

from dashboard import mentor_seed as _seed
from dashboard import mentor_pages as _mp

NARRATIVE_SECTIONS = ("life_and_work", "key_contribution", "lineage", "why_it_matters")

_MODEL = "claude-haiku-4-5-20251001"

_GROUND = (
    "Write ONLY from the provided source material about this person. Do not add "
    "biographical facts, dates, titles, publications, or claims that are not in the "
    "source. If the source does not support a section, write a short, honest, general "
    "paragraph without inventing specifics. No disease claims. No em dashes; use commas."
)

SECTION_BRIEFS = {
    "life_and_work": (
        "Write one or two warm paragraphs on who this person was: their field, era, "
        "institutional standing, and the shape of their career."
    ),
    "key_contribution": (
        "Write one or two paragraphs on their single most important contribution: what "
        "they discovered or built, and how, in plain language grounded in the source."
    ),
    "lineage": (
        "Write one paragraph placing this person in an intellectual lineage: who they "
        "drew on and who carried their work forward, only as supported by the source."
    ),
    "why_it_matters": (
        "Write one short paragraph on why this person matters to a modern integrative "
        "or wellness lineage, honest about what is established versus still emerging."
    ),
}


def build_section_prompt(section, mentor):
    """Return (system, user) for the grounded LLM path. mentor carries name + source text."""
    name = mentor.get("name", "")
    source = (mentor.get("source") or "").strip()
    brief = SECTION_BRIEFS.get(section, "")
    system = (
        "You are writing an educational mentor-and-lineage page for Dr. Glen Swartwout's "
        "site, honoring a teacher or scientist whose work informs his clinical approach. "
        "Voice: warm, clinically grounded, specific, no fluff. " + _GROUND
    )
    user = (
        f"Person: {name}\n\n"
        f"Source material:\n{source or '(none provided)'}\n\n"
        f"Task: {brief}\n\n"
        "Return only the copy itself, with no headings, labels, or preamble."
    )
    return system, user


def _apply_seed(cx, slug, seed, *, strip=None):
    strip = strip or (lambda s: s)
    for section, text in (seed.get("content") or {}).items():
        if text:
            _mp.upsert_section(cx, slug, section, strip(text).strip(), model="seed")
    _mp.set_name(cx, slug, seed.get("name") or slug)
    _mp.set_field(cx, slug, seed.get("field") or "")
    _mp.set_lifespan(cx, slug, seed.get("lifespan") or "")
    _mp.set_vital_status(cx, slug, seed.get("vital_status") or "")
    _mp.set_lineage(cx, slug, seed.get("lineage") or [])
    _mp.set_sources(cx, slug, seed.get("sources") or [])
    _mp.set_seo(cx, slug, seed.get("seo") or {})
    _mp.set_state(cx, slug, "draft")
    return {"slug": slug, "state": "draft", "source": "seed"}


def build_page(cx, slug, name="", *, client=None, retriever=None, strip=None):
    """Build (or rebuild) a mentor page's content. Never raises.

    retriever: optional callable(query:str) -> str, returning grounding context
    from the Pinecone `mentors` namespace. Required for the generated path.
    Returns a small status dict.
    """
    slug = (slug or "").strip().lower()
    if not slug:
        return {"ok": False, "error": "slug required"}
    strip = strip or (lambda s: s)

    seed = _seed.get_seed(slug)
    if seed:
        try:
            return _apply_seed(cx, slug, seed, strip=strip)
        except Exception as exc:  # noqa: BLE001
            print(f"[mentor-copy] seed apply failed for {slug}: {exc}", flush=True)
            return {"ok": False, "error": "seed_failed"}

    name = name or slug.replace("-", " ").title()
    _mp.set_name(cx, slug, name)

    source = ""
    if retriever is not None:
        try:
            source = (retriever(f"{name} biography contribution lineage") or "").strip()
        except Exception as exc:  # noqa: BLE001
            print(f"[mentor-copy] retriever failed for {slug}: {exc}", flush=True)
            source = ""
    if not source or client is None:
        # No grounding or no model: leave unbuilt rather than invent a real person's life.
        _mp.set_state(cx, slug, "pending")
        return {"ok": True, "state": "pending", "reason": "no_source_or_client"}

    mentor = {"name": name, "source": source}
    built = 0
    for section in NARRATIVE_SECTIONS:
        try:
            system, user = build_section_prompt(section, mentor)
            msg = client.messages.create(model=_MODEL, max_tokens=700, system=system,
                                          messages=[{"role": "user", "content": user}])
            text = "".join(getattr(b, "text", "") for b in msg.content
                           if getattr(b, "type", "") == "text")
            text = strip(text).strip()
            if text:
                _mp.upsert_section(cx, slug, section, text, model=_MODEL)
                built += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[mentor-copy] section {section} failed for {slug}: {exc}", flush=True)
    _mp.set_seo(cx, slug, {"title": f"{name}, mentor and lineage",
                           "meta_description": f"The life, work, and lineage of {name}."})
    _mp.set_state(cx, slug, "draft" if built else "pending")
    return {"ok": True, "state": "draft" if built else "pending", "sections_built": built}
