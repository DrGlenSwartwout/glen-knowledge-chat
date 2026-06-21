"""Canonical per-product remedy meaning store. One curated 1-2 sentence meaning
per catalog slug, applied to new Biofield reveals at ingest. All functions are
wrapped or pure; none raise into callers (propose_meaning returns "" on failure)."""
import json
from datetime import datetime, timezone

_MODEL = "claude-haiku-4-5-20251001"


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS biofield_remedy_meanings "
        "(slug TEXT PRIMARY KEY, meaning TEXT NOT NULL DEFAULT '', "
        "source TEXT NOT NULL DEFAULT '', updated_by TEXT, updated_at TEXT)")
    cx.commit()


def upsert(cx, slug, meaning, by, source):
    slug = (slug or "").strip()
    if not slug:
        return
    cx.execute(
        "INSERT INTO biofield_remedy_meanings (slug, meaning, source, updated_by, updated_at) "
        "VALUES (?,?,?,?,?) ON CONFLICT(slug) DO UPDATE SET "
        "meaning=excluded.meaning, source=excluded.source, updated_by=excluded.updated_by, updated_at=excluded.updated_at",
        (slug, (meaning or "").strip(), (source or "").strip(), (by or "").strip(), _now()))
    cx.commit()


def get_map(cx):
    """{slug: meaning} for non-empty meanings."""
    rows = cx.execute("SELECT slug, meaning FROM biofield_remedy_meanings").fetchall()
    return {r[0]: r[1] for r in rows if (r[1] or "").strip()}


def get_all(cx):
    rows = cx.execute(
        "SELECT slug, meaning, source, updated_at FROM biofield_remedy_meanings ORDER BY slug").fetchall()
    return [{"slug": r[0], "meaning": r[1], "source": r[2], "updated_at": r[3]} for r in rows]


def delete(cx, slug):
    cx.execute("DELETE FROM biofield_remedy_meanings WHERE slug=?", ((slug or "").strip(),))
    cx.commit()


def propose_meaning(product, client):
    """1-2 sentence meaning that LEADS with the remedy's major functions, warm lay
    voice, no disease claims. Never raises -> "" on any failure or no client."""
    if client is None:
        return ""
    try:
        name = product.get("name") or product.get("slug") or ""
        ingredients = product.get("ingredients") or []
        if isinstance(ingredients, list):
            ing = ", ".join(
                str(i.get("name") if isinstance(i, dict) else i) for i in ingredients[:20])
        else:
            ing = str(ingredients)
        benefits = product.get("benefits") or []
        ben = "; ".join(str(b) for b in benefits) if isinstance(benefits, list) else str(benefits)
        desc = product.get("description") or ""
        user = (
            f"Remedy: {name}\nKey ingredients: {ing}\nBenefits: {ben}\nDescription: {desc}\n\n"
            "Write a 1 to 2 sentence remedy 'meaning' that LEADS with this remedy's major functions, "
            "in warm, plain, lay language. No disease claims, no diagnosis, no hype. "
            "Return only the sentence or two, with no preamble.")
        msg = client.messages.create(
            model=_MODEL, max_tokens=160, messages=[{"role": "user", "content": user}])
        parts = getattr(msg, "content", None) or []
        return "".join(getattr(p, "text", "") for p in parts).strip()
    except Exception:
        return ""
