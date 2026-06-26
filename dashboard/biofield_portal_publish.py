"""Publish an authored Biofield Intake report to the illtowell.com client portal.

Pure / none-raising builder + an injectable prod POST. PHI stays local; only the
finished portal payload crosses to prod via the existing /admin/portal/upsert.
"""
import re
import secrets
import requests

from dashboard.practitioner_portal import name_to_slug
from dashboard import wholesale_pricing as _pricing
from dashboard.biofield_authoring import authored_report
from dashboard.biofield_narrative import get_narrative

# Protocol wordings that differ from the catalog. Keyed by alphanumeric-only,
# lowercased remedy text so "Focus, Neuromagnesium" and "Focus Neuro-Magnesium"
# collapse to the same key.
ALIAS_SLUGS = {
    "focusneuromagnesium": "neuro-magnesium",
    "communityspiritformulainterrainrestore": "terrain-restore",
}


def _norm_key(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def load_catalog():
    """The slug-keyed products map (data/products.json 'products')."""
    return _pricing._load_catalog()


def resolve_remedy_slug(name, catalog):
    """Resolve a protocol remedy name to a catalog slug: alias override first,
    then the in-repo fuzzy resolver. None when genuinely unresolvable."""
    if not (name or "").strip():
        return None
    alias = ALIAS_SLUGS.get(_norm_key(name))
    if alias:
        return alias
    return name_to_slug(name, catalog)


def _dosing(layer):
    parts = [(layer.get("dosage") or "").strip(),
             (layer.get("frequency") or "").strip(),
             (layer.get("timing") or "").strip()]
    return " ".join(p for p in parts if p)


def _cue_candidates(layer):
    """Ordered phrases to locate this layer in the narrative blob."""
    rem = (layer.get("remedy") or "").strip()
    out = []
    if rem:
        out.append(rem)
        first = rem.split(",")[0].strip()      # "Focus, Neuromagnesium" -> "Focus"
        if first and first != rem:
            out.append(first)
    head = (layer.get("head") or "").strip()
    if head:
        out.append(head)
    return out


def segment_narrative(narrative, layers):
    """Split the single narrative blob into one segment per layer, by locating
    each layer's cue (remedy, else its first word, else head) in increasing
    order. Returns a list aligned to ``layers``; ``[]`` when it cannot align."""
    text = narrative or ""
    if not text or not layers:
        return []
    low = text.lower()
    positions = []
    cursor = 0
    for layer in layers:
        found = -1
        for cue in _cue_candidates(layer):
            idx = low.find(cue.lower(), cursor)
            if idx != -1:
                found = idx
                break
        if found == -1:
            return []                          # a layer has no cue -> fall back
        positions.append(found)
        cursor = found + 1
    # positions are strictly increasing by construction (each search starts past
    # the previous hit). Slice between consecutive cue starts.
    segs = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        segs.append(text[start:end].strip())
    return segs


def build_portal_content(cx, test_id, *, special_price_cents, catalog=None,
                         audio_url=None, report_pdf_url=None):
    """Map an authored intake report to the portal content payload.

    Returns {email, name, scan_date, scan_id, content, unresolved}. Never raises
    on missing narrative (falls back to greeting=full narrative, blank meanings)."""
    cat = catalog if catalog is not None else load_catalog()
    rep = authored_report(cx, test_id)
    raw_layers = rep.get("layers") or []
    client = rep.get("client") or {}
    name = (client.get("name") or "").strip()
    first = name.split()[0] if name else ""

    narrative = get_narrative(cx, test_id) or ""
    segs = segment_narrative(narrative, raw_layers)
    if segs:
        greeting = f"Aloha {first}," if first else "Aloha,"
        meanings = segs
    else:
        greeting = narrative or (f"Aloha {first}," if first else "Aloha,")
        meanings = [""] * len(raw_layers)

    layers, reorder, seen, unresolved = [], [], set(), []
    for i, L in enumerate(raw_layers):
        remedy = (L.get("remedy") or "").strip()
        layers.append({
            "n": L.get("layer"),
            "title": (L.get("head") or "").strip(),
            "meaning": meanings[i] if i < len(meanings) else "",
            "remedy": remedy,
            "dosing": _dosing(L),
        })
        if not remedy:
            continue
        slug = resolve_remedy_slug(remedy, cat)
        if slug is None:
            if remedy not in unresolved:
                unresolved.append(remedy)
            continue
        if slug in seen:
            continue
        seen.add(slug)
        reorder.append({"slug": slug, "qty": 1, "price_cents": int(special_price_cents)})

    content = {
        "greeting": greeting,
        "video": {"url": "", "label": "Watch your message from Dr. Glen"},
        "layers": layers,
        "reorder_items": reorder,
        "pricing_note": "",
        "findings": [],
        "biofield_status": "confirmed",
    }
    if audio_url:
        content["audio"] = {"url": audio_url, "label": "Listen to your walkthrough"}
    if report_pdf_url:
        content["report_pdf"] = {"url": report_pdf_url}
    return {
        "email": (client.get("email") or "").strip().lower(),
        "name": name,
        "scan_date": rep.get("date") or "",
        "scan_id": "",
        "content": content,
        "unresolved": unresolved,
    }


def publish_to_portal(payload, *, base_url, console_key, send=False, http_post=None):
    """POST the portal payload to the prod /admin/portal/upsert.

    send=True asks the prod upsert to auto-email the portal link, but the upsert
    only emails when a NEW token is minted (first publish); re-publishing an
    existing portal returns token=None and never re-sends.
    Returns the parsed JSON (contains url/token). Raises RuntimeError on non-2xx."""
    post = http_post or requests.post
    url = f"{base_url.rstrip('/')}/admin/portal/upsert"
    body = {**payload, "send": bool(send)}
    r = post(url, json=body, headers={"X-Console-Key": console_key}, timeout=30)
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"portal upsert failed {r.status_code}: {r.text[:300]}")
    return r.json()


def _asset_name(ext):
    """Return an opaque portal asset filename: biofield-<16 hex chars>.<ext>."""
    return f"biofield-{secrets.token_hex(8)}.{ext}"


def upload_asset(data_bytes, filename, *, base_url, console_key, http_put=None):
    """PUT raw bytes to the prod /portal-asset/upload; return the served url.
    Raises RuntimeError on non-2xx. http_put injectable (defaults requests.put)."""
    put = http_put or requests.put
    url = f"{base_url.rstrip('/')}/portal-asset/upload?filename={filename}"
    r = put(url, data=data_bytes, headers={"X-Console-Key": console_key}, timeout=60)
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"asset upload failed {r.status_code}: {r.text[:300]}")
    return r.json()["url"]
