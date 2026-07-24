"""Extract structured clinical proposals + a client narrative from an uploaded
document, and write them as a DRAFT.

This module writes NOTHING to a live clinical store. That is the whole point of
the review gate: the console approval route is the only thing that writes
person_attributes / client_facts.

The fabrication guard is structural. Every extracted item must carry a
`source_quote` that actually appears in the document; anything else is dropped
before Glen ever sees it. Prompted output alone is not trusted.
"""
import json
import os
import re

_MODEL = os.environ.get("DOC_EXTRACT_MODEL", "claude-opus-4-8")

_PROMPT = (
    "You are reading a patient's medical record. Extract ONLY what the document "
    "actually states. Never infer, never generalize, never add a diagnosis that "
    "is not written down.\n\n"
    "Return STRICT JSON with these keys:\n"
    '  "document_text": a VERBATIM transcription of all text in the document, '
    "exactly as written. This is what every source_quote is checked against, so "
    "a quote that is not present here is discarded.\n"
    '  "narrative_md": a warm, plain-language summary for the patient '
    "(2-4 short paragraphs, markdown, no headings). Explain what the document "
    "says in everyday words. Do not give advice or recommend treatment.\n"
    '  "attributes": [{"field": one of "tags"|"conditions"|"terrain_concerns"'
    '|"body_systems"|"challenges"|"goals", "value": str, "source_quote": str}]\n'
    '  "facts": [{"fact_key": str, "value": true|false, "source_quote": str}]\n'
    '  "unstructured": [{"label": str, "value": str, "source_quote": str}] '
    "for lab results with numeric values and medications.\n\n"
    "EVERY item MUST include a source_quote copied VERBATIM from the document. "
    "An item without a verbatim quote will be discarded. No markdown fences, no "
    "prose outside the JSON."
)


def _norm_text(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def verify_quotes(items, source_text):
    """Split `items` into (kept, dropped) by whether each item's source_quote
    actually occurs in `source_text`. Whitespace- and case-insensitive, so the
    model is not punished for reflowing a line."""
    hay = _norm_text(source_text)
    kept, dropped = [], []
    for it in items or []:
        q = _norm_text((it or {}).get("source_quote"))
        (kept if (q and q in hay) else dropped).append(it)
    return kept, dropped


def _default_call_model(blob, content_type):
    """The real Anthropic call. Lazy-imported so tests that inject call_model
    never pull the SDK."""
    import base64
    import anthropic
    cli = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    b64 = base64.standard_b64encode(blob).decode("ascii")
    if (content_type or "").lower() == "application/pdf":
        doc = {"type": "document",
               "source": {"type": "base64", "media_type": "application/pdf",
                          "data": b64}}
    else:
        doc = {"type": "image",
               "source": {"type": "base64", "media_type": content_type,
                          "data": b64}}
    resp = cli.messages.create(
        model=_MODEL, max_tokens=4000,
        messages=[{"role": "user", "content": [doc, {"type": "text",
                                                     "text": _PROMPT}]}])
    text = resp.content[0].text.strip()
    if text.startswith("```"):                     # tolerate accidental fences
        text = text.split("```", 2)[1]
        if text.startswith("json\n"):
            text = text[5:]
    return json.loads(text)


def _source_text_for(payload):
    """The haystack the guard checks quotes against: the model's VERBATIM
    transcription of the document.

    Deliberately NOT the narrative. Checking the model's quotes against the
    model's own summary would make the guard vacuous — an invented diagnosis
    mentioned in the narrative would validate itself. Verifying against a
    separate transcription field means a fabricated quote must also be
    fabricated into the transcription, a meaningfully higher bar.

    Missing transcription returns "" and the guard FAILS CLOSED: with an empty
    haystack every quote fails and every item is dropped, so a model that omits
    the field yields an empty draft rather than an unchecked one.
    """
    return payload.get("document_text") or ""


def extract_document(cx, doc_id, call_model=None, source_text=None):
    """Run extraction for one document and write its draft. Returns a small
    summary dict, or None when extraction failed (document marked 'failed').

    Does NOT claim the document itself -- claiming is run_pending's job (see
    below). Called directly with an explicit doc_id, this always runs,
    regardless of the document's current extract_status; that is the
    supported path for a manual/forced re-extraction.
    """
    from dashboard import client_documents as _cd
    from dashboard import document_extractions as _dx
    from dashboard import canonical_tags as _ct

    doc = _cd.get(cx, doc_id)
    if not doc:
        return None
    call = call_model or _default_call_model
    try:
        payload = call(doc["blob"], doc["content_type"])
        if not isinstance(payload, dict):
            raise ValueError("model did not return an object")
    except Exception as e:                          # noqa: BLE001 - any failure
        print(f"[documents] extract failed for {doc_id}: {e!r}", flush=True)
        _cd.set_extract_status(cx, doc_id, "failed")
        return None

    hay = source_text if source_text is not None else _source_text_for(payload)
    attrs, d1 = verify_quotes(payload.get("attributes"), hay)
    facts, d2 = verify_quotes(payload.get("facts"), hay)
    unstruct, d3 = verify_quotes(payload.get("unstructured"), hay)

    # Drop out-of-vocabulary fields, then canonicalize so Glen reviews the exact
    # value that would be written.
    _ct.init_tables(cx)
    clean_attrs = []
    for a in attrs:
        field = (a.get("field") or "").strip()
        if field not in _ct.ALL_FIELDS:
            d1.append(a)
            continue
        value = _ct.resolve(cx, field, a.get("value"))
        if not value:
            d1.append(a)
            continue
        clean_attrs.append({"field": field, "value": value,
                            "source_quote": a.get("source_quote", "")})

    _dx.put_draft(cx, doc_id, doc["email"], payload.get("narrative_md") or "",
                  clean_attrs, facts, unstruct, _MODEL)
    _cd.set_extract_status(cx, doc_id, "drafted")
    dropped = len(d1) + len(d2) + len(d3)
    if dropped:
        print(f"[documents] dropped {dropped} ungrounded item(s) for {doc_id}",
              flush=True)
    return {"document_id": doc_id, "kept": len(clean_attrs) + len(facts)
            + len(unstruct), "dropped": dropped}


def run_pending(cx, limit=5, call_model=None):
    """Process up to `limit` pending documents. Returns how many drafted.

    Production runs multiple web instances against one database. Selecting
    pending documents and then extracting them (without a claim step) would
    let two instances pick up the SAME document -- duplicate paid Claude
    calls and racing writes into the draft store. So each candidate is first
    claimed with client_documents.claim_for_extraction, a single
    WHERE-guarded UPDATE; a claim that returns False means another instance
    already won that document, and it is skipped here (left in whatever
    state the winner puts it in) rather than extracted a second time.
    """
    from dashboard import client_documents as _cd
    done = 0
    for doc in _cd.pending(cx, limit=limit):
        if not _cd.claim_for_extraction(cx, doc["id"]):
            continue  # another instance claimed it first
        if extract_document(cx, doc["id"], call_model=call_model):
            done += 1
    return done
