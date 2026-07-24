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

# A `source_quote in document_text` substring check has no floor by itself: a
# quote of "." or "a" appears in nearly any document and would validate ANY
# fabricated fact attached to it. This is the minimum length (after whitespace
# collapse + lowercasing) a quote must clear before the substring check even
# runs -- see verify_quotes.
_MIN_QUOTE_LEN = 8

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


def _quote_is_substantial(q):
    """A quote must be long enough, and multi-token enough, that matching it
    means something. Without this floor, `"." in document_text` is true for
    almost any document, and a one-character source_quote validates any
    fabrication attached to it (see FINDING 1). Applied to the already
    whitespace-collapsed, lowercased quote."""
    if len(q) < _MIN_QUOTE_LEN:
        return False
    tokens = [t for t in q.split(" ") if t]
    alnum_tokens = [t for t in tokens if any(c.isalnum() for c in t)]
    return len(alnum_tokens) >= 2


def verify_quotes(items, source_text):
    """Split `items` into (kept, dropped) by whether each item's source_quote
    actually occurs in `source_text`. Whitespace- and case-insensitive, so the
    model is not punished for reflowing a line.

    Defensive about shape: a non-list `items` yields no items at all; a
    non-dict item, or one whose source_quote is not a string, is dropped
    rather than raising. A malformed model reply must fail closed into
    `dropped`, never crash the caller."""
    hay = _norm_text(source_text)
    kept, dropped = [], []
    if not isinstance(items, list):
        return kept, dropped
    for it in items:
        if not isinstance(it, dict):
            dropped.append(it)
            continue
        raw_quote = it.get("source_quote")
        if not isinstance(raw_quote, str):
            dropped.append(it)
            continue
        q = _norm_text(raw_quote)
        (kept if (_quote_is_substantial(q) and q in hay) else dropped).append(it)
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

    Also fails closed on a document_text of the wrong TYPE (e.g. a dict) --
    only a str is ever used as the haystack; anything else is treated as
    absent.
    """
    text = payload.get("document_text")
    return text if isinstance(text, str) else ""


def extract_document(cx, doc_id, call_model=None, source_text=None):
    """Run extraction for one document and write its draft. Returns a small
    summary dict, or None when extraction failed (document marked 'failed').

    Does NOT claim the document itself -- claiming is run_pending's job (see
    below). Called directly with an explicit doc_id, this always runs,
    regardless of the document's current extract_status; that is the
    supported path for a manual/forced re-extraction.

    The guarded region below covers the model call AND everything downstream
    of it (quote verification, canonicalization, draft write). A malformed
    reply -- attributes as a dict or bare string, non-string field/value/
    source_quote, a dict narrative_md, etc -- must fail closed to 'failed'
    exactly like a failed model call, never raise out of this function and
    leave the document wedged in 'extracting' forever (FINDING 2).
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
        # A whole item-collection of the wrong TYPE (a dict or bare string
        # instead of a list) means the reply's structured-extraction portion
        # is untrustworthy as a whole -- unlike a single bad ITEM inside an
        # otherwise-valid list (handled below by verify_quotes, which drops
        # just that item), there is nothing safe to salvage here. Fail
        # closed rather than silently draft an empty/degraded result that
        # would look to Glen like a clean, complete extraction.
        for _key in ("attributes", "facts", "unstructured"):
            _v = payload.get(_key)
            if _v is not None and not isinstance(_v, list):
                raise ValueError(f"model returned non-list '{_key}' "
                                 f"({type(_v).__name__})")

        hay = source_text if source_text is not None else _source_text_for(payload)
        attrs, d1 = verify_quotes(payload.get("attributes"), hay)
        facts, d2 = verify_quotes(payload.get("facts"), hay)
        unstruct, d3 = verify_quotes(payload.get("unstructured"), hay)

        # Drop out-of-vocabulary fields, then canonicalize so Glen reviews the
        # exact value that would be written. field/value must be strings --
        # a non-string of either (e.g. field=None, value=123) is dropped
        # rather than crashing canonical_tags.resolve.
        _ct.init_tables(cx)
        clean_attrs = []
        for a in attrs:
            field, value_raw = a.get("field"), a.get("value")
            if not isinstance(field, str) or not isinstance(value_raw, str):
                d1.append(a)
                continue
            field = field.strip()
            if field not in _ct.ALL_FIELDS:
                d1.append(a)
                continue
            value = _ct.resolve(cx, field, value_raw)
            if not value:
                d1.append(a)
                continue
            clean_attrs.append({"field": field, "value": value,
                                "source_quote": a.get("source_quote", "")})

        narrative = payload.get("narrative_md")
        narrative = narrative if isinstance(narrative, str) else ""

        _dx.put_draft(cx, doc_id, doc["email"], narrative, clean_attrs,
                      facts, unstruct, _MODEL)

        # FINDING 3: put_draft silently protects a 'confirmed' row -- it
        # writes NOTHING and returns the existing id. Report that honestly
        # instead of claiming N items were drafted when nothing was written.
        existing = _dx.get_for_document(cx, doc_id)
        if existing and existing.get("status") == "confirmed":
            print(f"[documents] extract for {doc_id}: existing draft is "
                  f"confirmed -- put_draft wrote nothing; this call's "
                  f"proposed items were discarded, not reported as kept",
                  flush=True)
            _cd.set_extract_status(cx, doc_id, "drafted")
            return {"document_id": doc_id, "kept": 0, "dropped": 0,
                    "skipped_confirmed": True}

        _cd.set_extract_status(cx, doc_id, "drafted")
        dropped = len(d1) + len(d2) + len(d3)
        if dropped:
            print(f"[documents] dropped {dropped} ungrounded item(s) for {doc_id}",
                  flush=True)
        return {"document_id": doc_id, "kept": len(clean_attrs) + len(facts)
                + len(unstruct), "dropped": dropped}
    except Exception as e:                          # noqa: BLE001 - any failure
        # A malformed reply must fail closed to 'failed', exactly like a
        # failed model call -- never leave the document wedged in
        # 'extracting' and never raise out of this function.
        print(f"[documents] extract failed for {doc_id}: {e!r}", flush=True)
        _cd.set_extract_status(cx, doc_id, "failed")
        return None


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

    Each document's processing is individually guarded: extract_document
    already fails closed internally, but this belt-and-suspenders try/except
    means even an unanticipated exception in the claim step or in
    extract_document cannot abort the rest of the batch (FINDING 2) --
    processing moves on to the next document.
    """
    from dashboard import client_documents as _cd
    done = 0
    for doc in _cd.pending(cx, limit=limit):
        try:
            if not _cd.claim_for_extraction(cx, doc["id"]):
                continue  # another instance claimed it first
            if extract_document(cx, doc["id"], call_model=call_model):
                done += 1
        except Exception as e:                      # noqa: BLE001 - one bad
            # document must not abort the rest of the batch.
            print(f"[documents] run_pending: document {doc['id']} raised, "
                  f"skipping: {e!r}", flush=True)
    return done
