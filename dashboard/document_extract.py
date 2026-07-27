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
    "An item without a verbatim quote will be discarded. Each source_quote must "
    "be a complete phrase or the whole line the finding appears on -- long "
    "enough to stand on its own -- never a bare single word or number (e.g. "
    "not just \"Glaucoma\" or \"6.4\"); quotes shorter than that will be "
    "discarded. No markdown fences, no prose outside the JSON."
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


def call_model_for_extraction(blob, content_type, call_model=None):
    """DB-free half of extraction: makes the (slow, tens-of-seconds) model
    call and returns the validated raw payload. Takes only bytes + a
    content type -- no `cx`, no document id -- so this can run with NO
    database lock held.

    Production runs `gunicorn --workers 2 --worker-class gevent --timeout
    120`; the application-wide write mutex (`_db_lock` in app.py) is used at
    hundreds of call sites. Holding it across this call would stall every
    DB-touching request in the worker for as long as the model takes.
    Callers (app.py) must call this OUTSIDE any lock, then take the lock
    again only for the write half below.

    Raises on any failure: a model error, a non-dict reply, or an
    item-collection (attributes/facts/unstructured) of the wrong TYPE (a
    dict or bare string instead of a list) -- that last case means the
    reply's structured-extraction portion is untrustworthy as a whole and
    is NOT salvaged (unlike a single bad ITEM inside an otherwise-valid
    list, which verify_quotes drops individually downstream). Callers must
    catch and route the failure to 'failed', exactly like extract_document
    always has.
    """
    call = call_model or _default_call_model
    payload = call(blob, content_type)
    if not isinstance(payload, dict):
        raise ValueError("model did not return an object")
    for _key in ("attributes", "facts", "unstructured"):
        _v = payload.get(_key)
        if _v is not None and not isinstance(_v, list):
            raise ValueError(f"model returned non-list '{_key}' "
                             f"({type(_v).__name__})")
    return payload


def write_extraction_draft(cx, doc_id, payload, source_text=None):
    """DB-only half of extraction: quote verification, canonicalization, the
    draft write, and the extract_status write. Takes an already-validated
    `payload` (see call_model_for_extraction above) plus `cx` -- no model
    call happens in here, so this is safe to run under `_db_lock`.

    Returns a small summary dict, or None if the document no longer exists.
    Does not itself catch exceptions from the DB writes below -- callers
    (extract_document, and app.py's split callers) are responsible for
    routing any exception here to 'failed', exactly as extract_document
    always has for the whole guarded region.
    """
    from dashboard import client_documents as _cd
    from dashboard import document_extractions as _dx
    from dashboard import canonical_tags as _ct

    doc = _cd.get(cx, doc_id)
    if not doc:
        return None

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


def extract_document(cx, doc_id, call_model=None, source_text=None):
    """Run extraction for one document and write its draft. Returns a small
    summary dict, or None when extraction failed (document marked 'failed').

    Does NOT claim the document itself -- claiming is run_pending's job (see
    below). Called directly with an explicit doc_id, this always runs,
    regardless of the document's current extract_status; that is the
    supported path for a manual/forced re-extraction.

    Composes the two halves above (call_model_for_extraction, then
    write_extraction_draft) against the SAME `cx` for the whole call --
    this function has no lock of its own to release, so callers that care
    about not holding a lock across the model call (app.py, under
    `_db_lock`) must call the two halves directly instead of this
    composition. This function's signature and behavior are otherwise
    unchanged: every existing caller/test keeps working exactly as before.

    The guarded region below covers the model call AND everything downstream
    of it (quote verification, canonicalization, draft write). A malformed
    reply -- attributes as a dict or bare string, non-string field/value/
    source_quote, a dict narrative_md, etc -- must fail closed to 'failed'
    exactly like a failed model call, never raise out of this function and
    leave the document wedged in 'extracting' forever (FINDING 2).
    """
    from dashboard import client_documents as _cd

    doc = _cd.get(cx, doc_id)
    if not doc:
        return None
    try:
        payload = call_model_for_extraction(doc["blob"], doc["content_type"],
                                            call_model)
        return write_extraction_draft(cx, doc_id, payload,
                                      source_text=source_text)
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


# --- Text-source Other-Ingredients extraction (purity Phase 2b) -----------
# Sibling of call_model_for_extraction, for an HTML/text source rather than an
# image/PDF blob. Same fabrication guard (verify_quotes), same fails-closed
# discipline. The page embeds several products' data, so the model is anchored
# on the target product's name/brand/SKU and instructed to return ONLY that
# product's Other Ingredients; verify_quotes then requires the returned line to
# be a verbatim substring of the fetched page, so a fabricated or wrong-block
# line fails closed to None.

_OTHER_ING_PROMPT = (
    "You are reading the text of a supplement catalog web page. It may contain "
    "several products. Find the ONE product that matches this identity:\n"
    "  name: {name}\n  brand: {brand}\n  sku: {sku}\n\n"
    "Return STRICT JSON with ONE of these shapes:\n"
    "  1. The product lists other ingredients -> "
    "{{\"other_ingredients_line\": \"...\"}} where the value is the VERBATIM "
    "'Other Ingredients' (or 'Non-Medicinal Ingredients') text for THAT product "
    "ONLY, copied exactly from the page, character for character. Do NOT include "
    "the 'Other Ingredients:' label, other products' ingredients, the "
    "Supplement/Medicinal Facts actives, dosages, or any warning text.\n"
    "  2. The product's block EXPLICITLY states it has no other ingredients "
    "(e.g. 'Other Ingredients: None') -> "
    "{{\"none_source_quote\": \"...\"}} where the value is that VERBATIM "
    "declaration copied exactly from the page, INCLUDING the 'Other Ingredients' "
    "label and the word 'None' (e.g. \"Other Ingredients: None\").\n"
    "  3. You cannot find this exact product's block on the page -> "
    "{{\"other_ingredients_line\": \"\"}}.\n"
    "Never guess or infer an ingredient that is not written on the page, and "
    "never claim 'None' unless the page literally says so for THIS product. No "
    "markdown fences, no prose outside the JSON."
)


def _default_call_model_text(source_text, name, brand, sku):
    """The real Anthropic text call. Lazy-imported so tests that inject
    call_model never pull the SDK."""
    import anthropic
    cli = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    prompt = _OTHER_ING_PROMPT.format(name=name or "", brand=brand or "", sku=sku or "")
    resp = cli.messages.create(
        model=_MODEL, max_tokens=1000,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": source_text[:120000]},
            {"type": "text", "text": prompt}]}])
    text = resp.content[0].text.strip()
    if text.startswith("```"):                      # tolerate accidental fences
        text = text.split("```", 2)[1]
        if text.startswith("json\n"):
            text = text[5:]
    return json.loads(text)


def _quote_near_anchor(source_text, quote, anchors, window=5000):
    """True iff `quote` occurs in `source_text` within `window` characters of an
    occurrence of one of `anchors` (the target product's name / brand / sku /
    slug).

    This ties an extracted quote to the target product's OWN region of a
    multi-product catalog page. Measured on a real Fullscript page: the target's
    own ingredients line sits ~660 chars from its brand and ~1400 from its slug,
    while a DIFFERENT product's block is ~140,000 chars away -- an enormous safe
    band. The window is set well above the intra-block spread (some anchors, like
    the seed display name, may not even appear near the line, or at all) and far
    below the inter-product distance, so a neighbor's identical-looking line --
    or its "Other Ingredients: None" -- is still rejected. Fails closed (returns
    False) when the quote is absent or no usable anchor (>=3 chars) appears on
    the page, so a product with no recoverable identity can never be greened off
    a borrowed quote.

    Matching is done in the whitespace-collapsed, lowercased space used by
    verify_quotes so offsets are consistent with that check."""
    hay = _norm_text(source_text)
    q = _norm_text(quote)
    if not q or q not in hay:
        return False
    apos = []
    for a in anchors:
        na = _norm_text(a)
        if len(na) < 3:
            continue
        i = hay.find(na)
        while i >= 0:
            apos.append(i)
            i = hay.find(na, i + 1)
    if not apos:
        return False
    i = hay.find(q)
    while i >= 0:
        if any(abs(ap - i) <= window for ap in apos):
            return True
        i = hay.find(q, i + 1)
    return False


def _ground_oi(payload, source_text, anchors):
    """Resolve a model payload {other_ingredients_line?, none_source_quote?} to a
    verified line / "" (verified explicit-none -> caller screens green) / None
    (miss -> unrated). Every non-None outcome must pass verify_quotes against
    source_text; when `anchors` is a list the quote must ALSO fall within a
    target anchor (_quote_near_anchor) -- for multi-product online pages.
    `anchors=None` skips the anchor: a single-product LABEL photo has no neighbor
    to borrow from, so verify_quotes against the label's own transcription is the
    whole guard."""
    line = payload.get("other_ingredients_line")
    if isinstance(line, str) and line.strip():
        kept, _d = verify_quotes([{"source_quote": line}], source_text)
        if kept and (anchors is None or _quote_near_anchor(source_text, line, anchors)):
            return line.strip()
        return None
    none_q = payload.get("none_source_quote")
    if isinstance(none_q, str) and none_q.strip():
        kept, _d = verify_quotes([{"source_quote": none_q}], source_text)
        nq = _norm_text(none_q)
        if (kept and "none" in nq and "ingredient" in nq
                and (anchors is None or _quote_near_anchor(source_text, none_q, anchors))):
            return ""
    return None


def extract_other_ingredients(source_text, *, name, brand, sku="", slug="", call_model=None):
    """Resolve the target product's Other Ingredients to one of three outcomes:

      - a NON-EMPTY str: the verified verbatim Other Ingredients line (the
        product lists excipients);
      - "" (EMPTY str): the product VERIFIABLY lists no other ingredients (its
        block literally reads e.g. "Other Ingredients: None") -- the caller
        screens this GREEN (pristine);
      - None: not found / unverifiable -> the caller screens UNRATED, never
        green.

    Both non-None outcomes are grounded the same TWO ways, so a mere model CLAIM
    is never trusted and a neighbor product's text on a multi-product page cannot
    be borrowed:
      1. verbatim: the quote must pass verify_quotes as a substring of
         source_text (and, for the none outcome, read as an
         other-ingredients-none declaration -- contains both 'ingredient' and
         'none');
      2. anchored: the quote must occur within a window of the TARGET product's
         own name/brand/sku (`_quote_near_anchor`), so a neighbor's line or its
         "Other Ingredients: None" -- which sits far from the target's name --
         is rejected.
    A product whose block was merely NOT FOUND yields no quote that clears both,
    so it fails closed to None -- 'not found' can never become green. The
    residual risk is a neighbor whose block happens to fall inside the target's
    anchor window; the human confirm gate is the backstop for that.

    Fails closed to None on a model error, a non-dict reply, or any output that
    clears neither grounded path."""
    call = call_model or _default_call_model_text
    try:
        payload = call(source_text, name, brand, sku)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return _ground_oi(payload, source_text, [name or "", brand or "", sku or "", slug or ""])


# --- Image-source Other-Ingredients extraction (purity Phase 2c) ----------
# Sibling of extract_other_ingredients, for a single-product LABEL PHOTO rather
# than a multi-product catalog page. There is no neighbor product to borrow a
# quote from, so grounding is against the model's OWN verbatim transcription of
# the label (`label_text`) with NO anchor -- verify_quotes against that
# transcription is the whole guard.

_OTHER_ING_IMAGE_PROMPT = (
    "You are reading a photo of a single dietary supplement product label. "
    "Return STRICT JSON with:\n"
    '  "label_text": a VERBATIM transcription of ALL text visible on the label, '
    "exactly as printed. Every other value is checked against this, so any text "
    "not present here is discarded.\n"
    "AND ONE of:\n"
    '  "other_ingredients_line": the VERBATIM "Other Ingredients" (or '
    '"Non-Medicinal Ingredients") text copied exactly, WITHOUT the label word; '
    "OR\n"
    '  "none_source_quote": if the label explicitly states it has none (e.g. '
    '"Other Ingredients: None"), that verbatim declaration INCLUDING the word '
    '"None"; OR\n'
    '  "other_ingredients_line": "" if the label is unreadable or shows no '
    "other-ingredients section.\n"
    "Never invent an ingredient not printed on the label. No markdown fences, no "
    "prose outside the JSON."
)


def _default_call_model_image(blob, content_type):
    """Real Anthropic vision call for a supplement-label photo. Lazy-imported so
    tests that inject call_model never pull the SDK. Mirrors _default_call_model's
    image/PDF source shaping."""
    import base64
    import anthropic
    cli = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    b64 = base64.standard_b64encode(blob).decode("ascii")
    ct = (content_type or "").lower()
    if ct == "application/pdf":
        doc = {"type": "document", "source": {"type": "base64",
               "media_type": "application/pdf", "data": b64}}
    else:
        # Anthropic's vision media_type is image/jpeg, not image/jpg -- the
        # route allows image/jpg (some browsers report it), so normalize or a
        # real JPEG would fail the SDK call and silently land 'unrated'.
        media = "image/jpeg" if ct == "image/jpg" else ct
        doc = {"type": "image", "source": {"type": "base64",
               "media_type": media, "data": b64}}
    resp = cli.messages.create(
        model=_MODEL, max_tokens=2000,
        messages=[{"role": "user", "content": [doc, {"type": "text",
                                                     "text": _OTHER_ING_IMAGE_PROMPT}]}])
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json\n"):
            text = text[5:]
    return json.loads(text)


def extract_other_ingredients_from_image(blob, content_type, *, name="", brand="",
                                         sku="", call_model=None):
    """Vision variant of extract_other_ingredients for a single-product LABEL
    photo. Returns a verified line / "" (verified explicit-none) / None (miss),
    grounded by verify_quotes against the model's OWN verbatim transcription
    (`label_text`) -- NO anchor, since a single label has no neighbor product to
    borrow from. Fails closed to None on a model error, non-dict reply, or a
    missing/blank/non-str transcription. name/brand/sku are accepted for parity
    with the text extractor but unused here (no anchor)."""
    call = call_model or _default_call_model_image
    try:
        payload = call(blob, content_type)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    src = payload.get("label_text")
    if not isinstance(src, str) or not src.strip():
        return None
    return _ground_oi(payload, src, None)
