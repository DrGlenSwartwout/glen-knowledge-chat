import sqlite3
from dashboard import client_documents as cd
from dashboard import document_extractions as dx
from dashboard import document_extract as de


def _cx():
    cx = sqlite3.connect(":memory:")
    cd.init_table(cx); dx.init_table(cx)
    return cx


def _doc(cx, email="c@x.com", blob=b"%PDF fake"):
    return cd.put(cx, email, blob, "labs.pdf", "application/pdf", "console")["id"]


SOURCE = "Patient reports taking AREDS2 daily. Assessment: glaucoma. HbA1c 6.4."


def _fake_model(payload):
    def call(blob, content_type):
        return payload
    return call


def test_verify_quotes_keeps_grounded_items_and_drops_invented_ones():
    items = [{"value": "Glaucoma", "source_quote": "Assessment: glaucoma"},
             {"value": "Diabetes", "source_quote": "Assessment: diabetes"}]
    kept, dropped = de.verify_quotes(items, SOURCE)
    assert [k["value"] for k in kept] == ["Glaucoma"]
    assert [d["value"] for d in dropped] == ["Diabetes"]


def test_verify_quotes_drops_items_with_no_quote_at_all():
    kept, dropped = de.verify_quotes([{"value": "Glaucoma"}], SOURCE)
    assert kept == [] and len(dropped) == 1


def test_verify_quotes_is_case_and_whitespace_insensitive():
    kept, _ = de.verify_quotes(
        [{"value": "X", "source_quote": "  ASSESSMENT:   GLAUCOMA "}], SOURCE)
    assert len(kept) == 1


# --- FINDING 1: a trivially short quote defeats the substring guard. A
# quote of "." or "a" appears in nearly any document and would otherwise
# validate ANY fabrication attached to it. ---

def test_verify_quotes_drops_a_one_character_quote():
    kept, dropped = de.verify_quotes(
        [{"value": "Metastatic melanoma", "source_quote": "."}], SOURCE)
    assert kept == [] and len(dropped) == 1


def test_verify_quotes_drops_a_single_letter_quote():
    kept, dropped = de.verify_quotes(
        [{"value": "X", "source_quote": "a"}], "Patient is a diabetic.")
    assert kept == [] and len(dropped) == 1


def test_verify_quotes_drops_a_whitespace_only_quote():
    kept, dropped = de.verify_quotes(
        [{"value": "X", "source_quote": "    "}], SOURCE)
    assert kept == [] and len(dropped) == 1


def test_verify_quotes_keeps_a_legitimate_short_multi_token_quote():
    """Guards against over-tightening: a real, short, two-token clinical
    quote like 'HbA1c 6.4' must still survive the new length/token floor."""
    kept, dropped = de.verify_quotes(
        [{"value": "6.4", "source_quote": "HbA1c 6.4"}], SOURCE)
    assert len(kept) == 1 and dropped == []


def test_extract_drops_a_trivially_short_quote_end_to_end():
    """The exact FINDING 1 repro: a '.' source_quote must not let a
    fabricated diagnosis survive into the draft Glen reviews."""
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"narrative_md": "n", "facts": [], "unstructured": [],
               "attributes": [{"field": "conditions",
                                "value": "Metastatic melanoma",
                                "source_quote": "."}]}
    out = de.extract_document(cx, doc_id, call_model=_fake_model(payload),
                              source_text=SOURCE)
    assert out["dropped"] == 1
    assert dx.get_for_document(cx, doc_id)["attributes"] == []


def test_extract_writes_a_draft_and_marks_the_document(monkeypatch):
    cx = _cx()
    doc_id = _doc(cx)
    payload = {
        "narrative_md": "Your panel showed a few things.",
        "attributes": [{"field": "conditions", "value": "glaucoma",
                        "source_quote": "Assessment: glaucoma"}],
        "facts": [{"fact_key": "on_areds2", "value": True,
                   "source_quote": "taking AREDS2 daily"}],
        "unstructured": [{"label": "HbA1c", "value": "6.4",
                          "source_quote": "HbA1c 6.4"}],
    }
    out = de.extract_document(cx, doc_id, call_model=_fake_model(payload),
                              source_text=SOURCE)
    assert out["dropped"] == 0
    draft = dx.get_for_document(cx, doc_id)
    assert draft["status"] == "ai_draft"
    assert draft["narrative_md"] == "Your panel showed a few things."
    assert draft["facts"][0]["fact_key"] == "on_areds2"
    assert draft["unstructured"][0]["label"] == "HbA1c"
    assert cd.get(cx, doc_id)["extract_status"] == "drafted"


def test_extract_drops_an_ungrounded_attribute_from_the_draft():
    """The fabrication guard: an invented diagnosis never reaches Glen's review."""
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"narrative_md": "n", "facts": [], "unstructured": [],
               "attributes": [
                   {"field": "conditions", "value": "glaucoma",
                    "source_quote": "Assessment: glaucoma"},
                   {"field": "conditions", "value": "lupus",
                    "source_quote": "Assessment: lupus"}]}
    out = de.extract_document(cx, doc_id, call_model=_fake_model(payload),
                              source_text=SOURCE)
    assert out["dropped"] == 1
    vals = [a["value"] for a in dx.get_for_document(cx, doc_id)["attributes"]]
    assert "lupus" not in [v.lower() for v in vals]


def test_production_path_verifies_against_the_models_transcription():
    """No source_text injected — the real call path. Quotes are checked against
    the model's `document_text`, NOT its narrative."""
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"document_text": SOURCE, "narrative_md": "n",
               "facts": [], "unstructured": [],
               "attributes": [
                   {"field": "conditions", "value": "glaucoma",
                    "source_quote": "Assessment: glaucoma"},
                   {"field": "conditions", "value": "lupus",
                    "source_quote": "Assessment: lupus"}]}
    out = de.extract_document(cx, doc_id, call_model=_fake_model(payload))
    assert out["dropped"] == 1
    vals = [a["value"] for a in dx.get_for_document(cx, doc_id)["attributes"]]
    assert [v.lower() for v in vals] == ["glaucoma"]


def test_narrative_alone_cannot_validate_a_quote():
    """The guard must not be self-validating: a diagnosis invented into the
    narrative, absent from the transcription, is still dropped."""
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"document_text": "Routine visit. Nothing remarkable.",
               "narrative_md": "Assessment: lupus was noted.",
               "facts": [], "unstructured": [],
               "attributes": [{"field": "conditions", "value": "lupus",
                               "source_quote": "Assessment: lupus"}]}
    de.extract_document(cx, doc_id, call_model=_fake_model(payload))
    assert dx.get_for_document(cx, doc_id)["attributes"] == []


def test_missing_transcription_fails_closed():
    """A model that omits document_text yields an EMPTY draft, never an
    unchecked one."""
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"narrative_md": "n", "facts": [], "unstructured": [],
               "attributes": [{"field": "conditions", "value": "glaucoma",
                               "source_quote": "Assessment: glaucoma"}]}
    out = de.extract_document(cx, doc_id, call_model=_fake_model(payload))
    assert out["dropped"] == 1
    assert dx.get_for_document(cx, doc_id)["attributes"] == []


def test_extract_canonicalizes_attribute_values_before_drafting():
    """Glen reviews the canonical form he will actually be approving."""
    cx = _cx()
    from dashboard import canonical_tags as ct
    ct.init_tables(cx)
    cx.execute("INSERT INTO canonical_vocab(field, alias_norm, canonical) "
               "VALUES(?,?,?)", ("conditions", "glaucoma", "Glaucoma (POAG)"))
    cx.commit()
    doc_id = _doc(cx)
    payload = {"narrative_md": "n", "facts": [], "unstructured": [],
               "attributes": [{"field": "conditions", "value": "glaucoma",
                               "source_quote": "Assessment: glaucoma"}]}
    de.extract_document(cx, doc_id, call_model=_fake_model(payload),
                        source_text=SOURCE)
    assert dx.get_for_document(cx, doc_id)["attributes"][0]["value"] == "Glaucoma (POAG)"


def test_extract_drops_attributes_with_an_out_of_vocabulary_field():
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"narrative_md": "n", "facts": [], "unstructured": [],
               "attributes": [{"field": "not_a_field", "value": "x",
                               "source_quote": "Assessment: glaucoma"}]}
    de.extract_document(cx, doc_id, call_model=_fake_model(payload),
                        source_text=SOURCE)
    assert dx.get_for_document(cx, doc_id)["attributes"] == []


def test_extract_marks_failed_and_writes_no_draft_when_the_model_raises():
    cx = _cx()
    doc_id = _doc(cx)

    def boom(blob, content_type):
        raise RuntimeError("api down")

    assert de.extract_document(cx, doc_id, call_model=boom, source_text=SOURCE) is None
    assert cd.get(cx, doc_id)["extract_status"] == "failed"
    assert dx.get_for_document(cx, doc_id) is None


def test_extract_marks_failed_on_unparseable_model_output():
    cx = _cx()
    doc_id = _doc(cx)
    assert de.extract_document(cx, doc_id, call_model=_fake_model("not a dict"),
                               source_text=SOURCE) is None
    assert cd.get(cx, doc_id)["extract_status"] == "failed"


def test_run_pending_processes_only_pending_documents():
    cx = _cx()
    a = _doc(cx, blob=b"one")
    b = _doc(cx, blob=b"two")
    cd.set_extract_status(cx, b, "drafted")
    payload = {"narrative_md": "n", "attributes": [], "facts": [],
               "unstructured": []}
    assert de.run_pending(cx, call_model=_fake_model(payload)) == 1
    assert cd.get(cx, a)["extract_status"] == "drafted"


# --- claim step: prevents two web instances from double-extracting the same
# document (duplicate paid Claude calls + racing draft writes). ---

def test_claim_for_extraction_succeeds_once_and_fails_on_second_attempt():
    cx = _cx()
    doc_id = _doc(cx)
    assert cd.claim_for_extraction(cx, doc_id) is True
    assert cd.get(cx, doc_id)["extract_status"] == "extracting"
    # Second claim on the same (now non-pending) document loses.
    assert cd.claim_for_extraction(cx, doc_id) is False


def test_claim_for_extraction_fails_for_a_nonexistent_document():
    cx = _cx()
    assert cd.claim_for_extraction(cx, 999999) is False


def test_run_pending_only_extracts_documents_it_successfully_claimed():
    """Simulates a second instance winning the claim race on one of the two
    pending documents: run_pending must skip it, not extract it a second time."""
    cx = _cx()
    a = _doc(cx, blob=b"one")
    b = _doc(cx, blob=b"two")
    # Another instance claims `b` first.
    assert cd.claim_for_extraction(cx, b) is True
    payload = {"narrative_md": "n", "attributes": [], "facts": [],
               "unstructured": []}
    assert de.run_pending(cx, call_model=_fake_model(payload)) == 1
    assert cd.get(cx, a)["extract_status"] == "drafted"
    # `b` was left alone -- still 'extracting', not touched by this run.
    assert cd.get(cx, b)["extract_status"] == "extracting"
    assert dx.get_for_document(cx, b) is None


# --- FINDING 2: a malformed model reply must fail closed to 'failed', never
# raise out of extract_document and never wedge the document in
# 'extracting'. A single bad ITEM inside an otherwise-valid list is dropped
# (salvage what's good); a whole item-collection of the wrong TYPE (a dict
# or bare string instead of a list) means the reply is untrustworthy as a
# whole and is NOT salvaged -- fail closed instead of drafting an empty
# result that would look to Glen like a clean, complete extraction. ---

def test_extract_fails_closed_when_attributes_is_a_dict():
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"narrative_md": "n", "facts": [], "unstructured": [],
               "attributes": {"field": "conditions", "value": "x",
                              "source_quote": "Assessment: glaucoma"}}
    assert de.extract_document(cx, doc_id, call_model=_fake_model(payload),
                               source_text=SOURCE) is None
    assert cd.get(cx, doc_id)["extract_status"] == "failed"
    assert dx.get_for_document(cx, doc_id) is None


def test_extract_fails_closed_when_attributes_is_a_bare_string():
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"narrative_md": "n", "facts": [], "unstructured": [],
               "attributes": "Assessment: glaucoma"}
    assert de.extract_document(cx, doc_id, call_model=_fake_model(payload),
                               source_text=SOURCE) is None
    assert cd.get(cx, doc_id)["extract_status"] == "failed"
    assert dx.get_for_document(cx, doc_id) is None


def test_extract_fails_closed_when_unstructured_is_a_dict():
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"narrative_md": "n", "attributes": [], "facts": [],
               "unstructured": {"label": "HbA1c", "value": "6.4",
                                "source_quote": "HbA1c 6.4"}}
    assert de.extract_document(cx, doc_id, call_model=_fake_model(payload),
                               source_text=SOURCE) is None
    assert cd.get(cx, doc_id)["extract_status"] == "failed"
    assert dx.get_for_document(cx, doc_id) is None


def test_extract_drops_a_bare_string_item_without_crashing_the_batch():
    """A single bad item, unlike a whole malformed collection, is dropped --
    the rest of a valid list is still salvaged, not treated as a fatal
    whole-document failure."""
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"narrative_md": "n", "facts": [], "unstructured": [],
               "attributes": ["Assessment: glaucoma",
                              {"field": "conditions", "value": "glaucoma",
                               "source_quote": "Assessment: glaucoma"}]}
    out = de.extract_document(cx, doc_id, call_model=_fake_model(payload),
                              source_text=SOURCE)
    assert out == {"document_id": doc_id, "kept": 1, "dropped": 1}
    assert cd.get(cx, doc_id)["extract_status"] == "drafted"
    vals = [a["value"] for a in dx.get_for_document(cx, doc_id)["attributes"]]
    assert vals == ["glaucoma"]


def test_extract_drops_an_item_with_a_list_source_quote_without_crashing():
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"narrative_md": "n", "facts": [], "unstructured": [],
               "attributes": [{"field": "conditions", "value": "glaucoma",
                               "source_quote": ["Assessment: glaucoma"]}]}
    out = de.extract_document(cx, doc_id, call_model=_fake_model(payload),
                              source_text=SOURCE)
    assert out == {"document_id": doc_id, "kept": 0, "dropped": 1}
    assert cd.get(cx, doc_id)["extract_status"] == "drafted"


def test_extract_drops_attributes_with_non_string_field_or_value():
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"narrative_md": "n", "facts": [], "unstructured": [],
               "attributes": [{"field": None, "value": "glaucoma",
                               "source_quote": "Assessment: glaucoma"},
                              {"field": "conditions", "value": 123,
                               "source_quote": "HbA1c 6.4"}]}
    out = de.extract_document(cx, doc_id, call_model=_fake_model(payload),
                              source_text=SOURCE)
    assert out == {"document_id": doc_id, "kept": 0, "dropped": 2}
    assert cd.get(cx, doc_id)["extract_status"] == "drafted"


def test_extract_treats_non_string_document_text_as_missing():
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"document_text": {"oops": "not a string"}, "narrative_md": "n",
               "facts": [], "unstructured": [],
               "attributes": [{"field": "conditions", "value": "glaucoma",
                               "source_quote": "Assessment: glaucoma"}]}
    out = de.extract_document(cx, doc_id, call_model=_fake_model(payload))
    assert out["dropped"] == 1
    assert cd.get(cx, doc_id)["extract_status"] == "drafted"
    assert dx.get_for_document(cx, doc_id)["attributes"] == []


def test_extract_treats_non_string_narrative_md_as_empty():
    cx = _cx()
    doc_id = _doc(cx)
    payload = {"narrative_md": {"oops": "dict"}, "facts": [], "unstructured": [],
               "attributes": [{"field": "conditions", "value": "glaucoma",
                               "source_quote": "Assessment: glaucoma"}]}
    out = de.extract_document(cx, doc_id, call_model=_fake_model(payload),
                              source_text=SOURCE)
    assert out["kept"] == 1
    assert dx.get_for_document(cx, doc_id)["narrative_md"] == ""


def test_run_pending_continues_past_a_document_that_raises(monkeypatch):
    """Even an exception extract_document itself does not internally catch
    (an unanticipated bug, not just a malformed-reply case) must not abort
    the rest of run_pending's batch."""
    cx = _cx()
    a = _doc(cx, blob=b"one")
    b = _doc(cx, blob=b"two")
    real_extract = de.extract_document

    def flaky_extract(cx, doc_id, call_model=None, source_text=None):
        if doc_id == a:
            raise RuntimeError("boom -- unanticipated bug")
        return real_extract(cx, doc_id, call_model=call_model,
                            source_text=source_text)

    monkeypatch.setattr(de, "extract_document", flaky_extract)
    payload = {"narrative_md": "n", "attributes": [], "facts": [],
               "unstructured": []}
    assert de.run_pending(cx, call_model=_fake_model(payload)) == 1
    assert cd.get(cx, b)["extract_status"] == "drafted"


# --- FINDING 3: put_draft correctly protects a 'confirmed' row by writing
# nothing and returning the existing id -- but extract_document must not
# then LIE about it by reporting kept/dropped counts for work that was
# discarded, or by claiming a fresh draft was written. ---

def test_reextracting_a_confirmed_document_reports_skipped_confirmed():
    cx = _cx()
    doc_id = _doc(cx)
    original_payload = {
        "narrative_md": "Original approved narrative.",
        "facts": [], "unstructured": [],
        "attributes": [{"field": "conditions", "value": "glaucoma",
                        "source_quote": "Assessment: glaucoma"}]}
    de.extract_document(cx, doc_id, call_model=_fake_model(original_payload),
                        source_text=SOURCE)
    extraction_id = dx.get_for_document(cx, doc_id)["id"]
    assert dx.confirm(cx, extraction_id, "Original approved narrative.",
                      "glen") is True

    new_payload = {
        "narrative_md": "A completely different, fabricated narrative.",
        "facts": [], "unstructured": [],
        "attributes": [{"field": "conditions", "value": "lupus",
                        "source_quote": "Assessment: glaucoma"}]}
    out = de.extract_document(cx, doc_id, call_model=_fake_model(new_payload),
                              source_text=SOURCE)
    assert out == {"document_id": doc_id, "kept": 0, "dropped": 0,
                    "skipped_confirmed": True}

    draft = dx.get_for_document(cx, doc_id)
    assert draft["status"] == "confirmed"
    assert draft["narrative_md"] == "Original approved narrative."
    vals = [a["value"] for a in draft["attributes"]]
    assert vals == ["glaucoma"]
    assert cd.get(cx, doc_id)["extract_status"] == "drafted"


# --- FINDING 4: the guard must not be self-validating. A quote present
# ONLY in narrative_md (never in document_text, which is entirely ABSENT
# here -- not just empty) must still be dropped. This test must fail RED
# under the mutation `_source_text_for` -> `payload.get("document_text") or
# payload.get("narrative_md") or ""`. ---

def test_narrative_cannot_substitute_for_a_wholly_absent_document_text():
    cx = _cx()
    doc_id = _doc(cx)
    payload = {
        # no "document_text" key at all
        "narrative_md": "Assessment: glaucoma was noted in the chart today.",
        "facts": [], "unstructured": [],
        "attributes": [{"field": "conditions", "value": "glaucoma",
                        "source_quote": "Assessment: glaucoma"}]}
    out = de.extract_document(cx, doc_id, call_model=_fake_model(payload))
    assert out["dropped"] == 1
    assert dx.get_for_document(cx, doc_id)["attributes"] == []
