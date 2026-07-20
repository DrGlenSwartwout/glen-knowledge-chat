"""Raw consult/class transcripts must not feed the model dollar figures.

24% of consultation vectors carry a $ figure, and on a price question 4-5 of the
top 5 consult results do. They are conversational noise — internal cost notes,
marketing asides, tangents — never a retail price. The NIR helmet's fabricated
"$754 ($622 + $132 shipping)" came verbatim from a consult transcript. This
strips them at retrieval so the model cannot quote one; authoritative prices
live in the product injection table.
"""
import app


def test_consultation_prices_are_redacted():
    meta = {"_source_ns": "consultations",
            "text": "Helmet 810 nm $754 ($622 + $132 shipping): Wholesale $997 Retail $1997"}
    out = app._redact_transcript_prices(meta["text"], meta)
    assert "$754" not in out and "$132" not in out and "$1997" not in out
    assert "$[amount omitted]" in out
    assert "Helmet 810 nm" in out  # clinical text preserved


def test_namespace_from_bare_metadata_also_redacts():
    """Older vectors carry namespace in their own metadata, not _source_ns."""
    meta = {"namespace": "consultations", "text": "we quoted $50,000 for that"}
    assert "$50,000" not in app._redact_transcript_prices(meta["text"], meta)


def test_authoritative_and_product_prices_untouched():
    for ns in ("specific-formulations", "clinical-qa", "business", ""):
        meta = {"_source_ns": ns, "text": "Terrain Restore is $69.97 list"}
        assert app._redact_transcript_prices(meta["text"], meta) == meta["text"], ns


def test_query_ns_stamps_source_namespace():
    """build_context relies on _source_ns being present; query_ns must set it."""
    captured = {}

    class _M:
        def __init__(self): self.metadata = {"text": "x"}

    class _Res:
        matches = [_M()]

    orig = app._idx
    try:
        app._idx = type("I", (), {"query": staticmethod(lambda **k: _Res())})()
        ms = app.query_ns([0.0], "consultations", 5)
        assert ms[0].metadata["_source_ns"] == "consultations"
    finally:
        app._idx = orig
