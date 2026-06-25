import sqlite3
from dashboard import sourcing as sc
from scripts.scan_supplier_quotes import looks_like_quote, extract_quote, _to_stage_row


def test_looks_like_quote():
    assert looks_like_quote("Re: HydroCurc quote", "Price is $334/kg, MOQ 25 kg, lead time 7-10 days")
    assert not looks_like_quote("Lunch?", "are you free thursday")


class _FakeClient:
    def __init__(self, payload): self._p = payload
    @property
    def messages(self): return self
    def create(self, **kw):
        class _Block:  # mimic a tool_use content block
            type = "tool_use"; name = "record_quote"
        b = _Block(); b.input = self._p
        class _Msg: content = [b]
        return _Msg()


def test_extract_quote_tooluse():
    payload = {"is_supplier_quote": True, "supplier_name": "Pharmako", "ingredient_name": "HydroCurc",
               "price": 334, "price_unit": "kg", "currency": "USD", "moq": 25, "moq_unit": "kg",
               "lead_time_days": 9, "confidence": 0.9}
    q = extract_quote("HydroCurc quote", "…", client=_FakeClient(payload))
    assert q["is_supplier_quote"] and q["price"] == 334 and q["ingredient_name"] == "HydroCurc"
    # a non-quote returns None
    assert extract_quote("hi", "…", client=_FakeClient({"is_supplier_quote": False})) is None


def test_stage_row_idempotent(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        from dashboard.ingredient_catalog import init_ingredients_schema
        init_ingredients_schema(cx); sc.init_sourcing_schema(cx); cx.commit()
        row = _to_stage_row("msg-1", "sales@x.com", "Quote",
                            {"supplier_name": "X", "ingredient_name": "Y", "price": 10, "price_unit": "kg", "confidence": 0.8})
        assert sc.stage_quotes(cx, [row]) == 1
        assert sc.stage_quotes(cx, [row]) == 0
        cx.commit()


def test_received_iso_and_row():
    from scripts.scan_supplier_quotes import _received_iso, _to_stage_row
    assert _received_iso("Wed, 25 Jun 2026 09:36:00 -1000") == "2026-06-25 09:36:00"
    assert _received_iso(None) is None and _received_iso("garbage") is None
    row = _to_stage_row("m9", "x@y.com", "Quote", {"price": 5}, received_at="2026-06-25 09:36:00")
    assert row["received_at"] == "2026-06-25 09:36:00" and row["gmail_msg_id"] == "m9"
