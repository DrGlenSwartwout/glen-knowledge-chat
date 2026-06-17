# tests/test_fmp_biofield.py
"""FMP causal-chain importer: pure transforms (no network)."""


def test_translate_uses_trailing_name():
    from dashboard import fmp_biofield as fb
    assert fb.translate_head_chain("ED9 Muscle Driver") == "Muscle Driver"
    assert fb.translate_head_chain("MB6 Liberator") == "Liberator"
    assert fb.translate_head_chain("EI8 Microbes/Liver") == "Microbes/Liver"


def test_translate_falls_back_to_code_map():
    from dashboard import fmp_biofield as fb
    # bare code, no trailing text -> use the shipped infoceutical map
    assert fb.translate_head_chain("ES13") == "COH Carbohydrate Metabolism"
    assert fb.translate_head_chain("") == ""


def test_build_layers_maps_and_orders():
    from dashboard import fmp_biofield as fb
    rows = [
        {"layer": "2", "head_chain": "ES2 Memory Imprinter", "remedy": "Nous Energy", "dosage": "1 cap daily"},
        {"layer": "1", "head_chain": "MB6 Liberator", "remedy": "Neuroprotect", "dosage": "1 cap daily"},
    ]
    layers = fb.build_layers(rows)
    assert [l["n"] for l in layers] == [1, 2]
    assert layers[0]["title"] == "Liberator"          # layer 1 first
    assert layers[0]["remedy"] == "Neuroprotect"
    assert layers[0]["dosing"] == "1 cap daily"
    assert layers[0]["meaning"] == ""                  # filled by draft_prose later


def test_build_layers_skips_consider_and_blank():
    from dashboard import fmp_biofield as fb
    rows = [
        {"layer": "1", "head_chain": "MB6 Liberator", "remedy": "Neuroprotect", "dosage": "x"},
        {"layer": "2", "head_chain": "ED9 Muscle Driver", "remedy": "Consider: Liver Support, Free & Easy", "dosage": ""},
        {"layer": "3", "head_chain": "ED8 Stomach", "remedy": "", "dosage": "y"},
    ]
    layers = fb.build_layers(rows)
    assert [l["title"] for l in layers] == ["Liberator"]   # consider + blank dropped
    assert layers[0]["n"] == 1


class _FakeCur:
    def __init__(self, rows): self._rows = rows; self.executed = None
    def execute(self, sql, params): self.executed = (sql, params)
    def fetchall(self): return self._rows


class _FakeCtx:
    def __init__(self, cur): self.cur = cur
    def __enter__(self): return self.cur
    def __exit__(self, *a): return False


def test_fetch_causal_chain_queries_by_email(monkeypatch):
    from dashboard import fmp_biofield as fb
    import db_supabase
    cur = _FakeCur([{"layer": "1", "head_chain": "MB6 Liberator", "remedy": "Neuroprotect", "dosage": "x"}])
    monkeypatch.setattr(db_supabase, "supabase_cursor", lambda: _FakeCtx(cur))
    out = fb.fetch_causal_chain("Backdoc.Molina@gmail.com")
    assert out == [{"layer": "1", "head_chain": "MB6 Liberator", "remedy": "Neuroprotect", "dosage": "x"}]
    assert cur.executed[1] == ("backdoc.molina@gmail.com",)          # lowercased param
    assert "fmp_newapp.client_causal_chain" in cur.executed[0]


def test_fetch_causal_chain_handles_errors(monkeypatch):
    from dashboard import fmp_biofield as fb
    import db_supabase
    def _boom(): raise RuntimeError("supabase down")
    monkeypatch.setattr(db_supabase, "supabase_cursor", _boom)
    assert fb.fetch_causal_chain("x@y.com") == []


def test_draft_prose_parses_llm_json(monkeypatch):
    from dashboard import fmp_biofield as fb
    import openai

    class _Msg:  # minimal OpenAI response shape
        def __init__(self, c): self.message = type("M", (), {"content": c})
    class _Resp:
        def __init__(self, c): self.choices = [_Msg(c)]
    class _Fake:
        def __init__(self, *a, **k):
            self.chat = type("C", (), {"completions": type("X", (), {
                "create": staticmethod(lambda **kw: _Resp(
                    '{"greeting":"Aloha Othon.","layers":[{"title":"Calm","meaning":"settle"}]}'))})()})
    monkeypatch.setattr(openai, "OpenAI", _Fake)
    out = fb.draft_prose([{"title": "Liberator", "remedy": "Neuroprotect", "dosing": "x"}], "Othon")
    assert out["greeting"] == "Aloha Othon."
    assert out["layers"][0]["meaning"] == "settle"


def test_import_content_merges_prose(monkeypatch):
    from dashboard import fmp_biofield as fb
    monkeypatch.setattr(fb, "fetch_causal_chain",
                        lambda e: [{"layer": "1", "head_chain": "MB6 Liberator", "remedy": "Neuroprotect", "dosage": "x"}])
    monkeypatch.setattr(fb, "draft_prose",
                        lambda layers, name, tags=None: {"greeting": "G", "layers": [{"title": "Calm mind", "meaning": "settle"}]})
    content = fb.import_content("e@x.com", "Othon")
    assert content["greeting"] == "G"
    assert content["layers"][0]["title"] == "Calm mind"   # warm title from prose
    assert content["layers"][0]["meaning"] == "settle"
    assert content["layers"][0]["remedy"] == "Neuroprotect"  # structural fields from FMP


def test_import_content_none_when_no_fmp(monkeypatch):
    from dashboard import fmp_biofield as fb
    monkeypatch.setattr(fb, "fetch_causal_chain", lambda e: [])
    assert fb.import_content("nobody@x.com", "X") is None
