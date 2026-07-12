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


def _snapshot_conn():
    """In-memory stand-in for the local FMP snapshot tables in chat_log.db,
    seeded to mirror a real client (Desiree, id 4576) with an OLDER test (217)
    and a NEWER test (235). fetch_causal_chain must return only the newest."""
    import sqlite3
    cx = sqlite3.connect(":memory:")
    cx.executescript(
        "CREATE TABLE fmp_clients (id_pk TEXT, email TEXT);"
        "CREATE TABLE fmp_snap_client_biofield_test (id_pk TEXT, id_fk_client TEXT, active TEXT, date_test TEXT);"
        "CREATE TABLE fmp_snap_client_remedy (id_pk TEXT, id_fk_client TEXT, id_fk_test TEXT, "
        "id_fk_causal_chain TEXT, layer TEXT, data_2 TEXT, remedy TEXT, dosage TEXT, zc_dosage_text TEXT);")
    cx.execute("INSERT INTO fmp_clients VALUES ('4576','desireedallaguardia@gmail.com')")
    cx.executemany("INSERT INTO fmp_snap_client_biofield_test (id_pk,id_fk_client,active,date_test) VALUES (?,?,?,?)",
                   [("217", "4576", "Yes", "2026-04-17"), ("235", "4576", "Yes", "2026-05-17")])
    cx.executemany(
        "INSERT INTO fmp_snap_client_remedy "
        "(id_pk,id_fk_client,id_fk_test,id_fk_causal_chain,layer,data_2,remedy,dosage,zc_dosage_text) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        [  # older test 217 — must NOT appear
            ("1445", "4576", "217", "1061", "1", "ED7 Lung Epigenetics", "Focus Neuro-Magnesium Powder", "1 scoop", "1 scoop - 2x/day"),
           # newer test 235 — layer 2 inserted before layer 1 to prove ordering
            ("9002", "4576", "235", "1171", "2", "MB8 Love", "Mauve Mullein Flower Essence", "", "5 drops - nightly"),
            ("9001", "4576", "235", "1170", "1", "ED5 Circulation", "Black Tourmaline", "2 drops", "2 drops - daily")])
    cx.commit()
    return cx


def test_fetch_causal_chain_reads_latest_snapshot_test():
    from dashboard import fmp_biofield as fb
    out = fb.fetch_causal_chain("Desireedallaguardia@GMAIL.com", conn=_snapshot_conn())
    # only the newest test (235), ordered by layer, dosing from composed zc_dosage_text
    assert out == [
        {"layer": "1", "head_chain": "ED5 Circulation", "remedy": "Black Tourmaline", "dosage": "2 drops - daily"},
        {"layer": "2", "head_chain": "MB8 Love", "remedy": "Mauve Mullein Flower Essence", "dosage": "5 drops - nightly"},
    ]
    # the older test's remedy must be excluded
    assert all("Focus Neuro-Magnesium" not in r["remedy"] for r in out)


def test_fetch_causal_chain_feeds_build_layers():
    from dashboard import fmp_biofield as fb
    layers = fb.build_layers(fb.fetch_causal_chain("desireedallaguardia@gmail.com", conn=_snapshot_conn()))
    assert [l["n"] for l in layers] == [1, 2]
    assert layers[0]["remedy"] == "Black Tourmaline"
    assert layers[0]["dosing"] == "2 drops - daily"


def test_fetch_causal_chain_falls_back_to_bare_dosage():
    from dashboard import fmp_biofield as fb
    cx = _snapshot_conn()
    cx.execute("UPDATE fmp_snap_client_remedy SET zc_dosage_text='' WHERE id_pk='9001'")
    cx.commit()
    out = fb.fetch_causal_chain("desireedallaguardia@gmail.com", conn=cx)
    assert out[0]["dosage"] == "2 drops"   # empty composed text -> bare dosage


def test_fetch_causal_chain_empty_and_error_safe():
    from dashboard import fmp_biofield as fb
    assert fb.fetch_causal_chain("nobody@x.com", conn=_snapshot_conn()) == []
    assert fb.fetch_causal_chain("", conn=_snapshot_conn()) == []
    import sqlite3
    bare = sqlite3.connect(":memory:")   # no snapshot tables -> OperationalError, swallowed
    assert fb.fetch_causal_chain("x@y.com", conn=bare) == []


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
