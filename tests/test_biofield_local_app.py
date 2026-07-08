"""Smoke tests for the local Biofield Analysis viewer (standalone Flask app)."""
import json
import sqlite3

import pytest
from biofield_local_app import create_app
from dashboard.biofield_stress import add_voice_stress, init_stress_tables


@pytest.fixture(autouse=True)
def _no_console_gate(monkeypatch):
    # Functional tests run without the console key (the gate is tested separately).
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)


def _seed(db):
    cx = sqlite3.connect(db)
    cx.executescript("""
      CREATE TABLE fmp_snap_clients (id_pk TEXT, email TEXT, name_first TEXT, name_last TEXT);
      CREATE TABLE fmp_snap_client_biofield_test (id_pk TEXT, id_fk_client TEXT, date_test TEXT);
      CREATE TABLE fmp_snap_client_active_main_stress (id_pk TEXT, layer TEXT);
      CREATE TABLE fmp_snap_client_causal_chain
        (id_pk TEXT, id_fk_test TEXT, id_fk_active_stress TEXT, head_chain TEXT, most_affected TEXT);
      CREATE TABLE fmp_snap_client_remedy
        (id_fk_causal_chain TEXT, remedy TEXT, dosage TEXT, frequency TEXT, timing TEXT);
    """)
    cx.execute("INSERT INTO fmp_snap_clients VALUES ('5','lz@x.com','Lewis','Zardo')")
    cx.execute("INSERT INTO fmp_snap_client_biofield_test VALUES ('10','5','2026-06-01')")
    cx.execute("INSERT INTO fmp_snap_client_active_main_stress VALUES ('100','1')")
    cx.execute("INSERT INTO fmp_snap_client_causal_chain VALUES ('200','10','100','Acid','Liver')")
    cx.execute("INSERT INTO fmp_snap_client_remedy VALUES ('200','Sterol Max','3 caps','daily','with food')")
    cx.commit()


def test_console_key_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("CONSOLE_SECRET", "k")
    db = str(tmp_path / "chat_log.db")
    _seed(db)
    client = create_app(db).test_client()
    assert client.get("/").status_code == 401            # blocked without the key
    assert client.get("/?key=k").status_code == 200      # the launcher's ?key= unlocks


def test_index_works_without_snapshot(tmp_path):
    # fresh machine: no fmp_snap_* tables -> home page still renders (no 500)
    db = str(tmp_path / "chat_log.db")
    r = create_app(db).test_client().get("/")
    assert r.status_code == 200


def test_index_lists_tests(tmp_path):
    db = str(tmp_path / "chat_log.db")
    _seed(db)
    client = create_app(db).test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert b"Lewis Zardo" in r.data and b"/test/10" in r.data


def test_report_page_renders(tmp_path):
    db = str(tmp_path / "chat_log.db")
    _seed(db)
    client = create_app(db).test_client()
    r = client.get("/test/10")
    assert r.status_code == 200
    assert b"Sterol Max" in r.data and b"Causal Chain Report" in r.data


def test_notes_save_generate_and_show(tmp_path):
    db = str(tmp_path / "chat_log.db")
    _seed(db)
    seen = {}

    def fake(system, user):
        seen["user"] = user
        return "Aloha Lewis,\n\nYour body identified the right starting points."

    client = create_app(db, complete=fake).test_client()
    assert client.post("/test/10/notes", json={"notes": "mercury hx"}).status_code == 200
    assert b"mercury hx" in client.get("/test/10").data            # notes prefilled
    g = client.post("/test/10/generate", json={"notes": "mercury hx"}).get_json()
    assert g["narrative"].startswith("Aloha Lewis")
    assert "mercury hx" in seen["user"]                            # notes reached the model
    assert b"Aloha Lewis" in client.get("/test/10").data           # saved + shown


def test_video_script_generate_and_audio(tmp_path):
    db = str(tmp_path / "chat_log.db")
    _seed(db)

    def fake_llm(system, user):
        return "Aloha Lewis, here is the short walkthrough."

    def fake_tts(text):
        return b"ID3" + text.encode("utf-8")[:8]   # pretend mp3 bytes

    client = create_app(db, complete=fake_llm, tts=fake_tts).test_client()
    g = client.post("/test/10/video-generate", json={"notes": ""}).get_json()
    assert g["script"].startswith("Aloha Lewis")
    assert b"short walkthrough" in client.get("/test/10").data       # saved + shown
    a = client.post("/test/10/audio", json={}).get_json()
    assert a["url"] == "/audio/test_10.mp3" and a["bytes"] > 0
    served = client.get("/audio/test_10.mp3")
    assert served.status_code == 200 and served.data.startswith(b"ID3")


def test_authoring_flow(tmp_path):
    db = str(tmp_path / "chat_log.db")
    _seed(db)
    client = create_app(db).test_client()
    r = client.post("/author/new")
    assert r.status_code in (302, 308)
    tid = r.headers["Location"].rstrip("/").rsplit("/", 1)[-1]
    assert tid.startswith("a")
    assert client.get("/author/" + tid).status_code == 200
    assert client.post("/author/" + tid + "/header",
                       json={"name": "Jane Doe", "email": "j@x.com", "date": "2026-06-23"}
                       ).status_code == 200
    rid = client.post("/author/" + tid + "/row",
                      json={"layer": "1", "head": "Acid", "most_affected": "Liver",
                            "remedy": "Sterol Max", "dosage": "3 caps",
                            "frequency": "daily", "timing": "with food"}).get_json()["rid"]
    rep = client.get("/test/" + tid)
    assert rep.status_code == 200 and b"Sterol Max" in rep.data and b"Jane Doe" in rep.data
    assert client.post(f"/author/{tid}/row/{rid}", json={"remedy": "Sterol Max XR"}).status_code == 200
    assert b"Sterol Max XR" in client.get("/test/" + tid).data
    assert client.post(f"/author/{tid}/row/{rid}/delete", json={}).status_code == 200
    cat = client.get("/api/catalog?q=x")
    assert cat.status_code == 200 and "catalog" in cat.get_json()


def test_deepgram_token_endpoint(tmp_path):
    db = str(tmp_path / "chat_log.db")
    _seed(db)
    client = create_app(db, deepgram_token=lambda: "tok-123").test_client()
    r = client.get("/api/deepgram-token")
    assert r.status_code == 200 and r.get_json()["key"] == "tok-123"


def test_session_transcript_saves_to_notes(tmp_path):
    db = str(tmp_path / "chat_log.db")
    _seed(db)
    client = create_app(db).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").rsplit("/", 1)[-1]
    r = client.post(f"/author/{tid}/session", json={"transcript": "BSI 21 phase 2 toxicity"})
    assert r.status_code == 200 and r.get_json()["ok"]
    assert b"BSI 21 phase 2 toxicity" in client.get("/test/" + tid).data  # -> notes -> narrative


def test_interpret_fills_chain_rows_from_transcript(tmp_path):
    db = str(tmp_path / "chat_log.db")
    _seed(db)

    def fake(system, user):
        return json.dumps({"header": "BSI 21x1/1x1; phase 2 toxicity", "layers": [
            {"layer": 1, "head": "Large Intestine Meridian",
             "most_affected": "Large Intestine Meridian", "remedy": "Microbiome"},
            {"layer": 2, "head": "Toxicity", "most_affected": "Toxicity",
             "remedy": "Neuro-Magnesium"}]})

    client = create_app(db, interpret_complete=fake).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").rsplit("/", 1)[-1]
    client.post(f"/author/{tid}/session", json={"transcript": "large intestine meridian head and tail balanced by microbiome"})
    r = client.post(f"/author/{tid}/interpret", json={}).get_json()
    assert r["added"] == 2
    report = client.get("/test/" + tid).data
    assert b"Microbiome" in report and b"Neuro-Magnesium" in report
    assert b"Large Intestine Meridian" in report


def test_delete_confirm_and_unconfirmed_from_interpret(tmp_path):
    db = str(tmp_path / "chat_log.db")
    _seed(db)

    def fake(system, user):
        return json.dumps({"layers": [
            {"layer": 1, "head": "Acid", "most_affected": "Liver", "remedy": "Sterol Max"}]})

    client = create_app(db, interpret_complete=fake).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").rsplit("/", 1)[-1]
    client.post(f"/author/{tid}/session", json={"transcript": "acid balanced by sterol max"})
    client.post(f"/author/{tid}/interpret", json={})
    assert b"rline unconf" in client.get("/author/" + tid).data        # voice rows highlighted
    assert client.post(f"/author/{tid}/confirm-all", json={}).status_code == 200
    assert b"rline unconf" not in client.get("/author/" + tid).data    # confirmed -> no highlight
    assert client.post(f"/author/{tid}/delete", json={}).status_code == 200
    assert ("/author/" + tid).encode() not in client.get("/").data     # gone from the list


def test_depth_match_flagged_in_report(tmp_path):
    db = str(tmp_path / "chat_log.db")
    _seed(db)
    client = create_app(db).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").rsplit("/", 1)[-1]
    rid = client.post(f"/author/{tid}/row",
                      json={"layer": "1", "head": "Mercury", "most_affected": "Brain",
                            "remedy": "Gut Binder", "dosage": "1", "frequency": "daily",
                            "timing": "with food"}).get_json()["rid"]
    client.post(f"/author/{tid}/depth", json={"rid": rid, "side": "stress", "rank": 5})
    client.post(f"/author/{tid}/depth", json={"rid": rid, "side": "remedy", "rank": 1})
    assert b"may not reach" in client.get("/test/" + tid).data
    # Depth selects are present per remedy line (hidden by default, shown via toggle).
    ed = client.get("/author/" + tid)
    assert b"Nucleoplasm" in ed.data and b"class=dcol" in ed.data and b"depthbtn" in ed.data


def test_report_view_shows_voice_stress_balanced_via_head_match(tmp_path):
    """Regression: report route must pass chain_rows (not bare remedy names) to
    list_stresses so the label-match path (head -> label) fires for voice stresses.
    Before the fix the voice stress appeared Active in the printed report even though
    the author/edit view correctly showed it Balanced."""
    db = str(tmp_path / "chat_log.db")
    _seed(db)
    client = create_app(db).test_client()
    # create an authored test with a chain row whose head matches a voice stress label
    tid = client.post("/author/new").headers["Location"].rstrip("/").rsplit("/", 1)[-1]
    client.post(f"/author/{tid}/row",
                json={"layer": "1", "head": "Acid", "most_affected": "Liver",
                      "remedy": "Sterol Max", "dosage": "3 caps",
                      "frequency": "daily", "timing": "with food"})
    # add a voice stress whose label matches the chain row head ("Acid")
    cx = sqlite3.connect(db)
    init_stress_tables(cx)
    add_voice_stress(cx, tid, "Acid")
    cx.close()
    # the report view must show the voice stress in the "Stresses balanced" section
    html = client.get("/test/" + tid).data.decode()
    assert "Stresses balanced" in html
    assert "Acid" in html.split("Stresses balanced")[1]


def test_add_one_from_remedy_set_sets_driver_head_full_tail_name_and_dosing(tmp_path):
    # Adding a remedy from the Minimal Remedy Set as a new layer must:
    #  - set the HEAD to a single functional term (the root 'Driver'-level stressor
    #    among those covered), while the TAIL keeps the full covered-stressor list,
    #  - store the CANONICAL catalog name (the set/coverage map holds a lowercased
    #    synthesis name like 'heart health' -> must become 'Heart Health'), and
    #  - auto-fill dosing from the FF, like scan-imported layers.
    from dashboard.biofield_authoring import init_auth_tables, create_test
    from dashboard import biofield_stress as st
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    init_auth_tables(cx)
    cx.execute("CREATE TABLE fmp_snap_products(id_pk TEXT,product_name TEXT,dosage TEXT,dosage_freq TEXT,dosage_timing TEXT)")
    cx.execute("INSERT INTO fmp_snap_products VALUES('1','Heart Health','1 capsule','daily','between meals')")
    tid = create_test(cx, "Test Pt", "t@x.com", "2026-07-08")   # -> "a1"
    tnum = tid.lstrip("a")
    st.seed_from_scan(cx, tnum,
                      [{"code": "ED5", "name": "Circulation Driver"},
                       {"code": "EI2", "name": "Heart – Lung Integrator"}],
                      {"heart health": ["ED5", "EI2"]})         # coverage keyed by the lowercased name
    st.save_remedy_set(cx, tnum, ["heart health"])
    cx.commit()
    client = create_app(db).test_client()
    r = client.post("/author/%s/remedy-set/add-one" % tid, json={"remedy": "heart health"})
    assert r.status_code == 200 and r.get_json()["added"] == 1
    head, tail, remedy, dosage, freq, timing = cx.execute(
        "SELECT head, most_affected, remedy, dosage, frequency, timing "
        "FROM biofield_auth_chain WHERE test_id=? AND layer=1", (int(tnum),)).fetchone()
    cx.close()
    assert head == "Circulation Driver"                        # single root driver stressor
    assert set(tail.split(", ")) == {"Circulation Driver", "Heart – Lung Integrator"}  # full list
    assert (remedy, dosage, freq, timing) == ("Heart Health", "1 capsule", "daily", "between meals")


def test_invoice_route_returns_order_id_and_links(tmp_path):
    # Slice 1 of the invoice-access panel: /author/<id>/invoice returns order_id +
    # print_url (the /invoice/<token> view/print/PDF link) so the panel can offer
    # Open-invoice and Edit-in-Orders actions.
    from dashboard.biofield_authoring import init_auth_tables, create_test, add_chain_row
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    init_auth_tables(cx)
    tid = create_test(cx, "Pt", "pt@x.com", "2026-07-08")
    add_chain_row(cx, tid, 1, "Head", "Tail", "Liver Support", "1 cap", "daily", "with food")
    cx.commit()

    def fake_catalog():
        return [{"name": "Liver Support", "slug": "liver-support"}]

    def fake_create(customer, lines, replace_open=False):
        return {"ok": True, "order_id": 42, "external_ref": "ER42", "total_cents": 30000,
                "accepted_slugs": [l["slug"] for l in lines]}

    def fake_link(order_id):
        return {"ok": True, "print_url": "https://illtowell.com/invoice/tok123?print=1"}

    client = create_app(db, invoice_fetch_catalog=fake_catalog,
                        invoice_create=fake_create, invoice_link=fake_link).test_client()
    j = client.post("/author/%s/invoice" % tid, json={}).get_json()
    assert j["ok"] is True
    assert j["order_id"] == 42
    assert j["print_url"] == "https://illtowell.com/invoice/tok123?print=1"
    assert "orders_url" in j                              # present (empty without CONSOLE_SECRET)


def test_default_orders_link():
    import importlib
    from dashboard import biofield_invoice as bi
    import os
    old = {k: os.environ.get(k) for k in ("CONSOLE_SECRET", "PUBLIC_BASE_URL")}
    try:
        os.environ["CONSOLE_SECRET"] = "sekret key"
        os.environ["PUBLIC_BASE_URL"] = "https://illtowell.com/"
        url = bi.default_orders_link(42)
        assert url == "https://illtowell.com/console/orders?order=42&key=sekret%20key"
        assert bi.default_orders_link(None) == ""
        os.environ.pop("CONSOLE_SECRET")
        assert bi.default_orders_link(42) == ""           # unconfigured -> empty
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_per_client_tabs_and_invoice_view_page(tmp_path):
    # The per-client tab strip (Edit / Report / Invoice / Portal) appears on the
    # editor and report pages, and the Invoice tab has its own page hosting the
    # fee/invoice panel.
    from dashboard.biofield_authoring import init_auth_tables, create_test
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    init_auth_tables(cx)
    tid = create_test(cx, "Bob", "bob@x.com", "2026-07-08")
    cx.commit()
    client = create_app(db).test_client()

    ed = client.get("/author/" + tid).data.decode()
    assert ("/author/" + tid + "/invoice-view") in ed       # Invoice tab target
    for lbl in (">Edit<", ">Report<", ">Invoice<", ">Portal<"):
        assert lbl in ed

    rep = client.get("/test/" + tid).data.decode()
    assert ("/author/" + tid + "/invoice-view") in rep      # tabs on the report page too

    inv = client.get("/author/" + tid + "/invoice-view")
    assert inv.status_code == 200
    body = inv.data.decode()
    assert "wfnav" in body                                   # per-client tab strip present
    assert ("Invoice &mdash; Bob" in body or "Invoice — Bob" in body)  # invoice page heading
    assert "feepanel" in body                                # hosts the fee/invoice panel


def test_invoice_publish_route_requires_order_id(tmp_path):
    # The local Publish button posts {order_id}; the route validates + delegates to
    # the prod publish endpoint. Without an order_id it 400s; with no console config
    # the delegated call reports unavailable (never crashes).
    from dashboard.biofield_authoring import init_auth_tables, create_test
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    init_auth_tables(cx)
    tid = create_test(cx, "Pt", "pt@x.com", "2026-07-08")
    cx.commit()
    client = create_app(db).test_client()
    r = client.post("/author/%s/invoice/publish" % tid, json={})
    assert r.status_code == 400 and r.get_json()["ok"] is False
    r2 = client.post("/author/%s/invoice/publish" % tid, json={"order_id": 42})
    assert r2.get_json()["ok"] is False        # no CONSOLE_SECRET -> unavailable, not a crash


def test_default_publish_invoice_unconfigured():
    import os
    from dashboard import biofield_invoice as bi
    old = os.environ.pop("CONSOLE_SECRET", None)
    try:
        assert bi.default_publish_invoice(42)["ok"] is False
        assert bi.default_publish_invoice(None)["ok"] is False
    finally:
        if old is not None:
            os.environ["CONSOLE_SECRET"] = old


def test_invoice_view_shows_client_options_reference(tmp_path):
    # Decision #3: the Invoice page also shows a secondary "Client options & pricing"
    # reference card (analysis price data-sourced from fee_state; $300 standard here).
    from dashboard.biofield_authoring import init_auth_tables, create_test
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    init_auth_tables(cx)
    tid = create_test(cx, "Ref Pt", "ref@x.com", "2026-07-08")
    cx.commit()
    body = create_app(db).test_client().get("/author/%s/invoice-view" % tid).data.decode()
    assert "Client options &amp; pricing" in body
    assert "$300" in body and "$997" in body          # data-sourced standard + value
    assert "No subscription" in body


def test_build_portal_seed_from_authored_flat_format():
    # Slice 1 (Hand off to Rae): build a portal-seed from the authored chain in the
    # composer's FLAT format (remedy=name string, title=head) — the automation of the
    # by-hand seed-build, so no nested-remedy "[object Object]" and no stale content.
    import sqlite3
    from dashboard.biofield_authoring import init_auth_tables, create_test, add_chain_row
    from dashboard import biofield_handoff
    cx = sqlite3.connect(":memory:")
    init_auth_tables(cx)
    tid = create_test(cx, "Bob Ross", "bob@x.com", "2026-07-08")
    add_chain_row(cx, tid, 1, "Liver Meridian", "Liver", "Liver Support", "1 capsule", "daily", "with food")
    add_chain_row(cx, tid, 2, "Mercury", "Brain", "Glutathione Syntropy", "1 capsule", "", "")
    cx.commit()
    slugs = {"Liver Support": "liver-support", "Glutathione Syntropy": "glutathione-syntropy"}
    content = biofield_handoff.build_portal_seed(cx, tid, lambda nm: slugs.get(nm), name="Bob Ross")
    assert content["greeting"].startswith("Aloha Bob")
    L = content["layers"]
    assert [x["remedy"] for x in L] == ["Liver Support", "Glutathione Syntropy"]
    assert L[0]["title"] == "Liver Meridian" and isinstance(L[0]["remedy"], str)   # flat, not object
    assert L[0]["dosing"] == "1 capsule daily with food"
    assert content["reorder_items"] == [
        {"slug": "liver-support", "name": "Liver Support"},
        {"slug": "glutathione-syntropy", "name": "Glutathione Syntropy"}]


def test_handoff_route_graceful(tmp_path):
    from dashboard.biofield_authoring import init_auth_tables, create_test, add_chain_row
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    init_auth_tables(cx)
    tid = create_test(cx, "Pt", "pt@x.com", "2026-07-08")
    add_chain_row(cx, tid, 1, "Head", "Tail", "Liver Support", "1 cap", "daily", "")
    cx.commit()
    client = create_app(db, invoice_fetch_catalog=lambda: [{"name": "Liver Support", "slug": "liver-support"}]).test_client()
    j = client.post("/author/%s/handoff" % tid, json={}).get_json()
    assert j["ok"] is False        # no CONSOLE_SECRET in test -> push reports failure, no crash
    # a test with no authored layers -> 400 before any push
    tid2 = create_test(cx, "Empty", "empty@x.com", "2026-07-08"); cx.commit()
    r2 = client.post("/author/%s/handoff" % tid2, json={})
    assert r2.status_code == 400


def test_report_remedies_for_invoice_qty():
    # qty = bottles for a 30-day program from the authored frequency + FMP doses/bottle.
    from dashboard import biofield_handoff, biofield_invoice
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE fmp_snap_products (product_name TEXT, doses_per_bottle INTEGER)")
    cx.execute("INSERT INTO fmp_snap_products VALUES ('Liver Support', 30)")  # 2/day*30/30 = 2
    cx.commit()
    # sqlite in-memory can't be reopened by path; use a temp file instead
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".db"); os.close(fd)
    dcx = sqlite3.connect(path)
    dcx.execute("CREATE TABLE fmp_snap_products (product_name TEXT, doses_per_bottle INTEGER)")
    dcx.execute("INSERT INTO fmp_snap_products VALUES ('Liver Support', 30)")
    dcx.commit(); dcx.close()
    rep = {"layers": [
        {"remedy": "Liver Support", "frequency": "twice a day"},   # 2*30/30 = 2 bottles
        {"remedy": "Infoceutical X", "frequency": "daily"},        # no FMP row -> qty 1
        {"remedy": "", "frequency": "daily"},                      # skipped (no name)
    ]}
    out = biofield_handoff.report_remedies_for_invoice(path, rep, biofield_invoice.bottles_needed)
    os.unlink(path)
    assert out == [{"name": "Liver Support", "qty": 2}, {"name": "Infoceutical X", "qty": 1}]


def test_build_invoice_lines_include_fee_false():
    # A client who already PAID the analysis: invoice remedies only (no fee line).
    from dashboard import biofield_invoice
    cat = [{"name": "Liver Support", "slug": "liver-support"}]
    built = biofield_invoice.build_invoice_lines(
        {"email": "x@x.com"}, [{"name": "Liver Support", "qty": 2}], cat, include_fee=False)
    assert built["lines"] == [{"slug": "liver-support", "qty": 2}]     # no biofield-analysis
    # no remedies + no fee -> empty (caller skips the raise)
    assert biofield_invoice.build_invoice_lines({}, [], cat, include_fee=False)["lines"] == []


def test_handoff_skips_raise_when_analysis_paid(tmp_path, monkeypatch):
    # Pre-paid analysis, no remedies (like Steve Fox): handoff pushes the analysis but
    # raises NO invoice, reporting already_paid + the paid order id.
    from dashboard.biofield_authoring import init_auth_tables, create_test, add_chain_row
    from dashboard import biofield_invoice
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    init_auth_tables(cx)
    tid = create_test(cx, "Steve Fox", "sf@x.com", "2026-07-08")
    add_chain_row(cx, tid, 1, "Head", "Tail", "Liver Support", "1 cap", "daily", "")
    cx.commit()
    monkeypatch.setattr(biofield_invoice, "default_handoff_push", lambda *a, **k: {"ok": True})
    created_calls = []
    client = create_app(
        db,
        invoice_fetch_catalog=lambda: [],   # remedy doesn't resolve -> skipped, like Steve's $300-only
        invoice_create=lambda c, lines, replace_open=False: created_calls.append(lines) or {"ok": True, "order_id": 9},
        invoice_paid_check=lambda email: {"paid": True, "order_id": 37},
    ).test_client()
    j = client.post("/author/%s/handoff" % tid, json={}).get_json()
    assert j["ok"] is True
    assert j["invoice"]["ok"] is False and j["invoice"]["already_paid"] is True
    assert j["invoice"]["order_id"] == 37
    assert created_calls == []        # no order raised for a paid analysis-only intake


def test_handoff_raises_remedies_only_when_paid(tmp_path, monkeypatch):
    # Pre-paid analysis WITH remedies: raise a remedy-only invoice (fee line dropped).
    from dashboard.biofield_authoring import init_auth_tables, create_test, add_chain_row
    from dashboard import biofield_invoice
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    init_auth_tables(cx)
    cx.execute("CREATE TABLE fmp_snap_products (product_name TEXT, doses_per_bottle INTEGER)")
    tid = create_test(cx, "Paid Pat", "pp@x.com", "2026-07-08")
    add_chain_row(cx, tid, 1, "Head", "Tail", "Liver Support", "1 cap", "daily", "")
    cx.commit()
    monkeypatch.setattr(biofield_invoice, "default_handoff_push", lambda *a, **k: {"ok": True})
    captured = {}
    client = create_app(
        db,
        invoice_fetch_catalog=lambda: [{"name": "Liver Support", "slug": "liver-support"}],
        invoice_create=lambda c, lines, replace_open=False: captured.update(lines=lines) or {"ok": True, "order_id": 5, "total_cents": 3000},
        invoice_paid_check=lambda email: {"paid": True, "order_id": 37},
    ).test_client()
    j = client.post("/author/%s/handoff" % tid, json={}).get_json()
    assert j["invoice"]["ok"] is True and j["invoice"]["already_paid"] is True
    slugs = [l["slug"] for l in captured["lines"]]
    assert "biofield-analysis" not in slugs and "liver-support" in slugs   # fee dropped


def test_handoff_route_raises_invoice(tmp_path, monkeypatch):
    # With the portal push succeeding, the handoff ALSO raises the invoice (proposed
    # order) from the authored remedies + the Biofield Analysis fee.
    from dashboard.biofield_authoring import init_auth_tables, create_test, add_chain_row
    from dashboard import biofield_invoice
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    init_auth_tables(cx)
    cx.execute("CREATE TABLE fmp_snap_products (product_name TEXT, doses_per_bottle INTEGER)")
    cx.execute("INSERT INTO fmp_snap_products VALUES ('Liver Support', 30)")
    tid = create_test(cx, "Pt", "pt@x.com", "2026-07-08")
    add_chain_row(cx, tid, 1, "Head", "Tail", "Liver Support", "1 cap", "twice a day", "")
    cx.commit()
    # portal push succeeds (bypass prod); capture the order lines the raise sends
    monkeypatch.setattr(biofield_invoice, "default_handoff_push", lambda *a, **k: {"ok": True})
    captured = {}
    def fake_create(cust, lines, replace_open=False):
        captured["lines"] = lines
        return {"ok": True, "order_id": 77, "total_cents": 130000, "external_ref": "INH-x"}
    client = create_app(
        db,
        invoice_fetch_catalog=lambda: [{"name": "Liver Support", "slug": "liver-support"}],
        invoice_create=fake_create,
    ).test_client()
    j = client.post("/author/%s/handoff" % tid, json={}).get_json()
    assert j["ok"] is True
    assert j["invoice"]["ok"] is True and j["invoice"]["order_id"] == 77
    slugs = [l["slug"] for l in captured["lines"]]
    assert slugs[0] == "biofield-analysis"           # fee always first
    assert {"slug": "liver-support", "qty": 2} in captured["lines"]   # 2 bottles for twice-daily
