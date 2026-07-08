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


def test_add_one_from_remedy_set_populates_head_tail_name_and_dosing(tmp_path):
    # Adding a remedy from the Minimal Remedy Set as a new layer must:
    #  - fill BOTH head and tail (most_affected) with the stresses it covers,
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
    st.seed_from_scan(cx, tnum, [{"code": "ED1", "name": "Membrane"}],
                      {"heart health": ["ED1"]})               # coverage keyed by the lowercased name
    st.save_remedy_set(cx, tnum, ["heart health"])
    cx.commit()
    client = create_app(db).test_client()
    r = client.post("/author/%s/remedy-set/add-one" % tid, json={"remedy": "heart health"})
    assert r.status_code == 200 and r.get_json()["added"] == 1
    row = cx.execute(
        "SELECT head, most_affected, remedy, dosage, frequency, timing "
        "FROM biofield_auth_chain WHERE test_id=? AND layer=1", (int(tnum),)).fetchone()
    cx.close()
    # head+tail = covered stress; name canonicalized; dosing auto-filled from the FF
    assert row == ("Membrane", "Membrane", "Heart Health", "1 capsule", "daily", "between meals")
