"""Smoke tests for the local Biofield Analysis viewer (standalone Flask app)."""
import sqlite3
from biofield_local_app import create_app


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
