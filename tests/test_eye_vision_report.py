import json
import sqlite3

from dashboard import eye_vision_report as report


def _e4l(tmp_path, primary=1):
    path = tmp_path / "e4l.db"
    cx = sqlite3.connect(path)
    cx.executescript("""
      CREATE TABLE e4l_clients(client_id INTEGER PRIMARY KEY, name TEXT, email TEXT);
      CREATE TABLE e4l_scans(scan_id INTEGER PRIMARY KEY, client_id INTEGER, scan_date TEXT);
      CREATE TABLE e4l_scan_results(id INTEGER PRIMARY KEY, scan_id INTEGER, item_code TEXT, priority_rank INTEGER);
      CREATE TABLE e4l_items(code TEXT PRIMARY KEY, name TEXT, full_name TEXT, e4l_description TEXT, category TEXT);
      CREATE TABLE e4l_pattern_structures(code TEXT, structure TEXT, stype TEXT, is_primary INTEGER, source_phrase TEXT);
    """)
    cx.execute("INSERT INTO e4l_clients VALUES(1,'A','a@example.com')")
    cx.execute("INSERT INTO e4l_scans VALUES(5,1,'2026-07-24')")
    cx.execute("INSERT INTO e4l_scan_results VALUES(1,5,'EI1',1)")
    cx.execute("INSERT INTO e4l_items VALUES('EI1','Eye pattern','Eye Pattern','','EI')")
    cx.execute("INSERT INTO e4l_pattern_structures VALUES('EI1','Eye','organ',?,?)",
               (primary, "Eye"))
    cx.commit()
    cx.close()
    return str(path)


def _portal():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    cx.execute("""CREATE TABLE intake_responses(
      email TEXT PRIMARY KEY, form_version TEXT, status TEXT, answers_json TEXT,
      created_at TEXT, submitted_at TEXT)""")
    return cx


def test_history_opens_by_default_and_saved_preference_wins(tmp_path):
    cx = _portal()
    e4l_path = _e4l(tmp_path)
    cx.execute("INSERT INTO intake_responses VALUES(?,?,?,?,?,?)", (
        "a@example.com", "1", "submitted",
        json.dumps({"diagnoses": [{"diagnosis": "Glaucoma"}]}), "", ""))
    block = report.build_portal_block(
        cx, "a@example.com", "2026-07-25", db_path=e4l_path)
    assert block["open"] is True
    assert block["default_open"] is True
    assert block["findings"][0]["classification"] == "direct_ocular"
    relationship = block["findings"][0]["integrated_relationships"][0]
    assert relationship["label"] == "Eye and visual system"
    assert relationship["eav_points"]
    assert relationship["information_relationships"]
    assert relationship["symptom_context"]
    assert relationship["condition_context"]
    assert "diagnose" in block["findings"][0]["history_context"]
    saved_closed = report.build_portal_block(
        cx, "a@example.com", "2026-07-25", saved_collapsed=True,
        db_path=e4l_path)
    assert saved_closed["open"] is False


def test_no_history_is_closed_and_associated_mapping_is_labeled(tmp_path):
    cx = _portal()
    block = report.build_portal_block(
        cx, "a@example.com", "2026-07-25", db_path=_e4l(tmp_path, primary=0))
    assert block["open"] is False
    assert block["findings"][0]["classification"] == "associated_ocular"


def test_non_ocular_mapping_is_excluded(tmp_path):
    path = _e4l(tmp_path)
    cx_e = sqlite3.connect(path)
    cx_e.execute("UPDATE e4l_pattern_structures SET structure='Liver'")
    cx_e.commit()
    cx_e.close()
    block = report.build_portal_block(
        _portal(), "a@example.com", "2026-07-25", db_path=path)
    assert block["status"] == "no_eye_patterns"
    assert block["findings"] == []
