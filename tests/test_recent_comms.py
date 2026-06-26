import json
import sqlite3
from datetime import datetime, timezone, timedelta
from dashboard.recent_comms import recent_comms


def _db(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    cx.executescript("""
        CREATE TABLE inbound_leads(id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT,
            source TEXT, first_name TEXT, raw_json TEXT);
        CREATE TABLE inquiries(id INTEGER PRIMARY KEY AUTOINCREMENT, client_email TEXT,
            main_challenge TEXT, main_goal TEXT, created_at TEXT);
        CREATE TABLE query_log(id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT,
            query TEXT, ts TEXT);
    """)
    return cx


def _iso(days_ago):
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def test_windows_chat_and_inquiries_keeps_latest_intake(tmp_path):
    cx = _db(tmp_path)
    # intake (old, but always kept)
    quiz = {"data": {"total_score": {"percent": 80},
                     "quiz_questions": [{"question": "Energy?", "answers": [{"answer": "Low"}]}]}}
    cx.execute("INSERT INTO inbound_leads(email,source,first_name,raw_json) VALUES(?,?,?,?)",
               ("j@x.com", "scoreapp", "Jane", json.dumps(quiz)))
    # inquiries: one recent, one old
    cx.execute("INSERT INTO inquiries(client_email,main_challenge,main_goal,created_at) VALUES(?,?,?,?)",
               ("j@x.com", "fatigue", "more energy", _iso(2)))
    cx.execute("INSERT INTO inquiries(client_email,main_challenge,main_goal,created_at) VALUES(?,?,?,?)",
               ("j@x.com", "ancient", "old", _iso(60)))
    # queries: one recent, one old
    cx.execute("INSERT INTO query_log(email,query,ts) VALUES(?,?,?)", ("j@x.com", "why tired", _iso(1)))
    cx.execute("INSERT INTO query_log(email,query,ts) VALUES(?,?,?)", ("j@x.com", "stale q", _iso(30)))
    cx.commit()
    out = recent_comms(cx, "j@x.com", days_window=7)
    assert "Energy?: Low" in out["intake_summary"] and "Jane" in out["intake_summary"]
    assert [i["main_challenge"] for i in out["recent_inquiries"]] == ["fatigue"]   # old excluded
    assert [q["question"] for q in out["recent_queries"]] == ["why tired"]          # old excluded


def test_question_column_fallback(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    cx.executescript("""CREATE TABLE query_log(id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT, question TEXT, ts TEXT);""")
    cx.execute("INSERT INTO query_log(email,question,ts) VALUES(?,?,?)",
               ("j@x.com", "from question col", _iso(1)))
    cx.commit()
    out = recent_comms(cx, "j@x.com")
    assert [q["question"] for q in out["recent_queries"]] == ["from question col"]


def test_empty_email_and_missing_tables(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))   # no tables at all
    assert recent_comms(cx, "") == {"intake_summary": "", "recent_inquiries": [], "recent_queries": [], "recent_feedback": []}
    assert recent_comms(cx, "j@x.com") == {"intake_summary": "", "recent_inquiries": [], "recent_queries": [], "recent_feedback": []}


import json as _json
from datetime import datetime, timezone, timedelta


def _fb_db(tmp_path):
    import sqlite3
    cx = sqlite3.connect(str(tmp_path / "fb.db"))
    cx.executescript("""
        CREATE TABLE users(id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT);
        CREATE TABLE personal_email_feedback(id INTEGER PRIMARY KEY AUTOINCREMENT,
            received_at TEXT, user_id INTEGER, raw_text TEXT, ai_summary TEXT,
            extracted_topics TEXT, extracted_conditions TEXT);
    """)
    cx.execute("INSERT INTO users(id,email) VALUES(1,'j@x.com')")
    cx.execute("INSERT INTO users(id,email) VALUES(2,'other@x.com')")
    return cx


def _ago(days):
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def test_recent_feedback_windowed_joined_parsed(tmp_path):
    cx = _fb_db(tmp_path)
    # in-window row for j@x.com
    cx.execute("INSERT INTO personal_email_feedback(received_at,user_id,raw_text,ai_summary,"
               "extracted_topics,extracted_conditions) VALUES(?,?,?,?,?,?)",
               (_ago(2), 1, "SECRET RAW REPLY", "feels exhausted lately",
                _json.dumps(["sleep", "energy"]), _json.dumps(["adrenal fatigue"])))
    # out-of-window row for j@x.com
    cx.execute("INSERT INTO personal_email_feedback(received_at,user_id,ai_summary) VALUES(?,?,?)",
               (_ago(60), 1, "old feedback"))
    # other user's row
    cx.execute("INSERT INTO personal_email_feedback(received_at,user_id,ai_summary) VALUES(?,?,?)",
               (_ago(1), 2, "not jane's"))
    cx.commit()
    out = recent_comms(cx, "J@X.com", days_window=7)               # case-insensitive join
    fb = out["recent_feedback"]
    assert len(fb) == 1
    assert fb[0]["summary"] == "feels exhausted lately"
    assert fb[0]["topics"] == ["sleep", "energy"]
    assert fb[0]["conditions"] == ["adrenal fatigue"]
    # raw_text must never be exposed
    assert "raw_text" not in fb[0] and "SECRET RAW REPLY" not in str(out)


def test_recent_feedback_bad_json_and_missing_table(tmp_path):
    cx = _fb_db(tmp_path)
    cx.execute("INSERT INTO personal_email_feedback(received_at,user_id,ai_summary,"
               "extracted_topics,extracted_conditions) VALUES(?,?,?,?,?)",
               (_ago(1), 1, "sum", "not json", ""))
    cx.commit()
    out = recent_comms(cx, "j@x.com")
    assert out["recent_feedback"][0]["topics"] == [] and out["recent_feedback"][0]["conditions"] == []
    # missing tables entirely -> empty section, no raise
    import sqlite3
    bare = sqlite3.connect(str(tmp_path / "bare.db"))
    assert recent_comms(bare, "j@x.com")["recent_feedback"] == []


def test_empty_email_includes_feedback_key(tmp_path):
    cx = _fb_db(tmp_path)
    assert recent_comms(cx, "")["recent_feedback"] == []
