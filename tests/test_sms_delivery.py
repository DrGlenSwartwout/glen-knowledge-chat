import sqlite3
from dashboard import sms_delivery as S


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db")); S.init_table(cx); return cx


def test_record_upserts_status_and_recent(tmp_path):
    cx = _cx(tmp_path)
    S.record(cx, "SM1", "+18085550001", "queued", "")
    S.record(cx, "SM1", "+18085550001", "delivered", "")     # same sid -> updates in place
    S.record(cx, "SM2", "+18085550002", "failed", "30007")
    rows = S.recent(cx, limit=10)
    by = {r["message_sid"]: r for r in rows}
    assert by["SM1"]["status"] == "delivered" and len(rows) == 2
    assert by["SM2"]["status"] == "failed" and by["SM2"]["error_code"] == "30007"


def test_recent_failed_only(tmp_path):
    cx = _cx(tmp_path)
    S.record(cx, "SM1", "+1", "delivered", "")
    S.record(cx, "SM2", "+2", "undelivered", "30008")
    failed = S.recent(cx, limit=10, failed_only=True)
    assert [r["message_sid"] for r in failed] == ["SM2"]
