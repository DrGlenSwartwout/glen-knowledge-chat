import sqlite3
from datetime import datetime, timedelta, timezone
from dashboard.chat_limits import client_ip, VelocityLimiter, LIMITS, tier_for, monthly_full_words, is_flagged

def test_client_ip_takes_first_xff_hop():
    assert client_ip("1.2.3.4, 5.6.7.8", "9.9.9.9") == "1.2.3.4"

def test_client_ip_falls_back_to_remote_addr():
    assert client_ip("", "9.9.9.9") == "9.9.9.9"

def test_client_ip_ipv6_normalized_to_64():
    # two addresses in the same /64 collapse to one key
    a = client_ip("2001:db8:abcd:1234:1::1", "")
    b = client_ip("2001:db8:abcd:1234:ffff::ff", "")
    assert a == b == "2001:db8:abcd:1234::/64"

def test_velocity_allows_under_limit():
    t = [1000.0]
    v = VelocityLimiter(clock=lambda: t[0])
    for _ in range(5):
        allowed, _ = v.check("ip", per_min=10, per_day=40)
        assert allowed

def test_velocity_blocks_over_per_minute():
    t = [1000.0]
    v = VelocityLimiter(clock=lambda: t[0])
    for _ in range(10):
        assert v.check("ip", 10, 40)[0]
    allowed, retry = v.check("ip", 10, 40)
    assert not allowed and retry > 0

def test_velocity_recovers_after_window():
    t = [1000.0]
    v = VelocityLimiter(clock=lambda: t[0])
    for _ in range(10):
        v.check("ip", 10, 40)
    t[0] += 61  # past the per-minute window
    assert v.check("ip", 10, 40)[0]

def test_limits_has_three_tiers():
    assert set(LIMITS) == {"anonymous", "registered", "member"}
    assert LIMITS["anonymous"]["per_min"] == 10

def test_tier_precedence():
    assert tier_for(True, True, True) == "member"
    assert tier_for(True, False, True) == "registered"
    assert tier_for(False, False, True) == "registered"   # email alone = registered
    assert tier_for(False, False, False) == "anonymous"

def _db(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    cx.execute("CREATE TABLE query_log (ts TEXT, email TEXT, mode TEXT, word_count INTEGER DEFAULT 0)")
    return cx

def _abuse_db(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "a.db"))
    cx.execute(
        "CREATE TABLE abuse_flags "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, ip TEXT, reason TEXT, ts TEXT)"
    )
    cx.execute("CREATE INDEX IF NOT EXISTS idx_abuse_flags_ts ON abuse_flags(ts)")
    cx.commit()
    return cx

def test_is_flagged_true_for_matching_session(tmp_path):
    cx = _abuse_db(tmp_path)
    now = datetime(2026, 6, 27, 12, 0, 0, tzinfo=timezone.utc)
    recent = (now - timedelta(hours=1)).isoformat()
    cx.execute("INSERT INTO abuse_flags (session_id, ip, reason, ts) VALUES (?,?,?,?)",
               ("sess-abc", "1.2.3.4", "velocity", recent))
    cx.commit()
    assert is_flagged(cx, "sess-abc", "9.9.9.9", now.isoformat()) is True

def test_is_flagged_true_for_matching_ip(tmp_path):
    cx = _abuse_db(tmp_path)
    now = datetime(2026, 6, 27, 12, 0, 0, tzinfo=timezone.utc)
    recent = (now - timedelta(hours=2)).isoformat()
    cx.execute("INSERT INTO abuse_flags (session_id, ip, reason, ts) VALUES (?,?,?,?)",
               ("sess-xyz", "5.5.5.5", "velocity", recent))
    cx.commit()
    assert is_flagged(cx, "sess-unrelated", "5.5.5.5", now.isoformat()) is True

def test_is_flagged_false_for_unrelated_session_and_ip(tmp_path):
    cx = _abuse_db(tmp_path)
    now = datetime(2026, 6, 27, 12, 0, 0, tzinfo=timezone.utc)
    recent = (now - timedelta(hours=1)).isoformat()
    cx.execute("INSERT INTO abuse_flags (session_id, ip, reason, ts) VALUES (?,?,?,?)",
               ("sess-other", "8.8.8.8", "velocity", recent))
    cx.commit()
    assert is_flagged(cx, "sess-abc", "1.2.3.4", now.isoformat()) is False

def test_is_flagged_false_for_old_row(tmp_path):
    cx = _abuse_db(tmp_path)
    now = datetime(2026, 6, 27, 12, 0, 0, tzinfo=timezone.utc)
    old = (now - timedelta(hours=25)).isoformat()
    cx.execute("INSERT INTO abuse_flags (session_id, ip, reason, ts) VALUES (?,?,?,?)",
               ("sess-abc", "1.2.3.4", "velocity", old))
    cx.commit()
    assert is_flagged(cx, "sess-abc", "1.2.3.4", now.isoformat()) is False

def test_monthly_full_words_sums_only_full_in_window(tmp_path):
    cx = _db(tmp_path)
    now = datetime(2026, 6, 27, tzinfo=timezone.utc)
    recent = now.isoformat(); old = (now - timedelta(days=40)).isoformat()
    rows = [(recent,"a@x","full",300),(recent,"a@x","brief",999),
            (recent,"a@x","full",200),(old,"a@x","full",500),(recent,"b@x","full",111)]
    cx.executemany("INSERT INTO query_log VALUES (?,?,?,?)", rows); cx.commit()
    assert monthly_full_words(cx, "a@x", now.isoformat()) == 500  # 300+200 only
