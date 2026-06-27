import sqlite3
from datetime import datetime, timedelta, timezone
from dashboard.chat_limits import client_ip, VelocityLimiter, LIMITS, tier_for, monthly_full_words

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

def test_monthly_full_words_sums_only_full_in_window(tmp_path):
    cx = _db(tmp_path)
    now = datetime(2026, 6, 27, tzinfo=timezone.utc)
    recent = now.isoformat(); old = (now - timedelta(days=40)).isoformat()
    rows = [(recent,"a@x","full",300),(recent,"a@x","brief",999),
            (recent,"a@x","full",200),(old,"a@x","full",500),(recent,"b@x","full",111)]
    cx.executemany("INSERT INTO query_log VALUES (?,?,?,?)", rows); cx.commit()
    assert monthly_full_words(cx, "a@x", now.isoformat()) == 500  # 300+200 only
