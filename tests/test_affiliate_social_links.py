import sqlite3
from dashboard import affiliate_dashboard as ad


def test_inserts_only_http_and_returns_count():
    cx = sqlite3.connect(":memory:")  # no table pre-created
    n = ad.add_social_links(cx, "amy7", "amy@x.com",
                            ["https://a.com/p", "ftp://nope", "http://b.com/q", "  notaurl "])
    assert n == 2
    rows = cx.execute("SELECT slug, email, url FROM affiliate_social_links ORDER BY id").fetchall()
    assert rows == [("amy7", "amy@x.com", "https://a.com/p"),
                    ("amy7", "amy@x.com", "http://b.com/q")]


def test_caps_at_10():
    cx = sqlite3.connect(":memory:")
    n = ad.add_social_links(cx, "amy7", "amy@x.com", [f"https://a.com/{i}" for i in range(15)])
    assert n == 10


def test_non_list_is_zero():
    cx = sqlite3.connect(":memory:")
    assert ad.add_social_links(cx, "amy7", "amy@x.com", None) == 0
    assert ad.add_social_links(cx, "amy7", "amy@x.com", []) == 0


def test_truncates_to_500():
    cx = sqlite3.connect(":memory:")
    long = "https://a.com/" + ("x" * 600)
    ad.add_social_links(cx, "amy7", "amy@x.com", [long])
    stored = cx.execute("SELECT url FROM affiliate_social_links").fetchone()[0]
    assert len(stored) == 500
