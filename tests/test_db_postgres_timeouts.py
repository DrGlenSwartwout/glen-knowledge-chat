def test_postgres_checkout_and_queries_share_caller_timeout(monkeypatch):
    from dashboard import db

    class Cursor:
        def __init__(self):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql, params=None):
            self.calls.append((sql, params))

    class Raw:
        def __init__(self):
            self.cursor_instance = Cursor()
            self.committed = False

        def cursor(self):
            return self.cursor_instance

        def commit(self):
            self.committed = True

    class Pool:
        def __init__(self):
            self.raw = Raw()
            self.checkout_timeout = None
            self.returned = []

        def getconn(self, *, timeout):
            self.checkout_timeout = timeout
            return self.raw

        def putconn(self, raw):
            self.returned.append(raw)

    pool = Pool()
    monkeypatch.setenv("PG_DSN", "postgresql://example.invalid/test")
    monkeypatch.setattr(db, "_get_pg_pool", lambda _dsn, _timeout: pool)
    monkeypatch.setattr(db, "_ensure_pg_schema",
                        lambda _raw, _dsn, _schema: None)

    cx = db._connect_postgres("chat_log.db", timeout=2.5)
    try:
        assert pool.checkout_timeout == 2.5
        assert pool.raw.committed is True
        calls = pool.raw.cursor_instance.calls
        assert (
            "SELECT set_config('statement_timeout', %s, false)",
            ("2500ms",),
        ) in calls
        assert (
            "SELECT set_config('lock_timeout', %s, false)",
            ("2500ms",),
        ) in calls
    finally:
        cx.close()

    assert pool.returned == [pool.raw]
