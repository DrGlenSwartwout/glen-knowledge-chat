"""run_upsert_many: batched write path shared by the bulk farm sources.

run_upsert() opens a connection per row (~14k SSL handshakes for USDA alone,
measured at 1.3 rows/s). run_upsert_many shares ONE connection for the batch
(measured 11.3 rows/s). These tests pin the two properties that matter: exactly
one connection is opened, and country normalization still happens per row."""
import scrapers.practitioner_finder.db as db


class _Cur:
    def __init__(self):
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))


class _Ctx:
    """Counts how many times a cursor/connection is opened."""

    def __init__(self, cur, counter):
        self.cur, self.counter = cur, counter

    def __enter__(self):
        self.counter.append(1)
        return self.cur

    def __exit__(self, *a):
        return False


def _patch(monkeypatch):
    cur, opens = _Cur(), []
    monkeypatch.setattr(db, "supabase_cursor", lambda: _Ctx(cur, opens))
    return cur, opens


def test_upsert_many_opens_one_connection_for_the_batch(monkeypatch):
    cur, opens = _patch(monkeypatch)
    rows = [{"source_url": f"u/{i}", "name": f"F{i}"} for i in range(25)]
    written = db.run_upsert_many(rows)
    assert written == 25
    assert len(cur.executed) == 25      # one statement per row
    assert len(opens) == 1              # ...but only ONE connection


def test_upsert_many_normalizes_country_per_row(monkeypatch):
    cur, _ = _patch(monkeypatch)
    db.run_upsert_many([
        {"source_url": "u/1", "name": "A", "country": "USA"},
        {"source_url": "u/2", "name": "B", "country": "United States"},
    ])
    # Both rows land as the ISO-2 code, same as run_upsert's boundary.
    for _sql, params in cur.executed:
        assert "US" in params


def test_upsert_many_empty_is_a_noop(monkeypatch):
    cur, opens = _patch(monkeypatch)
    assert db.run_upsert_many([]) == 0
    assert not cur.executed
    assert not opens                    # no connection opened at all


def test_upsert_many_matches_run_upsert_sql(monkeypatch):
    # The batch path must emit the SAME SQL as the single-row path.
    cur_many, _ = _patch(monkeypatch)
    row = {"source_url": "u/1", "name": "A", "city": "X"}
    db.run_upsert_many([dict(row)])
    many_sql = cur_many.executed[0][0]

    cur_one, _ = _patch(monkeypatch)
    db.run_upsert(dict(row))
    assert cur_one.executed[0][0] == many_sql
