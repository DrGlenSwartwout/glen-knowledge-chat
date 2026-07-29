"""A sqlite3 connection wrapper that reproduces Postgres' ABORTED-TRANSACTION state.

Why this exists: every `except Exception:` in the cart code that "self-heals" by
re-reading after a failed write is a no-op on Postgres unless it rolls back first.
psycopg leaves the transaction in an aborted state after a failed statement, so the
very next statement raises `InFailedSqlTransaction` -- the recovery read included.
sqlite3 does NOT behave this way: the connection stays perfectly usable, so a
SQLite-only test suite (this repo has no Postgres lane) proves those recovery paths
work when in production they raise.

This wrapper closes that gap without a database: once `abort()` is called (or a
statement matching `fail_on` is executed), every subsequent `execute` raises until
`rollback()` is called, exactly as psycopg does.
"""


class AbortedTransaction(Exception):
    """Stand-in for psycopg.errors.InFailedSqlTransaction."""


class AbortingConn:
    """Wraps a real sqlite3 connection and simulates psycopg's aborted transaction.

    `fail_on`: a substring; the first `execute` whose SQL contains it runs
    `on_fail(raw_connection)` (to stage whatever the racing request "won" with),
    then aborts the transaction and raises. Every later statement raises
    `AbortedTransaction` until `rollback()` clears the flag.
    """

    def __init__(self, raw, fail_on=None, on_fail=None):
        self._raw = raw
        self._fail_on = fail_on
        self._on_fail = on_fail
        self._fired = False
        self.aborted = False
        self.rollbacks = 0

    # -- the abort machinery -------------------------------------------------
    def abort(self):
        self.aborted = True

    def execute(self, sql, params=()):
        if self.aborted:
            raise AbortedTransaction(
                "current transaction is aborted, commands ignored until end of "
                "transaction block")
        if self._fail_on and not self._fired and self._fail_on in sql:
            self._fired = True
            if self._on_fail is not None:
                self._on_fail(self._raw)
            self.aborted = True
            raise AbortedTransaction("simulated unique violation on: %s" % sql[:40])
        return self._raw.execute(sql, params)

    def rollback(self):
        self.rollbacks += 1
        self.aborted = False
        return self._raw.rollback()

    def commit(self):
        if self.aborted:
            raise AbortedTransaction("cannot commit an aborted transaction")
        return self._raw.commit()

    # -- transparent pass-through -------------------------------------------
    def __getattr__(self, name):
        return getattr(self._raw, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # sqlite3's context manager commits on success / rolls back on error. An
        # aborted transaction can do neither, so swallow the commit attempt the way
        # a real driver's error handling would.
        if exc_type is None and not self.aborted:
            self._raw.commit()
        else:
            try:
                self._raw.rollback()
            except Exception:
                pass
        return False
