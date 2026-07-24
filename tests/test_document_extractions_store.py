import sqlite3
from dashboard import document_extractions as dx


def _cx():
    cx = sqlite3.connect(":memory:")
    dx.init_table(cx)
    return cx


def test_put_draft_round_trips_decoded_payloads():
    cx = _cx()
    eid = dx.put_draft(
        cx, 7, "c@x.com", "You had a panel done.",
        attributes=[{"field": "conditions", "value": "Glaucoma", "source_quote": "dx: glaucoma"}],
        facts=[{"fact_key": "on_areds2", "value": True, "source_quote": "taking AREDS2"}],
        unstructured=[{"label": "HbA1c", "value": "6.4", "source_quote": "HbA1c 6.4"}],
        model="claude-opus-4-8")
    got = dx.get_for_document(cx, 7)
    assert got["id"] == eid
    assert got["status"] == "ai_draft"
    assert got["email"] == "c@x.com"
    assert got["narrative_md"] == "You had a panel done."
    assert got["model"] == "claude-opus-4-8"
    assert got["attributes"][0]["value"] == "Glaucoma"
    assert got["facts"][0]["fact_key"] == "on_areds2"
    assert got["unstructured"][0]["label"] == "HbA1c"
    assert got["reviewed_at"] is None


def test_get_for_document_returns_none_when_absent():
    assert dx.get_for_document(_cx(), 999) is None


def test_confirm_sets_status_narrative_and_reviewer():
    cx = _cx()
    eid = dx.put_draft(cx, 7, "c@x.com", "draft text", [], [], [], "m")
    assert dx.confirm(cx, eid, "edited text", "glen") is True
    got = dx.get_for_document(cx, 7)
    assert got["status"] == "confirmed"
    assert got["narrative_md"] == "edited text"
    assert got["reviewed_by"] == "glen"
    assert got["reviewed_at"]


def test_confirm_is_idempotent_second_call_is_a_noop():
    cx = _cx()
    eid = dx.put_draft(cx, 7, "c@x.com", "draft", [], [], [], "m")
    assert dx.confirm(cx, eid, "first", "glen") is True
    assert dx.confirm(cx, eid, "second", "glen") is False
    assert dx.get_for_document(cx, 7)["narrative_md"] == "first"


def test_reject_sets_status_and_blocks_later_confirm():
    cx = _cx()
    eid = dx.put_draft(cx, 7, "c@x.com", "draft", [], [], [], "m")
    assert dx.reject(cx, eid, "glen") is True
    assert dx.get_for_document(cx, 7)["status"] == "rejected"
    assert dx.confirm(cx, eid, "x", "glen") is False


def test_put_draft_replaces_a_prior_draft_for_the_same_document():
    """Re-extraction must not leave two competing drafts on one document."""
    cx = _cx()
    dx.put_draft(cx, 7, "c@x.com", "old", [], [], [], "m")
    dx.put_draft(cx, 7, "c@x.com", "new", [], [], [], "m")
    got = dx.get_for_document(cx, 7)
    assert got["narrative_md"] == "new"
    rows = cx.execute("SELECT COUNT(*) FROM client_document_extractions "
                      "WHERE document_id=7").fetchone()
    assert rows[0] == 1


def test_put_draft_does_not_touch_a_confirmed_document():
    """A confirmed narrative is live in the client's portal and has no
    history table -- re-extraction must leave it completely alone."""
    cx = _cx()
    eid = dx.put_draft(cx, 7, "c@x.com", "draft", [], [], [], "m")
    dx.confirm(cx, eid, "approved narrative", "glen")

    new_id = dx.put_draft(cx, 7, "c@x.com", "re-extracted text",
                           [{"field": "x"}], [{"fact_key": "y"}],
                           [{"label": "z"}], "m2")

    assert new_id == eid
    got = dx.get_for_document(cx, 7)
    assert got["status"] == "confirmed"
    assert got["narrative_md"] == "approved narrative"
    assert got["reviewed_by"] == "glen"
    assert got["reviewed_at"]
    rows = cx.execute("SELECT COUNT(*) FROM client_document_extractions "
                      "WHERE document_id=7").fetchone()
    assert rows[0] == 1


class _SqlLog:
    """Wraps a connection to record the SQL text of every execute() call, in
    order, so a test can assert on statement SEQUENCING and CONTENT. This is
    how the TOCTOU regression (a status check split into a SELECT that runs
    BEFORE a separate DELETE) can be caught, even though a genuine two-
    connection race is not reproducible deterministically in one process --
    see the test docstring below."""

    def __init__(self, cx):
        self._cx = cx
        self.statements = []

    def execute(self, sql, params=()):
        self.statements.append(sql)
        return self._cx.execute(sql, params)

    def commit(self):
        self._cx.commit()


def test_put_draft_guards_the_delete_itself_not_a_preceding_select():
    """Regression guard for the TOCTOU fix: the status check must live in the
    DELETE statement's own WHERE clause, not in a SELECT that runs before it.

    Honest limitation: a genuine concurrent interleaving -- another
    connection's confirm() committing in the gap between a status-SELECT and
    the DELETE -- cannot be reproduced deterministically with sqlite3 in a
    single test process. Both connections execute fully sequentially here (no
    real thread scheduler to land a commit inside a gap), so a true race
    cannot be forced. What this test verifies instead is the structural
    property that makes that race impossible BY CONSTRUCTION: no statement
    reads `status` for this document_id before the guarded DELETE runs, so
    there is no gap left to interleave into. That is a meaningfully different
    (and passing) assertion from "the confirmed row is untouched" (already
    covered by test_put_draft_does_not_touch_a_confirmed_document) -- this
    test would FAIL if a future edit reintroduced a status-SELECT ahead of
    the DELETE, even if that edit still happened to pass the behavioral test
    on today's single-threaded, non-racing test harness.
    """
    cx = _cx()
    dx.put_draft(cx, 7, "c@x.com", "draft", [], [], [], "m")

    log = _SqlLog(cx)
    dx.put_draft(log, 7, "c@x.com", "new draft", [], [], [], "m2")

    delete_idx = next(i for i, s in enumerate(log.statements)
                       if s.strip().upper().startswith("DELETE"))
    delete_sql = log.statements[delete_idx].upper()
    assert "STATUS" in delete_sql
    assert "CONFIRMED" in delete_sql

    for s in log.statements[:delete_idx]:
        assert not (s.strip().upper().startswith("SELECT")
                    and "STATUS" in s.upper()), (
            "a SELECT of status ran before the guarded DELETE -- this is "
            "the TOCTOU shape the fix removed")


class _RaceInjector:
    """Wraps a connection to inject a competing writer's fully-committed row
    at a precise point in put_draft's own statement sequence -- simulating
    another connection's commit landing in a specific gap, deterministically
    and without real thread concurrency. Same technique as _SqlLog above,
    extended to also perform an action (not just record), so it can recreate
    the exact POST-CONDITION a genuine race leaves behind."""

    def __init__(self, cx, trigger_upper_prefix, inject):
        self._cx = cx
        self._trigger = trigger_upper_prefix
        self._inject = inject
        self._fired = False

    def execute(self, sql, params=()):
        cur = self._cx.execute(sql, params)
        if not self._fired and sql.strip().upper().startswith(self._trigger):
            self._fired = True
            self._inject(self._cx)
        return cur

    def commit(self):
        self._cx.commit()

    def rollback(self):
        self._cx.rollback()


def _insert_competing_draft(cx, document_id, narrative_md, email="racer@x.com"):
    cx.execute(
        "INSERT INTO client_document_extractions"
        "(document_id, email, status, narrative_md, attributes_json,"
        " facts_json, unstructured_json, model, created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (document_id, email, "ai_draft", narrative_md, "[]", "[]", "[]",
         "racer-model", "2020-01-01T00:00:00+00:00"))
    cx.commit()


def test_put_draft_defers_and_logs_when_a_non_confirmed_row_wins_the_race(capsys):
    """Drives the non-confirmed-survivor branch (the core finding): a
    competing writer's row appears in the gap between our guarded DELETE and
    our own SELECT. Per spec this is last-write-wins (both writers are
    extracting the same document) -- the call must not raise, must return the
    id that is actually in the table (the racer's), and must NOT silently
    discard this call's own payload while reporting success: it must log
    that the payload was dropped.

    Honest limitation: this does not exercise a genuine two-thread/two-
    connection race -- sqlite3 in a single process has no real scheduler to
    land a commit inside a gap. It instead injects the competing row
    synchronously right after the guarded DELETE executes, which reproduces
    the exact post-condition a real race leaves behind (a non-confirmed row
    present when put_draft's own status-SELECT runs). This is the same
    technique test_put_draft_guards_the_delete_itself_not_a_preceding_select
    already uses for a sibling gap, extended to actually inject a row rather
    than just observe statement order.
    """
    cx = _cx()
    eid = dx.put_draft(cx, 7, "c@x.com", "draft", [], [], [], "m")
    dx.reject(cx, eid, "glen")  # non-confirmed -> the guarded DELETE clears it

    racer = _RaceInjector(
        cx, "DELETE",
        lambda underlying: _insert_competing_draft(underlying, 7, "racer's payload"))
    result_id = dx.put_draft(racer, 7, "c@x.com", "our payload", [], [], [], "our-model")

    logged = capsys.readouterr().out
    assert "race" in logged.lower()
    assert "document_id=7" in logged
    assert "NOT written" in logged

    got = dx.get_for_document(cx, 7)
    assert got["id"] == result_id
    assert got["narrative_md"] == "racer's payload"
    assert got["email"] == "racer@x.com"
    rows = cx.execute("SELECT COUNT(*) FROM client_document_extractions "
                      "WHERE document_id=7").fetchone()
    assert rows[0] == 1


def test_put_draft_resolves_unique_violation_on_insert_without_raising(capsys):
    """Drives the second required fix: a competing INSERT lands in the gap
    between our own 'no survivor' SELECT and our own INSERT, so our INSERT
    hits the UNIQUE(document_id) index and must not propagate as an unhandled
    exception. Must resolve deterministically to the row that exists and log
    that this call's payload was not written.

    Honest limitation: same as above -- this recreates the post-condition of
    a real race (a row appearing immediately before our INSERT attempts to
    claim the same document_id) rather than a true concurrent interleaving,
    which sqlite3 in one process cannot schedule.
    """
    cx = _cx()
    # No prior row for document_id=9, so put_draft's DELETE is a no-op and its
    # own SELECT finds nothing -- the code proceeds to the try/INSERT branch.
    racer = _RaceInjector(
        cx, "SELECT",
        lambda underlying: _insert_competing_draft(underlying, 9, "racer's payload"))
    result_id = dx.put_draft(racer, 9, "c@x.com", "our payload", [], [], [], "our-model")

    logged = capsys.readouterr().out
    assert "unique" in logged.lower() or "race" in logged.lower()
    assert "document_id=9" in logged
    assert "NOT written" in logged

    got = dx.get_for_document(cx, 9)
    assert got["id"] == result_id
    assert got["narrative_md"] == "racer's payload"
    assert got["email"] == "racer@x.com"
    rows = cx.execute("SELECT COUNT(*) FROM client_document_extractions "
                      "WHERE document_id=9").fetchone()
    assert rows[0] == 1


def test_put_draft_replaces_a_rejected_document():
    """Rejection must not permanently block re-extraction (re-queuing)."""
    cx = _cx()
    eid = dx.put_draft(cx, 7, "c@x.com", "draft", [], [], [], "m")
    dx.reject(cx, eid, "glen")

    dx.put_draft(cx, 7, "c@x.com", "re-extracted text", [], [], [], "m2")

    got = dx.get_for_document(cx, 7)
    assert got["status"] == "ai_draft"
    assert got["narrative_md"] == "re-extracted text"
    rows = cx.execute("SELECT COUNT(*) FROM client_document_extractions "
                      "WHERE document_id=7").fetchone()
    assert rows[0] == 1
