"""`biofield_auth_chain.updated_at` — so a human edit is distinguishable from a model write.

The dose audit (2026-07-09) could not tell a row Glen hand-corrected from one the LLM
fabricated: chain rows carried only `created_at`. `updated_at` closes that.

Semantics, deliberately narrow:
  * a row is born with updated_at == created_at (never NULL)
  * update_chain_row (a VALUE edit: remedy/dosage/frequency/timing/head/layer) bumps it
  * confirm_row / confirm_all / reorder do NOT bump it — reviewing or re-ordering a row
    is not editing its content, and bumping there would mark every confirmed row "edited"
  * existing rows are backfilled updated_at = created_at, so the migration invents no edits
"""
import sqlite3

from dashboard.biofield_authoring import (
    add_chain_row, confirm_all, confirm_row, create_test, init_auth_tables,
    update_chain_row, was_edited,
)


def _cx():
    cx = sqlite3.connect(":memory:")
    init_auth_tables(cx)
    return cx


def _row(cx, rid):
    cx.row_factory = sqlite3.Row
    return cx.execute("SELECT created_at, updated_at, confirmed FROM biofield_auth_chain "
                      "WHERE id=?", (rid,)).fetchone()


def _seed(cx):
    tid = create_test(cx, "Debra Herndon", "chakamom1@gmail.com", "2026-07-09")
    return tid, add_chain_row(cx, tid, 1, "Lung Meridian", "Lung Meridian",
                              "Fiber Cleanse", "one", "a day", "with food", confirmed=0)


def test_column_exists_and_is_never_null_on_insert():
    cx = _cx()
    _, rid = _seed(cx)
    r = _row(cx, rid)
    assert r["updated_at"], "updated_at must be set on insert"
    assert r["updated_at"] == r["created_at"]


def test_migration_is_idempotent_and_backfills_existing_rows():
    """A pre-existing table without the column: adding it must not invent edits."""
    cx = sqlite3.connect(":memory:")
    cx.execute("""CREATE TABLE biofield_auth_chain(
        id INTEGER PRIMARY KEY AUTOINCREMENT, test_id INTEGER, layer INTEGER,
        head TEXT, most_affected TEXT, remedy TEXT, dosage TEXT, frequency TEXT,
        timing TEXT, sort_seq INTEGER, created_at TEXT, confirmed INTEGER DEFAULT 1,
        origin TEXT NOT NULL DEFAULT 'live')""")
    cx.execute("INSERT INTO biofield_auth_chain(test_id,layer,remedy,created_at) "
               "VALUES(6,1,'Fiber Cleanse','2026-07-09T07:31:18Z')")
    cx.commit()

    init_auth_tables(cx)          # migrate
    init_auth_tables(cx)          # again -> must not raise or re-backfill wrongly

    r = cx.execute("SELECT created_at, updated_at FROM biofield_auth_chain").fetchone()
    assert r[1] == r[0] == "2026-07-09T07:31:18Z"


def test_value_edit_bumps_updated_at_and_leaves_created_at():
    cx = _cx()
    _, rid = _seed(cx)
    cx.execute("UPDATE biofield_auth_chain SET created_at='2026-01-01T00:00:00Z',"
               "updated_at='2026-01-01T00:00:00Z' WHERE id=?", (rid,))
    cx.commit()

    update_chain_row(cx, rid, timing="with extra water, away from beneficial oils")
    r = _row(cx, rid)
    assert r["created_at"] == "2026-01-01T00:00:00Z"     # untouched
    assert r["updated_at"] > r["created_at"]             # bumped


def test_confirm_does_not_count_as_an_edit():
    """Reviewing a row is not editing it — otherwise every confirmed row reads 'edited'."""
    cx = _cx()
    tid, rid = _seed(cx)
    cx.execute("UPDATE biofield_auth_chain SET created_at='2026-01-01T00:00:00Z',"
               "updated_at='2026-01-01T00:00:00Z' WHERE id=?", (rid,))
    cx.commit()

    confirm_row(cx, rid)
    assert _row(cx, rid)["updated_at"] == "2026-01-01T00:00:00Z"
    assert _row(cx, rid)["confirmed"] == 1

    confirm_all(cx, tid)
    assert _row(cx, rid)["updated_at"] == "2026-01-01T00:00:00Z"


def test_no_op_update_does_not_bump():
    cx = _cx()
    _, rid = _seed(cx)
    before = _row(cx, rid)["updated_at"]
    update_chain_row(cx, rid)          # no fields -> early return
    assert _row(cx, rid)["updated_at"] == before


def test_reorder_and_arrange_do_not_count_as_edits():
    """Moving a row's layer is positioning, not a value edit."""
    from dashboard.biofield_authoring import reorder_chain
    cx = _cx()
    tid, rid = _seed(cx)
    add_chain_row(cx, tid, 2, "Liver", "Liver", "Liver Support", confirmed=0)
    cx.execute("UPDATE biofield_auth_chain SET created_at='2026-01-01T00:00:00Z',"
               "updated_at='2026-01-01T00:00:00Z'")
    cx.commit()
    reorder_chain(cx, tid, rid, 2)
    assert _row(cx, rid)["updated_at"] == "2026-01-01T00:00:00Z"


def test_known_limit_same_second_edit_is_invisible():
    """_now() is second-resolution: an edit inside the insert's second can't be seen.
    Documented rather than hidden — the audit under-claims edits, which is the safe
    direction (it never excuses a fabrication as 'a human wrote that')."""
    cx = _cx()
    _, rid = _seed(cx)
    update_chain_row(cx, rid, dosage="a pinch")   # same second as the insert
    r = _row(cx, rid)
    assert r["updated_at"] == r["created_at"]
    assert was_edited(cx, rid) is False            # the limit, asserted explicitly


def test_was_edited_is_the_audit_signal():
    cx = _cx()
    _, rid = _seed(cx)
    assert was_edited(cx, rid) is False          # as written by the interpreter
    cx.execute("UPDATE biofield_auth_chain SET created_at='2026-01-01T00:00:00Z',"
               "updated_at='2026-01-01T00:00:00Z' WHERE id=?", (rid,))
    cx.commit()
    update_chain_row(cx, rid, dosage="a pinch")
    assert was_edited(cx, rid) is True           # a human touched this value
