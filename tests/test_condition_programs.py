"""Slice 1: dashboard/condition_programs.py — sqlite store for Glen's
9 authored condition-support programs. seed_if_empty must be idempotent and
must never clobber an operator's upsert edit."""
import sqlite3

from dashboard import condition_programs as cp

SEED = {
    "dry-eye": {
        "label": "Dry Eye",
        "consult_recommended": False,
        "items": [
            {"slug": "aces-eye-drops", "name": "ACES Eye Drops",
             "alts": [{"slug": "ocuheal-eye-drops", "name": "OcuHeal Eye Drops"}]},
            {"slug": "wholomega", "name": "WholOmega", "dose": "4 capsules/day"},
        ],
    },
    "wet-amd": {
        "label": "Wet AMD",
        "consult_recommended": True,
        "items": [
            {"slug": "angiogenx", "name": "AngiogenX", "dose": "1 or more/day"},
        ],
    },
}


def _cx(tmp_db):
    cx = sqlite3.connect(tmp_db)
    cx.row_factory = sqlite3.Row
    return cx


def test_init_table_creates_schema(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    cols = {r[1] for r in cx.execute("PRAGMA table_info(condition_programs)")}
    assert cols == {"condition_key", "label", "consult_recommended", "items_json",
                     "modifiers_json", "updated_at"}


def test_seed_if_empty_populates_table(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    cp.seed_if_empty(cx, SEED)
    rows = cp.all(cx)
    assert len(rows) == 2
    keys = {r["condition_key"] for r in rows}
    assert keys == {"dry-eye", "wet-amd"}


def test_seed_if_empty_is_idempotent_no_duplicates(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    cp.seed_if_empty(cx, SEED)
    cp.seed_if_empty(cx, SEED)
    cp.seed_if_empty(cx, SEED)
    assert len(cp.all(cx)) == 2


def test_seed_if_empty_never_overwrites_operator_edit(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    cp.seed_if_empty(cx, SEED)
    cp.upsert(cx, "dry-eye", "Dry Eye (edited)", True,
              [{"slug": "moisturize", "name": "Moisturize"}])
    # a re-seed attempt (e.g. on a fresh process boot) must not stomp the edit
    cp.seed_if_empty(cx, SEED)
    got = cp.get(cx, "dry-eye")
    assert got["label"] == "Dry Eye (edited)"
    assert got["consult_recommended"] is True
    assert got["items"] == [{"slug": "moisturize", "name": "Moisturize"}]


def test_seed_marker_prevents_resurrection_after_table_cleared(tmp_db):
    """Mirrors the broad_benefit resurrection guard: once seeded, the
    persisted _seed_state marker means a later empty table (cleared by any
    means) is never mistaken for "not yet seeded"."""
    cx = _cx(tmp_db)
    cp.init_table(cx)
    cp.seed_if_empty(cx, SEED)
    cx.execute("DELETE FROM condition_programs")
    cx.commit()
    assert cp.all(cx) == []
    cp.seed_if_empty(cx, SEED)
    assert cp.all(cx) == []


def test_get_returns_none_for_missing_key(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    assert cp.get(cx, "nonexistent") is None


def test_get_shape(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    cp.seed_if_empty(cx, SEED)
    got = cp.get(cx, "wet-amd")
    assert got["condition_key"] == "wet-amd"
    assert got["label"] == "Wet AMD"
    assert got["consult_recommended"] is True
    assert got["items"] == SEED["wet-amd"]["items"]
    assert got["updated_at"]


def test_upsert_inserts_new_program(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    cp.upsert(cx, "dry-eye", "Dry Eye", False,
               [{"slug": "moisturize", "name": "Moisturize"}])
    got = cp.get(cx, "dry-eye")
    assert got["label"] == "Dry Eye"
    assert got["consult_recommended"] is False
    assert got["items"] == [{"slug": "moisturize", "name": "Moisturize"}]


def test_upsert_updates_existing_program_and_bumps_updated_at(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    cp.upsert(cx, "dry-eye", "Dry Eye", False, [{"slug": "a", "name": "A"}])
    first_updated = cp.get(cx, "dry-eye")["updated_at"]
    cp.upsert(cx, "dry-eye", "Dry Eye v2", True, [{"slug": "b", "name": "B"}])
    got = cp.get(cx, "dry-eye")
    assert got["label"] == "Dry Eye v2"
    assert got["consult_recommended"] is True
    assert got["items"] == [{"slug": "b", "name": "B"}]
    assert got["updated_at"] >= first_updated
    assert len(cp.all(cx)) == 1


def test_all_returns_list(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    cp.seed_if_empty(cx, SEED)
    rows = cp.all(cx)
    assert isinstance(rows, list)
    assert len(rows) == 2


CLINICAL_ORDER = [
    "glaucoma-elevated-iop", "glaucoma-normal-iop", "dry-amd", "wet-amd",
    "senile-cataract", "psc-cataract", "dry-eye", "retinitis-pigmentosa",
    "diabetic-retinopathy",
]


def test_all_returns_clinical_order_not_alphabetical(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    # Insert in reverse (also non-alphabetical) order so a passing test can't
    # be an accident of insertion order or of alphabetical sorting.
    for key in reversed(CLINICAL_ORDER):
        cp.upsert(cx, key, key, False, [])
    rows = cp.all(cx)
    assert [r["condition_key"] for r in rows] == CLINICAL_ORDER


def test_all_sorts_unknown_keys_last_by_key(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    cp.upsert(cx, "wet-amd", "Wet AMD", True, [])
    cp.upsert(cx, "zzz-unknown", "Unknown Z", False, [])
    cp.upsert(cx, "aaa-unknown", "Unknown A", False, [])
    rows = cp.all(cx)
    keys = [r["condition_key"] for r in rows]
    assert keys == ["wet-amd", "aaa-unknown", "zzz-unknown"]
