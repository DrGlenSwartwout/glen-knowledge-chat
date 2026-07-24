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


VISION_ITEMS = [
    {"slug": "wholomega-120-gelcaps", "name": "WholOmega 120 gelcaps", "dose": "4 times a day"},
    {"slug": "nous-energy", "name": "Nous Energy"},
]


def test_ensure_program_creates_when_absent(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    assert cp.get(cx, "vision-improvement") is None
    cp.ensure_program(cx, "vision-improvement", "Vision Improvement", VISION_ITEMS)
    got = cp.get(cx, "vision-improvement")
    assert got is not None
    assert got["label"] == "Vision Improvement"
    assert got["consult_recommended"] is False
    assert got["items"] == VISION_ITEMS


def test_ensure_program_is_idempotent_no_duplicates(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    cp.ensure_program(cx, "vision-improvement", "Vision Improvement", VISION_ITEMS)
    cp.ensure_program(cx, "vision-improvement", "Vision Improvement", VISION_ITEMS)
    cp.ensure_program(cx, "vision-improvement", "Vision Improvement", VISION_ITEMS)
    rows = [r for r in cp.all(cx) if r["condition_key"] == "vision-improvement"]
    assert len(rows) == 1


def test_ensure_program_never_overwrites_operator_edit(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    cp.ensure_program(cx, "vision-improvement", "Vision Improvement", VISION_ITEMS)
    cp.upsert(cx, "vision-improvement", "Vision Improvement (edited)", True,
              [{"slug": "operator-swap", "name": "Operator Swap"}])
    # A later ensure_program call (e.g. on a fresh boot after this key already
    # existed) must leave the operator's edit completely untouched.
    cp.ensure_program(cx, "vision-improvement", "Vision Improvement", VISION_ITEMS)
    got = cp.get(cx, "vision-improvement")
    assert got["label"] == "Vision Improvement (edited)"
    assert got["consult_recommended"] is True
    assert got["items"] == [{"slug": "operator-swap", "name": "Operator Swap"}]


# ---------------------------------------------------------------------------
# migrate_dry_eye_modifiers: one-time, marker-guarded restructure for stores
# seeded long ago with the OLD flat dry-eye shape (moisturize and
# moisture-eyes-night-oil as unconditional base items).
# ---------------------------------------------------------------------------

OLD_DRY_EYE_ITEMS = [
    {"slug": "aces-eye-drops", "name": "ACES Eye Drops",
     "alts": [{"slug": "ocuheal-eye-drops", "name": "OcuHeal Eye Drops"}]},
    {"slug": "moisturize", "name": "Moisturize"},
    {"slug": "wholomega", "name": "WholOmega", "dose": "4 capsules/day"},
    {"slug": "moisture-eyes-night-oil", "name": "Moisture Eyes Night Oil",
     "alts": [{"slug": "moisture-eyes-night-drops", "name": "Moisture Eyes Night Drops"}]},
]


def _seed_old_shape_dry_eye(cx):
    cp.init_table(cx)
    cp.upsert(cx, "dry-eye", "Dry Eye", False, OLD_DRY_EYE_ITEMS)


def test_migrate_dry_eye_modifiers_rewrites_old_shape_to_base_plus_modifiers(tmp_db):
    cx = _cx(tmp_db)
    _seed_old_shape_dry_eye(cx)

    cp.migrate_dry_eye_modifiers(cx)

    got = cp.get(cx, "dry-eye")
    assert [it["slug"] for it in got["items"]] == ["aces-eye-drops", "wholomega"]
    aces = got["items"][0]
    assert aces["alts"] == [{"slug": "ocuheal-eye-drops", "name": "OcuHeal Eye Drops"}]
    wholomega = got["items"][1]
    assert wholomega["dose"] == "4 capsules/day"

    mods = {m["when"]: m for m in got["modifiers"]}
    assert set(mods) == {"aqueous_deficiency", "severe"}

    aq = mods["aqueous_deficiency"]
    assert aq["action"] == "add"
    assert aq["source"] == "client-reported"
    assert aq["client_default"] is True
    assert aq["items"] == [{"slug": "moisturize", "name": "Moisturize"}]

    sev = mods["severe"]
    assert sev["action"] == "add"
    assert sev["source"] == "client-reported"
    assert sev["client_default"] is False
    assert sev["items"] == [
        {"slug": "moisture-eyes-night-oil", "name": "Moisture Eyes Night Oil",
         "alts": [{"slug": "moisture-eyes-night-drops", "name": "Moisture Eyes Night Drops"}]},
    ]


def test_migrate_dry_eye_modifiers_second_call_is_noop(tmp_db):
    cx = _cx(tmp_db)
    _seed_old_shape_dry_eye(cx)
    cp.migrate_dry_eye_modifiers(cx)
    first = cp.get(cx, "dry-eye")

    cp.migrate_dry_eye_modifiers(cx)
    second = cp.get(cx, "dry-eye")
    assert second["items"] == first["items"]
    assert second["modifiers"] == first["modifiers"]
    assert second["updated_at"] == first["updated_at"]  # upsert never re-ran


def test_migrate_dry_eye_modifiers_marker_holds_after_operator_edit(tmp_db):
    """Prove the migration runs AT MOST ONCE EVER: once an operator deletes an
    item via the console editor (upsert), a later migration call must NEVER
    resurrect it."""
    cx = _cx(tmp_db)
    _seed_old_shape_dry_eye(cx)
    cp.migrate_dry_eye_modifiers(cx)

    # Operator deletes the severe modifier's item entirely via the console.
    edited = cp.get(cx, "dry-eye")
    cp.upsert(cx, "dry-eye", edited["label"], edited["consult_recommended"],
              edited["items"], modifiers=[m for m in edited["modifiers"]
                                          if m["when"] != "severe"])

    cp.migrate_dry_eye_modifiers(cx)  # third call overall

    got = cp.get(cx, "dry-eye")
    assert {m["when"] for m in got["modifiers"]} == {"aqueous_deficiency"}


def test_migrate_dry_eye_modifiers_noop_when_program_absent(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    cp.migrate_dry_eye_modifiers(cx)  # must not raise
    assert cp.get(cx, "dry-eye") is None
    # marker recorded even though there was nothing to migrate
    cp.migrate_dry_eye_modifiers(cx)
    assert cp.get(cx, "dry-eye") is None


# ---------------------------------------------------------------------------
# migrate_cataract_brunescent: one-time, marker-guarded move of lens-zyme out
# of senile-cataract's unconditional items into a client-reported "brunescent"
# modifier. Mirrors seed_if_empty's once-ever _seed_state marker pattern.
# ---------------------------------------------------------------------------

LEGACY_SENILE_ITEMS = [
    {"slug": "golden-book", "name": "Golden Book"},
    {"slug": "crystalline-clarity", "name": "Crystalline Clarity"},
    {"slug": "clarity", "name": "Clarity"},
    {"slug": "clear-lens-eye-drops", "name": "Clear Lens Eye Drops",
     "alts": [{"slug": "ocuheal-eye-drops", "name": "OcuHeal Eye Drops"}]},
    {"slug": "lens-zyme", "name": "Lens-Zyme Brunescence Buster",
     "note": "Add for brunescent cataracts"},
]


def test_migrate_cataract_brunescent_moves_lens_zyme_into_modifier(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    cp.upsert(cx, "senile-cataract", "Senile (Age-Related) Cataract", False,
              LEGACY_SENILE_ITEMS)
    cp.migrate_cataract_brunescent(cx)
    got = cp.get(cx, "senile-cataract")
    slugs = [it["slug"] for it in got["items"]]
    assert "lens-zyme" not in slugs
    assert slugs == ["golden-book", "crystalline-clarity", "clarity", "clear-lens-eye-drops"]
    # every other item's content survives verbatim by slug
    drops = next(it for it in got["items"] if it["slug"] == "clear-lens-eye-drops")
    assert drops["alts"] == [{"slug": "ocuheal-eye-drops", "name": "OcuHeal Eye Drops"}]

    mods = got["modifiers"]
    assert len(mods) == 1
    m = mods[0]
    assert m["when"] == "brunescent"
    assert m["action"] == "add"
    assert m["source"] == "client-reported"
    assert m["client_default"] is False
    assert m["items"] == [{"slug": "lens-zyme", "name": "Lens-Zyme Brunescence Buster"}]


def test_migrate_cataract_brunescent_runs_only_once(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    cp.upsert(cx, "senile-cataract", "Senile (Age-Related) Cataract", False,
              LEGACY_SENILE_ITEMS)
    cp.migrate_cataract_brunescent(cx)
    cp.migrate_cataract_brunescent(cx)
    cp.migrate_cataract_brunescent(cx)
    got = cp.get(cx, "senile-cataract")
    assert len(got["modifiers"]) == 1  # not duplicated by repeated calls


def test_migrate_cataract_brunescent_never_resurrects_operator_deletion(tmp_db):
    """Mirrors the seed_if_empty resurrection guard: once an operator has
    edited the program (here, removing lens-zyme entirely before the
    migration ever runs), a later call must never bring it back."""
    cx = _cx(tmp_db)
    cp.init_table(cx)
    cp.upsert(cx, "senile-cataract", "Senile (Age-Related) Cataract", False,
              [{"slug": "golden-book", "name": "Golden Book"}])  # operator already removed lens-zyme
    cp.migrate_cataract_brunescent(cx)
    got = cp.get(cx, "senile-cataract")
    assert [it["slug"] for it in got["items"]] == ["golden-book"]
    assert got["modifiers"] == []

    # Simulate a later boot: marker already recorded, so even if lens-zyme
    # somehow reappeared in a fresh upsert, the migration must not re-fire.
    cp.upsert(cx, "senile-cataract", "Senile (Age-Related) Cataract", False,
              LEGACY_SENILE_ITEMS)
    cp.migrate_cataract_brunescent(cx)
    got2 = cp.get(cx, "senile-cataract")
    assert got2["modifiers"] == []  # migration did not run again
    assert any(it["slug"] == "lens-zyme" for it in got2["items"])  # untouched


def test_migrate_cataract_brunescent_noop_when_program_absent(tmp_db):
    cx = _cx(tmp_db)
    cp.init_table(cx)
    cp.migrate_cataract_brunescent(cx)  # must not raise
    assert cp.get(cx, "senile-cataract") is None
