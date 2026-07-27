"""Unit tests for _merge_canonical_into_person (app.py): read-through merge of
canonical clinical attributes into a /api/people row. Env-gated like the other
api tests (importing app builds OpenAI/Pinecone clients)."""
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("requires app env (use doppler run / CI)", allow_module_level=True)

import app  # noqa: E402
from dashboard import canonical_tags as ct  # noqa: E402


def _cx():
    cx = sqlite3.connect(":memory:")
    ct.init_tables(cx)
    return cx


def _seed_canon(cx, email, **fields):
    for f, vals in fields.items():
        vals = vals if isinstance(vals, (list, tuple)) else [vals]
        for v in vals:
            ct.set_attr(cx, email, f, v, source="test")
    cx.commit()


def test_discrete_union_preserves_json_string_shape():
    cx = _cx()
    _seed_canon(cx, "c@x.com", conditions="ocular hypertension")
    person = {"email": "c@x.com", "conditions": json.dumps(["glaucoma"]),
              "tags": json.dumps(["vip"])}
    out = app._merge_canonical_into_person(cx, person)
    assert json.loads(out["conditions"]) == ["glaucoma", "ocular hypertension"]
    assert isinstance(out["conditions"], str)          # still a JSON string


def test_discrete_dedup_is_case_insensitive_first_seen_wins():
    cx = _cx()
    _seed_canon(cx, "c@x.com", conditions="glaucoma")
    person = {"email": "c@x.com", "conditions": json.dumps(["Glaucoma"])}
    out = app._merge_canonical_into_person(cx, person)
    assert json.loads(out["conditions"]) == ["Glaucoma"]   # people casing kept, one item


def test_tags_is_never_merged():
    cx = _cx()
    _seed_canon(cx, "c@x.com", tags="from-canonical")
    person = {"email": "c@x.com", "tags": json.dumps(["crm-tag"])}
    out = app._merge_canonical_into_person(cx, person)
    assert json.loads(out["tags"]) == ["crm-tag"]          # canonical tag NOT added


def test_scalar_canonical_wins_when_present_else_people():
    cx = _cx()
    _seed_canon(cx, "a@x.com", challenges="fatigue")
    a = app._merge_canonical_into_person(cx, {"email": "a@x.com", "challenges": ""})
    assert a["challenges"] == "fatigue"
    # canonical empty -> people value kept
    b = app._merge_canonical_into_person(cx, {"email": "b@x.com", "goals": "sleep"})
    assert b["goals"] == "sleep"


def test_no_canonical_row_returns_person_unchanged():
    cx = _cx()
    person = {"email": "nobody@x.com", "conditions": json.dumps(["x"]),
              "challenges": "y"}
    out = app._merge_canonical_into_person(cx, dict(person))
    assert json.loads(out["conditions"]) == ["x"] and out["challenges"] == "y"


def test_empty_everything_leaves_field_untouched():
    cx = _cx()
    person = {"email": "c@x.com", "conditions": None}
    out = app._merge_canonical_into_person(cx, person)
    assert out["conditions"] is None                        # not forced to "[]"


def test_malformed_people_json_degrades_to_canonical():
    cx = _cx()
    _seed_canon(cx, "c@x.com", conditions="glaucoma")
    person = {"email": "c@x.com", "conditions": "not json"}
    out = app._merge_canonical_into_person(cx, person)
    assert json.loads(out["conditions"]) == ["glaucoma"]


def test_best_effort_get_person_raises_returns_unchanged(monkeypatch):
    cx = _cx()
    monkeypatch.setattr(ct, "get_person",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    person = {"email": "c@x.com", "conditions": json.dumps(["x"])}
    out = app._merge_canonical_into_person(cx, person)      # must not raise
    assert json.loads(out["conditions"]) == ["x"]


def test_blank_email_returns_unchanged():
    cx = _cx()
    person = {"email": "", "conditions": json.dumps(["x"])}
    assert app._merge_canonical_into_person(cx, person)["conditions"] == json.dumps(["x"])


def test_scalar_canonical_overwrites_different_people_value():
    """When both person and canonical have different non-empty scalar values,
    canonical wins (overwrites people value)."""
    cx = _cx()
    _seed_canon(cx, "c@x.com", challenges="canonical-fatigue")
    person = {"email": "c@x.com", "challenges": "people-fatigue"}
    out = app._merge_canonical_into_person(cx, person)
    assert out["challenges"] == "canonical-fatigue"


def test_discrete_field_already_list_is_preserved_and_unioned():
    """If person[field] is already a Python list (not a JSON string),
    it should be preserved and unioned with canonical, not silently discarded."""
    cx = _cx()
    _seed_canon(cx, "c@x.com", conditions="canonical-glaucoma")
    # person["conditions"] is a Python list, not a JSON string
    person = {"email": "c@x.com", "conditions": ["people-glaucoma"]}
    out = app._merge_canonical_into_person(cx, person)
    # The list should be preserved and unioned with canonical
    assert json.loads(out["conditions"]) == ["people-glaucoma", "canonical-glaucoma"]
    assert isinstance(out["conditions"], str)  # still serialized to JSON
