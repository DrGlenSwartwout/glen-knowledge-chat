import sqlite3
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _cx():
    from dashboard import cohorts as C
    cx = sqlite3.connect(":memory:")
    C.init_tables(cx)
    return C, cx


def test_cohort_crud_and_policy_validation():
    C, cx = _cx()
    C.upsert_cohort(cx, key="loyal", name="Stay the Course", policy={"type": "flat_ff", "cents": 5000})
    assert C.get_cohort(cx, "loyal")["policy"]["cents"] == 5000
    C.upsert_cohort(cx, key="loyal", name="Stay the Course", policy={"type": "flat_ff", "cents": 4500})
    assert C.get_cohort(cx, "loyal")["policy"]["cents"] == 4500     # upsert
    assert len(C.list_cohorts(cx)) == 1
    for bad in [{"type": "nope"}, {"type": "flat_ff"}, {"type": "per_sku"},
                {"type": "percent_off", "pct": "x"}]:
        try:
            C.upsert_cohort(cx, key="b", name="b", policy=bad); assert False
        except ValueError:
            pass


def test_membership_active_and_expiry():
    C, cx = _cx()
    C.upsert_cohort(cx, key="a", name="A", policy={"type": "percent_off", "pct": 10})
    C.upsert_cohort(cx, key="b", name="B", policy={"type": "percent_off", "pct": 20}, active=False)
    C.add_member(cx, "JC@x.com", "a")
    C.add_member(cx, "jc@x.com", "b")                      # cohort inactive -> excluded
    C.add_member(cx, "jc@x.com", "a", expires_at="2000-01-01T00:00:00+00:00")  # upsert -> expired
    got = C.member_cohorts(cx, "jc@x.com")
    assert got == []                                       # a is now expired, b inactive
    C.add_member(cx, "jc@x.com", "a")                      # re-add active (no expiry)
    keys = [c["key"] for c in C.member_cohorts(cx, "jc@x.com")]
    assert keys == ["a"]
    assert C.remove_member(cx, "jc@x.com", "a") is True
    assert C.member_cohorts(cx, "jc@x.com") == []


def test_policy_unit_cents_per_type():
    from dashboard import cohorts as C
    # flat_ff applies only to FFs
    assert C.policy_unit_cents({"type": "flat_ff", "cents": 5000}, slug="x", list_cents=6997, is_ff=True) == 5000
    assert C.policy_unit_cents({"type": "flat_ff", "cents": 5000}, slug="x", list_cents=3997, is_ff=False) is None
    # per_sku
    p = {"type": "per_sku", "prices": {"neuro-magnesium": 4200}}
    assert C.policy_unit_cents(p, slug="neuro-magnesium", list_cents=6997, is_ff=True) == 4200
    assert C.policy_unit_cents(p, slug="other", list_cents=6997, is_ff=True) is None
    # percent_off scope
    assert C.policy_unit_cents({"type": "percent_off", "pct": 10, "scope": "all"}, slug="x", list_cents=1000, is_ff=False) == 900
    assert C.policy_unit_cents({"type": "percent_off", "pct": 10, "scope": "ff"}, slug="x", list_cents=1000, is_ff=False) is None
    # volume/reorder resolve elsewhere -> None here
    assert C.policy_unit_cents({"type": "volume"}, slug="x", list_cents=6997, is_ff=True) is None


def test_best_cohort_price_is_lowest_applicable():
    from dashboard import cohorts as C
    cohorts = [
        {"policy": {"type": "flat_ff", "cents": 5000}},
        {"policy": {"type": "per_sku", "prices": {"nm": 4200}}},
        {"policy": {"type": "percent_off", "pct": 50, "scope": "ff"}},   # 3498 for a 6997 FF
    ]
    # FF slug 'nm': min(5000, 4200, 3498) = 3498
    assert C.best_cohort_price(cohorts, slug="nm", list_cents=6997, is_ff=True) == 3498
    # a non-FF product only the 'all'-less policies apply -> here none apply -> None
    assert C.best_cohort_price([{"policy": {"type": "flat_ff", "cents": 5000}}],
                               slug="drops", list_cents=3997, is_ff=False) is None
