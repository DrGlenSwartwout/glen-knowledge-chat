"""The set of flags that must be on lives in a committed file, not a hardcoded tuple.

PR #759 watched four flags. The root-cause dig that evening found EIGHT had been deleted
from the prod service — five still off, silently disabling a Stripe monthly checkout, the
two-door reveal, the prepay ladder, analysis-quota enforcement and the review flow. The
#759 watchdog would not have caught one of them.

Render sells no audit log on this plan (`audit logs not available for this plan`), and its
events API never records env changes. So attribution has to come from somewhere else:
with a committed baseline, turning a flag off is a PULL REQUEST. Git records who intended
it. That is the point; the alerting is secondary.
"""
import json
import pathlib

import scripts.surface_check as S

BASELINE = pathlib.Path("scripts/flags_expected.json")


def _baseline():
    return json.loads(BASELINE.read_text())["expected_on"]


def test_baseline_file_is_a_clean_sorted_unique_list():
    flags = _baseline()
    assert isinstance(flags, list)
    assert len(flags) == 38, len(flags)
    assert all(isinstance(f, str) and f for f in flags)
    assert flags == sorted(flags), "keep it sorted so a diff is readable"
    assert len(set(flags)) == len(flags), "no duplicates"


def test_no_underscore_aliases_in_the_baseline():
    """`_REVIEWS_ENABLED` & co are module-internal names whose env var is spelled without
    the underscore — they report env_present=False forever. Watching them is noise, not
    coverage; their public counterparts are already here."""
    assert [f for f in _baseline() if f.startswith("_")] == []


def test_every_baseline_entry_is_a_flag_name():
    assert all(f.endswith("_ENABLED") for f in _baseline())


def test_required_on_is_derived_from_the_file_not_hardcoded():
    assert set(S.REQUIRED_ON) == set(_baseline())
    assert len(S.REQUIRED_ON) == 38


def test_the_four_flags_759_watched_are_still_covered():
    """No coverage regression when generalising from the hardcoded tuple."""
    assert {"FIRESIDE_ENABLED", "REPERTOIRE_ENABLED",
            "INVOICE_PAYLINK_ENABLED", "SCAN_REQUEST_ENABLED"} <= set(S.REQUIRED_ON)


def test_the_five_flags_found_deleted_during_the_dig_are_covered():
    """These were off in prod for an unknown period and nothing would have said so."""
    assert {"PREPAY_LADDER_ENABLED", "CONTINUOUS_CARE_MONTHLY_ENABLED", "TWO_DOOR_ENABLED",
            "ANALYSIS_QUOTA_ENABLED", "REVIEWS_ENABLED"} <= set(S.REQUIRED_ON)


# ── behaviour over the baseline ──
def _on():
    return {"value": True, "env_present": True, "source": "import"}


def _payload(**flags):
    return {"ok": True, "data": {"flags": flags}}


def _fetch_ok(payload):
    def _f(url, key, timeout=0):
        return payload
    return _f


def test_a_baseline_flag_that_is_off_fails():
    p = _payload(**{n: _on() for n in S.REQUIRED_ON})
    p["data"]["flags"]["PREPAY_LADDER_ENABLED"] = {"value": False, "env_present": False,
                                                   "source": "import"}
    out = S.check_flags("https://x.test", "k", fetch=_fetch_ok(p))
    assert [f["flag"] for f in out] == ["PREPAY_LADDER_ENABLED"]
    assert "missing" in out[0]["reason"].lower()


def test_a_flag_outside_the_baseline_being_off_is_silent():
    """New flags default off. Alerting on them would punish every feature branch —
    and a watchdog that cries wolf gets ignored, which is how this stayed invisible."""
    p = _payload(**{n: _on() for n in S.REQUIRED_ON})
    p["data"]["flags"]["JOURNEY_QUEST_ENABLED"] = {"value": False, "env_present": True,
                                                   "source": "import"}
    assert S.check_flags("https://x.test", "k", fetch=_fetch_ok(p)) == []


def test_missing_baseline_file_is_could_not_load_never_a_silent_pass(monkeypatch):
    """A watchdog whose expectations vanished must SAY SO, not quietly expect nothing."""
    monkeypatch.setattr(S, "REQUIRED_ON", ())
    monkeypatch.setattr(S, "BASELINE_ERROR", "no such file: scripts/flags_expected.json")
    out = S.check_flags("https://x.test", "k", fetch=_fetch_ok(_payload()))
    assert len(out) == 1
    assert out[0]["flag"] == "*"
    assert "could not load baseline" in out[0]["reason"].lower()


def test_unparseable_baseline_is_could_not_load(monkeypatch):
    monkeypatch.setattr(S, "REQUIRED_ON", ())
    monkeypatch.setattr(S, "BASELINE_ERROR", "expected_on is str, expected list")
    out = S.check_flags("https://x.test", "k", fetch=_fetch_ok(_payload()))
    assert out[0]["flag"] == "*" and "baseline" in out[0]["reason"].lower()
