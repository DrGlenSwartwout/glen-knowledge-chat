# tests/test_cert_rules.py
from dashboard import cert_rules as cr


def _sub(modules, formats):
    return {"credited_modules": modules, "formats": formats}


def test_catalog_shape():
    assert len(cr.MODULES) == 12
    assert [m["id"] for m in cr.MODULES] == list(range(1, 13))
    assert cr.MODULES[0]["label"] == "Body"
    # every format has a kind we can test the written+video rule on
    kinds = {f["kind"] for f in cr.FORMATS}
    assert {"written", "video"} <= kinds
    assert cr.MIN_SUBMISSIONS == 12


def test_empty_is_incomplete():
    r = cr.evaluate([])
    assert r["complete"] is False
    assert r["approved_count"] == 0
    assert r["modules_covered"] == set()
    assert len(r["modules_missing"]) == 12
    assert r["has_written"] is False and r["has_video"] is False


def test_one_submission_can_cover_multiple_modules():
    r = cr.evaluate([_sub([1, 2, 3], ["case_report"])])
    assert r["modules_covered"] == {1, 2, 3}
    assert r["has_written"] is True and r["has_video"] is False
    assert r["complete"] is False  # <12 submissions, missing modules, no video


def test_written_and_video_detected_by_kind():
    subs = [_sub([1], ["article"]), _sub([2], ["talking_head_scripted"])]
    r = cr.evaluate(subs)
    assert r["has_written"] is True
    assert r["has_video"] is True
    assert r["multi_modality"] is True


def test_complete_when_all_rules_met():
    # 12 submissions, all 12 modules covered, both written + video present
    subs = [_sub([i], ["article"]) for i in range(1, 13)]
    subs[0]["formats"] = ["article", "talking_head_unscripted"]  # add a video
    r = cr.evaluate(subs)
    assert r["approved_count"] == 12
    assert r["modules_missing"] == []
    assert r["has_written"] and r["has_video"]
    assert r["complete"] is True
    assert r["reasons"] == []


def test_reasons_list_unmet_rules():
    r = cr.evaluate([_sub([1], ["article"])])
    joined = " ".join(r["reasons"]).lower()
    assert "12" in joined          # needs >=12
    assert "module" in joined      # missing modules
    assert "video" in joined       # no video yet
