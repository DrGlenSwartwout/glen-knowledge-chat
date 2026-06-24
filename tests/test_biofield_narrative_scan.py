"""The narrative/video prompts can fold in the client's recent E4L voice scan as
corroborating context. The scan arg is optional: omitting it reproduces the prior
prompt exactly (back-compat)."""
from dashboard.biofield_narrative import (
    build_narrative_prompt, generate_narrative,
    build_video_script_prompt, generate_video_script)


def _report():
    return {"test_id": "a1", "client": {"name": "Jane Doe", "email": "jane@x.com"},
            "date": "2026-06-24",
            "layers": [{"layer": 1, "head": "Night", "most_affected": "Night",
                        "remedy": "TMG", "dosage": "1 scoop", "frequency": "daily",
                        "timing": "at night"}],
            "schedule": {"slots": [], "entries": []}}


def _scan():
    return {"status": "fresh", "found": True, "days_ago": 4, "fresh": True,
            "scan_date": "2026-06-20",
            "findings": [{"rank": 1, "code": "LV3", "name": "Liver meridian",
                          "description": "detox and anger"}]}


def test_scan_block_appears_when_scan_given():
    p = build_narrative_prompt(_report(), "notes", scan=_scan())
    usr = p["user"]
    assert "E4L" in usr and "LV3" in usr and "Liver meridian" in usr
    assert "4 days ago" in usr
    # system prompt tells the model to only reference, not invent, scan findings
    assert "do not invent" in p["system"].lower()


def test_no_scan_block_without_scan_arg():
    p = build_narrative_prompt(_report(), "notes")
    assert "E4L" not in p["user"]
    # back-compat: no scan -> system prompt unchanged (no scan guidance bolted on)
    assert "E4L" not in p["system"] and "do not invent" not in p["system"].lower()


def test_none_scan_keeps_system_prompt_clean():
    scan = {"status": "none", "found": False, "findings": [], "days_ago": None}
    assert "E4L" not in build_narrative_prompt(_report(), "notes", scan=scan)["system"]


def test_stale_scan_labeled_stale_in_prompt():
    scan = _scan(); scan.update(status="stale", fresh=False, days_ago=40)
    usr = build_narrative_prompt(_report(), "notes", scan=scan)["user"]
    assert "stale" in usr.lower() and "40 days ago" in usr


def test_none_scan_adds_no_block():
    scan = {"status": "none", "found": False, "findings": [], "days_ago": None}
    assert "E4L" not in build_narrative_prompt(_report(), "notes", scan=scan)["user"]


def test_generate_narrative_passes_scan_to_llm():
    seen = {}
    def fake(system, user):
        seen["u"] = user
        return "Aloha Jane,"
    out = generate_narrative(_report(), "notes", fake, scan=_scan())
    assert "LV3" in seen["u"]
    assert out == "Aloha Jane,"


def test_video_prompt_also_supports_scan():
    usr = build_video_script_prompt(_report(), "notes", scan=_scan())["user"]
    assert "LV3" in usr
    # and is still optional
    assert "E4L" not in build_video_script_prompt(_report(), "notes")["user"]
