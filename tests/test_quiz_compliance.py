# tests/test_quiz_compliance.py
import json
from pathlib import Path
import quiz_engine

_BANNED_DISEASE = ["macular", "amd", "glaucoma", "cataract", "alzheimer", "dementia",
                   "cure", "treat ", "prevent ", "reverse "]
_EMDASH = "—"


def _all_quiz_text(include_disclaimer=False):
    # The DSHEA disclaimer is REQUIRED legal text and legitimately contains
    # "prevent ... disease"; it is excluded from the disease-CLAIM scan and
    # validated separately by test_disclaimer_is_dshea. The em-dash scan still
    # covers it (the disclaimer must contain no em-dash).
    cfg = quiz_engine.load_config()
    q = quiz_engine.get_quiz("eye-brain", cfg)
    parts = [q["title"], q["hook"]]
    if include_disclaimer:
        parts.append(q["disclaimer"])
    for qq in q["questions"]:
        parts.append(qq["prompt"])
        parts += [o["label"] for o in qq["options"]]
    for band in q["bands"].values():
        parts += [band["headline"], band["reasoning"]] + band["bullets"]
    return parts


def test_no_disease_claims_in_quiz_copy():
    for s in _all_quiz_text(include_disclaimer=False):
        low = s.lower()
        for b in _BANNED_DISEASE:
            assert b not in low, f"banned term {b!r} in: {s!r}"


def test_no_emdash_in_quiz_copy():
    for s in _all_quiz_text(include_disclaimer=True):
        assert _EMDASH not in s, f"em-dash in: {s!r}"


def test_disclaimer_is_dshea():
    q = quiz_engine.get_quiz("eye-brain")
    assert "not been evaluated by the Food and Drug Administration" in q["disclaimer"]
    assert "not intended to diagnose, treat, cure, or prevent any disease" in q["disclaimer"]


def test_static_result_and_quiz_pages_no_emdash():
    root = Path(__file__).resolve().parent.parent
    for name in ("begin-quiz.html", "begin-quiz-result.html"):
        txt = (root / "static" / name).read_text()
        assert _EMDASH not in txt, f"em-dash in static/{name}"
