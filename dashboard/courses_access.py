from __future__ import annotations

TIER_RANK = {"public": 0, "member": 1, "paid": 2}


def is_visible(lesson_access: str, member_level: int) -> bool:
    required = TIER_RANK.get((lesson_access or "").strip().lower(), 99)
    return member_level >= required


def lock_state(lesson_access: str, member_level: int) -> str:
    if is_visible(lesson_access, member_level):
        return "open"
    if (lesson_access or "").strip().lower() == "member":
        return "locked_register"
    return "locked_upgrade"
