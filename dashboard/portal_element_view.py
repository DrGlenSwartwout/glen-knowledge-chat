"""Shape a member's element state for the portal API: adds `setting` (the
lowercased deficient element = the Glendalf backdrop to show)."""
from dashboard import member_element_state


def element_view(cx, email):
    row = member_element_state.get(cx, email)
    if not row:
        return None
    dfc = row.get("deficient_element")
    row["setting"] = dfc.lower() if dfc else None
    # The member's explicit backdrop choice (element key) overriding `setting`;
    # None/absent means Automatic (use the computed deficient element).
    ov = (row.get("scene_override") or "").strip().lower()
    row["override"] = ov or None
    return row
