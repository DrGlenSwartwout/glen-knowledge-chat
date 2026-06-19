"""People-directory pure helpers (no DB, no Flask) so tag logic is unit-testable."""

import json

MAX_TAG_LEN = 64


def set_person_tags(current_tags, add=None, remove=None):
    """Apply remove-then-add to current_tags and return the new list.

    Normalization: each tag is str-stripped; empties dropped; added tags longer
    than MAX_TAG_LEN dropped; de-duplicated case-sensitively, first-seen order.
    Existing-tag order is preserved (minus removals); new adds are appended.
    """
    add = add or []
    remove = remove or []
    remove_set = {t.strip() for t in remove if isinstance(t, str)}

    result = []
    seen = set()
    for t in current_tags:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t or t in remove_set or t in seen:
            continue
        seen.add(t)
        result.append(t)
    for t in add:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t or len(t) > MAX_TAG_LEN or t in seen:
            continue
        seen.add(t)
        result.append(t)
    return result


def distinct_tags(tag_lists):
    """Union of all tags across people, de-duplicated, stripped, sorted ascending.

    Each item in tag_lists is either a list[str] or a JSON-encoded string of one;
    malformed/None/non-iterable entries are skipped, as are empty/whitespace tags.
    """
    out = set()
    for entry in tag_lists:
        if isinstance(entry, str):
            try:
                entry = json.loads(entry)
            except (ValueError, TypeError):
                continue
        if not isinstance(entry, list):
            continue
        for t in entry:
            if isinstance(t, str):
                t = t.strip()
                if t:
                    out.add(t)
    return sorted(out)
