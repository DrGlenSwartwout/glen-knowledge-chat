"""Remedy schedule: place each remedy into times-of-day slots with a food
relationship, derived from its frequency + timing.

The frequency x timing -> {slots, food} mapping encodes Glen's protocol logic and
is meant to be tuned against the real timing/frequency vocabulary in the FMP data.
Unknown combinations are never dropped; they fall back to an "as directed" row so
nothing silently disappears from a patient's schedule.
"""
import re

SLOTS = ["On waking", "Breakfast", "Mid-morning", "Lunch",
         "Mid-afternoon", "Dinner", "Bedtime"]

# count -> ordered slots, per food relationship
_MEAL = {1: ["Breakfast"], 2: ["Breakfast", "Dinner"],
         3: ["Breakfast", "Lunch", "Dinner"],
         4: ["Breakfast", "Lunch", "Dinner", "Bedtime"]}
_BETWEEN = {1: ["Mid-morning"], 2: ["Mid-morning", "Mid-afternoon"],
            3: ["Mid-morning", "Mid-afternoon", "Bedtime"]}


def _meal_slot(t):
    if "with breakfast" in t:
        return "Breakfast"
    if "with lunch" in t:
        return "Lunch"
    if "with dinner" in t:
        return "Dinner"
    return None


def _count(freq):
    f = (freq or "").lower()
    if "four" in f or "4 time" in f:
        return 4
    if "three" in f or "thrice" in f or "3 time" in f:
        return 3
    if "twice" in f or "two time" in f or "2 time" in f:
        return 2
    if "once" in f or "daily" in f or "a day" in f or "per day" in f or "1 time" in f:
        return 1
    return None


def _placement(freq, timing):
    """Return (slots, food). slots == [] means it could not be placed (as directed)."""
    t = (timing or "").lower().strip()
    # specific single meal ("with lunch")
    ms = _meal_slot(t)
    if ms:
        return [ms], "with food"
    # specific time of day
    if any(k in t for k in ("at night", "before bed", "bedtime", "evening")):
        return ["Bedtime"], ""
    if any(k in t for k in ("on rising", "upon rising", "on waking",
                            "first thing", "early in the day")):
        return ["On waking"], ""
    # food relationship -> count-based meal/between slots
    if any(k in t for k in ("before meal", "before food", "before eating")):
        food, table = "before meals", _MEAL
    elif any(k in t for k in ("between meal", "empty stomach", "away from food", "on an empty")):
        food, table = "between meals", _BETWEEN
    elif any(k in t for k in ("with food", "with meal", "with a meal", "with meals",
                              "with heavier", "with a heavier")):
        food, table = "with food", _MEAL
    else:
        return [], ""  # unknown timing -> as directed
    c = _count(freq)
    if c is None:
        return [], food  # known food, unknown count -> as directed (food noted)
    return list(table.get(c) or table[max(table)]), food


def _is_terrain_restore(name):
    return "in terrain restore" in (name or "").lower()


def _strip_terrain_restore(name):
    return re.sub(r"\s*in terrain restore\s*$", "", (name or "").strip(), flags=re.I).strip()


def build_schedule(remedies):
    """remedies: [{name, dosage, frequency, timing}] -> a schedule view.

    Liquid remedies named "[name] in Terrain Restore" (essences, homeopathics, tinctures,
    gemmotherapies, peptides, ORMUS) are individualized into ONE combined Terrain Restore
    bottle and shown as a single entry (`contains` lists what's in it), taken together.

    Returns {"slots": SLOTS, "entries": [{name, dosage, frequency, timing, slots, food,
    as_directed, contains:[...]}]}.
    """
    remedies = list(remedies or [])
    liquids = [r for r in remedies if _is_terrain_restore(r.get("name"))]
    work = [r for r in remedies if not _is_terrain_restore(r.get("name"))]
    if liquids:
        contains = [_strip_terrain_restore(r.get("name")) for r in liquids]
        freq = next((r.get("frequency") for r in liquids if (r.get("frequency") or "").strip()), "")
        timing = next((r.get("timing") for r in liquids if (r.get("timing") or "").strip()), "")
        work.append({"name": "Terrain Restore", "dosage": "contains: " + ", ".join(contains),
                     "frequency": freq, "timing": timing, "contains": contains})

    entries = []
    for r in work:
        slots, food = _placement(r.get("frequency"), r.get("timing"))
        entries.append({
            "name": (r.get("name") or "").strip(),
            "dosage": (r.get("dosage") or "").strip(),
            "frequency": (r.get("frequency") or "").strip(),
            "timing": (r.get("timing") or "").strip(),
            "slots": slots,
            "food": food,
            "as_directed": not slots,
            "contains": r.get("contains") or [],
        })
    return {"slots": SLOTS, "entries": entries}
