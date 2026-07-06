"""Remedy schedule: place each remedy into times-of-day slots with a food
relationship, derived from its dosage + frequency + timing.

Multiple capsules per day are DIVIDED across different times of day (one capsule
per slot) unless the timing explicitly directs taking them together at a single
time/meal (e.g. WholOmega "with dinner / the heaviest / an oil-rich meal").
Caps/day = capsules-in-dosage x times-in-frequency (each defaulting to 1); only a
clean whole-capsule dose is auto-split, so droppers/fractions/"as directed" are
left as a single entry. Unknown combinations are never dropped; they fall back to
an "as directed" row so nothing silently disappears from a patient's schedule.
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
# neutral spread when no food relationship is specified (anchored to meal times)
_SPREAD = {1: ["Breakfast"], 2: ["Breakfast", "Dinner"],
           3: ["Breakfast", "Lunch", "Dinner"],
           4: ["Breakfast", "Lunch", "Dinner", "Bedtime"]}

_WORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_UNIT = r"(?:cap|capsule|tab|tablet|pill|softgel|gel|lozenge)"


def _meal_slot(t):
    if "with breakfast" in t:
        return "Breakfast"
    if "with lunch" in t:
        return "Lunch"
    if "with dinner" in t or "with supper" in t:
        return "Dinner"
    return None


def _freq_count(freq):
    """Times per day from the frequency text. None if unknown. Handles number
    words and digits (so 'two a day' / '2 a day' -> 2, not the generic 'a day' -> 1)."""
    f = (freq or "").lower()
    if any(k in f for k in ("four", "4 time", "4x", "4/day", "qid")):
        return 4
    if any(k in f for k in ("three", "thrice", "3 time", "3x", "3/day", "tid")):
        return 3
    if any(k in f for k in ("twice", "two time", "2 time", "2x", "2/day", "bid",
                            "two a day", "two per day", "two daily")):
        return 2
    m = re.match(r"\s*(\d+)", f)           # leading digit: '2 a day', '2 daily'
    if m:
        n = int(m.group(1))
        if 1 <= n <= 6:
            return n
    if any(k in f for k in ("once", "daily", "a day", "per day", "every day",
                            "1 time", "1x", "qd", "one a day")):
        return 1
    return None


def _dose_count(dosage):
    """Whole capsules per administration, or None if the dose is not a clean
    whole-capsule count (droppers, fractions, 'as directed' -> not auto-split)."""
    d = (dosage or "").lower().strip()
    m = re.match(rf"(\d+)\s*{_UNIT}", d)
    if m:
        return int(m.group(1))
    m = re.match(rf"(one|two|three|four|five|six)\s+{_UNIT}", d)
    if m:
        return _WORD[m.group(1)]
    m = re.fullmatch(r"(\d+)", d)          # bare integer
    if m:
        return int(m.group(1))
    return None


def _caps(n):
    return f"{n} capsule" + ("" if n == 1 else "s")


def _split_n(times, dose):
    """(number of slots, capsules per slot). per_slot None -> show the raw dosage."""
    if times and times >= 2:            # honor an explicit multi-times-per-day frequency
        return times, dose
    if dose and dose >= 2:              # divide a multi-capsule daily dose across the day
        return dose, 1
    if times or dose:                   # a single daily administration
        return 1, dose
    return None, None


def _placement(freq, timing, dosage):
    """Return (slots, food, per_slot). slots == [] means it could not be placed
    (as directed). per_slot is capsules-per-slot (int) or None (show raw dosage)."""
    t = (timing or "").lower().strip()
    times = _freq_count(freq)
    dose = _dose_count(dosage)
    cpd = (dose * (times or 1)) if dose is not None else None  # caps/day (if dose known)

    def together(slot, food):
        # explicit single time/meal -> whole daily dose together at one slot
        return [slot], food, cpd

    ms = _meal_slot(t)
    if ms:
        return together(ms, "with food")
    if any(k in t for k in ("heaviest meal", "heavier meal", "with the heaviest",
                            "oil-rich", "oil rich", "fattiest", "fatty meal",
                            "largest meal", "biggest meal")):
        return together("Dinner", "with the heaviest meal")
    if any(k in t for k in ("at night", "before bed", "bedtime", "evening")):
        return together("Bedtime", "")
    if any(k in t for k in ("on rising", "upon rising", "on waking",
                            "first thing", "early in the day")):
        return together("On waking", "")

    # food relationship -> which slot names
    if any(k in t for k in ("before meal", "before food", "before eating")):
        food, table = "before meals", _MEAL
    elif any(k in t for k in ("between meal", "empty stomach", "away from food", "on an empty")):
        food, table = "between meals", _BETWEEN
    elif any(k in t for k in ("with food", "with meal", "with a meal", "with meals",
                              "with heavier", "with a heavier")):
        food, table = "with food", _MEAL
    else:
        food, table = "", _SPREAD       # no timing given -> neutral spread across the day

    n, per_slot = _split_n(times, dose)
    if n is None:
        return [], food, None           # unknown count -> as directed (food noted if any)
    return list(table.get(n) or table[max(table)]), food, per_slot


def _dose_str(per_slot, raw_dosage):
    if per_slot is not None:
        return _caps(per_slot)
    return (raw_dosage or "").strip()


def _is_terrain_restore(name):
    return "in terrain restore" in (name or "").lower()


def _strip_terrain_restore(name):
    return re.sub(r"\s*in terrain restore\s*$", "", (name or "").strip(), flags=re.I).strip()


def build_schedule(remedies):
    """remedies: [{name, dosage, frequency, timing}] -> a schedule view.

    Liquid remedies named "[name] in Terrain Restore" (essences, homeopathics, tinctures,
    gemmotherapies, peptides, ORMUS) are individualized into ONE combined Terrain Restore
    bottle and shown as a single entry (`contains` lists what's in it), taken together.

    Returns {"slots": SLOTS, "entries": [{name, dosage, per_slot, frequency, timing, slots,
    food, as_directed, contains:[...]}]}. `per_slot` is the amount to take at each slot
    (e.g. "1 capsule" when a multi-cap daily dose is divided across the day).
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
        slots, food, per_slot = _placement(r.get("frequency"), r.get("timing"), r.get("dosage"))
        entries.append({
            "name": (r.get("name") or "").strip(),
            "dosage": (r.get("dosage") or "").strip(),
            "per_slot": _dose_str(per_slot, r.get("dosage")),
            "frequency": (r.get("frequency") or "").strip(),
            "timing": (r.get("timing") or "").strip(),
            "slots": slots,
            "food": food,
            "as_directed": not slots,
            "contains": r.get("contains") or [],
        })
    return {"slots": SLOTS, "entries": entries}
