#!/usr/bin/env python3
"""Build data/bodymap-organclock.json — the Chinese organ (horary) clock.

The 24-hour meridian clock: each of the 12 organ meridians peaks in a 2-hour
window. Rendered as an annular ring of 12 sectors (unit-circle sector geometry,
like the iris), 30 degrees each, running clockwise from the top. The cycle starts
at the Lung (3-5 AM) at 12 o'clock so no window wraps past 0 degrees. Each sector
is coloured by its Five-Element phase and names its organ + time window, so a
client's organ finding lights that organ's clock sector.

Positions/times are the standard horary clock; a reference diagram for Glen's use.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GROUPS = [
    {"id": "metal", "label": "Metal — Lung / Large Intestine"},
    {"id": "earth", "label": "Earth — Stomach / Spleen"},
    {"id": "fire", "label": "Fire — Heart / Small Intestine / Pericardium / Triple Burner"},
    {"id": "water", "label": "Water — Kidney / Bladder"},
    {"id": "wood", "label": "Wood — Gallbladder / Liver"},
]

# (code, name, element, start_hour, time_label, function)
ORGANS = [
    ("LU", "Lung", "metal", 3, "3–5 AM", "Taking in, letting go; the breath and the day begin."),
    ("LI", "Large intestine", "metal", 5, "5–7 AM", "Elimination and release; a natural time for a bowel movement."),
    ("ST", "Stomach", "earth", 7, "7–9 AM", "Digestion at its strongest; the ideal time for the main meal."),
    ("SP", "Spleen", "earth", 9, "9–11 AM", "Transforming food to energy; peak mental focus."),
    ("HT", "Heart", "fire", 11, "11 AM–1 PM", "Circulation and joy; the Heart houses the spirit (Shen)."),
    ("SI", "Small intestine", "fire", 13, "1–3 PM", "Sorting pure from impure; assimilation and discernment."),
    ("BL", "Bladder", "water", 15, "3–5 PM", "Storing and clearing fluids; good time for study and work."),
    ("KI", "Kidney", "water", 17, "5–7 PM", "Storing essence (Jing); the root of vitality and will."),
    ("PC", "Pericardium", "fire", 19, "7–9 PM", "The Heart's protector; intimacy, relationship, circulation."),
    ("TE", "Triple burner", "fire", 21, "9–11 PM", "Balancing the whole; metabolism and thermoregulation settle for sleep."),
    ("GB", "Gallbladder", "wood", 23, "11 PM–1 AM", "Decisions and courage; deep regeneration begins."),
    ("LR", "Liver", "wood", 1, "1–3 AM", "Detoxification and blood storage; deepest repair."),
]


def _zone(code, name, element, start_hour, time_label, function):
    start = ((start_hour - 3) % 24) * 15  # Lung (3 AM) -> 0 deg at top; clockwise
    return {
        "id": f"clock-{code}", "side": "clock", "bilateral": False, "group": element,
        "radial": {"r_inner": 0.42, "r_outer": 0.95},
        "sector": {"start_deg": start, "end_deg": start + 30},
        "anatomy": f"{name} ({time_label})",
        "meaning_standard": f"{name} meridian peaks {time_label}. {function}",
        "meaning_glen": "",
        "layers": {"embryological_depth": None, "stress_affirmation": None, "touch_for_health": None},
    }


def main():
    zones = [_zone(*o) for o in ORGANS]
    data = {
        "system": "organclock", "reference_frame": "unit_circle",
        "side_noun": "clock", "group_noun": "element",
        "groups": GROUPS,
        "zones": zones,
    }
    out = ROOT / "data" / "bodymap-organclock.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    from collections import Counter
    print(f"wrote {out}: {len(zones)} organ windows", dict(Counter(z["group"] for z in zones)))


if __name__ == "__main__":
    main()
