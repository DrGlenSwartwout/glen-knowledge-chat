"""Remedy schedule: place each remedy into times-of-day slots with a food relationship,
derived from its frequency + timing. Unknown combos must never be dropped."""
from dashboard.biofield_schedule import build_schedule


def test_twice_daily_with_food_places_breakfast_and_dinner():
    sched = build_schedule([
        {"name": "Sterol Max", "dosage": "3 capsules",
         "frequency": "twice a day", "timing": "with food"}
    ])
    entry = sched["entries"][0]
    assert entry["food"] == "with food"
    assert set(entry["slots"]) == {"Breakfast", "Dinner"}
    assert entry["as_directed"] is False


def test_between_meals_uses_empty_stomach_slots():
    sched = build_schedule([
        {"name": "Calcium D-Glucarate", "dosage": "1 capsule",
         "frequency": "twice a day", "timing": "between meals"}
    ])
    entry = sched["entries"][0]
    assert entry["food"] == "between meals"
    assert set(entry["slots"]) == {"Mid-morning", "Mid-afternoon"}


def test_at_night_places_bedtime():
    sched = build_schedule([
        {"name": "TMG Powder", "dosage": "1 scoop",
         "frequency": "daily", "timing": "at night"}
    ])
    entry = sched["entries"][0]
    assert entry["slots"] == ["Bedtime"]


def test_before_meals_three_times_places_before_each_meal():
    sched = build_schedule([
        {"name": "Exaltation", "dosage": "10 drops", "frequency": "3 times a day",
         "timing": "ideally 30 minutes before meals with affirmation & visualization"}
    ])
    entry = sched["entries"][0]
    assert entry["food"] == "before meals"
    assert set(entry["slots"]) == {"Breakfast", "Lunch", "Dinner"}


def test_before_food_daily_is_one_before_meal():
    sched = build_schedule([
        {"name": "Glucose Tolerance", "dosage": "1 capsule",
         "frequency": "daily", "timing": "before food"}
    ])
    entry = sched["entries"][0]
    assert entry["food"] == "before meals"
    assert entry["slots"] == ["Breakfast"]


def test_with_heavier_meal_is_with_food():
    sched = build_schedule([
        {"name": "WholOmega", "dosage": "4 capsules", "frequency": "daily",
         "timing": "with heavier meal"}
    ])
    entry = sched["entries"][0]
    assert entry["food"] == "with food"
    assert entry["slots"] == ["Breakfast"]


def test_with_lunch_places_lunch_slot():
    sched = build_schedule([
        {"name": "X", "dosage": "1", "frequency": "daily", "timing": "with lunch"}
    ])
    assert sched["entries"][0]["slots"] == ["Lunch"]


def test_early_in_the_day_is_on_waking():
    sched = build_schedule([
        {"name": "Y", "dosage": "1", "frequency": "daily", "timing": "early in the day"}
    ])
    assert sched["entries"][0]["slots"] == ["On waking"]


def test_terrain_restore_liquids_combine_into_one_bottle():
    sched = build_schedule([
        {"name": "Microbiome", "dosage": "1 capsule", "frequency": "daily", "timing": "with food"},
        {"name": "Perelandra essence in Terrain Restore", "dosage": "in Terrain Restore",
         "frequency": "3 times a day", "timing": "between meals"},
        {"name": "Kidney Homeopathic in Terrain Restore", "dosage": "in Terrain Restore",
         "frequency": "3 times a day", "timing": "between meals"},
    ])
    names = [e["name"] for e in sched["entries"]]
    assert names.count("Terrain Restore") == 1          # the liquids merged into ONE bottle
    assert "Microbiome" in names                        # capsules untouched
    bottle = next(e for e in sched["entries"] if e["name"] == "Terrain Restore")
    assert "Perelandra essence" in bottle["contains"]
    assert "Kidney Homeopathic" in bottle["contains"]
    assert set(bottle["slots"]) == {"Mid-morning", "Mid-afternoon", "Bedtime"}  # 3x/day between meals


def test_no_terrain_restore_means_no_bottle():
    sched = build_schedule([{"name": "Microbiome", "dosage": "1 capsule",
                             "frequency": "daily", "timing": "with food"}])
    assert all(e["name"] != "Terrain Restore" for e in sched["entries"])


def test_unknown_combo_falls_back_to_as_directed_not_dropped():
    sched = build_schedule([
        {"name": "Mystery", "dosage": "1", "frequency": "every third tuesday",
         "timing": "qux"}
    ])
    assert len(sched["entries"]) == 1
    entry = sched["entries"][0]
    assert entry["slots"] == []
    assert entry["as_directed"] is True
