"""
TCM Five-Element resonance mapper.

Takes a 48-dim emotion vector (Hume's vocabulary, now produced by Haiku in
the T2 pipeline since Hume's Expression Measurement API is sunset
2026-06-14) and maps it to the TCM Five-Element framework + Shen.

In T2 this module is no longer the primary scorer — Haiku produces direct
TCM scores (elements + treasures) per entry. This mapper now serves as a
QA cross-check via `compare_haiku_to_mapper`, surfacing prompt-quality drift
when the rule-based mapping diverges from Haiku's direct read.

Note on Shen: in TCM, Shen is a Treasure (San Bao), not an Element. This
mapper retains Shen in its 6-bucket output for historical compatibility,
but `project_to_five_elements` collapses to the canonical 5-element axis
for comparison with the T2 schema.

Public surface:
    map_emotions_to_elements(emotion_scores) -> dict
        6-bucket output {Wood, Fire, Earth, Metal, Water, Shen} sum=100
    project_to_five_elements(elements_6) -> dict
        Drop Shen, renormalize remaining 5 to sum=100
    compare_haiku_to_mapper(haiku_emotions, haiku_elements) -> dict
        QA cross-check; returns mapper-derived elements + delta surface
"""

# Element targets and their weights for each Hume emotion.
# Weights for a single emotion sum to 1.0 — emotions can route to multiple
# elements (e.g. Anxiety bridges Water and Fire).
EMOTION_TO_ELEMENT = {
    # ----- Wood — Liver / Gallbladder (anger, frustration, irritation) -----
    "Anger":          {"Wood": 1.0},
    "Contempt":       {"Wood": 0.7, "Fire": 0.3},
    "Determination":  {"Wood": 0.6, "Fire": 0.4},
    "Disgust":        {"Wood": 0.8, "Earth": 0.2},
    "Envy":           {"Wood": 0.7, "Water": 0.3},
    "Pride":          {"Wood": 0.5, "Fire": 0.5},

    # ----- Fire — Heart / Small Intestine / Pericardium -----
    # joy in excess, mania, ecstasy, anxiety-as-overstimulation
    "Ecstasy":        {"Fire": 1.0},
    "Excitement":     {"Fire": 0.9, "Wood": 0.1},
    "Joy":            {"Fire": 0.6, "Shen": 0.4},
    "Triumph":        {"Fire": 0.7, "Wood": 0.3},
    "Romance":        {"Fire": 0.7, "Earth": 0.3},
    "Love":           {"Fire": 0.5, "Shen": 0.5},
    "Desire":         {"Fire": 0.6, "Wood": 0.4},
    "Craving":        {"Fire": 0.5, "Earth": 0.5},
    "Amusement":      {"Fire": 0.5, "Shen": 0.5},
    "Surprise (positive)": {"Fire": 0.6, "Shen": 0.4},

    # ----- Earth — Spleen / Stomach (worry, contemplation, rumination) -----
    "Contemplation":  {"Earth": 1.0},
    "Confusion":      {"Earth": 1.0},
    "Doubt":          {"Earth": 0.7, "Water": 0.3},
    "Sympathy":       {"Earth": 0.7, "Shen": 0.3},
    "Empathic Pain":  {"Earth": 0.6, "Metal": 0.4},
    "Boredom":        {"Earth": 0.7, "Metal": 0.3},
    "Concentration":  {"Earth": 0.5, "Wood": 0.5},
    "Interest":       {"Earth": 0.5, "Wood": 0.5},

    # ----- Metal — Lung / Large Intestine (sadness, grief, regret) -----
    "Sadness":        {"Metal": 1.0},
    "Nostalgia":      {"Metal": 0.8, "Earth": 0.2},
    "Disappointment": {"Metal": 0.8, "Earth": 0.2},
    "Pain":           {"Metal": 0.7, "Water": 0.3},
    "Tiredness":      {"Metal": 0.5, "Earth": 0.5},

    # ----- Water — Kidney / Bladder (fear, dread, distress) -----
    "Fear":           {"Water": 1.0},
    "Anxiety":        {"Water": 0.6, "Fire": 0.4},
    "Distress":       {"Water": 0.8, "Wood": 0.2},
    "Horror":         {"Water": 0.9, "Wood": 0.1},
    "Shame":          {"Water": 0.6, "Earth": 0.4},
    "Embarrassment":  {"Water": 0.5, "Earth": 0.5},
    "Guilt":          {"Water": 0.5, "Earth": 0.5},
    "Awkwardness":    {"Earth": 0.6, "Water": 0.4},
    "Surprise (negative)": {"Water": 0.6, "Wood": 0.4},

    # ----- Shen — transcendent / aesthetic / unifying -----
    "Awe":                    {"Shen": 1.0},
    "Aesthetic Appreciation": {"Shen": 1.0},
    "Calmness":               {"Shen": 0.8, "Earth": 0.2},
    "Contentment":            {"Shen": 0.6, "Earth": 0.4},
    "Satisfaction":           {"Shen": 0.5, "Earth": 0.5},
    "Relief":                 {"Shen": 0.5, "Metal": 0.3, "Water": 0.2},
    "Adoration":              {"Shen": 0.7, "Fire": 0.3},
    "Admiration":             {"Shen": 0.6, "Fire": 0.4},
    "Realization":            {"Shen": 0.6, "Earth": 0.4},
    "Entrancement":           {"Shen": 0.7, "Fire": 0.3},
}

ELEMENT_ORGANS = {
    "Wood":  "Liver / Gallbladder",
    "Fire":  "Heart / Small Intestine / Pericardium",
    "Earth": "Spleen / Stomach",
    "Metal": "Lung / Large Intestine",
    "Water": "Kidney / Bladder",
    "Shen":  "Transcendent / neutral bucket",
}


def map_emotions_to_elements(emotion_scores: dict, top_k: int = 3) -> dict:
    """
    Args:
        emotion_scores: {"Sadness": 0.78, "Anxiety": 0.42, ...} — Hume Prosody
            emotion-name → score (typically 0–1, but any non-negative scale works).
        top_k: how many top emotions to return.

    Returns:
        {
            "elements": {"Wood": 12.3, "Fire": 4.1, "Earth": 9.0, "Metal": 51.2,
                         "Water": 18.0, "Shen": 5.4},   # percentages, sum = 100
            "dominant": "Metal",
            "top_emotions": [{"name": "Sadness", "score": 0.78}, ...]
        }
    """
    elements = {"Wood": 0.0, "Fire": 0.0, "Earth": 0.0, "Metal": 0.0,
                "Water": 0.0, "Shen": 0.0}

    for emotion, score in emotion_scores.items():
        if score is None or score < 0:
            continue
        weights = EMOTION_TO_ELEMENT.get(emotion)
        if not weights:
            continue
        for element, weight in weights.items():
            elements[element] += float(score) * weight

    total = sum(elements.values())
    if total > 0:
        elements_pct = {k: round(v / total * 100, 1) for k, v in elements.items()}
    else:
        elements_pct = {k: 0.0 for k in elements}

    dominant = max(elements_pct, key=elements_pct.get) if total > 0 else None

    sorted_emotions = sorted(
        ((name, float(score)) for name, score in emotion_scores.items() if score),
        key=lambda x: x[1],
        reverse=True,
    )
    top_emotions = [{"name": n, "score": round(s, 3)} for n, s in sorted_emotions[:top_k]]

    return {
        "elements": elements_pct,
        "dominant": dominant,
        "top_emotions": top_emotions,
    }


def project_to_five_elements(elements_6: dict) -> dict:
    """Drop Shen, renormalize the other 5 to sum to 100.

    Used to align the legacy 6-bucket mapper output with the T2 schema's
    canonical Five-Element axis (Shen migrates to the Treasures axis in T2).
    """
    five = {k: float(elements_6.get(k, 0) or 0)
            for k in ("Wood", "Fire", "Earth", "Metal", "Water")}
    total = sum(five.values())
    if total <= 0:
        return {k: 0.0 for k in five}
    return {k: round(v / total * 100, 1) for k, v in five.items()}


def compare_haiku_to_mapper(haiku_emotions: dict,
                            haiku_elements: dict) -> dict:
    """QA cross-check for the T2 pipeline.

    Runs the rule-based mapper on Haiku's 48-emotion output and compares
    the result to Haiku's direct Five-Element scores. Large deltas surface
    drift between the mapper's curated weights and Haiku's clinical reading
    — useful for prompt-quality monitoring.

    Args:
        haiku_emotions:  Haiku's 48-emotion vector  (Hume vocabulary keys)
        haiku_elements:  Haiku's direct elements    {Wood, Fire, Earth, Metal, Water}

    Returns:
        {
            "mapper_elements": {Wood, Fire, Earth, Metal, Water},  # 5-bucket projection
            "delta": float,            # 0 (identical) … 100 (opposite)
            "delta_per_element": {Wood: ..., Fire: ..., ...},
        }
    """
    mapper_full = map_emotions_to_elements(haiku_emotions)
    mapper_proj = project_to_five_elements(mapper_full["elements"])

    delta_per = {
        el: round(abs(mapper_proj[el] - float(haiku_elements.get(el, 0) or 0)), 1)
        for el in mapper_proj
    }
    # L1 / 2 — bounded 0…100. (Two distributions on the same simplex differing
    # by total mass M have a sum-of-abs-differences of 2M, so /2 normalizes.)
    delta = round(sum(delta_per.values()) / 2, 1)

    return {
        "mapper_elements": mapper_proj,
        "delta": delta,
        "delta_per_element": delta_per,
    }


if __name__ == "__main__":
    # Quick smoke test
    sample_emotions = {
        "Sadness": 0.78,
        "Tiredness": 0.55,
        "Anxiety": 0.42,
        "Nostalgia": 0.30,
        "Calmness": 0.18,
        "Joy": 0.05,
    }
    result = map_emotions_to_elements(sample_emotions)
    print("6-bucket elements:", result["elements"])
    print("Dominant:", result["dominant"])
    print("Top emotions:", result["top_emotions"])

    five = project_to_five_elements(result["elements"])
    print("5-bucket projection:", five)

    # Simulate Haiku output that mostly agrees but slightly under-reads Metal
    haiku_elements = {"Wood": 5, "Fire": 5, "Earth": 15, "Metal": 55, "Water": 20}
    qa = compare_haiku_to_mapper(sample_emotions, haiku_elements)
    print("QA cross-check:", qa)
