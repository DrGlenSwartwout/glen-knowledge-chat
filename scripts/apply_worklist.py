#!/usr/bin/env python3
"""Apply Glen's Products-Worklist.md formula decisions to data/products.json.

Reproducible, idempotent edits for the products named in sections A, B, C of
`00 System/Products-Worklist.md`. Only the named products are touched.

Run:  python3 scripts/apply_worklist.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PRODUCTS_PATH = os.path.join(REPO, "data", "products.json")

# ---------------------------------------------------------------------------
# Section A: append confirmed "extra" ingredients (the Bioavailability Factors
# blend + a few others Glen confirmed belong). Doses unknown for the appended
# extras -> dose "". Existing ingredients preserved except explicit removals.
# ---------------------------------------------------------------------------
SECTION_A = {
    "brain-boost": {
        "add": ["Centrophenoxine", "Citicoline", "Fulvic Acid", "11 Spirit Minerals"],
        "remove": ["Coluracetam"],  # Glen: keep all except DHA and Coluracetam; DHA not added.
        "note": ("Section A: appended confirmed Bioavailability-Factors extras "
                 "(Centrophenoxine, Citicoline, Fulvic Acid, 11 Spirit Minerals); "
                 "removed Coluracetam per Glen; DHA intentionally NOT added. "
                 "Source: Glen's Products-Worklist.md."),
    },
    "longevity": {
        "add": [
            "EGb 761 (Ginkgo biloba)", "Vitamin B3 (Niacin)",
            "Moringa oleifera (Miracle Tree 100:1)", "Licorice (Glycyrrhiza glabra)",
            "Vinpocetine (Vinca minor)", "Gingerol 10% (Zingiber officinalis)",
            "Piperine (Piper nigrum)", "Fulvic Acid", "Humic Acid", "11 Spirit Minerals",
        ],
        "remove": [],
        "note": ("Section A: appended the 10 Bioavailability-Factors ingredients "
                 "per Glen. Source: Glen's Products-Worklist.md."),
    },
    "muscle-mass": {
        "add": [
            "Phosphatides 50% (Helianthus annua)", "EGb 761 (Ginkgo biloba)",
            "Vitamin B3 (Niacin)", "Moringa oleifera (Miracle Tree 100:1)",
            "Licorice (Glycyrrhiza glabra)", "Vinpocetine (Vinca minor)",
            "Gingerol 10% (Zingiber officinalis)", "Piperine (Piper nigrum)",
            "Fulvic Acid", "Humic Acid", "11 Spirit Minerals",
        ],
        "remove": [],
        "note": ("Section A: appended Bioavailability-Factors extras per Glen. "
                 "Source: Glen's Products-Worklist.md."),
    },
    "seaamino-syntropy": {
        "add": [
            "Vitamin B13: Phosphatides 50% (Helianthus annua)",
            "Fulvic Acid", "Humic Acid", "11 Spirit Minerals",
        ],
        "remove": [],
        "note": ("Section A: appended Bioavailability-Factors extras per Glen. "
                 "Source: Glen's Products-Worklist.md."),
    },
    "stone-solvent": {
        # Glen: bioavailability factors + the others are all included.
        "add": [
            "Magnesium (Magnesium Sulfate)", "Dandelion (Taraxacum officinale)",
            "Cynarin 10% (Cynara scolymus)", "Curcumin (Curcuma longa)",
            "Silymarin (Silybum marianum)", "EGb 761 (Ginkgo biloba)",
            "Vitamin B3 (Niacin)", "Moringa 100:1 (Moringa oleifera)",
            "Licorice Omnipotent (Glycyrrhiza glabra)",
            "Gingerol 10% (Zingiber officinalis)", "Piperine (Piper nigrum)",
            "Vinpocetine (Vinca minor)",
        ],
        "remove": [],
        "note": ("Section A: appended all GK extras per Glen (Bioavailability "
                 "Factors + Magnesium/Dandelion/Cynarin/Curcumin/Silymarin). "
                 "Source: Glen's Products-Worklist.md."),
    },
    "vitamin-d-syntropy": {
        "add": ["Fulvic Acid", "Humic Acid", "Spirit Minerals"],
        "remove": [],
        "note": ("Section A: appended Bioavailability-Factors extras per Glen "
                 "(Fulvic Acid, Humic Acid, Spirit Minerals). "
                 "Source: Glen's Products-Worklist.md."),
    },
}

# ---------------------------------------------------------------------------
# Section B: replace ingredients with Glen's pasted authoritative formula.
# ---------------------------------------------------------------------------
HEART_HEALTH_INGREDIENTS = [
    {"name": "Selenium (as Methylselenocysteine)", "dose": "100 mcg"},
    {"name": "Magnesium (as Taurate and Ascorbate)", "dose": "8.2 mg"},
    {"name": "Vitamin C (as Magnesium Ascorbate)", "dose": "60 mg"},
    {"name": "Vitamin D3 (Cholecalciferol)", "dose": "1000 IU"},
    {"name": "Vitamin E (Mixed Tocopherols)", "dose": "30 IU"},
    {"name": "CoQ10 (Ultra Absorption)", "dose": "10 mg"},
    {"name": "DHEA-S (Dehydroepiandrosterone Sulfate)", "dose": "5 mg"},
    {"name": "Lumbrokinase (Lumbricus rubellus) 178,000 LKU", "dose": "10 mg"},
    {"name": "Curcumin (Curcuma longa)", "dose": "50 mg"},
    {"name": "Inosine", "dose": "10 mg"},
    {"name": "MCP (Modified Citrus Pectin)", "dose": "30 mg"},
    {"name": "Salvianolic Acid 20% (Salvia miltiorrhiza)", "dose": "10 mg"},
    {"name": "Astaxanthin (Haematococcus pluvialis)", "dose": "4 mg"},
    {"name": "Flavonoids 85% (Crataegus oxyacantha)", "dose": "20 mg"},
    {"name": "Reishi 30:1 (Ganoderma lucidum)", "dose": "20 mg"},
    {"name": "Catechins 80% (Camelia sinensis)", "dose": "10 mg"},
    {"name": "Humic Acid", "dose": "1 mg"},
    # Bioavailability Factors:
    {"name": "EGb 761 (Ginkgo biloba)", "dose": "13 mg"},
    {"name": "Vitamin B3 (as Niacin)", "dose": "8 mg"},
    {"name": "Moringa 100:1 (Moringa oleifera)", "dose": "5 mg"},
    {"name": "Licorice Omnipotent (Glycyrrhiza glabra)", "dose": "3 mg"},
    {"name": "Gingerol 10% (Zingiber officinalis)", "dose": "2 mg"},
    {"name": "Piperine (Piper nigrum)", "dose": "1 mg"},
    {"name": "Vinpocetine (Vinca minor)", "dose": "1 mg"},
]

C15_INGREDIENTS = [
    {"name": "C15 (Pentadecanoic Acid)", "dose": "100 mg"},
    {"name": "Sodium Propionate", "dose": "100 mg"},
    {"name": "ALCAR (Acetyl L-Carnitine) [HYGROSCOPIC]", "dose": "5 mg"},
    {"name": "Bioavailability Blend", "dose": "34.5 mg"},
    {"name": "Pullulan 00 vegi capsule", "dose": "1 ea"},
    {"name": "Vitamin B13: Alpha GPC 99%", "dose": "65 mg"},
    {"name": "Vitamin B13: DMAE (Dimethylaminoethanol) Bitartrate", "dose": "65 mg"},
    {"name": "Vitamin B13: Centrophenoxine 99%", "dose": "65 mg"},
    {"name": "Vitamin B13: Choline Bitartrate", "dose": "65 mg"},
]

SERENITY_INGREDIENTS = [
    {"name": "Monk Fruit Mogroside V 50% (Siraitia grosvenorii)", "dose": "20 mg"},
    {"name": "Yuzu Spray-Dried Fruit Juice (Citrus junos)", "dose": "50 mg"},
    {"name": "Barley (Hordeum vulgare) Grass Juice Powder Organic", "dose": "100 mg"},
    {"name": "Potassium Citrate", "dose": "100 mg"},
    {"name": "Spirulina (Spirulina platensis)", "dose": "100 mg"},
    {"name": "Phycocyanin 99% (Spirulina platensis)", "dose": "30 mg"},
    {"name": "Collard (Brassica oleracea folia)", "dose": "100 mg"},
]

# ---------------------------------------------------------------------------
# Section C: bundles.
# ---------------------------------------------------------------------------
DRY_EYE_DESC = (
    "The Dry Eye Relief Program provides an automatic monthly supply of key "
    "remedies to support eye comfort and healthy tear production including:\n\n"
    "ACES Eye Drops (1 drop in each eye, AM and PM)\n\n"
    "Moisturize capsules (1 per day with food)\n\n"
    "WholOmega capsules (4 per day with the heaviest meal)"
)

GLUCOSE_DESC = (
    "About one-third of aging-related changes are directly related to sugar "
    "regulation. If you have an issue with sugar regulation such as metabolic "
    "syndrome, pre-diabetes, or diabetes, adding a regular program to "
    "specifically support healing your sugar regulation and reverse related "
    "degenerative tissue changes is essential.\n\n"
    "The Glucose Tolerance Program supplies key remedies monthly to support "
    "healthy sugar metabolism:\n\n"
    "Glucose Tolerance\n\n"
    "Reverse AGE\n\n"
    "You can order either or both of these remedies individually on their "
    "product pages or you can order the program from this page. Each product is "
    "designed to last for one month, and you can order the program to be "
    "automatically delivered monthly. We extend savings to you when you set up "
    "an automatic shipment, and you can always cancel at any time. If you know "
    "you will be using a product for at least 6 months or a year, you can access "
    "even greater savings by purchasing in bulk.\n\n"
    "In any case, whatever challenges you are facing, if you are not achieving "
    "your health and wellness goals, including maintenance or recovery of "
    "vision, the next step beyond a program like this designed to support and "
    "optimize your healing processes in the face of a particular issue or "
    "symptom, you can apply for a consultation with us. We can evaluate what "
    "your healing powers are focused on achieving right now and design a "
    "precision program tailor-made to your body's intelligent healing processes "
    "that are active - in real time. We typically see the reversal of about one "
    "year of aging and degeneration in a one-month program using our systems, "
    "which is why we call it Accelerated Self Healing™."
)

MACULAR_DESC = (
    "The Macular Wellness Program supplies:\n\n"
    "Macular Wellness Lycopene (with a different meal than WholOmega and "
    "Astaxanthin)\n\n"
    "Macular Wellness Astaxanthin (with a different meal than WholOmega and "
    "Lycopene)\n\n"
    "Lipid Cleanse capsules - with meals\n\n"
    "Lipid Zyme capsules - between meals \n\n"
    "WholOmega capsules (4/day with the heaviest meal, and separate from the "
    "Macular Wellness formulas)\n\n"
    "If you are having trouble with night vision, take our Night Vision "
    "formula.\n\n"
    "If your Macular Pigment Optical Density (MPOD) test is low, add the "
    "appropriate Macular Wellness formulation with the carotenoid that the body "
    "stores in the area(s) of the retina showing visual or tissue changes. The "
    "fovea is responsible for sharp visual acuity better than 20/60. Any "
    "distortion of vision can be mapped with an Amsler Grid to track the "
    "location to the Parafoveal or outer Macula area. Some tests can "
    "distinguish between low Lutein and low levels of the other macular "
    "carotenoids.\n\n"
    "Macula: Macular Wellness Lutein\n\n"
    "Parafovea: Macular Wellness Zeaxanthin\n\n"
    "Fovea: Macular Wellness Meso-Zeaxanthin\n\n"
    "If your Macular Degeneration is moderately advanced, or already progressed "
    "into the wet stage, please also apply for consultation with us to develop "
    "an individualized program.\n\n"
    "If there is scar tissue in the retina, add Clear the Way capsules between "
    "meals.\n\n"
    "If there is leakage of fluids in the retina or the growth of new blood "
    "vessels, consider adding AngiogenX in consultation with your trusted "
    "source of health and wellness guidance. Avoid taking AngiogenX before, "
    "during, and after surgeries or after any other trauma or tissue damage "
    "that requires cicatrization (wound healing) and growth of new blood "
    "vessels in the process of repair."
)

SECTION_C = {
    "dry-eye-relief-program": {
        "components": ["ACES Eye Drops", "Moisturize", "WholOmega"],
        "description": DRY_EYE_DESC,
        "note": ("Section C: marked as bundle (multi-product monthly program), "
                 "not a single formula. Source: Glen's Products-Worklist.md."),
    },
    "glucose-tolerance-program": {
        "components": ["Glucose Tolerance", "Reverse AGE"],
        "description": GLUCOSE_DESC,
        "note": ("Section C: marked as bundle (multi-product monthly program), "
                 "not a single formula. Source: Glen's Products-Worklist.md."),
    },
    "macular-wellness-program": {
        "components": [
            "Macular Wellness Lycopene", "Macular Wellness Astaxanthin",
            "Lipid Cleanse", "Lipid Zyme", "WholOmega",
        ],
        "description": MACULAR_DESC,
        "note": ("Section C: marked as bundle (multi-product monthly program), "
                 "not a single formula. Source: Glen's Products-Worklist.md."),
    },
}

GK_FLAGS_TO_CLEAR = ("gk_has_extra", "gk_extra_accepted", "gk_stale",
                     "gk_stale_reason")


def main():
    with open(PRODUCTS_PATH) as f:
        data = json.load(f)
    prods = data["products"]

    changed = []

    # --- Section A ---
    for slug, spec in SECTION_A.items():
        p = prods[slug]
        before = len(p.get("ingredients") or [])
        ings = list(p.get("ingredients") or [])
        # remove
        for rm in spec["remove"]:
            ings = [i for i in ings if i.get("name") != rm]
        # append (idempotent: skip if name already present)
        existing_names = {i.get("name") for i in ings}
        for name in spec["add"]:
            if name not in existing_names:
                ings.append({"name": name, "dose": ""})
                existing_names.add(name)
        p["ingredients"] = ings
        p["ingredients_source"] = "manual"
        p["enrichment_note"] = spec["note"]
        for fk in GK_FLAGS_TO_CLEAR:
            p.pop(fk, None)
        changed.append((slug, before, len(ings)))

    # --- Section B ---
    b_specs = {
        "heart-health": {
            "name": "Heart Health",
            "ingredients": HEART_HEALTH_INGREDIENTS,
            "note": ("Section B: replaced with Glen's authoritative 'Heart Health "
                     "4' formula (Formulations DB 6/14/23). NOTE: Rhythm Restore "
                     "(TCM Wenxin Keli) is a SEPARATE product needing its own "
                     "catalog entry - ingredients: Codonopsis pilosula, "
                     "Polygonatum sibiricum, Panax notoginseng (Notoginsenosides "
                     "80%), Amber (Succinic Acid), Nardostachys jatamansi "
                     "(Spikenard). Those are NOT included here. "
                     "Source: Glen's Products-Worklist.md."),
        },
        "c15-syntropy-pentadecanoic-acid": {
            "name": None,
            "ingredients": C15_INGREDIENTS,
            "note": ("Section B: replaced with Glen's authoritative C15 formula. "
                     "Dose-to-ingredient pairing is best-effort (Glen pasted "
                     "doses and names in separate columns) - awaiting Glen's "
                     "confirmation. Source: Glen's Products-Worklist.md."),
        },
        "serenity": {
            "name": "Serene Blue Green",
            "ingredients": SERENITY_INGREDIENTS,
            "note": ("Section B: replaced with Glen's authoritative Serene Blue "
                     "Green formula. Dose-to-ingredient pairing is best-effort "
                     "(doses/names pasted in separate columns) - awaiting Glen's "
                     "confirmation. Source: Glen's Products-Worklist.md."),
        },
    }
    for slug, spec in b_specs.items():
        p = prods[slug]
        before = len(p.get("ingredients") or [])
        if spec["name"] is not None:
            p["name"] = spec["name"]
        p["ingredients"] = [dict(i) for i in spec["ingredients"]]
        p["ingredients_source"] = "manual"
        p["enrichment_note"] = spec["note"]
        for fk in GK_FLAGS_TO_CLEAR:
            p.pop(fk, None)
        changed.append((slug, before, len(p["ingredients"])))

    # --- Section C ---
    for slug, spec in SECTION_C.items():
        p = prods[slug]
        before = len(p.get("ingredients") or [])
        p["bundle"] = True
        p["bundle_components"] = list(spec["components"])
        p["bundle_description"] = spec["description"]
        p["ingredients"] = []
        p["ingredients_source"] = "manual"
        p["enrichment_note"] = spec["note"]
        changed.append((slug, before, 0))

    with open(PRODUCTS_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("Applied worklist. Changed products (slug, before_ings, after_ings):")
    for slug, b, a in changed:
        print(f"  {slug}: {b} -> {a}")


if __name__ == "__main__":
    main()
