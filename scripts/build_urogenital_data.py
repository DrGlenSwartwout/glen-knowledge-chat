#!/usr/bin/env python3
"""Build data/bodymap-urogenital.json — the urogenital system atlas.

One map with a MALE / FEMALE view toggle (the `side` axis). The urinary tract
(kidneys, ureters, bladder, urethra) appears in both views; the reproductive
organs are shown per sex. Each named so a client's finding (Kidney, Bladder,
Prostate, Ovaries, Uterus...) lights the structure.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bodymap_atlas_lib as lib

GROUPS = [
    {"id": "urinary", "label": "Urinary tract"},
    {"id": "reproductive", "label": "Reproductive organs"},
]
e, p, z, cr = lib.ellipse, lib.path, lib.zone, lib.catmull

# urinary tract, shown in BOTH views
URINARY = [
    ("kidney-r", "Kidney (right)", e(0.450, 0.424, 0.022, 0.032), "Right kidney — filters blood, makes urine."),
    ("kidney-l", "Kidney (left)", e(0.550, 0.424, 0.022, 0.032), "Left kidney — filters blood, makes urine."),
    ("ureter-r", "Ureter (right)", p(cr([(0.456, 0.452), (0.470, 0.500), (0.486, 0.528)])), "Right ureter — kidney to bladder."),
    ("ureter-l", "Ureter (left)", p(cr([(0.544, 0.452), (0.530, 0.500), (0.514, 0.528)])), "Left ureter — kidney to bladder."),
    ("bladder", "Bladder", e(0.500, 0.538, 0.026, 0.024), "Urinary bladder — stores urine."),
    ("urethra", "Urethra", p(cr([(0.500, 0.560), (0.500, 0.575), (0.500, 0.590)])), "Urethra — carries urine out."),
]
Z = []
for view in ("female", "male"):
    for slug, name, geom, meaning in URINARY:
        Z.append(z(f"uro-{view}-{slug}", name, view, "urinary", geom, meaning))

# female reproductive
Z += [
    z("uro-female-uterus", "Uterus", "female", "reproductive", e(0.500, 0.552, 0.020, 0.018), "Uterus — the womb."),
    z("uro-female-cervix", "Cervix", "female", "reproductive", e(0.500, 0.572, 0.010, 0.010), "Cervix — neck of the uterus."),
    z("uro-female-ovary-r", "Ovary (right)", "female", "reproductive", e(0.458, 0.548, 0.013, 0.013), "Right ovary — eggs and sex hormones."),
    z("uro-female-ovary-l", "Ovary (left)", "female", "reproductive", e(0.542, 0.548, 0.013, 0.013), "Left ovary — eggs and sex hormones."),
    z("uro-female-tube-r", "Fallopian tube (right)", "female", "reproductive", p(cr([(0.470, 0.548), (0.484, 0.546), (0.494, 0.550)])), "Right fallopian tube."),
    z("uro-female-tube-l", "Fallopian tube (left)", "female", "reproductive", p(cr([(0.530, 0.548), (0.516, 0.546), (0.506, 0.550)])), "Left fallopian tube."),
    z("uro-female-vagina", "Vagina", "female", "reproductive", e(0.500, 0.588, 0.010, 0.014), "Vagina."),
]
# male reproductive
Z += [
    z("uro-male-prostate", "Prostate", "male", "reproductive", e(0.500, 0.560, 0.016, 0.012), "Prostate gland — surrounds the urethra."),
    z("uro-male-seminal", "Seminal vesicles", "male", "reproductive", e(0.500, 0.550, 0.018, 0.008), "Seminal vesicles."),
    z("uro-male-testis-r", "Testis (right)", "male", "reproductive", e(0.474, 0.618, 0.014, 0.016), "Right testis — sperm and testosterone."),
    z("uro-male-testis-l", "Testis (left)", "male", "reproductive", e(0.526, 0.618, 0.014, 0.016), "Left testis — sperm and testosterone."),
    z("uro-male-epididymis-r", "Epididymis (right)", "male", "reproductive", e(0.486, 0.628, 0.008, 0.008), "Right epididymis."),
    z("uro-male-epididymis-l", "Epididymis (left)", "male", "reproductive", e(0.514, 0.628, 0.008, 0.008), "Left epididymis."),
    z("uro-male-vas-r", "Vas deferens (right)", "male", "reproductive", p(cr([(0.480, 0.610), (0.490, 0.580), (0.498, 0.562)])), "Right vas deferens."),
    z("uro-male-vas-l", "Vas deferens (left)", "male", "reproductive", p(cr([(0.520, 0.610), (0.510, 0.580), (0.502, 0.562)])), "Left vas deferens."),
    z("uro-male-penis", "Penis", "male", "reproductive", e(0.500, 0.600, 0.010, 0.020), "Penis."),
]

lib.write_system("urogenital", GROUPS, Z, side_noun="sex", group_noun="system", views=("female", "male"))
