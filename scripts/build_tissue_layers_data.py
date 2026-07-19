#!/usr/bin/env python3
"""Build data/bodymap-tissue-layers.json — the organ -> tissue-sub-layer assignment
map for Dr. Glen Swartwout's 5 Embryological Tissue Layers (the tissue 5 C's).

The 5 layers (deepest -> surface) and their 2 sub-layers each are FIXED (defined in
bodymap_store.TISSUE_LAYERS). This file holds the editable part: which canonical
organ/tissue sits in which sub-layer, plus the keywords that resolve any zone's
anatomy label to its organ. Glen edits the assignments in the tissue-layer editor
(/admin/body-map/tissue-layers); the embryological depth-peel reads them.

Seed assignments are a best-guess starting point from the tissue-layer skill; Glen
refines. A zone resolves to the organ whose LONGEST matching keyword wins, so
'large intestine' beats 'intestine'.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (organ display name, sublayer id, [keywords])  — keywords are lowercase, matched
# on word boundaries against a zone's anatomy label.
ORGANS = [
    # ---- Compression / Urogenital (1a) ----
    ("Kidney", "urogenital", ["kidney", "renal"]),
    ("Bladder", "urogenital", ["bladder"]),
    ("Ureter", "urogenital", ["ureter"]),
    ("Urethra", "urogenital", ["urethra"]),
    ("Prostate", "urogenital", ["prostate"]),
    ("Testis", "urogenital", ["testis", "testes", "testicle"]),
    ("Ovary", "urogenital", ["ovary", "ovaries"]),
    ("Uterus", "urogenital", ["uterus", "womb"]),
    ("Fallopian tube", "urogenital", ["fallopian"]),
    ("Cervix", "urogenital", ["cervix"]),
    ("Vagina", "urogenital", ["vagina"]),
    ("Penis", "urogenital", ["penis"]),
    ("Epididymis", "urogenital", ["epididymis"]),
    ("Seminal vesicles", "urogenital", ["seminal"]),
    ("Reproductive organs", "urogenital", ["reproductive", "genital", "gonad", "vas deferens"]),
    # ---- Compression / Muscle (1b) ----
    ("Skeletal muscle", "muscle", ["muscle", "muscular", "deltoid", "biceps", "triceps",
        "pectoralis", "trapezius", "latissimus", "quadriceps", "hamstring", "gluteus",
        "gastrocnemius", "calf", "sternocleidomastoid", "oblique", "abdominis", "erector",
        "tibialis", "sartorius", "forearm flexor"]),
    # ---- Connection / Bone & connective tissue (2a) ----
    ("Skull", "bone", ["skull", "cranium", "occiput", "mandible"]),
    ("Spine", "bone", ["spine", "vertebra", "vertebrae", "cervical", "thoracic", "lumbar", "sacrum", "sacral", "coccyx"]),
    ("Ribs & sternum", "bone", ["rib", "sternum", "clavicle", "scapula"]),
    ("Pelvis", "bone", ["pelvis", "ilium", "iliac crest"]),
    ("Limb bones", "bone", ["femur", "patella", "tibia", "fibula", "humerus", "radius", "ulna", "hand bones", "foot bones"]),
    ("Joints", "bone", ["joint", "cartilage", "articular"]),
    ("Tendons & ligaments", "bone", ["tendon", "ligament"]),
    ("Fascia", "bone", ["fascia"]),
    ("Connective tissue", "bone", ["connective", "dermis (skin)"]),
    # ---- Connection / Cardiovascular & immune (2b) ----
    ("Heart", "cardiovascular", ["heart"]),
    ("Great vessels", "cardiovascular", ["aorta", "vena cava", "pulmonary vessel", "pulmonary"]),
    ("Arteries", "cardiovascular", ["artery", "arteries", "carotid"]),
    ("Veins", "cardiovascular", ["vein", "portal", "jugular"]),
    ("Blood vessels", "cardiovascular", ["blood vessel", "vessel", "circulation", "vascular"]),
    ("Lymph nodes", "cardiovascular", ["lymph node", "node"]),
    ("Lymphatic vessels", "cardiovascular", ["lymphatic", "duct", "trunk", "cisterna", "watershed"]),
    ("Spleen", "cardiovascular", ["spleen", "splenic"]),
    ("Thymus", "cardiovascular", ["thymus"]),
    ("Tonsils", "cardiovascular", ["tonsil", "adenoid", "waldeyer"]),
    ("Bone marrow", "cardiovascular", ["marrow"]),
    ("Gut-associated lymphoid (GALT)", "cardiovascular", ["galt", "peyer", "malt", "mucosa-associated"]),
    # ---- Conversion / Digestive (3a) ----
    ("Stomach", "digestive", ["stomach", "gastric"]),
    ("Liver", "digestive", ["liver", "hepatic"]),
    ("Gallbladder", "digestive", ["gallbladder", "gall bladder", "bile"]),
    ("Pancreas", "digestive", ["pancreas", "pancreatic"]),
    ("Small intestine", "digestive", ["small intestine", "duodenum", "jejunum", "ileum"]),
    ("Large intestine", "digestive", ["large intestine", "colon", "cecum", "appendix", "ileocecal", "sigmoid", "rectum"]),
    ("Esophagus", "digestive", ["esophagus", "oesophagus"]),
    ("Salivary glands", "digestive", ["salivary"]),
    # ---- Conversion / Respiratory (3b) ----
    ("Lungs", "respiratory", ["lung"]),
    ("Bronchi & trachea", "respiratory", ["bronchi", "bronchus", "trachea"]),
    ("Larynx", "respiratory", ["larynx"]),
    ("Pharynx", "respiratory", ["pharynx"]),
    ("Nose & sinuses", "respiratory", ["nose", "nasal", "sinus"]),
    ("Pleura", "respiratory", ["pleura"]),
    ("Diaphragm", "respiratory", ["diaphragm"]),
    # ---- Communication / Nerve (4a) ----
    ("Brain", "nerve", ["brain", "cerebrum", "cerebral", "frontal lobe", "occipital", "limbic", "subcortex"]),
    ("Cerebellum & brainstem", "nerve", ["cerebellum", "brainstem", "medulla", "pons"]),
    ("Spinal cord", "nerve", ["spinal cord", "cord", "cauda equina"]),
    ("Peripheral nerves", "nerve", ["nerve", "sciatic", "plexus", "femoral nerve"]),
    ("Autonomic & ganglia", "nerve", ["autonomic", "ganglia", "ganglion", "sympathetic", "vagus", "solar plexus", "celiac plexus"]),
    ("Cranial & special senses", "nerve", ["cranial nerve", "optic", "sensory"]),
    # ---- Communication / Endocrine (4b) ----
    ("Pituitary", "endocrine", ["pituitary"]),
    ("Pineal", "endocrine", ["pineal"]),
    ("Thyroid & parathyroid", "endocrine", ["thyroid", "parathyroid"]),
    ("Adrenal glands", "endocrine", ["adrenal", "suprarenal"]),
    ("Hypothalamus", "endocrine", ["hypothalamus"]),
    ("Endocrine glands", "endocrine", ["endocrine", "hormone", "islet"]),
    # ---- Containment / Oroderm (5a) ----
    ("Mouth & oral cavity", "oroderm", ["mouth", "oral", "palate", "throat"]),
    ("Tongue", "oroderm", ["tongue"]),
    ("Teeth", "oroderm", ["tooth", "teeth", "incisor", "canine", "premolar", "molar", "gum"]),
    ("Mucous membranes", "oroderm", ["mucous", "mucosa", "mucous membrane"]),
    # ---- Containment / Integument (5b) ----
    ("Skin", "integument", ["skin", "epidermis", "integument"]),
    ("Hair & nails", "integument", ["hair", "nail"]),
    ("Sensory skin (eye/ear)", "integument", ["eye", "iris", "sclera", "ear", "auricle", "retina", "cornea"]),
]


def _slug(name):
    import re
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def main():
    organs = [{"id": _slug(name), "name": name, "sublayer": sub, "keywords": kw}
              for (name, sub, kw) in ORGANS]
    # id uniqueness guard
    ids = [o["id"] for o in organs]
    assert len(ids) == len(set(ids)), "duplicate organ id"
    data = {"organs": organs}
    out = ROOT / "data" / "bodymap-tissue-layers.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    from collections import Counter
    print(f"wrote {out}: {len(organs)} organs", dict(Counter(o['sublayer'] for o in organs)))


if __name__ == "__main__":
    main()
