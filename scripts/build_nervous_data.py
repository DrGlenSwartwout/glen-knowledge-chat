#!/usr/bin/env python3
"""Build data/bodymap-nervous.json — the nervous system atlas (front + back).

CNS (brain regions + spinal cord), the major peripheral nerves as stroked paths,
and the autonomic ganglia. Each named so a client's nervous-system finding (Brain,
Sciatic Nerve, Optic Nerve, Vagus, Solar plexus...) lights the structure.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bodymap_atlas_lib as lib

GROUPS = [
    {"id": "brain", "label": "Brain"},
    {"id": "cord", "label": "Spinal cord"},
    {"id": "peripheral", "label": "Peripheral nerves"},
    {"id": "autonomic", "label": "Autonomic & ganglia"},
    {"id": "special", "label": "Cranial & special senses"},
]

Z = []
e, p, z, cr = lib.ellipse, lib.path, lib.zone, lib.catmull

# ---- FRONT ----
Z += [
    z("nerve-cerebrum", "Cerebrum (brain)", "front", "brain", e(0.500, 0.050, 0.046, 0.048), "The cerebral hemispheres."),
    z("nerve-frontal", "Frontal lobe", "front", "brain", e(0.500, 0.032, 0.030, 0.016), "Frontal lobe — executive function."),
    z("nerve-hypothalamus", "Hypothalamus & limbic", "front", "brain", e(0.500, 0.066, 0.016, 0.012), "Hypothalamus and limbic system."),
    z("nerve-brainstem", "Brainstem", "front", "brain", e(0.500, 0.088, 0.012, 0.016), "Midbrain, pons, medulla."),
    z("nerve-optic", "Optic nerve", "front", "special", e(0.478, 0.058, 0.010, 0.008), "Optic nerve (CN II)."),
    z("nerve-cranial", "Cranial nerves", "front", "special", e(0.522, 0.078, 0.012, 0.010), "The twelve cranial nerves."),
    z("nerve-vagus", "Vagus nerve", "front", "autonomic", p(cr([(0.512, 0.120), (0.516, 0.200), (0.520, 0.300), (0.516, 0.380)])), "Vagus nerve (CN X) — parasympathetic to the viscera."),
    z("nerve-cord", "Spinal cord", "front", "cord", p(cr([(0.500, 0.130), (0.500, 0.250), (0.500, 0.380), (0.500, 0.480)])), "The spinal cord within the vertebral canal."),
    z("nerve-brachial-r", "Brachial plexus (right)", "front", "peripheral", e(0.398, 0.212, 0.016, 0.012), "Right brachial plexus — nerves to the arm."),
    z("nerve-brachial-l", "Brachial plexus (left)", "front", "peripheral", e(0.602, 0.212, 0.016, 0.012), "Left brachial plexus — nerves to the arm."),
    z("nerve-median-r", "Arm nerves (right)", "front", "peripheral", p(cr([(0.372, 0.240), (0.340, 0.360), (0.312, 0.480)])), "Right median / radial / ulnar nerves."),
    z("nerve-median-l", "Arm nerves (left)", "front", "peripheral", p(cr([(0.628, 0.240), (0.660, 0.360), (0.688, 0.480)])), "Left median / radial / ulnar nerves."),
    z("nerve-solar", "Solar plexus (celiac)", "front", "autonomic", e(0.500, 0.360, 0.018, 0.014), "Celiac (solar) plexus — the abdominal autonomic hub."),
    z("nerve-lumbar-plexus", "Lumbar plexus", "front", "peripheral", e(0.500, 0.500, 0.020, 0.014), "Lumbar plexus — nerves to the pelvis and legs."),
    z("nerve-femoral-r", "Femoral nerve (right)", "front", "peripheral", p(cr([(0.462, 0.560), (0.456, 0.680), (0.456, 0.800)])), "Right femoral nerve — front of the thigh."),
    z("nerve-femoral-l", "Femoral nerve (left)", "front", "peripheral", p(cr([(0.538, 0.560), (0.544, 0.680), (0.544, 0.800)])), "Left femoral nerve — front of the thigh."),
]
# ---- BACK ----
Z += [
    z("nerve-cerebellum", "Cerebellum", "back", "brain", e(0.500, 0.072, 0.034, 0.030), "Cerebellum — coordination and balance."),
    z("nerve-occipital", "Occipital lobe", "back", "brain", e(0.500, 0.048, 0.030, 0.018), "Occipital lobe — vision."),
    z("nerve-cord-cerv", "Cervical cord", "back", "cord", p(cr([(0.500, 0.130), (0.500, 0.175), (0.500, 0.210)])), "Cervical spinal cord."),
    z("nerve-cord-thor", "Thoracic cord", "back", "cord", p(cr([(0.500, 0.220), (0.500, 0.300), (0.500, 0.375)])), "Thoracic spinal cord."),
    z("nerve-cord-lumb", "Lumbar cord & cauda equina", "back", "cord", p(cr([(0.500, 0.385), (0.500, 0.440), (0.500, 0.490)])), "Lumbar cord and cauda equina."),
    z("nerve-sympathetic-r", "Sympathetic chain (right)", "back", "autonomic", p(cr([(0.476, 0.200), (0.476, 0.320), (0.476, 0.440)])), "Right paravertebral sympathetic chain."),
    z("nerve-sympathetic-l", "Sympathetic chain (left)", "back", "autonomic", p(cr([(0.524, 0.200), (0.524, 0.320), (0.524, 0.440)])), "Left paravertebral sympathetic chain."),
    z("nerve-sciatic-r", "Sciatic nerve (right)", "back", "peripheral", p(cr([(0.454, 0.540), (0.452, 0.660), (0.456, 0.800), (0.458, 0.900)])), "Right sciatic nerve — the largest nerve, down the back of the leg."),
    z("nerve-sciatic-l", "Sciatic nerve (left)", "back", "peripheral", p(cr([(0.546, 0.540), (0.548, 0.660), (0.544, 0.800), (0.542, 0.900)])), "Left sciatic nerve — the largest nerve, down the back of the leg."),
]

lib.write_system("nervous", GROUPS, Z, group_noun="region")
