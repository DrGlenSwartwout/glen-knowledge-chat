#!/usr/bin/env python3
"""Build data/bodymap-lymph.json — the lymphatic system map on the body.

A dedicated lymphatic chart (distinct from the lymph *elements* inside the iris /
sclera / EAV systems): node clusters as `point`s, the main ducts/trunks as
`path` lines, and the drainage watershed divides as `path` lines, on the body
figure. Front + back views reuse the meridian silhouette + per-view outlines.
Paired node clusters are bilateral (emitted both sides, mirrored x -> 1-x).

Groups (colour/filter): nodes, lymphoid organs, ducts/trunks, watershed divides.
Positions are a standards-based rendering for Glen's clinical refinement, not
gospel; a starter set, expandable.
"""
import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("bbo", ROOT / "scripts" / "build_body_outline.py")
bbo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bbo)

GROUPS = [
    {"id": "nodes", "label": "Lymph node clusters"},
    {"id": "organs", "label": "Lymphoid & immune organs"},
    {"id": "ducts", "label": "Ducts / trunks"},
    {"id": "watershed", "label": "Watershed divides"},
    {"id": "connective", "label": "Connective tissue"},
]

# NODES / ORGANS: (slug, name, view, group, bilateral, x, y)
POINTS = [
    # ---- FRONT nodes ----
    ("cervical", "Cervical nodes", "front", "nodes", True, 0.435, 0.150, "Deep cervical node chain, along the neck."),
    ("submandibular", "Submandibular nodes", "front", "nodes", True, 0.462, 0.118, "Submandibular nodes, under the jaw."),
    ("supraclavicular", "Supraclavicular nodes", "front", "nodes", True, 0.405, 0.186, "Supraclavicular nodes, above the clavicle."),
    ("axillary", "Axillary nodes", "front", "nodes", True, 0.352, 0.246, "Axillary nodes, drain the arm and breast."),
    ("cubital", "Cubital nodes", "front", "nodes", True, 0.690, 0.446, "Cubital (epitrochlear) nodes, at the elbow."),
    ("mediastinal", "Mediastinal nodes", "front", "nodes", False, 0.500, 0.286, "Mediastinal nodes, central chest."),
    ("mesenteric", "Mesenteric nodes", "front", "nodes", False, 0.500, 0.400, "Mesenteric nodes, drain the gut."),
    ("para-aortic", "Para-aortic nodes", "front", "nodes", False, 0.500, 0.454, "Para-aortic / lumbar nodes."),
    ("inguinal", "Superficial inguinal nodes", "front", "nodes", True, 0.440, 0.520, "Superficial inguinal nodes, drain the leg and perineum."),
    ("preauricular", "Preauricular nodes", "front", "nodes", True, 0.425, 0.100, "Preauricular nodes, in front of the ear."),
    ("submental", "Submental nodes", "front", "nodes", False, 0.500, 0.128, "Submental nodes, under the chin."),
    ("jugulodigastric", "Jugulodigastric node", "front", "nodes", True, 0.445, 0.140, "Jugulodigastric (tonsillar) node."),
    ("infraclavicular", "Infraclavicular nodes", "front", "nodes", True, 0.420, 0.216, "Infraclavicular / deltopectoral nodes."),
    ("parasternal", "Parasternal nodes", "front", "nodes", True, 0.472, 0.300, "Parasternal (internal thoracic) nodes."),
    ("hilar", "Bronchopulmonary (hilar) nodes", "front", "nodes", False, 0.500, 0.312, "Hilar / bronchopulmonary nodes."),
    ("tracheobronchial", "Tracheobronchial nodes", "front", "nodes", False, 0.500, 0.332, "Tracheobronchial nodes, at the carina."),
    ("celiac", "Celiac nodes", "front", "nodes", False, 0.500, 0.372, "Celiac nodes, upper abdomen."),
    ("gastric", "Gastric nodes", "front", "nodes", False, 0.482, 0.382, "Gastric nodes, along the stomach."),
    ("hepatic", "Hepatic nodes", "front", "nodes", False, 0.532, 0.372, "Hepatic nodes, porta hepatis."),
    ("splenic", "Splenic / pancreatic nodes", "front", "nodes", False, 0.565, 0.368, "Splenic and pancreatic nodes."),
    ("sup-mesenteric", "Superior mesenteric nodes", "front", "nodes", False, 0.500, 0.415, "Superior mesenteric nodes."),
    ("inf-mesenteric", "Inferior mesenteric nodes", "front", "nodes", False, 0.500, 0.435, "Inferior mesenteric nodes."),
    ("common-iliac", "Common iliac nodes", "front", "nodes", True, 0.460, 0.485, "Common iliac nodes."),
    ("external-iliac", "External iliac nodes", "front", "nodes", True, 0.442, 0.508, "External iliac nodes."),
    ("deep-inguinal", "Deep inguinal nodes", "front", "nodes", True, 0.458, 0.534, "Deep inguinal nodes (node of Cloquet)."),
    # ---- FRONT organs ----
    ("tonsils", "Tonsils (Waldeyer's ring)", "front", "organs", False, 0.500, 0.135, "Waldeyer's tonsillar ring, throat."),
    ("thymus", "Thymus", "front", "organs", False, 0.500, 0.250, "Thymus, upper anterior mediastinum."),
    ("spleen", "Spleen", "front", "organs", False, 0.578, 0.360, "Spleen, left upper abdomen."),
    ("cisterna", "Cisterna chyli", "front", "organs", False, 0.500, 0.462, "Cisterna chyli, origin of the thoracic duct."),
    ("peyers", "Peyer's patches (GALT)", "front", "organs", False, 0.500, 0.425, "Gut-associated lymphoid tissue (Peyer's patches)."),
    ("marrow", "Red bone marrow", "front", "organs", False, 0.560, 0.510, "Red bone marrow, primary lymphoid tissue (pelvis / long bones)."),
    ("adenoids", "Adenoids", "front", "organs", False, 0.500, 0.106, "Nasopharyngeal tonsil (adenoids), immune tissue of the upper airway."),
    ("appendix-galt", "Appendix (GALT)", "front", "organs", False, 0.442, 0.492, "Appendix — gut-associated lymphoid tissue."),
    ("malt", "Mucosa-associated tissue (MALT)", "front", "organs", False, 0.470, 0.400, "MALT — immune tissue lining the mucous membranes."),
    # ---- connective tissue ----
    ("fascia", "Fascia (connective tissue)", "front", "connective", False, 0.500, 0.320, "Fascia — the body-wide connective-tissue web."),
    ("tendons", "Tendons & ligaments", "front", "connective", True, 0.360, 0.500, "Tendons and ligaments — dense connective tissue at the joints."),
    ("cartilage", "Cartilage", "back", "connective", True, 0.456, 0.740, "Cartilage — connective tissue cushioning the joints."),
    ("dermis", "Dermis (skin)", "back", "connective", False, 0.500, 0.250, "Dermis — the connective-tissue layer of the skin."),
    # ---- BACK nodes ----
    ("occipital", "Occipital nodes", "back", "nodes", False, 0.500, 0.062, "Occipital nodes, back of the head."),
    ("post-cervical", "Posterior cervical nodes", "back", "nodes", True, 0.437, 0.146, "Posterior cervical node chain."),
    ("post-axillary", "Posterior axillary nodes", "back", "nodes", True, 0.355, 0.250, "Posterior axillary nodes."),
    ("lumbar", "Lumbar nodes", "back", "nodes", False, 0.500, 0.450, "Lumbar node chain along the spine."),
    ("gluteal", "Gluteal / sacral nodes", "back", "nodes", False, 0.500, 0.560, "Gluteal and sacral nodes."),
    ("popliteal", "Popliteal nodes", "back", "nodes", True, 0.560, 0.782, "Popliteal nodes, behind the knee."),
    ("postauricular", "Postauricular nodes", "back", "nodes", True, 0.440, 0.092, "Postauricular (mastoid) nodes."),
    ("intercostal", "Intercostal nodes", "back", "nodes", True, 0.440, 0.300, "Intercostal nodes, posterior thorax."),
    ("common-iliac-b", "Common iliac nodes", "back", "nodes", True, 0.458, 0.485, "Common iliac nodes."),
    ("internal-iliac-b", "Internal iliac nodes", "back", "nodes", True, 0.478, 0.520, "Internal iliac nodes, pelvis."),
    ("sacral-b", "Sacral nodes", "back", "nodes", False, 0.500, 0.540, "Sacral nodes, in front of the sacrum."),
]

# PATHS: (slug, name, view, group, [waypoints])
PATHS = [
    # ---- FRONT ducts ----
    ("thoracic-duct", "Thoracic duct", "front", "ducts",
     [(0.500, 0.462), (0.487, 0.40), (0.478, 0.30), (0.452, 0.222), (0.420, 0.192)]),
    ("right-duct", "Right lymphatic duct", "front", "ducts",
     [(0.556, 0.244), (0.572, 0.206), (0.582, 0.190)]),
    ("intestinal-trunk", "Intestinal trunk", "front", "ducts",
     [(0.500, 0.402), (0.500, 0.460)]),
    ("jugular-trunk", "Jugular trunk", "front", "ducts",
     [(0.448, 0.158), (0.430, 0.184), (0.420, 0.192)]),
    ("subclavian-trunk", "Subclavian trunk", "front", "ducts",
     [(0.368, 0.246), (0.400, 0.206), (0.420, 0.192)]),
    ("bronchomediastinal", "Bronchomediastinal trunk", "front", "ducts",
     [(0.500, 0.300), (0.486, 0.236), (0.470, 0.198)]),
    ("lumbar-trunk", "Lumbar trunks", "front", "ducts",
     [(0.462, 0.506), (0.482, 0.480), (0.500, 0.462)]),
    # ---- FRONT watershed divides ----
    ("clavicular-ws", "Clavicular watershed", "front", "watershed",
     [(0.345, 0.205), (0.655, 0.205)]),
    ("umbilical-ws", "Umbilical watershed", "front", "watershed",
     [(0.415, 0.410), (0.585, 0.410)]),
    ("midline-ws", "Median watershed", "front", "watershed",
     [(0.500, 0.140), (0.500, 0.520)]),
    # ---- BACK ducts / watershed ----
    ("thoracic-duct-b", "Thoracic duct", "back", "ducts",
     [(0.500, 0.560), (0.490, 0.45), (0.485, 0.32), (0.478, 0.22), (0.472, 0.180)]),
    ("spine-ws", "Median (spinal) watershed", "back", "watershed",
     [(0.500, 0.130), (0.500, 0.560)]),
    ("waist-ws-b", "Lumbar watershed", "back", "watershed",
     [(0.420, 0.440), (0.580, 0.440)]),
]


def _catmull_open(pts):
    if len(pts) < 2:
        return ""
    p = [pts[0]] + list(pts) + [pts[-1]]
    d = f"M{pts[0][0]:.4f} {pts[0][1]:.4f} "
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        d += (f"C{p1[0]+(p2[0]-p0[0])/6:.4f} {p1[1]+(p2[1]-p0[1])/6:.4f} "
              f"{p2[0]-(p3[0]-p1[0])/6:.4f} {p2[1]-(p3[1]-p1[1])/6:.4f} "
              f"{p2[0]:.4f} {p2[1]:.4f} ")
    return d.strip()


def _mk(zid, name, view, group, geometry, meaning):
    return {
        "id": zid, "side": view, "bilateral": False, "group": group,
        "geometry": geometry, "anatomy": name,
        "meaning_standard": meaning, "meaning_glen": "",
        "layers": {"embryological_depth": None, "stress_affirmation": None, "touch_for_health": None},
    }


def main():
    zones = []
    for slug, name, view, group, bilateral, x, y, meaning in POINTS:
        geo = lambda xx: {"type": "point", "x": round(xx, 4), "y": y}
        if bilateral:
            zones.append(_mk(f"lymph-{slug}-R", name, view, group, geo(x), meaning))
            zones.append(_mk(f"lymph-{slug}-L", name, view, group, geo(1 - x), meaning))
        else:
            zones.append(_mk(f"lymph-{slug}", name, view, group, geo(x), meaning))
    for slug, name, view, group, wps in PATHS:
        zones.append(_mk(f"lymph-{slug}", name, view, group,
                         {"type": "path", "d": _catmull_open(wps)}, name + "."))
    front = bbo.build_outline()
    data = {
        "system": "lymph",
        "reference_frame": "body_outline",
        "side_noun": "view",
        "group_noun": "type",
        "outline": front,
        "outlines": {"front": front, "back": front},
        "groups": GROUPS,
        "anchors": [
            {"key": "head-f", "view": "front", "template": {"x": 0.50, "y": 0.02}, "hint": "Tap the top of the head."},
            {"key": "feet-f", "view": "front", "template": {"x": 0.50, "y": 0.985}, "hint": "Tap the point between the ankles."},
            {"key": "head-b", "view": "back", "template": {"x": 0.50, "y": 0.02}, "hint": "Tap the top of the head."},
            {"key": "feet-b", "view": "back", "template": {"x": 0.50, "y": 0.985}, "hint": "Tap the point between the ankles."},
        ],
        "zones": zones,
    }
    out = ROOT / "data" / "bodymap-lymph.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    from collections import Counter
    print(f"wrote {out}: {len(zones)} zones", dict(Counter(z["side"] for z in zones)),
          dict(Counter(z["geometry"]["type"] for z in zones)))


if __name__ == "__main__":
    main()
