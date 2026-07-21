import os


def write_sample_course(root):
    """Write an 'ash-intro' course: public out-takes + one member lesson."""
    base = os.path.join(root, "ash-intro")
    os.makedirs(os.path.join(base, "01-intro"), exist_ok=True)
    with open(os.path.join(base, "course.yaml"), "w") as f:
        f.write(
            "title: ASH Intro\n"
            "description: A free introduction.\n"
            "modules:\n"
            "  - slug: 01-intro\n"
            "    title: Introduction\n"
            "    lessons:\n"
            "      - 01-out-takes\n"
            "      - 02-welcome\n"
        )
    with open(os.path.join(base, "01-intro", "01-out-takes.md"), "w") as f:
        f.write(
            "---\ntitle: Out-takes\naccess: public\nrumble_id: v1abcd\n"
            "downloads: []\n---\nBloopers transcript here.\n"
        )
    with open(os.path.join(base, "01-intro", "02-welcome.md"), "w") as f:
        f.write(
            "---\ntitle: Welcome\naccess: member\nrumble_id: v2efgh\n"
            "downloads: []\n---\nWelcome transcript here.\n"
        )
    return root
