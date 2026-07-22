import os


def write_sample_course(root):
    """Write an 'ash-intro' course: public out-takes + one member lesson.

    Stage 1.5 content model: lesson bodies are sanitized HTML (Rumble/YouTube
    embeds + formatting + transcript interwoven) instead of markdown + a
    separate rumble_id field.
    """
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
            "---\ntitle: Out-takes\naccess: public\ndownloads: []\n---\n"
            '<h2>Out-takes</h2>\n'
            '<iframe src="https://rumble.com/embed/v1abcd/" width="640" height="360" '
            'frameborder="0" allowfullscreen></iframe>\n'
            "<p>Bloopers transcript here.</p>\n"
        )
    with open(os.path.join(base, "01-intro", "02-welcome.md"), "w") as f:
        f.write(
            "---\ntitle: Welcome\naccess: member\ndownloads: []\n---\n"
            '<h2>Welcome</h2>\n'
            '<iframe src="https://www.youtube.com/embed/v2efgh" width="640" height="360" '
            'frameborder="0" allowfullscreen></iframe>\n'
            "<p>Welcome transcript here.</p>\n"
        )
    return root
