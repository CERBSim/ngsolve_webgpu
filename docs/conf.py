#!/usr/bin/env python3
# -*- coding: utf-8 -*-

extensions = ["sphinx.ext.autodoc", "sphinx.ext.mathjax", "sphinx.ext.todo", "nbsphinx"]

import os

os.environ["WEBGPU_EXPORTING"] = "1"
master_doc = "index"
source_suffix = [".rst", ".md"]
language = "python"

html_theme = "pydata_sphinx_theme"
# html_theme = "piccolo_theme"

html_static_path = ["_static"]

html_logo = "_static/logo.svg"

html_theme_options = {
    "logo": {
        "text": "Webgpu Docs",
        "image_dark": "_static/logo_dark.png",
        "image_light": "_static/logo.svg",
    },
    "navigation_depth": 2,
    "show_nav_level": 2,
    # Show all 5 sections directly in the header (no "More" dropdown).
    "header_links_before_dropdown": 6,
    # Show child pages as a dropdown when hovering a top-level header link.
    "navbar_align": "left",
}

# nbsphinx: execute notebooks at build time so the WEBGPU export hook fires
# and inline interactive scenes are emitted into the HTML.
nbsphinx_execute = "auto"
nbsphinx_allow_errors = False
nbsphinx_timeout = 300
exclude_patterns = ["_build", "**/.ipynb_checkpoints"]

# Auto-generate the landing-page scene if it doesn't exist yet.
_landing = os.path.join(os.path.dirname(__file__), "_static", "landing.html")
if not os.path.exists(_landing):
    import subprocess
    import sys
    print("Generating landing scene (this requires Playwright/Chrome)...")
    try:
        subprocess.check_call(
            [sys.executable, os.path.join(os.path.dirname(__file__), "create_landing.py")],
            env={**os.environ, "WEBGPU_EXPORTING": "1"},
        )
    except Exception as e:
        print(f"Warning: failed to generate landing scene: {e}")
        # Write a placeholder so the build doesn't fail outright.
        os.makedirs(os.path.dirname(_landing), exist_ok=True)
        with open(_landing, "w") as _fh:
            _fh.write(
                "<p><em>Landing scene not yet generated. Run "
                "<code>python docs/create_landing.py</code>.</em></p>"
            )

todo_include_todos = True

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
# autodoc_typehints = "description"

# Introduce line breaks if the line length is greater than 80 characters.
python_maximum_signature_line_length = 80


def setup(app):
    app.add_css_file("custom.css")
