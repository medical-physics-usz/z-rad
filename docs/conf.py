from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zrad import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Z-Rad'
copyright = '2016 - 2026, USZ Department of Radiation Oncology'
author = 'USZ Department of Radiation Oncology'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_design",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'reference/_generate_api.rst']
autodoc_member_order = 'bysource'
autoclass_content = 'both'
autosummary_generate = False



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_title = f"{project} {release}"
html_logo = "logos/ZRadLogo.jpg"
html_favicon = "logos/icon.ico"
html_css_files = ["custom.css"]
html_theme_options = {
    "logo": {
        "text": "Z-Rad",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "header_links_before_dropdown": 8,
    "show_nav_level": 1,
    "navigation_with_keys": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/medical-physics-usz/z-rad",
            "icon": "fa-brands fa-github",
        },
    ],
}
