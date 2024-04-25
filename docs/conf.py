# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys


import sphinx_gallery  # noqa
from sphinx_gallery.sorting import ExplicitOrder

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = "rlberry"
copyright = "2023, rlberry team"
author = "rlberry team"

import rlberry

release = rlberry.__version__
# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx_math_dollar",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.video",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
    "sphinx_copybutton",
]

myst_enable_extensions = ["amsmath"]


autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}
# generate autosummary even if no references
autosummary_generate = True
autodoc_inherit_docstrings = True

# highlight_language = 'none'


# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "themes"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# Copied from scikit-learn:
# For maths, use mathjax by default and svg if NO_MATHJAX env variable is set
# (useful for viewing the doc offline)
if os.environ.get("NO_MATHJAX"):
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
    mathjax_path = ""
else:
    extensions.append("sphinx.ext.mathjax")
    mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "scikit-learn-fork"

html_theme_options = {"mathjax_path": mathjax_path}

html_theme_path = ["themes"]

html_logo = "../assets/logo_wide.svg"


# syntax highlight

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
html_extra_path = ["_video"]

# numpydoc_validation_checks = {"all"} # can be uncommented to get the warnings from numpy.

sphinx_gallery_conf = {
    "doc_module": "rlberry",
    "backreferences_dir": os.path.join("generated"),
    "reference_url": {"rlberry": None},
    "matplotlib_animations": True,
    "remove_config_comments": True,
    "subsection_order": ExplicitOrder(
        [
            "../examples/demo_env",
            "../examples/demo_agents",
            "../examples/demo_bandits",
            "../examples/demo_examples",
        ]
    ),
}

pygments_style = "sphinx"
highlight_language = "python3"
