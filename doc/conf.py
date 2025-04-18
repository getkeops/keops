#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# KeOps documentation build configuration file, created by
# sphinx-quickstart on Thu Sep 13 14:50:06 2018.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("sphinxext"))

try:
    import pykeops
except:
    sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "pykeops"))
    import pykeops

from pykeops import __version__

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx-prompt",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",
    "sphinxcontrib.httpdomain",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.napoleon",
    # 'sphinx.ext.viewcode',
    "sphinx.ext.linkcode",
]


# sphinx.ext.linkcode
def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/main/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect
        import os

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(pykeops.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "pykeops/%s#L%d-L%d" % find_source()
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"

    return "https://github.com/getkeops/keops/tree/main/pykeops/%s" % filename


# def linkcode_resolve(domain, info):
# from sphinx.util import get_full_modname
# if domain != 'py':
# return None
# if not info['module']:
# return None
# filename = get_full_modname(info['module'], info['fullname']).replace('.', '/')
# return "https://github.com/getkeops/keops/tree/main/%s.py" % filename

from sphinx_gallery.sorting import FileNameSortKey

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": [
        "../pykeops/pykeops/tutorials",
        "../pykeops/pykeops/benchmarks",
        "../pykeops/pykeops/examples",
    ],
    # path where to save gallery generated examples
    "gallery_dirs": ["_auto_tutorials", "_auto_benchmarks", "./_auto_examples"],
    # order of the Gallery
    "within_subsection_order": FileNameSortKey,
    # Add patterns
    # 'filename_pattern': r'../pykeops/pykeops/tutorials/*',
    "ignore_pattern": r"__init__\.py|benchmark_utils\.py|dataset_utils\.py",
}

# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = False
autodoc_member_order = "bysource"


def skip(app, what, name, obj, would_skip, options):
    if name == "__call__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)


# Include the example source for plots in API docs
# plot_include_source = True
# plot_formats = [("png", 90)]
# plot_html_show_formats = False
# plot_html_show_source_link = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

source_parsers = {
    ".md": "recommonmark.parser.CommonMarkParser",
}

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The root toctree document.
root_doc = "index"

# General information about the project.
project = "KeOps"

copyright = f"2018-{time.strftime("%Y")}, Benjamin Charlier, Jean Feydy, Joan A. Glaunès"
author = "Benjamin Charlier, Jean Feydy, Joan A. Glaunès."

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False
exclude_patterns = ["readme_first.md"]

# display broken internal links
nitpicky = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_context = {
    "display_github": True,  # Integrate Github
    "github_user": "getkeops",  # Username
    "github_repo": "keops",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/doc/",  # Path in the checkout to the docs root
}
# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "KeOps"

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "KeOps documentation"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/logo.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/logo.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

html_show_sphinx = False

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "KeOpsdoc"

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
    # Latex figure (float) alignment
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        root_doc,
        "KeOps.tex",
        "KeOps Documentation",
        "Benjamin Charlier, Jean Feydy, Joan A. Glaunès",
        "manual",
    ),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(root_doc, "keops", "KeOps Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        root_doc,
        "KeOps",
        "KeOps Documentation",
        author,
        "KeOps",
        "One line description of project.",
        "Miscellaneous",
    ),
]


def setup(app):
    app.add_css_file("css/custom.css")
