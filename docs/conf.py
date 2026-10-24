import seaborn

clean_bkgd = {"axes.facecolor": "none", "figure.facecolor": "none"}
seaborn.set_theme(style="ticks", rc=clean_bkgd)

project = "paramnormal"
copyright = "2015 - 2024, Paul Hobson"
author = "Paul Hobson"
version = release = "0.4.4"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    # Renders the `.. plot::` blocks embedded in the API docstrings.
    "matplotlib.sphinxext.plot_directive",
    # numpydoc renders the docstrings (all numpydoc-style after the 2025
    # clean-up) into the API reference pages.
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    # myst-nb executes the tutorial notebooks during the build.
    "myst_nb",
    # sphinx-gallery turns examples/*.py into an image gallery.
    # "sphinx_gallery.gen_gallery",
]

source_suffix = {".md": "myst-nb", ".ipynb": "myst-nb"}
master_doc = "index"
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
myst_enable_extensions = ["dollarmath", "amsmath"]
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 300
suppress_warnings = ["myst.header"]

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
autodoc_member_order = "bysource"
todo_include_todos = True
pygments_style = "sphinx"



# Options for the matplotlib .. plot:: directive (used in API docstrings)
plot_include_source = True
plot_html_show_formats = False
plot_html_show_source_link = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
}

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

# # -- myst-nb (tutorial notebooks) --------------------------------------------

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
htmlhelp_basename = "paramnormaldoc"
# html_theme = "pydata_sphinx_theme"
# html_theme_options = {
#     "github_url": "https://github.com/phobson/paramnormal",
#     "logo": {"text": "paramnormal"},
# }
# htmlhelp_basename = "paramnormaldoc"
