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
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_nb",
]

source_suffix = {".md": "myst-nb", ".ipynb": "myst-nb"}
master_doc = "index"
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
myst_enable_extensions = ["dollarmath", "amsmath"]
nb_execution_mode = "off"
suppress_warnings = ["myst.header"]

numpydoc_show_class_members = False
autodoc_member_order = "bysource"
todo_include_todos = True
pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"
htmlhelp_basename = "paramnormaldoc"

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
