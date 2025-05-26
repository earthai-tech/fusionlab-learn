# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import date

# -- Path setup --------------------------------------------------------------
# Add the project root directory (parent of 'docs') to the Python path
# so Sphinx can find the 'fusionlab' package.
sys.path.insert(0, os.path.abspath('../..'))

# -- Dynamically get version info from package ---
try:
    import fusionlab
    release = fusionlab.__version__
    # The short X.Y version
    version = '.'.join(release.split('.')[:2])
except ImportError:
    print("Warning: Could not import fusionlab to determine version.")
    print("Setting version and release to defaults.")
    release = '0.1.1' # Default fallback
    version = '0.1'   # Default fallback

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fusionlab'
# Use current year automatically
copyright = f'{date.today().year}, earthai-tech'
author = 'earthai-tech' # Or your preferred author name

# The full version, including alpha/beta/rc tags (set dynamically above)
# release = release
# The short X.Y version (set dynamically above)
# version = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add required Sphinx extension module names.
extensions = [
    'sphinx.ext.autodoc',       # Core library for html generation from docstrings
    'sphinx.ext.autosummary',   # Create neat summary tables
    'sphinx.ext.napoleon',      # Support for Google and NumPy style docstrings
    'sphinx.ext.intersphinx',   # Link to other projects' documentation
    'sphinx.ext.viewcode',      # Add links to source code from documentation
    'sphinx.ext.githubpages',   # Creates .nojekyll file for GitHub Pages deployment
    'sphinx.ext.mathjax',       # Render math equations (via MathJax)
    'sphinx_copybutton',        # Adds a 'copy' button to code blocks
    'myst_parser',              # Allow parsing Markdown files (like README.md)
    'sphinx_design',            # Enable design elements like cards, buttons, grids
                                # e.g., 'sphinx_gallery.gen_gallery'
    # 'sphinxcontrib.bibtex',   # Add BibTeX support <--- ADD THIS LINE
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
# Use a dictionary for multiple parsers if needed (e.g., MyST for .md)
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown', # If using myst_parser for markdown files
}
# Or just '.rst' if only using reStructuredText
# source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# ... (other general configuration like templates_path, exclude_patterns, etc.)

# -- BibTeX Configuration ----------------------------------------------------
# List of BibTeX files relative to the source directory
bibtex_bibfiles = ['references.bib'] 

# Choose the citation style: 'label', 'author_year', 'super', 'foot'
# 'label' uses the BibTeX key (e.g., [Lim21])
# 'author_year' uses (Author, Year)
bibtex_reference_style = 'label'#  (or choose another style)

# Choose the bibliography style (like LaTeX styles)
bibtex_default_style = 'plain' # (or 'unsrt', 'alpha', etc.)


# -- Custom Roles for Release Notes (rst_epilog) -------------------------
# These substitutions are appended to the end of every processed RST file.
# Defines roles like |Feature|, |Fix|, etc., using CSS classes for styling.
# Allow shorthand references for main function interface
#rst_prolog = wx_rst_epilog
rst_epilog = """
.. role:: bdg-success(raw)
   :format: html

.. role:: bdg-danger(raw)
   :format: html

.. role:: bdg-info(raw)
   :format: html

.. role:: bdg-warning(raw)
   :format: html

.. role:: bdg-primary(raw)
   :format: html

.. role:: bdg-secondary(raw)
   :format: html

.. |Feature| replace:: :bdg-success:`Feature`
.. |New| replace:: :bdg-success:`Feature`
.. |Fix| replace:: :bdg-danger:`Fix`
.. |Enhancement| replace:: :bdg-info:`Enhancement`
.. |Breaking| replace:: :bdg-warning:`Breaking`
.. |API Change| replace:: :bdg-warning:`API Change`
.. |Docs| replace:: :bdg-secondary:`Docs`
.. |Build| replace:: :bdg-primary:`Build`
.. |Tests| replace:: :bdg-primary:`Tests`

"""

# -- Options for HTML output -------------------------------------------------

# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.
html_theme = 'furo'


# -- Options for HTML output -------------------------------------------------
# Theme options are theme-specific and customize the look and feel.
# We are putting most CSS customizations in custom.css

html_theme_options = {
    # ── VCS “Edit this page” -----------------------------------------------
    "source_repository":  "https://github.com/earthai-tech/fusionlab-learn/",
    "source_branch":      "main",
    "source_directory":   "docs/source/",

    # ── Layout tweaks -------------------------------------------------------
    "sidebar_hide_name":      True,   # hide the project name over the logo
    "navigation_with_keys":   True,   # ← / → to browse pages

    # # ── Logos ---------------------------------------------------------------
    # "light_logo": "fusionlab.png",
    # "dark_logo":  "fusionlab.png",   # add this SVG to _static/

    # ── Footer icons --------------------------------------------------------
    "footer_icons": [
        {
            "name": "GitHub",
            "url":  "https://github.com/earthai-tech/fusionlab-learn",
            "html": """
                <svg viewBox="0 0 24 24" aria-hidden="true"
                     height="1.35em" width="1.35em">
                  <path fill="currentColor"
                        d="M12 .3a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2
                           c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1.2-.8.1-.8.1-.8
                           1.3.1 2 1.3 2 1.3 1.2 2 3.1 1.4 3.8 1.1.1-.9.5-1.4.9-1.8
                           -2.7-.3-5.4-1.4-5.4-6.3 0-1.4.5-2.5 1.3-3.4 0-.3-.6-1.5.1-3
                           0 0 1-.3 3.3 1.3a11.4 11.4 0 0 1 6 0c2.2-1.6 3.3-1.3 3.3-1.3
                           .7 1.5.1 2.7.1 3A5 5 0 0 1 21 13c0 4.9-2.7 6-5.4 6.3
                           .6.5 1.1 1.4 1.1 2.8v4.2c0 .3.2.7.8.6A12 12 0 0 0 12 .3Z"/>
                </svg>
            """,
            "class": "",
        },
        {
            "name": "PyPI",
            "url":  "https://pypi.org/project/fusionlab-learn/",
            "html": "<span class='fa fa-box-open'></span>",
            "class": "",
        },
    ],

    # ── Brand colours via Furo’s CSS variables -----------------------------
    # light_css_variables and dark_css_variables removed, using custom.css instead
    # "light_css_variables": {
    #     "color-brand-primary":   "#2E3191",
    #     "color-brand-content":   "#2E3191",
    #     "color-sidebar-link-text--top-level": "#242774",
    #     "color-sidebar-background-hover":     "rgba(46,49,145,0.08)",
    # },
    
    # "dark_css_variables": {
    #     "color-brand-primary":   "#000000",
    #     "color-brand-content":   "#787BC4",
    #     "color-sidebar-link-text--top-level": "#787BC4",
    #     "color-sidebar-background-hover":     "rgba(120,123,196,0.12)",
    # },
}

# Add any paths that contain custom static files (such as style sheets or logo)
html_static_path = ['_static']

# List of CSS files to include. Relative to html_static_path.
html_css_files = [
    'css/custom.css', # Your custom styles including variable overrides
]

# The name of an image file (relative to this directory, within _static path)
# Place your logo file at 'docs/source/_static/fusionlab.png'
html_logo = '_static/fusionlab_b.png'

# The name of an image file (relative to this directory, within _static path)
# Place your favicon file at 'docs/source/_static/favicon.ico'
html_favicon = '_static/favicon.ico'

# -- HTML output --------------------------------------------------------------

# # 1) add the theme to your environment
# #    pip install sphinx-book-theme

# # 2) make sure it’s in the extension list so Sphinx fails fast
# extensions += ["sphinx_book_theme"]

# # 3) set the theme + options
# html_theme = "sphinx_book_theme"

# html_logo    = "_static/fusionlab.svg"      # light-mode logo
# html_favicon = "_static/favicon.ico"

# html_theme_options = {
#     # ----- Repository buttons -----
#     "repository_url":      "https://github.com/earthai-tech/fusionlab",
#     "repository_branch":   "main",          # default: 'main'
#     "path_to_docs":        "docs/source",   # from repo root
#     "use_edit_page_button": True,           # “Suggest edit”
#     "use_repository_button": True,          # link to repo
#     "use_issues_button":     True,          # open issue
#     "use_source_button":     True,          # show raw source

#     # ----- Layout tweaks -----
#     "home_page_in_toc": True,     # show “Home” in sidebar
#     "show_navbar_depth": 2,       # how deep the nav expands
#     "extra_navbar": "",           # custom HTML if you’d like
#     "extra_footer": "",           # likewise for footer

#     # ----- Launch buttons (optional) -----
#     # "launch_buttons": {
#     #     "binderhub_url": "https://mybinder.org",
#     #     "colab_url": "https://colab.research.google.com",
#     # },
# }

# # keep your custom CSS (it still applies)
# html_static_path = ["_static"]
# html_css_files   = ["custom.css"]


# -- Extension configuration -------------------------------------------------

# -- Options for autodoc --
autodoc_member_order = 'bysource' # Order members by source code order
autodoc_default_options = {
    'members': True,            # Document members (methods, attributes)
    'member-order': 'bysource', # Order members by source order
    'special-members': '__init__',# Include __init__ docstring if present
    'undoc-members': False,     # DO NOT include members without docstrings
    'show-inheritance': True,   # Show base classes
    # 'exclude-members': '__weakref__' # Example: Exclude specific members
}
autodoc_typehints = "description" # Show typehints in description, not signature
autodoc_class_signature = "separated" # Class signature on separate line

# -- Options for autosummary --
autosummary_generate = True     # Enable automatic generation of stub files
autosummary_imported_members = False # Don't list imported members in summary

# -- Options for napoleon (Google/NumPy docstrings) --
napoleon_google_docstring = True
napoleon_numpy_docstring = True # Set False if not using NumPy style
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False # Exclude private members (_*)
napoleon_include_special_with_doc = True # Include special members like __call__
napoleon_use_admonition_for_examples = True # Use .. admonition:: for examples
napoleon_use_admonition_for_notes = True    # Use .. admonition:: for notes
napoleon_use_admonition_for_references = True # Use .. admonition:: for references
napoleon_use_ivar = True       # Use :ivar: role for instance variables
napoleon_use_param = True      # Use :param: role for parameters
napoleon_use_rtype = True      # Use :rtype: role for return types
napoleon_preprocess_types = True # Process type strings into links
napoleon_type_aliases = None # Dictionary to map type names
napoleon_attr_annotations = True # Use PEP 526 annotations for attributes

# -- Options for MyST Parser (Markdown) --
myst_enable_extensions = [
    "colon_fence",      # Allow ``` fenced code blocks
    "deflist",          # Allow definition lists
    "smartquotes",      # Use smart quotes
    "replacements",     # Apply textual replacements
    "linkify",        # Automatically identify URL links (optional)
    "dollarmath",     # Allow $...$ and $$...$$ for math (ifcd  not using mathjax)
]
myst_heading_anchors = 3 # Automatically add anchors to headings up to level 3

# -- Options for intersphinx extension --
# Link to other projects' documentation.
intersphinx_mapping = {
    'python': ('[https://docs.python.org/3/](https://docs.python.org/3/)', None),
    'numpy': ('[https://numpy.org/doc/stable/](https://numpy.org/doc/stable/)', None),
    'scipy': ('[https://docs.scipy.org/doc/scipy/](https://docs.scipy.org/doc/scipy/)', None),
    'sklearn': ('[https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)', None),
    'pandas': ('[https://pandas.pydata.org/pandas-docs/stable/](https://pandas.pydata.org/pandas-docs/stable/)', None),
    'tensorflow': ('[https://www.tensorflow.org/api_docs/python](https://www.tensorflow.org/api_docs/python)', '[https://www.tensorflow.org/api_docs/python/objects.inv](https://www.tensorflow.org/api_docs/python/objects.inv)'),
    'keras': ('[https://keras.io/api/](https://keras.io/api/)', None), #
    'matplotlib': ('[https://matplotlib.org/stable/](https://matplotlib.org/stable/)', None),
}

# -- Options for copybutton extension --
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

