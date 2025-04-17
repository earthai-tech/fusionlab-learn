# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import date

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# Point to the project root directory (one level up from 'docs')
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fusionlab'
copyright = f'{date.today().year}, earthai-tech' # Use current year automatically
author = 'earthai-tech' # Or 'Laurent Kouadio' if you prefer
release = '0.1.0' # The full version, including alpha/beta/rc tags
version = '0.1'   # The short X.Y version


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',      # Core library for html generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables
    'sphinx.ext.napoleon',     # Support for Google and NumPy style docstrings
    'sphinx.ext.intersphinx',  # Link to other projects' documentation
    'sphinx.ext.viewcode',     # Add links to source code from documentation
    'sphinx.ext.githubpages',  # Creates .nojekyll file for GitHub Pages deployment
    'sphinx_copybutton',       # Adds a 'copy' button to code blocks
    'furo.sphinxext',          # If using Furo theme directly (alternative to setting html_theme)

]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
#
# Popular options: 'furo', 'sphinx_rtd_theme', 'pydata_sphinx_theme', 'alabaster'
html_theme = 'furo'

# Theme options are theme-specific and customize the look and feel.
# For Furo theme options, see: https://pradyunsg.me/furo/customisation/
html_theme_options = {
    "light_css_variables": {
        # Add Furo customization here if desired
    },
    "dark_css_variables": {
        # Add Furo customization here if desired
    },
    # Example: Add source repository links (optional)
    # "source_repository": "https://github.com/earthai-tech/fusionlab/",
    # "source_branch": "main",
    # "source_directory": "docs/source/",
}

# --- To use Sphinx RTD Theme instead of Furo ---
# 1. Make sure you have run: pip install sphinx-rtd-theme
# 2. Comment out or remove: html_theme = 'furo'
# 3. Uncomment the following line:
# html_theme = 'sphinx_rtd_theme'
# 4. Make sure 'sphinx_rtd_theme' is in the `extensions` list (or add it).
# 5. Comment out or remove the Furo-specific `html_theme_options` above.
# 6. (Optional) Add RTD theme options if needed:
# html_theme_options = {
#     'logo_only': False,
#     'display_version': True,
#     'prev_next_buttons_location': 'bottom',
#     'style_external_links': False,
#     # Toc options
#     'collapse_navigation': True,
#     'sticky_navigation': True,
#     'navigation_depth': 4,
#     'includehidden': True,
#     'titles_only': False
# }
# -----------------------------------------------


# Add any paths that contain custom static files (such as style sheets or your logo)
# here, relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The name of an image file (relative to this directory, within _static path)
# to place at the top of the sidebar.
# Make sure 'logo.png' is inside the 'docs/source/_static/' directory
html_logo = '_static/logo.png'

# The name of an image file (relative to this directory) to use as a favicon of
# the docs. This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = '_static/favicon.ico' # Optional: Add a favicon if you have one


# -- Extension configuration -------------------------------------------------

# -- Options for autodoc ----------------------------------------------------
autodoc_member_order = 'bysource'  # Order members by source code order
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__', # Include __init__ methods
    'undoc-members': True,        # Include members without docstrings (use with caution)
    'show-inheritance': True,     # Show base classes
}
# autosummary_generate = True  # Turn on generating stub files for autosummary

# -- Options for napoleon ---------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True # Set to True if you use NumPy style docstrings
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'tensorflow': ('https://www.tensorflow.org/api_docs/python', 'https://www.tensorflow.org/api_docs/python/objects.inv'),
    'keras': ('https://keras.io/api/', None), # Check if Keras provides an objects.inv file
    # Add others as needed, e.g., 'matplotlib': ('https://matplotlib.org/stable/', None)
}

# -- Options for copybutton extension ---------------------------------------
# Remove prompts ($) and output (>>>) from copied code blocks
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True