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
    release = '0.1.0' # Default fallback
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
    # Add other extensions here if needed, e.g., 'sphinx_gallery.gen_gallery'
    'sphinxcontrib.bibtex',   # Add BibTeX support <--- ADD THIS LINE
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


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.
html_theme = 'furo'

# Theme options are theme-specific and customize the look and feel.
# We are putting most CSS customizations in custom.css
html_theme_options = {
    # Keep source repository links for Furo's "Edit on GitHub" feature
    "source_repository": "https://github.com/earthai-tech/fusionlab/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    # light_css_variables and dark_css_variables removed, using custom.css instead
}

# Add any paths that contain custom static files (such as style sheets or logo)
html_static_path = ['_static']

# List of CSS files to include. Relative to html_static_path.
html_css_files = [
    'custom.css', # Your custom styles including variable overrides
]

# The name of an image file (relative to this directory, within _static path)
# Place your logo file at 'docs/source/_static/fusionlab.png'
html_logo = '_static/fusionlab.png'

# The name of an image file (relative to this directory, within _static path)
# Place your favicon file at 'docs/source/_static/favicon.ico'
html_favicon = '_static/favicon.ico'


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
# napoleon_type_aliases = None # Dictionary to map type names
napoleon_attr_annotations = True # Use PEP 526 annotations for attributes

# -- Options for MyST Parser (Markdown) --
myst_enable_extensions = [
    "colon_fence",      # Allow ``` fenced code blocks
    "deflist",          # Allow definition lists
    "smartquotes",      # Use smart quotes
    "replacements",     # Apply textual replacements
    # "linkify",        # Automatically identify URL links (optional)
    "dollarmath",     # Allow $...$ and $$...$$ for math (if not using mathjax)
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
    'keras': ('[https://keras.io/api/](https://keras.io/api/)', None), # Verify Keras objects.inv availability
    'matplotlib': ('[https://matplotlib.org/stable/](https://matplotlib.org/stable/)', None),
    # Add other relevant libraries here
}

# -- Options for copybutton extension --
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True