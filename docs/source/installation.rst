.. _installation:

==============
Installation
==============

This page covers how to install the ``fusionlab-learn`` library.

Prerequisites
---------------

Before installing ``fusionlab-learn``, ensure you have the following:

* **Python:** Version 3.9 or higher. You can check your Python
  version by running ``python --version`` or ``python3 --version``.

* **pip:** The Python package installer. Pip usually comes with
  Python. You can update it using ``pip install --upgrade pip``.

* **TensorFlow:** ``fusionlab-learn``'s core neural network models (like
  TFT, XTFT) currently rely heavily on TensorFlow. You need a
  working installation of TensorFlow (version 2.10 or higher is
  recommended for compatibility with recent features and Python versions).

Installation from PyPI (Recommended)
--------------------------------------

The easiest way to install ``fusionlab-learn`` is using ``pip``, which
will fetch the latest stable release from the Python Package Index
(PyPI):

.. code-block:: bash

   pip install fusionlab-learn

.. note::
   This command will install ``fusionlab-learn`` and its core
   dependencies. However, **TensorFlow itself might need to be
   installed or managed separately**, especially if you require a
   specific version (e.g., GPU-enabled) or are managing packages
   within a virtual environment.

   It's often recommended to install TensorFlow first, following the
   official TensorFlow installation guide:
   `Install TensorFlow <https://www.tensorflow.org/install>`_.

   For a typical CPU-only installation of TensorFlow, you can often use:

   .. code-block:: bash

     pip install tensorflow


Optional Dependencies (Extras)
--------------------------------

``fusionlab-learn`` offers optional features that require additional
dependencies. These can be installed using "extras".

**`k-diagram` for Advanced Visualization:**

For enhanced uncertainty visualization and model diagnostics using
polar plots, you can install the `k-diagram`_ package alongside
``fusionlab-learn``.

.. _k-diagram: https://k-diagram.readthedocs.io/

To install ``fusionlab-learn`` with the `k-diagram` extra:

.. code-block:: bash

   pip install fusionlab-learn[k-diagram]

This will install `fusionlab-learn` and also pull in the `k-diagram`
package, enabling you to use visualization utilities that depend on it
(e.g., via `from fusionlab.kdiagram.plot import ...` if `k-diagram`
is installed).

**Development Dependencies:**

If you plan to contribute to ``fusionlab-learn`` development or run
tests, you can install development dependencies:

.. code-block:: bash

   # After cloning the repository (see "Installation from Source")
   pip install -e .[dev]
   # To install all optional dependencies including k-diagram and dev tools:
   pip install -e .[full]


Installation from Source (for Development)
--------------------------------------------

If you want to work with the latest development version, contribute
to the project, or modify the code, you can install ``fusionlab-learn``
directly from the source code on GitHub:

1.  **Clone the repository:**

    .. code-block:: bash

       git clone https://github.com/earthai-tech/fusionlab-learn.git
       cd fusionlab-learn

2.  **Install in editable mode:**
    This command installs the package, but allows you to edit the
    code directly without reinstalling.

    .. code-block:: bash

       pip install -e .

    To include optional dependencies like `k-diagram` or development
    tools when installing from source, you can specify the extras:

    .. code-block:: bash

       pip install -e .[k-diagram]
       pip install -e .[dev]
       pip install -e .[full] # Installs all extras

Verify Installation
---------------------

To quickly check if ``fusionlab-learn`` is installed correctly, you can
try importing it in Python and printing its version:

.. code-block:: bash

   python -c "import fusionlab; print(fusionlab.__version__)"

If this command executes without errors and prints a version
number, the basic installation was successful. To check if optional
dependencies like `k-diagram` are accessible through `fusionlab`,
you can try:

.. code-block:: python

   import fusionlab
   try:
       from fusionlab import kdiagram # Try accessing the proxy
       from fusionlab.kdiagram.plot import plot_coverage_diagnostic # Example
       print("k-diagram seems accessible via fusionlab.kdiagram")
   except ImportError as e:
       print(f"k-diagram not available via fusionlab.kdiagram: {e}")
       print("You might need to install it: pip install fusionlab-learn[kdiagram]")

