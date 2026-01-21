.. _geoprior_v3_installation:

========================
Installation and startup
========================

This page explains how to install the dependencies required by GeoPrior
v3 and how to launch the GUI either from a development checkout or from
an installed ``fusionlab-learn`` environment.

GeoPrior v3 is shipped as part of the FusionLab-Learn codebase and
depends on a working Python environment with **PyQt5** and the
``fusionlab`` stack available.

Prerequisites
-------------

Operating system
^^^^^^^^^^^^^^^^

GeoPrior v3 is a desktop GUI application. It is expected to run on any
platform supported by PyQt5 (Windows, macOS, Linux). For the best
experience, use a recent Python distribution (e.g., Conda/Miniforge,
system Python, or venv) and avoid mixing multiple Qt installations in
the same environment.

Python
^^^^^^

Use a dedicated environment (recommended). GeoPrior v3 targets Python
3.x as defined by your project configuration (see ``pyproject.toml``).
If you work across multiple projects, keeping an isolated environment
prevents Qt and scientific stack conflicts.

Dependencies
^^^^^^^^^^^^

At minimum you need:

- **PyQt5** (GUI framework)
- **fusionlab / fusionlab-learn** installed in the same environment
- The scientific stack required by the GeoPrior pipeline (TensorFlow or
  other backend, NumPy, pandas, etc.), depending on how you run Stage-2

If your environment is missing PyQt5, the application will fail at
import time with an error similar to ``ModuleNotFoundError: No module
named 'PyQt5'``.

Installation options
--------------------

There are two supported workflows:

1) **Development checkout (editable install)**: recommended if you are
   actively modifying the GUI or pipeline code.
2) **Installed package (wheel/standard install)**: recommended for
   users who only need to run the application.

Development checkout (recommended for contributors)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Clone the repository that contains ``fusionlab-learn`` (or your
   monorepo that provides it).

2. Create and activate a fresh environment.

   **Conda / Mamba example**::

      conda create -n fusionlab-gui python=3.10
      conda activate fusionlab-gui

   **venv example**::

      python -m venv .venv
      . .venv/bin/activate

3. Install PyQt5 (if it is not already installed).

   **pip**::

      pip install PyQt5

   **conda**::

      conda install -c conda-forge pyqt

4. Install the project in editable mode from the repository root::

      pip install -e .

This makes the GUI importable as a module and reflects code changes
immediately without reinstalling.

Installed environment (end users)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you distribute wheels, install them in a clean environment. The exact
command depends on your packaging/distribution method. A typical pip
install looks like::

   pip install fusionlab-learn

If you do not provide a public wheel, follow your internal packaging
instructions and ensure the runtime environment includes PyQt5.

Launching GeoPrior v3
---------------------

GeoPrior v3 can be launched as a module. This is the most reliable
method in development, because it uses your environment’s import
resolution and ensures relative imports work correctly.

Development entry point
^^^^^^^^^^^^^^^^^^^^^^^

From an activated environment where ``fusionlab-learn`` is installed::

   python -m fusionlab.tools.app.geoprior.app

If the GUI starts, you should see the main window with the tab strip
(Data, Experiment Setup, Preprocess, Train, Tune, Inference, Transfer,
Results, Map, Tools) and a status indicator at the bottom.

Console scripts (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Some distributions expose the GUI as a console script (for example,
``geoprior-v3``). If your ``pyproject.toml`` defines such an entry point,
document it here. For example::

   geoprior-v3

If you are unsure whether a console script is available, check the
installed scripts in your environment or inspect the project scripts
section in ``pyproject.toml``.

Common startup issues
---------------------

Qt binding conflicts
^^^^^^^^^^^^^^^^^^^^

If you have multiple Qt bindings installed (e.g., PyQt5 *and* PySide2 /
PySide6), you may run into import or runtime issues. Use a clean
environment and install only the Qt binding you intend to use.

Missing plugins / platform errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On Linux, Qt may fail with a platform plugin error (often mentioning
``xcb``). This is an OS-level dependency issue, not a FusionLab issue.
Install the missing system libraries required by your Qt build or use a
Conda-based Qt package which bundles compatible plugins.

GPU / backend issues
^^^^^^^^^^^^^^^^^^^^

Stage-2 workflows (training/tuning/inference) may require a specific ML
backend (e.g., TensorFlow) and optional GPU drivers. If the GUI launches
but Stage-2 fails, consult your GeoPrior backend installation notes and
verify your ML backend independently from the GUI.

First-run configuration
-----------------------

Results root
^^^^^^^^^^^^

On first launch, GeoPrior v3 will ask you to choose (or it will create)
a **results root** directory. This directory is the workspace where the
GUI stores outputs produced by Stage-1/Stage-2/Stage-5 workflows.

All run artifacts are written under this root using a consistent folder
layout. To understand where datasets, manifests, checkpoints, exports,
and plots are written, see :doc:`reference/file_layout_outputs`.

Permissions and portability
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Choose a results root that you can write to reliably (avoid protected
system folders). If you move the results root between machines, ensure
you also move the associated run folders so that the GUI can locate
manifests and exported outputs consistently.
