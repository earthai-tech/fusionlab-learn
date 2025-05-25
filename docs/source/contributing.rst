.. _contributing:

============
Contributing
============

We welcome contributions to ``fusionlab-learn``! Whether you're fixing a
bug, adding a new feature, improving documentation, or suggesting
ideas, your help is valued. Thank you for your interest in making
``fusionlab-learn`` better.

Getting Started
---------------

* **Issues Tracker:** The best place to start is the
  `GitHub Issues page <https://github.com/earthai-tech/fusionlab/issues>`_.
  Look for existing issues labeled ``bug``, ``enhancement``,
  ``documentation``, or ``good first issue``.
* **Ideas:** If you have an idea for a new feature or improvement,
  feel free to open a new issue to discuss it first. This helps
  ensure it aligns with the project's goals before significant
  work is done.
* **Questions:** If you have questions about usage or contributing,
  you can also use the `GitHub Issues <https://github.com/earthai-tech/fusionlab/issues>`_
  page.

Setting up for Development
----------------------------

To make changes to the code or documentation, you'll need to set up
a development environment. Please follow the instructions in the
:ref:`installation` guide under the section
**Installation from Source (for Development)**. This typically
involves:

1. Forking the repository on GitHub (``earthai-tech/fusionlab``).
2. Cloning your fork locally (e.g.,
   :code:`git clone https://github.com/[YourUsername]/fusionlab.git`).
3. Installing the package in editable mode with development
   dependencies (:code:`pip install -e .[dev]`). Using a virtual environment
   (like ``venv`` or ``conda``) is highly recommended.

Making Changes
--------------

1.  **Create a Branch:** Create a new branch from the ``main`` branch
    (or the current primary development branch) for your changes.
    Use a descriptive name (e.g., ``fix/lstm-state-bug`` or
    ``feature/add-transformer-encoder``).

    .. code-block:: bash

       # Make sure your main branch is up-to-date with the upstream repo
       # (First time setup: git remote add upstream https://github.com/earthai-tech/fusionlab.git)
       git checkout main
       git pull upstream main
       # Create your new feature branch
       git checkout -b your-descriptive-branch-name

2.  **Code Style:** Please follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_
    guidelines and strive for code consistency with the existing
    codebase. Using a linter like Flake8 (included in dev dependencies)
    is encouraged.

3.  **Docstrings:** Write clear and informative docstrings for any new
    functions or classes. This project generally follows the **NumPy
    docstring standard**. Ensure existing docstrings are updated if
    function signatures or behavior change.

4.  **Testing:** ``fusionlab-learn`` uses `pytest` for testing.
    * Add new tests for any new features you implement in the relevant
      `tests/` directory.
    * Add or update tests to cover any bug fixes, ensuring the bug
      is resolved and doesn't reappear.
    * Ensure all tests pass before submitting your changes. Run tests
      from the project root directory:

    .. code-block:: bash

       pytest fusionlab/tests # Or simply 'pytest' if configured

5.  **Documentation:** If your changes affect the user interface, add
    new features, or change behavior, please update the relevant
    documentation files (in `docs/source/`). Build the documentation
    locally to check formatting and ensure links work correctly:

    .. code-block:: bash

       # Navigate to the docs directory
       cd docs
       # Build the HTML documentation
       make clean html
       # Open _build/html/index.html in your browser

6.  **Commit Changes:** Make clear, concise commit messages that explain
    the "what" and "why" of your changes.

Submitting a Pull Request
---------------------------

1.  **Push to Fork:** Push your changes to your forked repository on
    GitHub:

    .. code-block:: bash

       git push origin your-descriptive-branch-name

2.  **Open Pull Request:** Go to the original ``fusionlab-learn`` repository
    on GitHub (`earthai-tech/fusionlab`) and open a Pull Request (PR)
    from your branch to the ``fusionlab-learn`` `main` branch (or the
    designated development branch).

3.  **Describe PR:** Write a clear description of the changes you made
    and why they are needed. Link to the relevant GitHub issue(s) using
    `#issue-number` notation (e.g., "Closes #123").

4.  **Checks:** Ensure any automated checks (Continuous Integration tests,
    linters) configured for the repository pass on your PR. Address any
    failures.

5.  **Review:** Your PR will be reviewed by the maintainers. Be
    prepared to discuss your changes and make adjustments based on
    feedback. Respond to comments and push updates to your branch as
    needed (the PR will update automatically).

Code of Conduct
---------------

All participants in the ``fusionlab-learn`` project (contributors,
maintainers, users in community spaces like the issue tracker) are
expected to adhere to the project's :doc:`Code of Conduct <code_of_conduct>`. 
Please review this document to understand the expected standards of behavior.

We strive to foster an open, welcoming, and respectful community.

Thank you again for contributing to ``fusionlab-learn``!