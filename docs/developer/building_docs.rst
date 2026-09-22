Building and editing the docs
=============================

Run commands from the repository root in your activated Python environment.
The ``dev`` extra already includes the documentation tools. For a docs-only
setup, install:

.. code-block:: bash

   python -m pip install -e ".[docs]"

Build and preview
-----------------

Build the HTML site with warnings treated as errors:

.. code-block:: bash

   python -m sphinx -b html -W docs docs/_build/html

Open ``docs/_build/html/index.html`` in a browser. Check the changed pages,
code examples, tables, cross-references, and sidebar navigation. The build
checks Sphinx references and syntax but does not execute documented examples.

For a local HTTP preview, run:

.. code-block:: bash

   python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html

Open ``http://127.0.0.1:8000`` and stop the server with ``Ctrl+C`` when finished.
If you prefer Make, ``make -C docs html SPHINXOPTS="-W"`` builds the same output.

Rebuild from scratch
--------------------

After changing navigation or moving pages, clean the generated output and
rebuild every page so sidebars do not mix old and new structures:

.. code-block:: bash

   python -m sphinx -M clean docs docs/_build
   python -m sphinx -b html -E -a -W docs docs/_build/html

The clean step removes generated build files. ``-E`` discards Sphinx's saved
environment and ``-a`` writes every page. Refresh the browser after rebuilding;
use a hard refresh if it still displays older content.

Choose where to edit
--------------------

* ``docs/user/`` explains how to use the GUI and Python API.
* ``docs/examples/`` contains worked configurations and examples.
* ``docs/developer/`` describes contributor and maintainer tasks.
* ``docs/ibsi/`` explains validation coverage and reference limitations.
* ``docs/reference/`` organizes the API reference with ``autosummary`` entries.
  API descriptions and parameter details come from docstrings in ``zrad/``.

Update the user guide when behavior changes, docstrings and reference entries
when public APIs change, and examples when a new workflow needs illustration.
Add troubleshooting guidance for common user-facing failures. Explain when to
use a feature before listing its parameters.

Add a new narrative page as an ``.rst`` file and include its name, without the
extension, in the relevant parent page's ``toctree``. For example, the GUI
pages are listed in ``docs/user/gui_workflows.rst``. Use ``:doc:`` for links to
pages and explicit labels with ``:ref:`` for sections that other pages need to
reference. Preserve existing labels when moving content.

For a new public class, add its ``autosummary`` entry to the appropriate
reference page and document the class in its source docstring. Sphinx generates
reference stubs under ``docs/reference/generated/``; edit the source entries and
docstrings rather than generated stubs or HTML in ``docs/_build/``.

Fix build problems
------------------

* **Sphinx or an extension is missing:** activate the intended environment and
  install ``.[docs]`` there. Use ``python -m sphinx`` to use that interpreter.
* **An autodoc import fails:** install Z-Rad and its dependencies in the same
  environment, then inspect the import exception in the build output.
* **A page or section cannot be found:** check its ``toctree`` entry, link, and
  label. Update links after renaming pages or moving sections.
* **Sidebars or deleted pages remain visible:** perform the clean rebuild above
  and confirm the browser is showing this build's output.
* **The displayed version is unexpected:** check ``zrad.__version__`` in
  ``zrad/__init__.py`` and rebuild. Sphinx and package metadata both read this
  value; change it only when intentionally updating the project version.

The documentation workflow runs on pushes to ``master`` and manual dispatch,
not on pull requests. Build documentation changes locally before review;
see :doc:`ci` for deployment details.
