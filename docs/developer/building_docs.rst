Building The Docs
=================

Requirements
------------

Install the documentation toolchain into the same Python environment you use to
run the project:

.. code-block:: bash

   python3 -m pip install -e ".[docs]"

For a full local development environment with docs, tests, and lint tooling,
install the development extra instead:

.. code-block:: bash

   python3 -m pip install -e ".[dev]"

Build Commands
--------------

Build the HTML site with warnings treated as errors:

.. code-block:: bash

   python3 -m sphinx -b html -W docs docs/_build/html

For iterative local builds, either of these commands can also be used from the
repository root:

.. code-block:: bash

   python3 -m sphinx -M html docs docs/_build

or:

.. code-block:: bash

   make -C docs html

Output
------

The generated documentation is written to ``docs/_build/html``.

When To Update Docs
-------------------

Update documentation together with code changes when:

* user-facing workflows or behavior change; update the relevant user-guide page
* public classes, functions, or arguments change; update the API reference
* a new contributor-visible workflow is introduced; add or update an example
* a new common user-facing failure mode is found; add troubleshooting guidance

Good documentation changes are usually narrative first and reference second:
explain when a feature should be used before listing every parameter.

Fixing Build Problems
---------------------

Common causes of docs failures include:

* Sphinx installed in a different Python environment than the one used to build
* missing runtime dependencies required by autodoc imports
* stale references in ``toctree`` blocks
