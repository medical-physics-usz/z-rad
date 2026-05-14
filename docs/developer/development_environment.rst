Development Environment
=======================

Python Version
--------------

Z-Rad supports Python 3.11 and newer. The continuous integration test matrix
currently runs on Python 3.11, 3.12, 3.13, and 3.14.

Virtual Environment
-------------------

Use an isolated virtual environment from the repository root:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate

On Windows, activate the environment with:

.. code-block:: powershell

   .venv\Scripts\Activate.ps1

Contributor Install
-------------------

Install the full contributor toolchain with the ``dev`` extra:

.. code-block:: bash

   python3 -m pip install --upgrade pip
   python3 -m pip install -e ".[dev]"
   python3 -m pip check

The ``pip check`` command verifies that installed dependencies do not have
conflicting requirements.

Optional Dependency Groups
--------------------------

The project defines these optional dependency groups:

* ``docs`` installs Sphinx and documentation theme dependencies.
* ``test`` installs pytest, coverage, and parallel test tooling.
* ``lint`` installs Ruff.
* ``dev`` installs the docs, test, and lint toolchains together.

Use ``dev`` for regular contributor work so tests, docs, and code-quality
checks run from the same environment.
