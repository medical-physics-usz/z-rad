Contributing
============

Development Environment
-----------------------

Z-Rad supports Python 3.11 and newer. For local development, use an isolated
virtual environment from the repository root:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate

On Windows, activate the environment with:

.. code-block:: powershell

   .venv\Scripts\Activate.ps1

Install the full contributor toolchain with the ``dev`` extra. This includes
the runtime dependencies, documentation tooling, test dependencies, coverage
tools, and Ruff:

.. code-block:: bash

   python3 -m pip install --upgrade pip
   python3 -m pip install -e ".[dev]"
   python3 -m pip check

Before Opening A Pull Request
-----------------------------

Run the focused test set for the area you changed before opening a pull
request. Unit tests should pass for every code change:

.. code-block:: bash

   pytest -m unit

Run integration tests when changing preprocessing, filtering, radiomics, IBSI
behavior, test data handling, or any code path that affects end-to-end feature
calculation:

.. code-block:: bash

   pytest -m integration

The pull request test workflow runs unit and integration tests with coverage
enabled. To reproduce the coverage sequence locally, run:

.. code-block:: bash

   pytest -m unit --cov=zrad
   pytest -m integration --cov=zrad --cov-append
   coverage report -m --skip-covered
   coverage html

The terminal report shows uncovered lines. The HTML coverage report is written
to ``htmlcov/``.

Linting
-------

Ruff is the local Python lint and format gate used by
``.github/workflows/python-lint.yml``. Check formatting with:

.. code-block:: bash

   ruff format --check zrad tests main.py generate_executable.py

Then run the linter:

.. code-block:: bash

   ruff check zrad tests main.py generate_executable.py

Fix Ruff failures before requesting review. If you intentionally change
formatting, run ``ruff format`` on the affected Python files and re-run the
checks above.

Super-Linter
------------

Super-Linter runs in GitHub Actions through ``.github/workflows/lint.yml``.
It validates changed files only and currently checks GitHub Actions, YAML, YAML
Prettier formatting, Checkov, Gitleaks, JSCPD duplicate detection, and merge
conflict markers.

The repository does not currently provide a local one-command Super-Linter
wrapper. Inspect the Super-Linter result on the pull request, fix any reported
files, and wait for the check to pass before requesting review.

General Expectations
--------------------

* keep user-facing behavior and documentation aligned
* add or update tests when changing preprocessing, filtering, or radiomics logic
* preserve IBSI validation coverage when modifying feature calculations

Documentation Expectations
--------------------------

When adding new functionality:

* update the relevant user-guide page
* extend the API reference if the public surface changes
* add an example when the change introduces a new workflow

Good documentation changes are usually narrative first and reference second:
explain when a feature should be used before listing every parameter.
