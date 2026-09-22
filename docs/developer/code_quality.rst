Code quality
============

Run these checks from the repository root in the :doc:`development_environment`.
Ruff checks Python code locally and in CI; Super-Linter checks other repository
content in GitHub Actions.

Check Python formatting and lint
--------------------------------

Run both checks before requesting review:

.. code-block:: bash

   python -m ruff format --check zrad tests main.py generate_executable.py
   python -m ruff check zrad tests main.py generate_executable.py

These use the same file selection as ``.github/workflows/python-lint.yml``.
Rules and the Ruff version are defined in ``pyproject.toml``.

To fix formatting, run ``python -m ruff format`` with the affected file paths.
Review the diff and rerun both checks. For lint findings, read the diagnostic
and fix its cause; if you use ``python -m ruff check --fix``, review those edits
before including them in the change.

Inspect super-linter results
----------------------------

``.github/workflows/lint.yml`` runs Super-Linter on changed files. There is no
local wrapper in this repository; inspect its job output on the pull request
and rerun CI after addressing findings.

The enabled checks cover:

* GitHub Actions workflow syntax and usage
* YAML validity and Prettier formatting
* Checkov checks for configuration security issues
* Gitleaks detection of committed secrets
* JSCPD detection of duplicated content
* unresolved merge-conflict markers

See :doc:`ci` for workflow triggers and where to investigate failed jobs.
