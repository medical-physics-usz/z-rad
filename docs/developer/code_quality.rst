Code Quality
============

Ruff
----

Ruff is the local Python formatting and linting gate used by
``.github/workflows/python-lint.yml``.

Check formatting with:

.. code-block:: bash

   ruff format --check zrad tests main.py generate_executable.py

Run the linter with:

.. code-block:: bash

   ruff check zrad tests main.py generate_executable.py

Fix Ruff failures before requesting review. If a change intentionally updates
Python formatting, run ``ruff format`` on the affected files and then re-run
the checks above.

Super-Linter
------------

Super-Linter runs in GitHub Actions through ``.github/workflows/lint.yml``.
It validates changed files only. The repository does not currently provide a
local one-command Super-Linter wrapper.

The currently enabled Super-Linter checks are:

* GitHub Actions
* YAML
* YAML Prettier formatting
* Checkov
* Gitleaks
* JSCPD duplicate detection
* merge-conflict markers

Inspect the Super-Linter result on the pull request, fix any reported files,
and wait for the check to pass before requesting review.
