Continuous Integration
======================

Pull Request Checks
-------------------

GitHub Actions runs the main contributor checks on pull requests to
``master``:

* tests and coverage on Python 3.11, 3.12, 3.13, and 3.14
* Ruff formatting and linting on Python 3.11
* Super-Linter on changed files

Contributors should reproduce the unit tests, relevant integration tests,
coverage sequence, and Ruff checks locally before requesting review.
Super-Linter is normally reviewed in the pull request CI result because the
repository does not currently provide a local wrapper.

Documentation Build
-------------------

The documentation workflow builds the Sphinx site on Python 3.12 with warnings
treated as errors:

.. code-block:: bash

   python -m sphinx -b html -W docs docs/_build/html

On pushes to ``master``, the workflow uploads the generated site and deploys it
to GitHub Pages.

Publishing
----------

Publishing to PyPI is maintainer-focused. The publish workflow runs when a
GitHub release is published. It builds the package and uploads the distribution
artifacts with Twine.
