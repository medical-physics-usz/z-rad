Contributing
============

Start by setting up a local development environment with the contributor
toolchain. See :doc:`development_environment` for Python version, virtual
environment, and dependency installation guidance.

Before Opening A Pull Request
-----------------------------

Before opening a pull request:

* run the relevant test set and coverage checks described in :doc:`testing`
* run the Ruff formatting and lint checks described in :doc:`code_quality`
* update and build documentation when user-facing behavior, public APIs, or
  examples change; see :doc:`building_docs`
* inspect the pull request CI results and fix failures before requesting review;
  see :doc:`ci`

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
