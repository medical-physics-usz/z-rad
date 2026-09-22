Contributing
============

Use this sequence to take a change from a local checkout to a pull request.
Run commands from the repository root in your development environment.

Set up and make the change
--------------------------

Follow :doc:`development_environment` to install the contributor tools. Keep
the change focused on one problem so its behavior and tests are easy to review.
The repository map in :doc:`index` can help you find the relevant code.

Add or update tests when changing preprocessing, filtering, or radiomics logic.
Preserve IBSI validation coverage when modifying feature calculations; see
:doc:`ibsi_validation` for reference handling.

Check the change locally
------------------------

Before opening a pull request:

* Run relevant tests during development, then the unit suite and any integration
  tests affected by the change. Review coverage as described in :doc:`testing`.
* Run the formatting and lint checks in :doc:`code_quality`.
* Update the documentation for changed behavior, public APIs, and examples,
  then build it using :doc:`building_docs`. That page explains where each kind
  of documentation belongs.
* If the change targets performance, compare measurements using
  :doc:`benchmarking` and retain the reference and candidate results.

Open the pull request
---------------------

Explain the problem, what changes for users or developers, and how you checked
it. Mention any relevant limitations or checks that you could not run.

After opening the pull request, inspect its :doc:`CI results <ci>` and address
failures before requesting review. Repeat the affected checks after revisions.
The documentation workflow does not run on pull requests, so build changed
documentation locally even when the other CI checks pass.
