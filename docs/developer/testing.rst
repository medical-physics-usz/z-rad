Testing
=======

Run tests from the repository root in the :doc:`development_environment`.
Use focused tests while developing, then run the suites relevant to your change.

Choose the test scope
---------------------

* **Unit tests** check individual operations, validation, and regressions without
  large data-backed workflows.
* **Integration tests** exercise data-backed workflows, including IBSI validation
  and end-to-end feature calculations. Run them for changes affecting feature
  results, preprocessing, filtering, or test-data handling.
* **Performance benchmarks** measure runtime and memory separately; use
  :doc:`benchmarking` rather than ordinary test commands.

Run the correctness suites with:

.. code-block:: bash

   python -m pytest -m unit
   python -m pytest -m integration

The repository's ``pytest.ini`` enables parallel workers (``-n auto``), coverage
collection, terminal and HTML coverage reports, strict marker/configuration
checks, and shortened tracebacks. Performance benchmarks are skipped by default.
These settings also apply when you run a single test.

Run a focused test
------------------

Run one file:

.. code-block:: bash

   python -m pytest tests/test_filtering.py

Run one test:

.. code-block:: bash

   python -m pytest tests/test_filtering.py::test_concrete_filter_constructor_valid_mean

For debugging in the current process, disable parallel workers and coverage.
``--tb=long`` shows a full traceback; add ``--pdb`` to enter the debugger on
failure:

.. code-block:: bash

   python -m pytest tests/test_filtering.py -n 0 --no-cov --tb=long

Review coverage
---------------

To reproduce the CI coverage sequence, start with the unit suite and append the
integration results:

.. code-block:: bash

   python -m pytest -m unit --cov=zrad
   python -m pytest -m integration --cov=zrad --cov-append
   python -m coverage report -m --skip-covered
   python -m coverage html

Open ``htmlcov/index.html`` to inspect uncovered lines relevant to
your change. Coverage is collected for review; ``--cov-fail-under=0`` means no
minimum percentage is enforced. A passing coverage command does not establish
that a changed behavior has been tested.

Add or update tests
-------------------

Mark correctness tests with ``@pytest.mark.unit`` or
``@pytest.mark.integration``, or apply the corresponding marker to the module.
The CI jobs explicitly select these markers, so an unmarked test is not included
in either selection. Additional markers such as ``gui`` do not replace them.
Declare any new custom marker in ``pytest.ini`` because strict markers are enabled.

Check observable behavior, including relevant error paths. Use explicit tolerances
for floating-point feature values so the expected precision is visible. Use exact
array comparisons when exact values are part of the behavior, such as discrete
masks, labels, or deterministic integer-valued arrays.

IBSI fixtures unpack archived data from ``tests/data/`` automatically. Preserve
its licensing and attribution information in ``tests/data/README.md`` when
changing datasets. For extraction integrity checks, reference precision, and
report generation, see :doc:`ibsi_validation`.
