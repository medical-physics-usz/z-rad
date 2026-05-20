Testing
=======

Test Categories
---------------

Unit tests are fast, isolated checks for preprocessing, filtering, radiomics
helpers, validation behavior, and regression coverage. They should not require
large data-backed workflows.

Integration tests exercise data-backed workflows, especially IBSI validation
and end-to-end feature calculations. Run them when changing preprocessing,
filtering, radiomics, IBSI behavior, test data handling, or any code path that
affects feature results.

Standard Commands
-----------------

Run unit tests with:

.. code-block:: bash

   pytest -m unit

Run integration tests with:

.. code-block:: bash

   pytest -m integration

Coverage
--------

The pull request test workflow runs unit and integration tests with coverage
enabled. To reproduce that sequence locally, run:

.. code-block:: bash

   pytest -m unit --cov=zrad
   pytest -m integration --cov=zrad --cov-append
   coverage report -m --skip-covered
   coverage html

The terminal report shows uncovered lines. The HTML coverage report is written
to ``htmlcov/``.

IBSI Test Data
--------------

IBSI fixtures unpack archived test data from ``tests/data/`` during test runs.
When adding or changing test data, preserve the licensing and attribution
information documented in ``tests/data/README.md``.

Numerical Assertions
--------------------

Radiomics and image-processing tests often compare floating point values. Use
explicit tolerances for floating point feature values so the expected precision
is visible in the test. Use exact array checks only when exact values are part
of the intended behavior, such as discrete masks, labels, or deterministic
integer-valued arrays.
