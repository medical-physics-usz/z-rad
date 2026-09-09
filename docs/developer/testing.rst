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

IBSI Benchmark Reports
----------------------

CI saves per-case integration results and an IBSI summary for each Python
version as ``ibsi-results-python-*`` artifacts, including failures and skips.
To produce the same report locally:

.. code-block:: bash

   pytest tests/test_ibsi_1.py tests/test_ibsi_2.py tests/test_pet_suv.py -m integration --junitxml=reports/integration.xml
   python scripts/ibsi_report.py reports/integration.xml reports/ibsi.md

The report records each executed benchmark case from JUnit. Revision,
working-tree state, environment, dependency versions, and reference hashes
describe report generation, so generate the report immediately after testing
in the same checkout and environment. This metadata does not authenticate an
older or imported JUnit file. It is execution evidence, not a certification.
The reference-coverage matrix and limitations are in :doc:`../ibsi/index`.

Reference loaders reject empty selections, duplicate tags, unexpected blanks,
non-finite references, and negative tolerances. Feature comparisons require
all expected keys for the selected aggregation mode, independently of the
keys returned by extraction. IBSI I preprocessing diagnostics are compared
separately from radiomic features.

Invalid aggregation modes and modes without texture references fail selection.
If a previously unavailable reference gains a value or tolerance, loading
fails until its documented exception has been reviewed.

Extracted archives are checked against their SHA-256 fingerprint and each
member's CRC. Missing, modified, or outdated files trigger extraction under
a process lock; a completion marker is written only after successful extraction.
