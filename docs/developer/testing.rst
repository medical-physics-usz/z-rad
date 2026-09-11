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

Extracted archives are checked against their SHA-256 fingerprint and each
member's CRC. Missing, modified, or outdated files trigger extraction under
a process lock; a completion marker is written only after successful extraction.

Numerical Assertions
--------------------

Radiomics and image-processing tests often compare floating point values. Use
explicit tolerances for floating point feature values so the expected precision
is visible in the test. Use exact array checks only when exact values are part
of the intended behavior, such as discrete masks, labels, or deterministic
integer-valued arrays.

.. _ibsi-benchmark-reports:

IBSI Benchmark Reports
----------------------

CI saves per-case integration results and an IBSI summary for each Python
version as ``ibsi-results-python-*`` artifacts, including failures and skips.
To produce the same report locally:

.. code-block:: bash

   pytest -m integration --junitxml=reports/integration.xml
   python tests/ibsi_report.py reports/integration.xml reports/ibsi.md

This uses the same integration-test selection as CI, including the 53
supplemental IBSI asset and filter checks. The report selects IBSI benchmark
and supplemental results from the JUnit file and lists their totals separately;
unrelated integration tests are not included in the IBSI summary.

The report records each executed benchmark case from JUnit. Revision,
working-tree state, environment, dependency versions, and reference hashes
describe report generation, so generate the report immediately after testing
in the same checkout and environment. This metadata does not authenticate an
older or imported JUnit file. It is execution evidence, not a certification.
The reference-coverage matrix and limitations are in :doc:`../ibsi/index`.

.. _ibsi-reference-validation:

IBSI Reference Validation
-------------------------

The :ref:`comparison rules <ibsi-comparison-rules>` define benchmark agreement.
The following checks keep reference selection and precision handling explicit.

Reference loaders reject empty selections, duplicate tags, unexpected blanks,
non-finite references, and negative tolerances. Feature comparisons require
all expected keys for the selected aggregation mode, independently of the
keys returned by extraction. IBSI I preprocessing diagnostics are compared
separately from radiomic features.

Invalid aggregation modes and modes without texture references fail selection.
If a previously unavailable reference gains a value or tolerance, loading
fails until its documented exception has been reviewed.

Trailing zeros in plain integer references are treated as place holders
(thus ``1494.6`` matches ``1490`` at three significant figures). Decimal or
scientific notation retains explicitly written precision; ``1.490e3`` records
four significant figures. References containing more significant digits,
such as a voxel count of ``125256``, retain those digits. A zero reference
with zero tolerance requires an exact zero.

.. _ibsi-supplemental-checks:

Supplemental IBSI Checks
------------------------

``tests/test_ibsi_supplemental.py`` contains 53 cases using IBSI II digital
assets. They exercise loading, geometry, and filter behavior independently
of published consensus comparisons:

* Nine DICOM/NIfTI geometry checks cover all supplied phantoms, including
  orientation, noise, empty, and patterns 2 and 3. NIfTI images and the eight
  supplied masks are loaded through Z-Rad; DICOM geometry is inspected through
  SimpleITK. The formats have different origins and their voxel arrays agree
  after reversing the slice axis; they are not identical physical grids.
* Nine Z-Rad DICOM loading checks require successful decoding of the synthetic
  CT series, checking voxel values, dimensions, spacing, direction, and origin.
  Entirely nonnegative decoded CT intensities are accepted: intensity sign alone
  does not establish whether HU conversion succeeded. SimpleITK applies the
  declared DICOM rescaling; separate DICOM regression tests verify known slopes
  and intercepts, including fractional and nonnegative outputs.
* Twenty-four zero-input cases check mean, Laplacian-of-Gaussian, and signed
  Laws filtering in 2D and 3D with constant, nearest, wrap, and reflect padding.
* Two mean-filter cases compare orientation and noise outputs to explicit
  periodic neighbourhood averages and check geometry and input preservation.
* Nine directional Laws cases compare the orientation and pattern 2/3 outputs
  with explicit three-tap convolution stencils along each axis. These check
  axis assignment and response sign without using generated golden outputs.
