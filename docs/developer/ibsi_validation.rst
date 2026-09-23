IBSI validation
===============

Use this guide when investigating numerical agreement with IBSI references or
maintaining the validation tests. For everyday test commands, see
:doc:`testing`; for implementation coverage and unavailable references, see
:doc:`../ibsi/index`.

These checks compare numerical results. Execution time and memory measurements
are covered separately in :doc:`benchmarking`.

.. _ibsi-benchmark-reports:

IBSI validation reports
-----------------------

CI saves per-case integration results and an IBSI summary for each Python
version as ``ibsi-results-python-*`` artifacts, including failures and skips.
To produce the same report locally:

.. code-block:: bash

   python -m pytest -m integration --junitxml=reports/integration.xml
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

Test data integrity
-------------------

IBSI fixtures unpack archives from ``tests/data/``. Extracted files are checked
against the archive's SHA-256 fingerprint and each member's CRC. Missing,
modified, or outdated files trigger extraction under a process lock; a completion
marker is written only after successful extraction. Preserve the licensing and
attribution information in ``tests/data/README.md`` when changing test data.

.. _ibsi-reference-validation:

Reference validation
--------------------

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

Trailing zeros in plain integer references are treated as placeholders
(thus ``1494.6`` matches ``1490`` at three significant figures). Decimal or
scientific notation retains explicitly written precision; ``1.490e3`` records
four significant figures. References containing more significant digits,
such as a voxel count of ``125256``, retain those digits. A zero reference
with zero tolerance requires an exact zero.

.. _ibsi-supplemental-checks:

Supplemental checks
-------------------

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
