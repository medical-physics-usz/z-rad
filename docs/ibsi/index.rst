=======================
IBSI Benchmark Coverage
=======================

Overview
--------

Z-Rad is developed around IBSI-oriented radiomics workflows and includes test
data and regression tests derived from IBSI reference material.

Repository Assets
-----------------

The repository includes:

* IBSI test data under ``tests/data``
* reference feature values for IBSI I and IBSI II
* automated tests in ``tests/test_ibsi_1.py`` and ``tests/test_ibsi_2.py``

The feature comparisons require every expected reference feature for the
selected configuration and aggregation mode. Expected features are selected
from reference metadata, independently of the extracted result keys. Missing
expected features fail validation. Diagnostic measurements, absent references,
and the CT morphology-correlation coverage gap are accounted for separately.

Coverage Matrix
---------------

.. list-table:: Implemented benchmark coverage
   :header-rows: 1
   :widths: 25 30 45

   * - Benchmark
     - Cases / comparisons
     - Scope and limitations
   * - IBSI I CT A and B
     - 4 aggregation modes each; 167 features per mode
     - 344 distinct reference tags per configuration. Moran's I and Geary's C
       are not benchmarked on CT; see below.
   * - IBSI I CT C, D and E
     - 2 aggregation modes each; 167 features per mode
     - 208 distinct reference tags per configuration; same CT limitation.
   * - IBSI I CT diagnostics A--E
     - Initial, interpolated and resegmented stages, separately reported
     - All 60 diagnostic rows per configuration, including image and ROI
       dimensions, voxel spacing, bounding boxes, voxel counts and intensities.
   * - IBSI I digital phantom
     - 6 aggregation modes; 169 features per mode
     - 482 distinct reference tags, including Moran's I and Geary's C.
   * - IBSI II phase I
     - All 33 bundled published response maps
     - Includes 10.b.1. Each map is a separate case; shape and finite values
       are required, and every voxel must satisfy the 1% reference-range tolerance.
   * - IBSI II phase II 1.A--9.B
     - 18 configurations; 323 reference feature comparisons
     - Includes unfiltered 1.A/1.B. 8.B has 17 comparisons; other configurations
       have 18. ``stat_qcod`` for 8.B has no consensus.
   * - IBSI II phase II 10.A/B and 11.A/B
     - No published feature references
     - No benchmark-agreement claim for these configurations.
   * - IBSI-SUV
     - 43 valid and 15 intentionally invalid digital reference objects
     - Valid cases check ROI minimum, median and maximum to two decimals;
       invalid cases require an exception. These are not radiomic-feature tests.

Counts of distinct reference tags include aggregation suffixes and should not
be interpreted as counts of independent feature definitions. The suite does
not establish universal compliance across all inputs or processing options,
or cover the IBSI II phase III clinical validation study.

IBSI I Reference Availability and Coverage Limits
-------------------------------------------------

The bundled IBSI I tables leave reference values and tolerances blank for
``morph_vol_dens_ombb``, ``morph_area_dens_ombb``, ``morph_vol_dens_mvee``,
``morph_area_dens_mvee``, and ``ivh_auc``. These five rows are explicitly
classified as lacking references and are excluded from numerical comparisons.
Unexpected blanks fail reference loading. Previously unavailable rows gaining
values also fail loading, requiring review of the documented exception.

Moran's I (``morph_moran_i``) and Geary's C (``morph_geary_c``) have published
references and are tested on the digital phantom. They remain unbenchmarked
for CT A--E: the opt-in implementation creates quadratic-size pairwise
matrices that are impractical for the full CT ROIs. This is a coverage gap,
not an absence of reference values or demonstrated agreement on CT.

Reference Precision
-------------------

For IBSI I configuration A, all participants submitted ``0.0455`` for the
intensity-histogram quartile coefficient of dispersion (``ih_qcod``). This
agreement at the reported precision accounts for the zero tolerance in the
reference table.

Z-Rad's observed quartiles are 21 and 23, giving
``(23 - 21) / (23 + 21) = 1/22 = 0.0454545...``, which rounds to ``0.0455``.
The validation therefore requires exact agreement with the reference after
rounding the computed value to four decimal places. This checks agreement
at the published precision; it does not require the unrounded value to equal
the rounded reference. The feature must be present for the comparison to
pass. Other comparisons retain their existing reference-value and tolerance
checks.

The digital-phantom CSV records zero tolerance for many rounded reference
values. These entries are compared at three significant figures; nonzero
tolerances retain the published numerical intervals. Zero-tolerance diagnostic
measurements are checked at the decimal precision recorded in the CSV (for
example, spacing ``0.9765625`` mm is reported as ``0.977`` mm). These precision
rules do not alter production feature values.

IBSI II Reference Availability
------------------------------

For IBSI II configuration 8.B (3D Simoncelli filtering, decomposition level 1),
Table 7.16 of the `IBSI II reference manual
<https://doi.org/10.48550/arXiv.2006.05470>`_ explicitly reports consensus as
``none`` for the quartile coefficient of dispersion (``stat_qcod``). IBSI
therefore provides neither a reference value nor a tolerance for this
feature/filter combination. The blank fields in the bundled reference CSV
match the `official IBSI reference data
<https://github.com/theibsi/ibsi_2_reference_data/blob/main/reference_feature_values/reference_values.csv>`_.

The 8.B test excludes only ``stat_qcod`` from the reference comparison and
continues to check the other 17 reference features. This is an intentional
exception due to the absence of IBSI consensus, not missing repository data
or evidence of a calculation defect. A passing 8.B comparison does not
establish IBSI agreement for ``stat_qcod``.

For IBSI II phase II configurations 10.A, 10.B, 11.A, and 11.B, IBSI provides
no reference values or tolerances for any of the 18 features. These entire
configurations are absent from the official reference-feature CSV and its
bundled copy, which cover configurations 1.A through 9.B. Consequently, no
IBSI phase II feature agreement can be established for 10.A, 10.B, 11.A, or
11.B using the published reference data.

This limitation concerns phase II feature values. Phase I response-map
comparisons are separate tests and do not establish phase II feature agreement
for these configurations.

Licensing
---------

The bundled IBSI datasets use multiple open licenses depending on the specific
component. See ``tests/data/README.md`` for the exact attribution and license
terms of each dataset subset.

Why This Matters
----------------

The IBSI tests provide a reproducibility baseline for:

* preprocessing choices
* filter definitions
* discretization behavior
* radiomics feature calculations

When changing feature code or preprocessing behavior, these tests should remain
part of the release and CI validation workflow.

Verification Evidence
---------------------

CI publishes an IBSI execution report and JUnit results for each Python
version. Reports identify individual passed, failed or skipped cases, plus
the revision, working-tree state and environment at report-generation time.
Generate reports immediately after testing; see :doc:`../developer/testing`
for reproduction commands. A passing result applies only to the comparisons
in the matrix above; line coverage is a separate metric.

Last local verification: 2026-09-09, Python 3.14.6 on macOS, with the working
tree changes described here. All 144 benchmark cases passed with zero skips:
14 CT feature cases, 15 CT diagnostic cases, 6 digital-phantom cases,
33 phase I response maps, 18 phase II feature cases, and 58 IBSI-SUV cases.
The combined benchmark, reference-helper and radiomics run passed 237 tests.
The full unit suite, including the subsequent exception and aggregation
selection checks, passed 270 tests. This is a local
verification snapshot; CI reports provide evidence for subsequent revisions
and other Python versions.

The new digital-phantom tests also identified and verified two corrections:
histogram gradients now preserve empty grey-level bins, and global intensity
peaks normalize boundary neighbourhoods by the number of available image
voxels. These changes can affect features on images with missing histogram
bins or intensity-peak neighbourhoods extending beyond the image boundary.
