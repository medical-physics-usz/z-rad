=======================
IBSI Benchmark Coverage
=======================

Z-Rad is developed around IBSI-oriented radiomics workflows and includes test
data and regression tests derived from IBSI reference material. The datasets
and reference values are under ``tests/data``. Automated benchmarks are in
``tests/test_ibsi_1.py``, ``tests/test_ibsi_2.py``, and ``tests/test_pet_suv.py``.

The feature comparisons require every expected reference feature for the
selected configuration and aggregation mode. Expected features are selected
from reference metadata, independently of the extracted result keys. Missing
expected features fail validation. Diagnostic measurements and absent references
are accounted for separately.

Implementation Coverage
-----------------------

Z-Rad supports all IBSI I preprocessing operations and radiomic features,
and all IBSI II filters. Implementation coverage describes available
operations; benchmark validation describes numerical agreement for specific
reference configurations. They should be assessed separately.

.. list-table:: Implementation guides
   :header-rows: 1
   :widths: 25 50 25

   * - Area
     - Available operations
     - Guide
   * - IBSI I preprocessing
     - Image and mask interpolation, resegmentation, and intensity discretization.
     - :doc:`../user/preprocessing`
   * - IBSI I features
     - Morphology, local intensity, intensity statistics, histograms,
       intensity-volume histograms, and texture. Moran's I and Geary's C
       are included in default 3D morphology extraction.
     - :doc:`../user/radiomics` and :doc:`../reference/radiomics`
   * - IBSI II filters
     - Mean, LoG, Laws, Gabor, separable wavelets, Simoncelli wavelets,
       and Riesz transforms.
     - :doc:`../user/filtering`

Reference Tests and Data
------------------------

* `IBSI I tests <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_1.py>`_
  define the digital-phantom and CT A--E configurations and aggregation modes.
  The `IBSI I reference tables <https://github.com/medical-physics-usz/z-rad/tree/master/tests/data/ibsi_1_reference_data>`_
  provide the expected values and tolerances.
* `IBSI II tests <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py>`_
  define the phase I filter settings and response-map comparisons, and
  phase II CT feature comparisons. The phase I and phase II configuration
  numbers belong to separate benchmarks and are not interchangeable filter IDs.
  The `IBSI II reference data <https://github.com/medical-physics-usz/z-rad/tree/master/tests/data/ibsi_2_reference_data>`_
  contain the response-map archive and phase II feature table.
* `IBSI-SUV tests <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_pet_suv.py>`_
  define the valid and intentionally invalid digital reference cases.

Coverage Matrix
---------------

.. list-table:: Implemented benchmark coverage
   :header-rows: 1
   :widths: 25 30 45

   * - Benchmark
     - Cases / comparisons
     - Scope and limitations
   * - IBSI I digital phantom
     - 6 aggregation modes; 169 features per mode
     - 482 distinct reference tags.
   * - IBSI I CT diagnostics A--E
     - Initial, interpolated and resegmented stages, separately reported
     - All 60 diagnostic rows per configuration, including image and ROI
       dimensions, voxel spacing, bounding boxes, voxel counts and intensities.
   * - IBSI I CT A and B
     - 4 aggregation modes each; 169 features per mode
     - 346 distinct reference tags per configuration.
   * - IBSI I CT C, D and E
     - 2 aggregation modes each; 169 features per mode
     - 210 distinct reference tags per configuration.
   * - IBSI II phase I
     - All 33 bundled published response maps
     - Each map is a separate case; shape and finite values
       are required, and every voxel must satisfy the 1% reference-range tolerance.
   * - IBSI II phase II 1.A--9.B
     - 18 configurations; 323 reference feature comparisons
     - ``stat_qcod`` for 8.B has no IBSI consensus therefore is not benchmarked.
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

Supplemental Asset and Filter Checks
------------------------------------

``tests/test_ibsi_supplemental.py`` contains 53 supplemental cases using the
IBSI II digital assets. These do not add published consensus comparisons to
the coverage matrix:

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

Execution reports list supplemental cases and their status counts separately
from consensus benchmarks. The integration-test selection runs both groups.

IBSI I Reference Availability and Coverage Limits
-------------------------------------------------

The bundled IBSI I tables leave reference values and tolerances blank for
``morph_vol_dens_ombb``, ``morph_area_dens_ombb``, ``morph_vol_dens_mvee``,
``morph_area_dens_mvee``, and ``ivh_auc``. These five rows are explicitly
classified as lacking references and are excluded from numerical comparisons.
Unexpected blanks fail reference loading. Previously unavailable rows gaining
values also fail loading, requiring review of the documented exception.

Comparison Rules
----------------

All scalar IBSI reference-table comparisons use the same tolerance policy,
including IBSI I CT configurations A--E, the digital phantom, preprocessing
diagnostics, and IBSI II phase II features.

When the published tolerance is zero, the suite requires exact agreement
after rounding the computed value to the reference precision. It uses at
least three significant figures, retaining any finer precision recorded in
the reference text. For example, ``2.148648...`` matches ``2.15``,
``0.0454545...`` matches ``0.0455``, and ``0.9765625`` matches ``0.977``.
The rule uses significant figures, not a fixed number of decimal places.

Trailing zeros in plain integer references are treated as place holders
(thus ``1494.6`` matches ``1490`` at three significant figures). Decimal or
scientific notation retains explicitly written precision; ``1.490e3`` records
four significant figures. References containing more significant digits,
such as a voxel count of ``125256``, retain those digits. A zero reference
with zero tolerance requires an exact zero.

For positive tolerances, the unrounded result must lie in the inclusive
interval ``[reference - tolerance, reference + tolerance]``. Missing expected
features, NaN, and infinite results fail validation. This policy changes only
benchmark comparisons, not the extracted feature values or reference files.
Rows without published references remain subject to the documented exclusions.

Phase I response-map comparisons use the separate voxel-wise 1% range rule.
IBSI-SUV checks use their specified two-decimal comparison; neither reads a
scalar reference-table tolerance field.

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

Test Results and Reproduction
-----------------------------

CI publishes an IBSI execution report and JUnit results in
``ibsi-results-python-*`` artifacts for each Python version. Reports identify
individual passed, failed or skipped cases, plus
the revision, working-tree state and environment at report-generation time.
Generate reports immediately after testing; see :doc:`../developer/testing`
for reproduction commands. A passing result applies only to the comparisons
in the matrix above; line coverage is a separate metric.

For reproducible studies, retain the Z-Rad version, image and mask geometry,
and exact preprocessing, filtering, discretization, and aggregation settings
alongside the extracted feature table.

Licensing
---------

The bundled IBSI datasets use multiple open licenses depending on the specific
component. See `tests/data/README.md
<https://github.com/medical-physics-usz/z-rad/blob/master/tests/data/README.md>`_
for the exact attribution and license terms of each dataset subset.
