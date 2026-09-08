============================
IBSI coverage and validation
============================

Implementation Coverage
-----------------------

Z-Rad supports all IBSI I preprocessing operations and radiomic features,
and all IBSI II filters. Implementation coverage describes the available
operations; benchmark validation describes numerical comparisons for specific
reference configurations. They should be assessed separately.

The matrix below identifies what the reference suites exercise, rather than
reporting a test result. Use the linked configurations and the test report to
assess numerical agreement for a particular version.

.. list-table:: Implementation and reference-test scope
   :header-rows: 1
   :widths: 20 40 40

   * - Area
     - Implemented
     - Reference-test scope
   * - IBSI I preprocessing
     - Image and mask interpolation, resegmentation, and discretization;
       see :doc:`../user/preprocessing`.
     - CT phantom configurations A–E exercise selected settings through
       downstream feature comparisons. See `IBSI I reference configurations`_.
   * - IBSI I morphology
     - Morphological features; see :doc:`../user/radiomics`.
     - Default morphology features are requested in the A–E suite.
       Reference tags are compared only when returned by extraction.
   * - Moran's I and Geary's C
     - Optional 3D ``morphology_correlation`` family, also selected by
       ``families="all"``; excluded from default extraction due to cost.
     - Not requested by the IBSI I reference suite. Implementation availability
       should not be read as reference validation by that suite.
   * - IBSI I intensity
     - Local intensity, intensity statistics, intensity histograms,
       and intensity-volume histograms; see :doc:`../reference/radiomics`.
     - A–E compare returned features, subject to preparation and the
       configuration A exclusion in `Current Benchmark Limitations`_.
   * - IBSI I texture
     - GLCM, GLRLM, GLSZM, GLDZM, NGTDM, and NGLDM with applicable
       aggregation methods; see :doc:`../user/radiomics`.
     - A/B exercise 2D and 2.5D aggregation; C/D/E exercise 3D aggregation.
       The linked tests specify the exact settings and compared tags.
   * - IBSI II filters
     - Mean, LoG, Laws, Gabor, separable wavelets, Simoncelli wavelets,
       and Riesz transforms; see :doc:`../user/filtering`.
     - Phase I compares selected digital-phantom response maps; phase II
       compares intensity statistics on filtered CT images. See
       `IBSI II reference configurations`_ for filter-specific checks.

IBSI I Reference Configurations
-------------------------------

CT phantom test functions: `A <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_1.py#L187>`_, `B <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_1.py#L236>`_, `C <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_1.py#L283>`_, `D <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_1.py#L312>`_, `E <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_1.py#L339>`_.

Reference values and per-feature tolerances: `A <https://github.com/medical-physics-usz/z-rad/blob/master/tests/data/ibsi_1_reference_values_config_A.csv>`_, `B <https://github.com/medical-physics-usz/z-rad/blob/master/tests/data/ibsi_1_reference_values_config_B.csv>`_, `C <https://github.com/medical-physics-usz/z-rad/blob/master/tests/data/ibsi_1_reference_values_config_C.csv>`_, `D <https://github.com/medical-physics-usz/z-rad/blob/master/tests/data/ibsi_1_reference_values_config_D.csv>`_, `E <https://github.com/medical-physics-usz/z-rad/blob/master/tests/data/ibsi_1_reference_values_config_E.csv>`_.

The `digital-phantom reference table <https://github.com/medical-physics-usz/z-rad/blob/master/tests/data/ibsi_1_reference_values_digital_phantom.csv>`_
is bundled, but is not read by the current ``tests/test_ibsi_1.py`` suite.
Its presence does not establish an automated digital-phantom comparison.

IBSI II Reference Configurations
--------------------------------

Phase I links below point to the test functions containing the exact
configuration IDs, filter parameters, and response-map filenames. The maps
are distributed in `IBSI_II.zip <https://github.com/medical-physics-usz/z-rad/blob/master/tests/data/IBSI_II.zip>`_ under
``Ph_I/response_maps`` after extraction.

.. list-table:: Filter-specific reference checks
   :header-rows: 1
   :widths: 30 35 35

   * - Implemented filter
     - Phase I response-map tests
     - Phase II CT feature tests

   * - Mean
     - `Group 1 <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L139>`_
     - No CT feature comparison in this suite.

   * - Laplacian of Gaussian (LoG)
     - `Group 2 <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L156>`_
     - `2.A <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L437>`_, `2.B <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L449>`_

   * - Laws
     - `Group 3 <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L175>`_
     - `3.A <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L461>`_, `3.B <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L475>`_, `4.A <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L489>`_, `4.B <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L510>`_

   * - Gabor
     - `Group 4 <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L265>`_
     - `5.A <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L532>`_, `5.B <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L555>`_

   * - Separable wavelets (Daubechies, Coiflet, Haar)
     - `Group 5 <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L301>`_, `Group 6 <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L321>`_, `Group 7 <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L341>`_
     - `6.A <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L578>`_, `6.B <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L598>`_, `7.A <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L618>`_, `7.B <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L638>`_

   * - Simoncelli
     - `Group 8 <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L361>`_
     - `8.A <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L658>`_, `8.B <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L672>`_, `9.A <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L688>`_, `9.B <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L702>`_

   * - Riesz-transformed LoG
     - `Group 9 <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L380>`_
     - No CT feature comparison in this suite.

   * - Riesz-transformed Simoncelli
     - `Group 10 <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L415>`_ (10.b.1; conditional skip)
     - No CT feature comparison in this suite.

The `IBSI II feature table <https://github.com/medical-physics-usz/z-rad/blob/master/tests/data/ibsi_2_reference_values.csv>`_ supplies phase II consensus values and tolerances,
selected by ``filter_id``. The current suite executes configurations 2.A–9.B;
the table also contains 1.A/1.B entries that this suite does not execute.
The phase I and phase II configuration numbers belong to separate benchmark
phases and should not be treated as interchangeable filter IDs.

Reference Tests and Tolerances
------------------------------

* `IBSI I comparison helper <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_1.py#L26>`_: reference
  value plus or minus the per-feature tolerance in the corresponding CSV.
* `IBSI II response-map comparison helper <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L34>`_:
  voxel-wise absolute tolerance of 1% of the reference map's intensity range.
* `IBSI II feature comparison helper <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py#L46>`_:
  consensus value plus or minus the per-feature tolerance in the CSV.

The `CI test workflow <https://github.com/medical-physics-usz/z-rad/actions/workflows/test.yml>`_
executes the unit and integration suites. A passing workflow reports the checks
that ran; it is not a percentage measure of IBSI implementation or validation
coverage.

Current Benchmark Limitations
-----------------------------

.. list-table:: Explicit benchmark exceptions
   :header-rows: 1
   :widths: 15 25 30 30

   * - Configuration
     - Affected feature or test
     - Reason / evidence
     - Practical implication
   * - IBSI I A
     - ``ih_qcod``: intensity-histogram quartile coefficient of dispersion,
       a measure of relative spread based on the lower and upper quartiles.
     - The comparison helper explicitly excludes it. The bundled table has
       a reference value (0.0455) and tolerance (0), but the reason for the
       exclusion is not documented in the test.
     - A passing configuration A test does not validate this feature.
       The exclusion requires investigation before agreement can be claimed.
   * - IBSI II 8.B
     - ``stat_qcod``: quartile coefficient of dispersion calculated from
       filtered-image intensities.
     - The bundled phase II table leaves both the consensus value and
       tolerance blank; the test also notes the empty reference entry.
     - This feature is excluded from the 8.B comparison. That comparison
       cannot establish agreement without a reference value and tolerance.
   * - IBSI II 10.b.1
     - Riesz-transformed Simoncelli response-map comparison.
     - The test skips if ``10_b_1-ValidCRM.nii`` is absent from the
       bundled response maps.
     - A skipped test supplies no numerical validation for this configuration.
       It runs when the expected reference map is available.

Both feature-comparison helpers check reference tags only when they are
present in the extracted result. Missing feature tags therefore do not fail
these comparisons. In particular, the default IBSI I extraction calls do not
request the optional ``morphology_correlation`` family. These suites alone
should not be interpreted as exhaustive validation of every implemented
feature or every parameter combination.

Reproducing the Checks
----------------------

From a repository checkout, install the test dependencies and run:

.. code-block:: bash

   python -m pip install -e ".[test]"
   python -m pytest tests/test_ibsi_1.py tests/test_ibsi_2.py -ra

The test fixtures unpack the bundled phantom archives. The report includes
failures and skipped tests, which should be reviewed alongside the reference
configuration and its tolerance.

For reproducible studies, retain the Z-Rad version, image and mask geometry,
and exact preprocessing, filtering, discretization, and aggregation settings
alongside the extracted feature table.

Data Attribution
----------------

The bundled IBSI datasets use multiple open licenses depending on the
component. See `tests/data/README.md <https://github.com/medical-physics-usz/z-rad/blob/master/tests/data/README.md>`_
for the attribution and license terms of each subset.
