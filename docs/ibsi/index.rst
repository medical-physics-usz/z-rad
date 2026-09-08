===============
IBSI Compliance
===============

Implementation Coverage
-----------------------

Z-Rad supports all IBSI I preprocessing operations and radiomic features,
and all IBSI II filters. Implementation coverage describes the available
operations; benchmark validation describes numerical comparisons for specific
reference configurations. They should be assessed separately.

.. list-table:: Implementation coverage and validation entry points
   :header-rows: 1
   :widths: 20 45 35

   * - Area
     - Implemented scope
     - Documentation and reference checks
   * - IBSI I preprocessing
     - Image and mask interpolation, intensity resegmentation, and
       intensity discretization
     - :doc:`../user/preprocessing`; CT phantom configurations A–E in
       ``tests/test_ibsi_1.py``
   * - IBSI I morphology
     - Morphological features, including Moran's I and Geary's C
     - :doc:`../user/radiomics`; phantom comparisons in
       ``tests/test_ibsi_1.py`` (see benchmark limitations below)
   * - IBSI I intensity
     - Local intensity, intensity statistics, intensity histograms,
       and intensity-volume histograms
     - :doc:`../reference/radiomics`; digital and CT phantom comparisons
       in ``tests/test_ibsi_1.py``
   * - IBSI I texture
     - GLCM, GLRLM, GLSZM, GLDZM, NGTDM, and NGLDM, with applicable
       2D, 2.5D, and 3D aggregation methods
     - :doc:`../user/radiomics`; phantom comparisons in
       ``tests/test_ibsi_1.py``
   * - IBSI II filters
     - Mean, Laplacian of Gaussian, Laws, Gabor, separable wavelets,
       non-separable Simoncelli wavelets, and Riesz transforms
     - :doc:`../user/filtering`; digital phantom response maps and
       filtered CT phantom features in ``tests/test_ibsi_2.py``

Moran's I and Geary's C are available for 3D ROIs through the
``morphology_correlation`` family or ``families="all"``. They are not
included in default extraction because they can be computationally expensive.

Reference Tests and Tolerances
------------------------------

The repository includes IBSI digital and CT phantom data under ``tests/data``,
reference feature tables, and two integration test suites:

* `IBSI I tests <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_1.py>`_
  compare extracted features with the reference value plus or minus the
  per-feature tolerance stored in the corresponding CSV table.
* `IBSI II tests <https://github.com/medical-physics-usz/z-rad/blob/master/tests/test_ibsi_2.py>`_
  compare filter response maps voxel by voxel with an absolute tolerance of
  1% of the reference map's intensity range. Filtered-image feature comparisons
  use the consensus values and per-feature tolerances in the reference CSV.

The `CI test workflow <https://github.com/medical-physics-usz/z-rad/actions/workflows/test.yml>`_
executes the unit and integration suites. A passing workflow reports the checks
that ran; it is not a percentage measure of IBSI implementation or validation
coverage.

Current Benchmark Limitations
-----------------------------

The reference suites contain explicit exceptions:

* IBSI I configuration A excludes ``ih_qcod`` from comparison.
* IBSI II configuration 8.B excludes ``stat_qcod`` from comparison.
* IBSI II configuration 10.b.1 is skipped when its response map is absent
  from the bundled data.

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
