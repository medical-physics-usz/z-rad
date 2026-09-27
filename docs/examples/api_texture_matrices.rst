Calculate and inspect texture matrices
======================================

All six texture calculators expose ``calculate_matrices`` independently of
feature evaluation. It accepts a discretized NumPy array or ``Image`` and returns
an immutable ``TextureMatrixCollection``. Its matrices retain grey-level labels,
original slice indices, direction offsets, voxel denominators, and merge
provenance. No discretization or minimum-ROI filtering happens in this method.

A hand-counted GLCM
-------------------

Use positive integer grey levels, with ``NaN`` outside the intensity ROI. A
separate optional ``mask`` further restricts the valid voxels. Values are never
rounded, discretized, or relabelled implicitly.

.. code-block:: python

   import numpy as np
   from zrad.radiomics.glcm import GLCM

   levels = np.array([
       [1, 1, 2],
       [2, 3, 3],
       [1, 2, 3],
   ])
   calculator = GLCM(aggr_dim="2D", aggr_method="AVER")
   directed = calculator.calculate_matrices(
       levels, directions=[(0, 1)], symmetric=False,
   )
   matrix = directed.only()
   print(matrix.axes)
   print(matrix.values)

The axes are both ``(0, 1, 2, 3)``. The zero row and column are retained so that
array indices remain grey-level labels. The directed counts are::

   [[0 0 0 0]
    [0 1 2 0]
    [0 0 0 2]
    [0 0 0 1]]

An offset ``(0, 1)`` means the right neighbour in NumPy ``(row, column)`` order.
The reversed offset ``(0, -1)`` produces the transpose. Symmetrization explicitly
adds the transpose, including doubling the diagonal:

.. code-block:: python

   symmetric = directed.symmetrize()
   print(symmetric.only().values)
   print(symmetric.only().probabilities())
   features = calculator.calculate_features_from_matrices(symmetric)
   assert np.isclose(features["cm_contrast"], 2 / 3)

The symmetric counts are::

   [[0 0 0 0]
    [0 2 2 0]
    [0 2 0 2]
    [0 0 2 2]]

``symmetric=True`` is the GLCM builder default. Asymmetric matrices can be
inspected, but feature evaluation rejects them because the current formulas
assume symmetry. ``probabilities()`` returns a separate read-only array; it does
not replace counts. Empty count matrices remain inspectable but cannot be
normalized or evaluated.

The same calculation using Image
--------------------------------

New matrix builders use ``(z, y, x)`` for 3D arrays and ``Image.array``. Directions
always have as many components as the input array has axes. They are voxel
offsets, not physical vectors. Image origin, spacing, and orientation, when
available, are recorded separately in the collection configuration.

.. code-block:: python

   from zrad.image import Image

   image = Image(array=levels[np.newaxis, ...], shape=(3, 3, 1))
   from_image = calculator.calculate_matrices(
       image, directions=[(0, 0, 1)], symmetric=False,
   )
   np.testing.assert_array_equal(from_image.only().values, matrix.values)
   assert from_image.only().slice_index == 0

For a volume in 2D or 2.5D mode, ``slice_axis=0`` selects axial array slices.
Standalone builders also accept another slice axis; offsets must lie in that
plane. Empty slices are listed in ``skipped_slices`` and remaining slices retain
their original indices. A genuinely 2D input needs no singleton dimension.

The existing family ``calculate_features(array)`` methods retain their legacy
``(x, y, z)`` convention. Prefer ``calculate_matrices`` followed by
``calculate_features_from_matrices`` for new code. High-level ``Radiomics``
continues to accept prepared ``RoiData`` in its existing convention.

All supported families
----------------------

.. code-block:: python

   from zrad.radiomics.glrlm import GLRLM
   from zrad.radiomics.glszm import GLSZM
   from zrad.radiomics.gldzm import GLDZM
   from zrad.radiomics.ngtdm import NGTDM
   from zrad.radiomics.ngldm import NGLDM

   calculators = [
       calculator,
       GLRLM(aggr_dim="2D", aggr_method="AVER"),
       GLSZM(aggr_dim="2D"),
       GLDZM(aggr_dim="2D"),
       NGTDM(aggr_dim="2D"),
       NGLDM(aggr_dim="2D"),
   ]
   for calc in calculators:
       options = {"morphological_mask": np.ones_like(levels)} if isinstance(calc, GLDZM) else {}
       matrices = calc.calculate_matrices(levels, **options)
       for item in matrices:
           print(item.family, item.direction, item.axis_names, item.values)

GLCM and GLRLM support explicit directions. GLCM accepts arbitrary nonzero
integer offsets, including larger distances; GLRLM accepts adjacent offsets
with components in ``{-1, 0, 1}``. Defaults comprise four unique directions in
2D and thirteen in 3D. Duplicate offsets, and redundant opposite directions for
symmetric GLCM or GLRLM, are rejected.

Other families use neighbourhoods or connected zones rather than directional
matrices. Unsupported controls raise errors. Their current settings are recorded
in ``configuration``: 8/26-connected zones, taxicab GLDZM distance, Chebyshev
radius-one neighbourhoods, and zero NGLDM dependence tolerance. These settings
are currently descriptive metadata, not configurable alternatives.

GLDZM requires a separate morphological mask. This mask may extend beyond the
intensity ROI after resegmentation and must contain every valid intensity voxel.
When both inputs are Images, supplied geometry must agree. No implicit
resampling occurs.

NGTDM's two columns are different statistics, not homogeneous counts:

.. code-block:: python

   ngtdm = NGTDM(aggr_dim="2D").calculate_matrices(levels).only()
   print(ngtdm.n_i)  # Counts of voxels having at least one valid neighbour
   print(ngtdm.s_i)  # Accumulated absolute differences from neighbour means
   print(ngtdm.p_i)  # n_i / sum(n_i)

Run length, zone size, distance and dependence-size columns are labelled starting
at one. NGLDM dependence size includes the centre voxel.

Merging versus averaging
------------------------

Builders return unmerged matrices, independently of the calculator's configured
feature aggregation method. Collections support selection and explicit merging:

.. code-block:: python

   volume = np.stack([levels, np.flip(levels, axis=0)])
   matrices = calculator.calculate_matrices(volume)
   first_slice = matrices.select(slice_index=0)
   rightward = matrices.select(direction=(0, 0, 1))
   per_slice = matrices.merge(over="directions")
   per_direction = matrices.merge(over="slices")
   combined = matrices.merge(over=("slices", "directions")).only()
   print(combined.source_ids)

Slice-wise matrices can be reused with either a 2D or 2.5D calculator without
rebuilding them. A 3D calculator requires matrices constructed in 3D.

Merging sums counts before normalization. Feature averaging evaluates each matrix
first, then reduces feature values. These operations generally differ. Calling
``inspect_matrices`` applies the calculator's aggregation and exposes the exact
inputs and reductions:

.. code-block:: python

   trace = calculator.inspect_matrices(matrices)
   print(trace.features)
   print(trace.aggregation)
   for item, values in zip(trace.feature_inputs, trace.per_matrix_features):
       print(item.source_ids, item.values, values)

For merged GLRLM matrices, ``voxel_count`` includes direction multiplicity and is
the feature denominator; ``roi_voxel_count`` counts each slice only once.
``feature_inputs`` contain normalized probabilities for GLCM and the original
matrix statistics for the other families. Merging normalized matrices is rejected.

Inspect an actual extraction
----------------------------

.. code-block:: python

   from zrad.preprocessing import IntensityMaskBuilder, RoiData, TextureDiscretizer
   from zrad.radiomics import Radiomics

   volume = np.stack([levels, np.flip(levels, axis=0), levels])
   image = Image(array=volume, shape=(3, 3, 3))
   mask = Image(array=np.ones_like(volume), shape=(3, 3, 3))
   roi = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))
   roi = TextureDiscretizer(number_of_bins=3).apply(roi)
   extractor = Radiomics(aggr_dim="3D", aggr_method="MERG")
   families = ["glcm", "glrlm", "glszm", "gldzm", "ngtdm", "ngldm"]

   # No feature evaluation: returns unmerged matrices for all six families.
   all_matrices = extractor.calculate_texture_matrices(roi)

   result = extractor.extract_with_details(roi, families=families)
   print(result.features)
   print(result.texture["glcm"].raw[0].values)
   print(result.texture["glcm"].feature_inputs.only().values)
   print(result.excluded_slices)

``extract_with_details`` preserves directed GLCM counts in ``raw`` and the actual
symmetric, merged, normalized inputs in ``feature_inputs``. Effective prepared
arrays are available as ``discretized_image`` and ``morphological_mask``.
``excluded_slices`` identifies originally nonempty slices removed by ROI validation.
The standalone family builders intentionally do not enforce those geometry checks.

Both extraction methods accept per-family construction options, for example
``texture_options={"glcm": {"directions": [(0, 0, 1)]}}``. High-level extraction
uses axial slice validation and consequently restricts ``slice_axis`` to zero.
Only ``calculate_texture_matrices`` allows ``symmetric=False`` at this level.
Custom direction results retain existing feature names; record the trace's
configuration when comparing results.

Ordinary ``extract_features`` still returns a flat dictionary and does not retain
matrices after returning. Inspection explicitly retains all matrices and a copy
of the effective ROI arrays, so it requires more memory. ``retain_matrices="all"``
is currently the only retention mode for ``extract_with_details``.
