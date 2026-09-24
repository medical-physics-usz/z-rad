Feature extraction concepts
===========================

Feature extraction combines an image with a region-of-interest (ROI) mask.
The morphological mask defines its shape; the intensity mask contains the
voxel values used for intensity and texture features. The guides for
:doc:`resegmentation_guidelines` and :doc:`discretization_guidelines` explain
how to prepare that intensity population.

The concepts below apply to both the GUI and Python API. See :doc:`results`
for feature names and output metadata.

Choose texture aggregation
--------------------------

The dimension controls whether texture neighbourhoods stay within slices or
extend through the volume:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Dimension
     - Texture calculation
     - How slices are combined
   * - ``2D``
     - Calculate texture within each slice.
     - Combine the resulting feature values across slices.
   * - ``2.5D``
     - Calculate texture within each slice.
     - Merge matrices across slices before calculating features.
   * - ``3D``
     - Calculate texture across the volume, including between slices.
     - Use the volume's texture matrices.

For directional features such as the grey level co-occurrence matrix (GLCM)
and grey level run length matrix (GLRLM), ``averaged``
calculates features for each direction and averages the values; ``merged``
combines matrices before calculating features. ``2D, slice-merged`` merges
directions within each slice, while ``2.5D, direction-merged`` merges slices
for each direction. Other texture families use their own dimension-specific
aggregation rules.

For 2D extraction, ``Slice Averaging`` offers ``Mean``, ``Weighted Mean``
(weighted by ROI voxel count), and ``Median``. Keep the dimension, aggregation,
and slice-averaging settings consistent across cases and record them with the
results.

GUI and Python aggregation settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Python column names below are arguments to ``Radiomics``. Batch extraction
uses ``aggregation_dimension`` and ``aggregation_method`` for the same values.

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - GUI selection
     - ``aggr_dim``
     - ``aggr_method``
   * - ``2D, averaged``
     - ``"2D"``
     - ``"AVER"``
   * - ``2D, slice-merged``
     - ``"2D"``
     - ``"SLICE_MERG"``
   * - ``2.5D, direction-merged``
     - ``"2.5D"``
     - ``"DIR_MERG"``
   * - ``2.5D, merged``
     - ``"2.5D"``
     - ``"MERG"``
   * - ``3D, averaged``
     - ``"3D"``
     - ``"AVER"``
   * - ``3D, merged``
     - ``"3D"``
     - ``"MERG"``

For 2D slice averaging, the Python defaults select the mean. Set
``slice_weighting=True`` for the voxel-weighted mean or ``slice_median=True``
for the median; these options are mutually exclusive.

.. _extraction-feature-families:

Feature families
----------------

The available families depend on the image dimensionality and prepared ROI
data. GUI and batch extraction select the supported families automatically;
in the single-ROI Python API, use ``families`` or ``features`` to select them:

* morphology
* local intensity
* intensity statistics
* intensity histogram
* intensity-volume histogram (IVH)
* grey level co-occurrence matrix (GLCM)
* grey level run length matrix (GLRLM)
* grey level size zone matrix (GLSZM)
* grey level distance zone matrix (GLDZM)
* neighbourhood grey tone difference matrix (NGTDM)
* neighbouring grey level dependence matrix (NGLDM)

For the preparation required by each family, see :doc:`api_workflows` and
:doc:`discretization_guidelines`.
Morphology requires a 3D ROI. See :doc:`../reference/radiomics` for the full API.

.. _extraction-mask-requirements:

ROI size requirements
---------------------

For volumetric images, Z-Rad validates the morphological mask for the requested
feature families. Texture analysis uses the selected aggregation dimension:

* For ``3D`` extraction, the mask must contain at least ``27`` valid voxels,
  and the bounding box of the nonzero mask region must be at least ``3``
  voxels wide in every dimension.
* For ``2D`` and ``2.5D`` extraction, Z-Rad validates each slice
  independently. A slice is discarded if it contains fewer than ``9`` valid
  voxels or if its nonzero bounding box is smaller than ``3`` voxels in either
  in-plane dimension.
* If no slice satisfies these ``2D`` or ``2.5D`` requirements, radiomics
  extraction is aborted for that mask.

Morphology and other families that use a volumetric ROI retain their 3D
validation rules even when texture aggregation is slice-wise. Single-slice
images follow a separate 2D extraction path. Re-segmentation can further reduce
the voxels available to intensity-based features; an empty intensity ROI cannot
be used for those calculations. See :doc:`troubleshooting` for rejected masks.
