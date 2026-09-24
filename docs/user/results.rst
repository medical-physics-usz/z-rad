Understanding results
=====================

GUI and Python batch extraction write ``radiomics.csv`` in the selected output
directory. Each data row represents one successfully extracted case and mask
for that run's image and settings. The single-ROI Python API returns a dictionary
of feature names and values instead.

Read the metadata
-----------------

The CSV starts with the following columns:

.. list-table:: Output metadata
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Meaning
   * - ``pat_id``
     - Case-folder name, not a patient identifier read from DICOM metadata.
   * - ``mask_id``
     - Requested structure or mask name.
   * - ``bounding_box_min``
     - Shortest side of the morphological ROI's bounding box, in voxels.
       This is a side length, not a coordinate or a length in millimetres.
   * - ``no_voxels``
     - Number of nonzero voxels in the morphological mask used for analysis.
       This is not the number of intensity voxels retained by re-segmentation.
   * - ``no_bins``
     - Number of distinct occupied grey levels in the discretized intensity
       image used for histogram or texture calculation. It can be smaller than
       the requested bin count. It does not describe IVH discretization.
       The single-ROI API reports zero when the
       selected families do not use that discretized image.

The bounding-box and voxel-count fields describe the analysis mask after
validation, including any slices removed for 2D or 2.5D texture analysis.
See :ref:`extraction-mask-requirements` for those checks.

In Python, pass ``include_metadata=True`` to ``Radiomics.extract_features``
to add the three numeric metadata fields. Case and mask identifiers are added
by the batch workflow; the single-ROI API does not infer them from file paths.

Read feature names and values
-----------------------------

For example, the :doc:`api_quickstart` returns an intensity-statistics
dictionary containing an entry approximately equal to:

.. code-block:: python

   {"stat_mean": -46.88}  # Selected entry, rounded for display; CT intensity in HU.

Feature prefixes identify families: ``stat_`` denotes intensity statistics,
``morph_`` morphology, ``cm_`` co-occurrence-matrix features, and ``ivh_``
intensity-volume histogram features. GUI and batch CSV files include six
``ivh_`` columns. Units depend on the feature and input image: the CT mean is
in HU, whereas the mean of a filtered image uses that filter's response units.
The texture name ``cm_contrast_3D_avg`` means GLCM contrast calculated with
3D neighbourhoods and averaged across directions. GLCM and GLRLM names end in
a dimension and aggregation suffix:

* ``2D``, ``2_5D``, or ``3D`` identifies the texture dimension.
* ``avg`` identifies ``AVER`` or ``DIR_MERG`` aggregation.
* ``comb`` identifies ``MERG`` or ``SLICE_MERG`` aggregation.

Other texture families use a dimension suffix without ``avg`` or ``comb``.
See :doc:`extraction_concepts` for the supported combinations. Column names
do not encode all settings, such as bin width or slice weighting; retain the
configuration with the output and use it when comparing runs.

The :doc:`../reference/radiomics` family classes provide
``get_feature_names()`` for listing their feature keys. For GLCM and GLRLM,
these are base names; extraction adds the configured aggregation suffixes.

Check completeness
------------------

Compare the actual ``(pat_id, mask_id)`` pairs with the cases and structures you
requested. A missing or rejected mask produces no feature row. Other masks in
the same case can still succeed. If every extraction is skipped, batch
extraction still creates an empty CSV file.

A non-finite feature value, such as ``NaN``, is different from a missing row:
extraction returned that feature, but a numeric value may be undefined for the
input. For example, spatial autocorrelation is undefined for constant
intensities. Check the feature's reference documentation and the input ROI
before interpreting such values.

For GUI runs, check the completion message and logs. For batch runs, inspect
both case-level errors and ``skipped_structures`` as shown in :doc:`api_batch`.
A processed case count does not guarantee that every requested structure
produced a row. See :doc:`troubleshooting` for missing results.
