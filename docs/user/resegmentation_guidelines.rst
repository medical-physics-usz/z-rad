Re-segmentation guidelines
==========================

Re-segmentation selects which voxels inside a region of interest (ROI)
contribute to intensity and texture analysis. Apply it after resampling and
before discretization. It can exclude voxels outside a selected intensity
range or remove intensity outliers.

Which mask changes?
-------------------

Z-Rad follows the Image Biomarker Standardisation Initiative (IBSI) distinction
between two masks:

.. list-table::
   :header-rows: 1

   * - Mask
     - Purpose
     - Effect of re-segmentation
   * - Morphological mask
     - Defines the ROI shape used for morphology.
     - Unchanged.
   * - Intensity mask
     - Selects voxels for intensity and texture analysis.
     - Voxels can be removed, leaving holes or disconnected regions.

Intensity statistics, intensity histogram, intensity-volume histogram (IVH),
and most texture features use the intensity mask. Grey level distance zone
matrix (GLDZM) features use both masks: the morphological mask defines distances
to the ROI boundary. See :doc:`extraction_concepts` for the feature families.

Range re-segmentation
---------------------

Range re-segmentation keeps voxels whose image intensities fall within the
selected lower and upper bounds, including both endpoints. For example, a
CT protocol might use ``[-50, 150]`` Hounsfield units (HU), while a PET protocol
might use a lower standardized uptake value (SUV) threshold such as
``[3, infinity)``. These are examples, not default settings for every study.

Use a range that is meaningful for the modality and analysis:

* CT and PET have calibrated units, so choose and report a range appropriate
  to the tissue and study protocol.
* Raw MRI intensities depend on acquisition and scanner settings. Use a common
  range only when the intensity scale has been standardized and the range can
  be justified.

In Z-Rad, range selection uses the original image supplied for extraction,
even when a filtered image supplies the intensities for feature calculation.
This is one reason filtered-image extraction also requires the original image.

Outlier removal
---------------

Outlier removal uses the mean and standard deviation of the valid intensity
values inside the ROI. For example, a setting of ``3`` keeps values within
``mean - 3 * standard deviation`` and ``mean + 3 * standard deviation``.
The accepted interval therefore depends on each ROI's intensity distribution.
For filtered-image extraction, these statistics use the filtered intensities.

When both methods are enabled, Z-Rad applies the range first and calculates
outlier statistics from the remaining voxels:

1. Keep voxels whose original image intensities fall within the selected range,
   for example ``[-50, 150]`` HU.
2. Calculate the mean and standard deviation of the retained intensity-mask
   values.
3. Remove values outside the selected standard-deviation interval.

The final intensity mask contains only voxels accepted by both rules. Check
that enough voxels remain for extraction; see :doc:`extraction_concepts` for mask-size
requirements.

Configure re-segmentation
-------------------------

In the GUI's ``Radiomics`` tab, use ``Intensity Range`` and ``Outlier Removal``.
In Python, apply ``Resegmenter`` after ``IntensityMaskBuilder`` and before
``TextureDiscretizer`` or ``IVHIntensityDiscretizer``. Re-segmentation clears
previously prepared texture and IVH images because the intensity population
has changed. See :doc:`api_workflows` for a complete pipeline.

A configured intensity range is also used by discretization: its lower bound
anchors fixed-bin-size bins, and its bounds help define the IVH intensity
range. See :doc:`discretization_guidelines`.

What to report
--------------

Record these settings with the extracted features:

* whether you used range re-segmentation, outlier removal, both, or neither
* the intensity bounds and units, including whether the upper bound was finite
* the standard-deviation multiplier for outlier removal
* that re-segmentation followed resampling and, when both methods were used,
  range selection preceded outlier statistics
* whether feature intensities came from the original or a filtered image
* that re-segmentation changed the intensity mask while retaining the
  morphological mask
