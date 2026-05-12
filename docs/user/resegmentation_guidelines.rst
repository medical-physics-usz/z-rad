Re-segmentation Guidelines
==========================

Overview
--------

Re-segmentation changes which voxels inside the region of interest are used
for intensity and texture analysis. It is applied after interpolation.

The goal is to remove voxels that should not contribute to intensity-based
features. Common examples are:

* air or bone voxels inside a tumour mask on CT
* low-activity voxels on PET
* extreme intensity outliers
* voxels outside a meaningful physical or biological range

Morphological Mask and Intensity Mask
-------------------------------------

Z-Rad follows the IBSI distinction between two masks:

.. list-table::
   :header-rows: 1

   * - Mask
     - Meaning
     - Changed by re-segmentation?
   * - Morphological mask
     - The original anatomical or geometrical ROI shape.
     - No
   * - Intensity mask
     - The voxels used for intensity and texture feature calculation.
     - Yes

This distinction is important. Re-segmentation can remove internal voxels. It
can also split the ROI into disconnected parts. The original ROI shape is still
kept for morphology. Only the intensity mask is updated.

Effect on Feature Families
--------------------------

Most feature families use the intensity mask after re-segmentation. There are
some important exceptions:

* morphological features use the morphological mask
* GLDZM features use both the morphological mask and the intensity mask
* intensity statistics, histogram features, IVH features, and most texture
  matrices use the intensity mask

This means that the geometry used for morphology can differ from the voxels
used for intensity and texture features.

Range Re-segmentation
---------------------

Range re-segmentation keeps only voxels inside a selected intensity range.
Voxels outside the range are removed from the intensity mask.

Examples:

* CT: keep ``[-50, 150]`` HU to exclude air and bone, if this range is suitable
  for the application
* PET: keep voxels above a SUV threshold, for example ``[3, infinity)``
* CT or PET: use ranges with physical meaning when the image units are
  calibrated

Range re-segmentation is most useful when the intensity scale has physical
meaning. This is usually true for HU in CT and SUV in PET.

For arbitrary units, such as raw MRI intensities or many filtered images, there
is no general range that fits every dataset. A range should only be used if the
intensity scale has been standardized and the range can be justified.

Intensity Outlier Filtering
---------------------------

Outlier filtering removes extreme values based on the intensity distribution
inside the ROI. A common rule is ``[mu - 3 sigma, mu + 3 sigma]``.

Here, ``mu`` is the mean intensity inside the ROI, and ``sigma`` is the
standard deviation. Voxels outside this interval are removed from the intensity
mask.

When range re-segmentation and outlier filtering are both configured, Z-Rad
applies the range first, then calculates ``mu`` and ``sigma`` from the
remaining valid intensity-mask voxels.

In the Python pipeline, run re-segmentation before ``TextureDiscretizer`` or
``IVHIntensityPreparer``. Re-segmentation changes the valid intensity
population and therefore clears any prepared texture or IVH images.

This method is data-driven. The accepted range can differ between patients or
lesions. It is not the same as using a fixed physical range.

Combining Methods
-----------------

Range re-segmentation and outlier filtering can be combined. In that case, the
final intensity mask contains only voxels accepted by all selected rules.

Example:

* range re-segmentation keeps ``[-50, 150]`` HU
* outlier filtering calculates ``mu`` and ``sigma`` from those retained voxels
  and keeps ``[mu - 3 sigma, mu + 3 sigma]``
* the final intensity mask keeps only voxels that satisfy both rules

Practical Guidance
------------------

Use re-segmentation only when it matches the image modality and the analysis
goal.

For CT:

* range re-segmentation can be useful because HU values have physical meaning
* choose the range according to the tissue and disease being studied
* report the exact HU range

For PET:

* range re-segmentation can be useful because SUV values have physical meaning
* thresholds such as ``[3, infinity)`` may be used when justified by the study
  protocol
* report the exact SUV range

For raw MRI:

* global range re-segmentation is usually not meaningful
* MRI intensities often depend on scanner settings and acquisition parameters
* use a range only if intensities have been standardized and the range is
  justified

For filtered images:

* intensities often have transformed or arbitrary units
* avoid fixed physical ranges unless the filtered scale has a clear meaning
* outlier filtering may be more suitable, but it should still be reported

What to Report
--------------

Report the re-segmentation settings so the analysis can be reproduced.

Include:

* whether re-segmentation was used
* the method: range, outlier filtering, or both
* the exact range and units
* whether the range is closed, such as ``[a, b]``, or half-open, such as
  ``[a, infinity)``
* that re-segmentation was applied after interpolation
* if outlier filtering was combined with range re-segmentation, that outlier
  statistics were calculated after range re-segmentation
* whether the morphological mask and intensity mask may differ

Main Point
----------

Re-segmentation decides which voxels inside the ROI are included in
intensity-based analysis. It does not redefine the original ROI shape used for
morphological features.
