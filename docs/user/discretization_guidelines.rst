Discretization guidelines
=========================

Discretization groups retained voxel intensities into grey-level bins before
histogram and texture feature calculation. It reduces the number of intensity
levels and groups small intensity differences together. Re-segmentation selects
which voxels are included; discretization assigns their intensities to bins.

Intensity histogram and all six texture families listed in :doc:`extraction_concepts`
require discretization. Morphology, local intensity, and intensity statistics
do not. Intensity-volume histogram (IVH) features use a separate intensity
preparation step, described below.

Texture and histogram grey levels start at ``1`` because some feature
definitions do not allow a grey level of ``0``.

Choose a texture discretization method
--------------------------------------

Use either fixed bin number or fixed bin size. The choice depends on whether
absolute intensity values are comparable across your images.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - How it works
     - Main tradeoff
   * - Fixed bin number
     - Divides each ROI's observed intensity range into a selected number of
       bins, such as ``32``.
     - Accommodates arbitrary intensity scales, but the same bin can represent
       different intensity ranges in different ROIs.
   * - Fixed bin size
     - Uses a constant intensity width, such as ``25`` Hounsfield units (HU)
       or ``0.5`` standardized uptake value (SUV), from a chosen lower origin.
     - Preserves a shared intensity scale when the width and origin are
       consistent, but requires meaningful units and bounds.

For CT and PET, fixed bin size can preserve the meaning of calibrated
intensities across cases. Fixed bin number is also available, but normalizes
each ROI's intensity range. For raw MRI and many filtered images, fixed bin
number is usually more appropriate because the intensities have arbitrary or
transformed units. Use fixed bin size for those images only when the scale and
origin have a clear, reproducible meaning.

These numerical examples illustrate the methods; choose the actual parameters
according to your analysis protocol.

Set the bin origin
------------------

Z-Rad requires an intensity range for fixed-bin-size texture discretization.
The lower bound becomes the bin origin. For example, a CT range starting at
``-500`` HU anchors the bins there. Keep the width and origin consistent across
cases if their grey levels are to represent the same intensity intervals.

In the GUI, select ``Bin Size`` and configure ``Intensity Range``. In Python,
run ``Resegmenter(intensity_range=...)`` before
``TextureDiscretizer(bin_size=...)``. If you use fixed bin number, select
``Number of Bins`` in the GUI or pass ``number_of_bins`` to
``TextureDiscretizer``; a fixed lower bound is not required.

For the order of processing steps, see :doc:`api_workflows` and
:doc:`resegmentation_guidelines`.

.. _ivh-discretization:

IVH-specific discretization
---------------------------

IVH features describe the fraction of ROI voxels with intensity at least a
selected level. Their intensity preparation is independent of the texture and
histogram bins set by the GUI's ``Discretization`` control.

GUI and batch extraction use the defaults below for unfiltered images and
``1000`` fixed-number bins for filtered images, regardless of modality.
In the single-ROI API, use ``IVHIntensityDiscretizer`` with the listed arguments.
Batch callers can override the defaults with ``ivh_method`` and, where needed,
``ivh_bin_size`` or ``ivh_number_of_bins``.

.. list-table:: Automatic IVH settings
   :header-rows: 1
   :widths: 20 40 40

   * - Selected modality
     - Automatic preparation
     - Equivalent single-ROI Python arguments
   * - CT
     - Use retained intensities directly with a step of ``1`` (HU for
       unfiltered CT).
     - ``method="direct"``
   * - PET and RTDOSE
     - Use fixed-width bins of ``0.1`` (SUV for PET or Gy for physical dose).
     - ``method="fixed_bin_size"``, ``bin_size=0.1``
   * - MRI, MG, and US
     - Divide the retained intensity range into ``1000`` bins.
     - ``method="fixed_bin_number"``, ``number_of_bins=1000``

Range and interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

For unfiltered images, re-segmentation bounds help define the direct or
fixed-bin-size IVH range. For fixed bin size, the lower bound anchors the bins;
GUI and batch extraction use the observed ROI minimum when no range is set.
The single-ROI API requires a preceding re-segmentation range for fixed bin size.
Fixed-width bins are represented by their centres. Fixed-bin-number preparation
uses the discretized range, such as ``[1, 1000]`` for ``1000`` bins.

For filtered images, GUI and batch extraction use the retained filtered
intensities to define the IVH range, including with custom batch settings.
Re-segmentation still selects voxels using the original image, but its bounds
do not define the filtered IVH axis.

``V10`` and ``V90`` are volume fractions at ``10%`` and ``90%`` of the intensity
range; ``I10`` and ``I90`` are intensities corresponding to ``10%`` and ``90%``
volume fractions. Keep the range definition consistent across cases, since it
affects these features. In particular, RTDOSE ``V10`` refers to a relative
intensity threshold, not the clinical dose-volume-histogram ``V10 Gy``.

What to report
--------------

Record these settings with the extracted features:

* the image modality, intensity units, and any filtering or standardization
* the texture/histogram discretization method and bin count or width
* the lower bin origin for fixed bin size
* for IVH, the separate method, bin count or width, intensity range, and the
  source of that range
* the reason for choosing these settings
