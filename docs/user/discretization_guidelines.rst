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
selected level. They need a fine, ordered intensity axis, so choose their
settings separately from texture discretization. A coarse texture setting
such as ``32`` bins should not automatically be reused for IVH.

GUI extraction from an unfiltered image and Python batch extraction prepare IVH
intensities from the selected imaging modality unless the batch API receives
custom IVH settings. For a filtered image, the GUI uses ``1000`` fixed-number
bins regardless of modality.
In the single-ROI Python API, prepare them with ``IVHIntensityDiscretizer``
before requesting IVH features.

Automatic settings and Python customization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following defaults apply to unfiltered GUI extraction and to
``BatchRadiomicsExtractor`` unless its IVH settings are overridden. The listed
Python arguments reproduce them in the single-ROI API. Batch callers can
override them with ``ivh_method`` and the corresponding ``ivh_bin_size`` or
``ivh_number_of_bins``.

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

The GUI's ``Discretization`` control sets the texture bins, which are also
used by the ordinary intensity-histogram family. It never supplies the IVH
bin width or count. Use the batch or single-ROI Python API for custom IVH
settings.

For filtered-image extraction in the GUI, IVH bins span the retained filtered
intensities in each ROI. A GUI re-segmentation range selects voxels using the
original image, but its bounds do not define the filtered IVH intensity axis.

For fixed-bin-size IVH, a GUI or batch re-segmentation range supplies the lower
bin origin and, if finite, the upper IVH bound. If no range is supplied, GUI
and batch extraction use each ROI's observed minimum as the lower origin. In the
single-ROI Python API, ``IVHIntensityDiscretizer`` requires a preceding
re-segmentation range for fixed bin size.

The fixed-bin-size intensity axis uses bin-centre values: with a lower bound
of ``0`` SUV and a bin width of ``0.1`` SUV, the centres start at
``0.05, 0.15, 0.25, ...`` SUV.
Fixed-bin-number IVH preparation uses the discretized range, such as
``[1, 1000]`` for ``1000`` bins.

The GUI's ``0.1`` Gy RTDOSE width is a starting setting, not a universal dose
resolution. If the endpoint needs another width, use the batch or single-ROI
Python API and keep the chosen dose range and width consistent across cases.
Mammography and B-mode ultrasound intensities depend on image processing and
acquisition settings; keep those settings consistent and assess feature
repeatability. A calibrated ultrasound map may need a strategy suited to its
actual intensity units rather than the GUI's modality-based setting.

Interpret the IVH range
~~~~~~~~~~~~~~~~~~~~~~~

Two fractions describe the curve:

* **Volume fraction:** the fraction of ROI voxels with intensity at least the
  selected level.
* **Intensity fraction:** the position of that level within the full IVH
  intensity range.

For a single-slice image, the reported fraction is the fraction of ROI pixels.

``V10`` and ``V90`` are volume fractions at ``10%`` and ``90%`` of the intensity
range. ``I10`` and ``I90`` are intensities corresponding to ``10%`` and ``90%``
volume fractions. Differences such as ``V10 - V90`` and ``I10 - I90`` summarize
the separation between these points.

For direct and fixed-bin-size IVH, the re-segmentation range helps define the
IVH range; it is not a separate IVH-only voxel-selection step. For example,
direct CT preparation with a range of ``[-500, 400]`` HU uses those bounds and
an interval of ``1`` HU. Fixed-bin-number IVH instead uses the observed range
of retained intensities after discretization.

``V10`` means the volume fraction above the level at ``10%`` of the IVH range,
not above an intensity value of ``10``. For illustration, before any bin-centre
adjustment:

* a range of ``[0, 20]`` SUV places the ``10%`` level at ``2`` SUV
* a range of ``[2, 12]`` SUV places it at ``3`` SUV

The same lesion can therefore have different ``V10`` and ``V90`` values when
the IVH range changes. Keep the range definition consistent across cases and
report whether it came from re-segmentation bounds or observed ROI values.
For RTDOSE, IVH ``V10`` is not the clinical dose-volume-histogram ``V10 Gy``.

What to report
--------------

Record these settings with the extracted features:

* the image modality, intensity units, and any filtering or standardization
* the texture/histogram discretization method and bin count or width
* the lower bin origin for fixed bin size
* for IVH, the separate method, bin count or width, intensity range, and the
  source of that range
* the reason for choosing these settings
