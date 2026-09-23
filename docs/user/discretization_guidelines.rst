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

IVH-specific discretization
---------------------------

IVH features describe the fraction of ROI voxels with intensity at least a
selected level. They need a fine, ordered intensity axis, so choose their
settings separately from texture discretization. A coarse texture setting
such as ``32`` bins should not automatically be reused for IVH.

IVH extraction is available through the Python API. Prepare its intensities
with ``IVHIntensityDiscretizer`` before requesting IVH features.

Choose an IVH strategy
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Intensity type
     - Strategy
     - Example arguments
   * - Discrete calibrated values, such as integer CT HU
     - Use retained intensities directly with a step of ``1``.
     - ``method="direct"``
   * - Continuous calibrated values, such as PET SUV
     - Use a small fixed bin width and a consistent lower origin.
     - ``method="fixed_bin_size"``,
       ``bin_size=0.1``
   * - Arbitrary units, such as raw MRI
     - Use a fine fixed bin number.
     - ``method="fixed_bin_number"``,
       ``number_of_bins=1000``

Fixed-bin-size IVH preparation also requires a preceding re-segmentation range.
Its intensity axis uses bin-centre values: with a lower bound of ``0`` SUV and
a bin width of ``0.1`` SUV, the centres start at ``0.05, 0.15, 0.25, ...`` SUV.
Fixed-bin-number IVH preparation uses the discretized range, such as
``[1, 1000]`` for ``1000`` bins.

Interpret the IVH range
~~~~~~~~~~~~~~~~~~~~~~~

Two fractions describe the curve:

* **Volume fraction:** the fraction of ROI voxels with intensity at least the
  selected level.
* **Intensity fraction:** the position of that level within the full IVH
  intensity range.

``V10`` and ``V90`` are volume fractions at ``10%`` and ``90%`` of the intensity
range. ``I10`` and ``I90`` are intensities corresponding to ``10%`` and ``90%``
volume fractions. Differences such as ``V10 - V90`` and ``I10 - I90`` summarize
the separation between these points.

The re-segmentation range helps define the IVH range; it is not a separate
IVH-only voxel-selection step. For example, direct CT preparation with a range
of ``[-500, 400]`` HU uses those bounds and an interval of ``1`` HU.

``V10`` means the volume fraction above the level at ``10%`` of the IVH range,
not above an intensity value of ``10``. For illustration, before any bin-centre
adjustment:

* a range of ``[0, 20]`` SUV places the ``10%`` level at ``2`` SUV
* a range of ``[2, 12]`` SUV places it at ``3`` SUV

The same lesion can therefore have different ``V10`` and ``V90`` values when
the IVH range changes. Keep the range definition consistent across cases and
report whether it came from re-segmentation bounds or observed ROI values.

What to report
--------------

Record these settings with the extracted features:

* the image modality, intensity units, and any filtering or standardization
* the texture/histogram discretization method and bin count or width
* the lower bin origin for fixed bin size
* for IVH, the separate method, bin count or width, intensity range, and the
  source of that range
* the reason for choosing these settings
