Resampling guidelines
=====================

Resampling places an image and its region-of-interest (ROI) mask on a chosen
voxel grid before filtering or feature extraction. It changes the sampled image
intensities and can change which voxels belong to the ROI. Choose a target
spacing and interpolation methods for the study, then use them consistently
across cases.

When to resample
----------------

Resample when images or masks need a common analysis grid or when the protocol
calls for a particular voxel spacing. If the source grid already meets the
protocol, resampling is optional. Align the mask with its image in physical
space; matching array shapes alone does not establish alignment. The GUI and
batch workflows process an image and its masks together. For an existing target
image grid in Python, see :ref:`resampling-existing-grid`.

Choose the target spacing
-------------------------

``Resample Resolution`` sets the target spacing in millimetres. Choose it with
the source resolution and analysis protocol in mind. A finer grid creates more
voxels but does not recover detail absent from the source image. A coarser grid
can remove small image or ROI details. Check the resulting image and mask before
extraction.

The GUI and batch preprocessing offer two dimension choices:

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Resample dimension
     - Target spacing
     - Typical reason to choose it
   * - ``2D``
     - Apply the requested spacing in-plane while preserving the original
       through-plane (slice) spacing.
     - Choose when through-plane resolution is much poorer than in-plane
       resolution, or when slices are widely spaced. This avoids creating
       interpolated slices between regions that were not directly sampled.
   * - ``3D``
     - Apply the requested spacing on all three axes to create an isotropic
       voxel grid.
     - Choose when the source data sample all three axes adequately and the
       protocol calls for common volumetric spacing, such as for 3D texture
       analysis or filtering. This makes spatial relationships more comparable
       across directions and datasets.

This resampling dimension is separate from the ``2D``, ``2.5D``, or ``3D``
texture aggregation setting described in :doc:`extraction_concepts`. One
controls the voxel grid; the other controls how texture features are
calculated and combined.

In Python API, ``ImageResampler`` and ``MaskResampler`` accept a single spacing or
an ``(x, y, z)`` spacing tuple. To reproduce 2D preprocessing, supply the target
in-plane spacing and the original slice spacing as the third tuple value. Use
the same target grid for the image and mask.

Choose image interpolation
--------------------------

Image interpolation estimates intensities at positions on the new grid. The
GUI and resampler classes support these methods:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Method
     - Effect
   * - Nearest neighbour (``NN``)
     - Takes the nearest source voxel value; does not create intermediate
       intensities, but may give a block-like appearance.
   * - Linear
     - Blends nearby voxel values; a straightforward choice for continuous
       image intensities.
   * - B-spline (``BSpline``)
     - Produces a smoother interpolated image; inspect edges and intensity
       ranges after resampling.
   * - Gaussian
     - Smooths values while interpolating; may reduce fine detail.

Select a method that fits the source image and protocol. Keep the choice
consistent across comparable cases. In batch preprocessing, resampled CT
intensities are rounded to integers and saved as signed 16-bit values; MR and
PET intensities remain floating point. The Python ``ImageResampler`` keeps
floating-point values unless ``intensity_rounding="nearest_integer"`` is set.

Choose mask interpolation
-------------------------

Masks need a separate interpolation choice because their output must remain
binary. ``NN`` assigns the nearest source mask value. Linear, B-spline, and
Gaussian interpolation can produce intermediate values; Z-Rad then includes
voxels whose interpolated value is greater than or equal to the configured
threshold and excludes the rest. The default threshold is ``0.5``. Changing
the method or threshold can change the ROI boundary and voxel count.

Inspect the resampled mask overlaid on the resampled image, especially thin
structures and edges. Check for an empty or unexpectedly changed ROI before
feature extraction. See :doc:`troubleshooting` for ROI size requirements and
missing results.

Configure and report the settings
---------------------------------

In the GUI, use the ``Preprocessing`` tab's resolution, dimension, image
interpolation, mask interpolation, and mask threshold controls; see
:doc:`preprocessing`. In Python, use ``ImageResampler`` and ``MaskResampler`` in
the order shown in :doc:`api_workflows`, or set the corresponding options in
:doc:`api_batch`.

Record whether resampling was performed, the target spacing, the 2D or 3D
choice, both interpolation methods, and the mask threshold when applicable.
Resample before re-segmentation and discretization; see
:doc:`resegmentation_guidelines` and :doc:`discretization_guidelines`.
