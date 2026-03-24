Preprocessing
=============

Overview
--------

Preprocessing covers resampling and image or mask normalization into the target
voxel grid expected by the downstream filtering and radiomics steps.

Main Parameters
---------------

The ``Preprocessing`` class requires:

* ``input_imaging_modality``: one of ``CT``, ``MR``, or ``PT``
* ``resample_resolution``: positive scalar output spacing
* ``resample_dimension``: ``2D`` or ``3D``
* ``interpolation_method``: ``Linear``, ``NN``, ``BSpline``, or ``Gaussian``
* ``interpolation_threshold``: mask threshold applied after mask resampling

Behavior
--------

Z-Rad computes:

* the output spacing from the requested resolution and dimensionality
* a centered output origin that preserves spatial alignment
* the output voxel grid size from input shape and spacing

Image and mask resampling share the same spatial transform, then diverge in the
final output handling:

* images are cast according to imaging modality
* masks are thresholded back to a binary label map

Notes
-----

* ``2D`` resampling preserves the original slice spacing in the third axis.
* CT images are rounded and stored as signed 16-bit integers after resampling.
* MR and PT images remain floating-point volumes.
