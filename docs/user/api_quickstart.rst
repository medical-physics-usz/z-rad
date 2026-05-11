API Quickstart
==============

The Python API mirrors the main GUI workflows through public preprocessing,
filtering, and radiomics interfaces:

* preprocessing step classes such as ``ImageResampler`` and ``MaskResampler``
* concrete filters created via ``create_filter(...)``
* ``Radiomics``

Recommended Workflow
--------------------

The typical Python workflow is:

1. Load the image and mask into ``zrad.image.Image`` objects with aligned
   geometry.
2. Resample them so image and mask share the intended
   voxel spacing.
3. Apply a configured filter if the experiment requires a filtered representation.
4. Run ``Radiomics.extract_features()`` and collect the returned feature dictionary for storage
   in a table or downstream analysis pipeline.
   Use ``include_metadata=True`` if you also want summary fields such as
   bounding-box size, voxel count, and discretized-bin count in the result.
5. Keep the exact preprocessing, filtering, and discretization settings next
   to the extracted features so the run remains reproducible.

Minimal Example
---------------

.. code-block:: python

   from zrad.filtering import create_filter
   from zrad.image import Image
   from zrad.preprocessing import (
       ImageFilter,
       ImageResampler,
       IntensityMaskBuilder,
       MaskResampler,
       Pipeline,
       Resegmenter,
       RoiCropper,
       RoiData,
   )
   from zrad.radiomics import Radiomics

   image = Image.from_nifti("path/to/image.nii.gz")
   mask = Image.from_nifti_mask("path/to/mask.nii.gz", reference=image)

   filt = create_filter(
       filtering_method="Laplacian of Gaussian",
       padding_type="reflect",
       sigma_mm=1.0,
       cutoff=4.0,
       dimensionality="3D",
   )

   roi_data = RoiData(
       image=image,
       morphological_mask=mask,
   )

   pipeline = Pipeline([
       ("image_resampler", ImageResampler(
           resolution=(2.0, 2.0, 2.0),
           method="tricubic_spline",
           intensity_rounding="nearest_integer",
       )),
       ("mask_resampler", MaskResampler(
           resolution=(2.0, 2.0, 2.0),
           method="trilinear",
           partial_volume_threshold=0.5,
       )),
       ("filter", ImageFilter(filt)),
       ("intensity_mask_builder", IntensityMaskBuilder()),
       ("resegmenter", Resegmenter(
           intensity_range=(-500, 400),
           outlier_range=3.0,
       )),
       ("cropper", RoiCropper(padding=1)),
   ])

   roi_data = pipeline.apply(roi_data)

   rad = Radiomics(
       aggr_dim="3D",
       aggr_method="AVER",
       number_of_bins=32,
   )
   features = rad.extract_features(
       roi_data=roi_data,
   )

Data Model Expectations
-----------------------

The processing classes operate on ``zrad.image.Image`` instances. In practice,
that means your workflow needs to provide:

* voxel data as a NumPy array
* image spacing
* image origin
* image direction
* image shape

For correct feature extraction, images and masks must refer to the same spatial
frame and voxel grid.

For parameter details, combine this quickstart with the dedicated user-guide
pages and the API reference.
