Python image workflows
======================

Start with :doc:`api_quickstart` for a runnable example. This page explains how
to prepare individual images and regions of interest (ROIs) for extraction.
For datasets organized into case folders, see :doc:`api_batch`.

Image geometry
--------------

``Image.from_nifti`` and the DICOM loaders populate voxel data and physical
geometry. NumPy arrays use ``(z, y, x)`` order; ``Image.spacing``,
``Image.origin``, and ``Image.shape`` use SimpleITK's ``(x, y, z)`` order.
``Image.shape`` is therefore not the same ordering as ``Image.array.shape``.
Supply the geometry fields when constructing an ``Image`` from an array.

Images and masks must share a physical frame and voxel grid for extraction.
The mask loaders align masks to the supplied reference image; matching array
shapes alone does not establish alignment. See :doc:`../reference/image`.

Recommended workflow
--------------------

The typical Python workflow is:

1. Load the image and mask into ``zrad.image.Image`` objects with aligned
   geometry.
2. Resample them if needed so the image and mask share the intended voxel grid.
3. Apply a configured filter if the experiment requires a filtered representation.
4. Build the intensity mask and apply any re-segmentation, then prepare
   texture and intensity-volume histogram (IVH) images in ``RoiData`` when
   those feature families are needed.
5. Run ``Radiomics.extract_features()`` on the prepared ``RoiData`` and collect
   the returned feature dictionary for storage in a table or downstream
   analysis pipeline.
6. Keep the exact preprocessing, filtering, and discretization settings next
   to the extracted features so the run remains reproducible.

Build a preprocessing pipeline
------------------------------

This CT example shows the order of processing steps from loading an image and
mask to extracting features. The parameter values illustrate the API; choose
values that match your analysis protocol. The ``RoiData`` object carries the
image and masks through preprocessing. This example uses unfiltered CT
intensities; to add filtering, insert a filter before ``IntensityMaskBuilder``
and choose discretization settings suited to the filtered intensities.

.. code-block:: python

   from zrad.image import Image
   from zrad.preprocessing import (
       ImageResampler,
       IntensityMaskBuilder,
       IVHIntensityDiscretizer,
       MaskResampler,
       Pipeline,
       Resegmenter,
       RoiCropper,
       RoiData,
       TextureDiscretizer,
   )
   from zrad.radiomics import Radiomics

   image = Image.from_nifti("path/to/image.nii.gz")
   mask = Image.from_nifti_mask("path/to/mask.nii.gz", reference=image)

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
       ("intensity_mask_builder", IntensityMaskBuilder()),
       ("resegmenter", Resegmenter(
           intensity_range=(-500, 400),
           outlier_range=3.0,
       )),
       ("ivh_discretizer", IVHIntensityDiscretizer(
           method="direct",
       )),
       ("texture_discretizer", TextureDiscretizer(
           number_of_bins=32,
       )),
       ("cropper", RoiCropper(padding=1)),
   ])

   roi_data = pipeline.apply(roi_data)

   rad = Radiomics(
       aggr_dim="3D",
       aggr_method="AVER",
   )
   features = rad.extract_features(
       roi_data=roi_data,
       families=["morphology", "intensity_statistics", "glcm", "ivh"],
   )

How pipeline steps update ROI data
----------------------------------

The optional preprocessing pipeline operates on ``RoiData``. Each step receives
the current ``RoiData`` and returns an updated ``RoiData``:

* ``ImageResampler`` updates ``roi_data.image``.
* ``MaskResampler`` updates ``roi_data.morphological_mask``.
* Concrete filters update ``roi_data.filtered_image``.
* ``IntensityMaskBuilder`` updates ``roi_data.intensity_mask`` from
  ``roi_data.filtered_image`` if present, otherwise from ``roi_data.image``.
* ``Resegmenter`` updates ``roi_data.intensity_mask``.
* ``TextureDiscretizer`` updates ``roi_data.texture_discretized_image``.
* ``IVHIntensityDiscretizer`` updates ``roi_data.ivh_intensity_image`` and
  IVH metadata.
* ``RoiCropper`` crops all present images and masks.

Steps that change the image, feature image, morphology mask, or intensity mask
clear prepared texture and IVH fields. Run re-segmentation before texture or
IVH preparation. Fixed-bin-size texture and IVH discretization reuse the lower
bound stored by ``Resegmenter`` as the discretization anchor.


Choose feature families and metadata
------------------------------------

``Radiomics.extract_features(roi_data=...)`` returns a dictionary. Use
``families`` to select feature groups, or omit it to extract the default
families available for the prepared ROI. Prepare texture discretization before
requesting histogram or texture features, and prepare IVH intensities before
requesting IVH features. IVH extraction is available through the Python API.
Use ``include_metadata=True`` to include the shortest bounding-box side length, voxel count, and
discretized-bin count.

See :doc:`extraction_concepts`, :doc:`resegmentation_guidelines`, and
:doc:`discretization_guidelines` for help choosing settings. Use :doc:`results`
to interpret feature names and metadata, and :doc:`../reference/radiomics`
for extraction options.

Resampling to an existing image grid
------------------------------------

To align one ``Image`` directly to the complete physical grid of another
``Image`` (rather than selecting a new voxel spacing), use
``Image.resample_to_target``. The operation returns a new image, uses the
moving image's minimum intensity outside its physical extent, and leaves both
input images unchanged. Save the result separately when a NIfTI file is
needed.

.. code-block:: python

   import SimpleITK as sitk

   from zrad.image import Image

   moving = Image.from_nifti("path/to/moving.nii.gz")
   target = Image.from_nifti("path/to/target.nii.gz")

   resampled = moving.resample_to_target(
       target,
       interpolator=sitk.sitkLinear,
   )
   resampled.save_as_nifti("path/to/resampled.nii.gz")

For filter examples, continue with :doc:`api_filtering`. The
:doc:`../reference/preprocessing` reference describes each pipeline step.
