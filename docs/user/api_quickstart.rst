API Quickstart
==============

The Python API mirrors the main GUI workflows through public preprocessing,
filtering, and radiomics interfaces:

* preprocessing step classes such as ``ImageResampler`` and ``MaskResampler``
* concrete filters such as ``Mean``, ``LoG``, ``Gabor``, ``Laws``, and wavelets
* ``Radiomics``

Recommended Workflow
--------------------

The typical Python workflow is:

1. Load the image and mask into ``zrad.image.Image`` objects with aligned
   geometry.
2. Resample them so image and mask share the intended
   voxel spacing.
3. Apply a configured filter if the experiment requires a filtered representation.
4. Prepare texture and IVH images in ``RoiData`` when those feature families
   are needed.
5. Run ``Radiomics.extract_features()`` on the prepared ``RoiData`` and collect
   the returned feature dictionary for storage in a table or downstream
   analysis pipeline.
   Use ``include_metadata=True`` if you also want summary fields such as
   bounding-box size, voxel count, and discretized-bin count in the result.
6. Keep the exact preprocessing, filtering, and discretization settings next
   to the extracted features so the run remains reproducible.

Canonical Full Pipeline Example
-------------------------------

This example shows the recommended explicit workflow from loading an image and
mask to extracting features. The intermediate ``RoiData`` object carries the
current image, optional filtered image, morphological mask, and intensity mask
through preprocessing.

.. code-block:: python

   from zrad.filtering import Mean
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

   image_filter = Mean(
       padding_type="reflect",
       support=3,
       dimensionality="3D",
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
       ("filter", image_filter),
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
       families=["morphology", "intensity_statistics", "glcm"],
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

Batch APIs
----------

Use ``zrad.batch`` when you want to process a folder cohort and write results
to disk. The lower-level modules remain the single-image or single-ROI APIs:

* ``BatchPreprocessor`` writes preprocessed images and masks.
* ``BatchFilter`` writes filtered images.
* ``BatchRadiomicsExtractor`` writes one ``radiomics.csv`` file.

All batch workflows return ``BatchResult``:

.. code-block:: python

   result = batch.run()

   print(result.processed_count, result.skipped_count, result.failed_count)
   for case in result.errors:
       print(case.case_name, case.error)

Batch Preprocessing
-------------------

Use ``zrad.batch.BatchPreprocessor`` when you want to run the preprocessing
workflow over many case folders and write the processed images and masks to
disk. The lower-level ``zrad.preprocessing`` classes remain the recommended API
for single image/mask pairs and interactive experiments.

DICOM example:

.. code-block:: python

   from zrad.batch import BatchPreprocessor

   result = BatchPreprocessor(
       input_directory="path/to/dicom_cases",
       output_directory="path/to/preprocessed_cases",
       input_data_type="dicom",
       modality="CT",
       number_of_threads=8,
       structures=["CTV", "liver"],
       resample_resolution=1.0,
       resample_dimension="3D",
       image_interpolation_method="linear",
       mask_interpolation_method="linear",
       mask_interpolation_threshold=0.5,
   ).run()

   print(result.processed_count, result.failed_count)
   for case in result.errors:
       print(case.case_name, case.error)

NIfTI example:

.. code-block:: python

   from zrad.batch import BatchPreprocessor

   result = BatchPreprocessor(
       input_directory="path/to/nifti_cases",
       output_directory="path/to/preprocessed_cases",
       input_data_type="nifti",
       modality="CT",
       nifti_image_name="imageCT",
       structures=["CTV", "liver"],
       resample_resolution=1.0,
       resample_dimension="2D",
       image_interpolation_method="linear",
       mask_interpolation_method="NN",
   ).run()

``BatchPreprocessor`` is a save-to-disk batch API. It reports per-case status
through ``BatchResult``.

Batch Filtering
---------------

Use ``zrad.batch.BatchFilter`` when you want to apply one configured image
filter to every selected case folder and write the filtered images to disk.
The lower-level ``zrad.filtering`` classes remain the recommended API for
single-image filtering.

.. code-block:: python

   from zrad.batch import BatchFilter

   result = BatchFilter(
       input_directory="path/to/preprocessed_cases",
       output_directory="path/to/filtered_cases",
       input_data_type="nifti",
       modality="CT",
       nifti_image_name="image",
       number_of_threads=8,
       filter_type="Mean",
       filter_dimension="3D",
       padding_type="reflect",
       mean_support=3,
   ).run()

   print(result.processed_count, result.failed_count)

``BatchFilter`` is a save-to-disk batch API.

Batch Radiomics
---------------

Use ``zrad.batch.BatchRadiomicsExtractor`` when you want to extract radiomics
features for many case folders and write one ``radiomics.csv`` file. The
lower-level ``zrad.radiomics.Radiomics`` class remains the recommended API for
single prepared ROIs.

.. code-block:: python

   from zrad.batch import BatchRadiomicsExtractor

   result = BatchRadiomicsExtractor(
       input_directory="path/to/preprocessed_cases",
       output_directory="path/to/radiomics_output",
       input_data_type="nifti",
       modality="CT",
       nifti_image_name="image",
       structures=["CTV", "liver"],
       number_of_threads=8,
       aggregation_dimension="3D",
       aggregation_method="MERG",
       discretization_method="Number of Bins",
       number_of_bins=64,
   ).run()

   print(result.processed_count, result.skipped_count, result.failed_count)
   for case in result.errors:
       print(case.case_name, case.error)

``BatchPreprocessor``, ``BatchFilter``, and ``BatchRadiomicsExtractor`` are
save-to-disk batch APIs. They all report aggregate and per-case status through
``BatchResult``.

For parameter details, combine this quickstart with the dedicated user-guide
pages and the API reference.
