Python batch workflows
======================

Use ``zrad.batch`` to process datasets organized as described in
:doc:`data_structure`. Each class writes results to disk:

* ``BatchPreprocessor`` writes images and masks into case subfolders.
* ``BatchFilter`` writes filtered images into case subfolders.
* ``BatchRadiomicsExtractor`` writes one ``radiomics.csv`` file.

For individual images or ROIs, use :doc:`api_workflows` instead.
The examples below are independent recipes; replace the paths and filenames
with those for your dataset.

Batch preprocessing
-------------------

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

See :doc:`resampling_guidelines` for help choosing resolution, dimension, and
image and mask interpolation settings for either example.

Batch filtering
---------------

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

Batch radiomics
---------------

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

``BatchRadiomicsExtractor`` prepares IVH intensities for every ROI using the
selected ``modality`` and writes six ``ivh_`` columns to ``radiomics.csv``.
These settings are independent of texture discretization. Set
``intensity_range`` when your study has a common re-segmentation interval;
it also supplies the fixed-bin-size IVH origin for PET and RTDOSE. See
:ref:`ivh-discretization` for the automatic settings and range behavior.

Inspect the result
------------------

All three workflows return ``BatchResult``. Counts describe cases, while
``result.errors`` contains case results with an error message:

.. code-block:: python

   print(result.processed_count, result.skipped_count, result.failed_count)
   for case in result.errors:
       print(case.case_name, case.error)

Preprocessing and radiomics also report individual skipped structures. Inspect
these even when ``failed_count`` is zero:

.. code-block:: python

   # Use with BatchPreprocessor or BatchRadiomicsExtractor results.
   for case in result.case_results:
       if case.skipped_structures:
           print(case.case_name, "Skipped structures:", case.skipped_structures)

A radiomics case is counted as processed if at least one structure produces
features. Another structure in that case can be skipped without a case-level
error. Check the requested case/mask pairs against the CSV; see :doc:`results`
for metadata and feature-name explanations.

Preprocessing saves each image as ``image.nii.gz`` and each mask under its
structure name. Use those names when configuring the next step. Filtering
saves only the filtered image; if extracting features from it, place it beside
the original image and masks in each extraction case folder and set
``nifti_filtered_image_name`` to its filename without the extension.

See :doc:`../reference/batch` for all parameters and result fields.
