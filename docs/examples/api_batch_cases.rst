Process case folders with the batch API
=======================================

Use a directory containing one subfolder per case, with a consistently named
NIfTI image and ROI mask in each. This example preprocesses the cases, extracts
features into one CSV file, and checks for failed cases or skipped structures.
Replace the example paths and names with your dataset layout; see
:doc:`../user/data_structure`.

.. code-block:: python

   import csv
   from pathlib import Path

   from zrad.batch import BatchPreprocessor, BatchRadiomicsExtractor

   # Each case folder contains phantom.nii.gz and GTV-1.nii.gz.
   preprocessed = BatchPreprocessor(
       input_directory="path/to/nifti_cases",
       output_directory="output/preprocessed",
       input_data_type="nifti",
       modality="CT",
       nifti_image_name="phantom",
       structures=["GTV-1"],
       resample_resolution=2.0,
       resample_dimension="3D",
       image_interpolation_method="linear",
       mask_interpolation_method="linear",
       mask_interpolation_threshold=0.5,
   ).run()

   extracted = BatchRadiomicsExtractor(
       input_directory="output/preprocessed",
       output_directory="output/features",
       input_data_type="nifti",
       modality="CT",
       nifti_image_name="image",  # Name written by BatchPreprocessor.
       structures=["GTV-1"],
       aggregation_dimension="3D",
       aggregation_method="AVER",
       discretization_method="Number of Bins",
       number_of_bins=32,
   ).run()

   # A processed case can still contain a skipped structure.
   for stage, result in (("preprocessing", preprocessed), ("radiomics", extracted)):
       print(f"{stage}: {result.processed_count} processed, {result.failed_count} failed")
       for case in result.case_results:
           if case.error or case.skipped_structures:
               print(case.case_name, case.error or case.skipped_structures)

   with Path("output/features/radiomics.csv").open(newline="") as csv_file:
       for row in csv.DictReader(csv_file):
           print(row["pat_id"], row["mask_id"])

Batch preprocessing writes one image and mask per case under
``output/preprocessed``. Extraction writes ``output/features/radiomics.csv``.
Check the output rows against the requested cases and structures; see
:doc:`../user/api_batch` and :doc:`../user/results`.
