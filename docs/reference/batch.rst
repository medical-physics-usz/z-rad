Batch
=====

``zrad.batch`` contains save-to-disk workflows for processing many case
folders with one configuration. Use these classes when you want the same
operation that the GUI performs over a cohort of patients.

The batch API currently includes:

* ``BatchPreprocessor`` for DICOM/NIfTI preprocessing and mask export.
* ``BatchFilter`` for applying one image filter to every selected case.
* ``BatchRadiomicsExtractor`` for writing one radiomics CSV from many cases.

The lower-level ``zrad.preprocessing``, ``zrad.filtering``, and
``zrad.radiomics`` modules remain the single-image or single-ROI APIs.

All batch workflows return ``BatchResult``:

.. code-block:: python

   result = batch.run()

   print(result.processed_count, result.skipped_count, result.failed_count)
   for case in result.errors:
       print(case.case_name, case.error)

.. currentmodule:: zrad

.. autosummary::
   :toctree: generated

   ~batch.filtering.BatchFilter
   ~batch.filtering.FilteringCaseResult
   ~batch.preprocessing.BatchPreprocessor
   ~batch.preprocessing.PreprocessingCaseResult
   ~batch.radiomics.BatchRadiomicsExtractor
   ~batch.radiomics.RadiomicsCaseResult
   ~batch.results.BatchResult
