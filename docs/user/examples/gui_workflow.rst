GUI Workflow Example
====================

This example describes a typical GUI-driven workflow for one patient study.

1. Open Z-Rad from the repository root with ``python main.py`` or start the
   packaged executable on Windows.
2. In the ``Preprocessing`` tab, load the image and ROI mask, select modality,
   and configure the target spacing.
3. If you need a transformed image, move to ``Filtering`` and choose a filter
   family and its parameters.
4. In ``Radiomics``, choose the aggregation dimension, discretization strategy,
   and any intensity restrictions.
5. Run the extraction and inspect the log output for the applied settings.

This workflow is most useful when you are iterating on parameters interactively
or validating settings before automating them through the API.
