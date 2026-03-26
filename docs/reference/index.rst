API Reference
=============

This section documents the main Python API exposed by Z-Rad. The primary
workflow is centered around preprocessing, filtering, radiomics extraction, and
the ``Image`` data structure shared between these steps.

.. currentmodule:: zrad

Core Workflow Classes
---------------------

.. autosummary::
   :toctree: generated

   ~preprocessing.preprocessing.Preprocessing
   ~filtering.filtering.Filtering
   ~radiomics.radiomics.Radiomics

Implementation Modules
----------------------

These modules contain the lower-level filter and feature-definition
implementations used by the workflow classes above.

.. autosummary::
   :toctree: generated

   ~filtering.filtering_definitions
   ~radiomics.radiomics_definitions

Data Model
----------

The image module handles reading DICOM and NIfTI data and writing NIfTI files.

.. autosummary::
   :toctree: generated

   ~image.Image