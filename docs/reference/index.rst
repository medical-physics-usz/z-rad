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

Data Model
----------

The image module handles reading DICOM and NIfTI data and writing NIfTI files.

.. autosummary::
   :toctree: generated

   ~image.Image

.. toctree::
   :hidden:

   generated/zrad.filtering.filtering_definitions
   generated/zrad.radiomics.radiomics_definitions

Filtering Definition Classes
----------------------------

.. autosummary::
   ~filtering.filtering_definitions.Mean
   ~filtering.filtering_definitions.LoG
   ~filtering.filtering_definitions.Laws
   ~filtering.filtering_definitions.Gabor
   ~filtering.filtering_definitions.Wavelets2D
   ~filtering.filtering_definitions.Wavelets3D

Radiomics Definition Classes
----------------------------

.. autosummary::
   ~radiomics.radiomics_definitions.MorphologicalFeatures
   ~radiomics.radiomics_definitions.LocalIntensityFeatures
   ~radiomics.radiomics_definitions.IntensityBasedStatFeatures
   ~radiomics.radiomics_definitions.IntensityVolumeHistogramFeatures
   ~radiomics.radiomics_definitions.GLCM
   ~radiomics.radiomics_definitions.GLRLM_GLSZM_GLDZM_NGLDM
   ~radiomics.radiomics_definitions.NGTDM
