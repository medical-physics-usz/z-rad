GUI Radiomics Extraction
====================

Overview
--------

The ``Radiomics`` class computes morphological, intensity, histogram, and
texture features from an image-mask pair. The extracted feature set depends on
the aggregation dimensionality and discretization choices.

.. figure:: ../images/Rad_tab.png
   :alt: Z-Rad radiomics tab
   :width: 900

   Radiomics extraction tab in the GUI.

GUI Workflow
------------

The radiomics tab combines dataset selection with the extraction-specific
controls:

* ``Data Type`` chooses DICOM or NIfTI input.
* ``Intensity Range`` restricts analysis to a selected voxel-value interval.
* ``Outlier Removal`` excludes intensities outside a chosen standard-deviation
  range. This can create holes in the effective ROI and should be used
  carefully.
* ``Texture Aggregation Method`` defines whether texture matrices are computed
  in 2D, 2.5D, or 3D and how slice or direction merges are handled.
* ``Discretization`` controls whether texture features use fixed bin size or
  fixed bin number binning.

After execution, Z-Rad writes feature tables as ``.csv`` files. Each row starts
with case and mask metadata and is followed by the extracted radiomic features.

Feature Families
----------------

The implementation includes features from these groups:

* shape and morphology
* local intensity
* intensity statistics
* intensity histogram
* GLCM
* GLRLM
* GLSZM
* GLDZM
* NGTDM
* NGLDM
* optional intensity-volume histogram features

In the GUI, the standard workflow exposes morphological, local intensity,
first-order, histogram, GLCM, GLRLM, GLSZM, and GLDZM features. More specialized
features such as intensity-volume histogram metrics and the computationally
expensive Moran's I and Geary's C measures remain API-only.

Key Parameters
--------------

Important configuration options include:

* ``aggr_dim``: ``2D``, ``2.5D``, or ``3D``
* ``aggr_method``: ``MERG``, ``AVER``, ``SLICE_MERG``, or ``DIR_MERG``
* ``intensity_range`` and ``outlier_range``
* discretization through ``number_of_bins`` or ``bin_size``
* IVH-specific discretization through ``ivh_number_of_bins`` or ``ivh_bin_size``
* optional slice weighting and slice median handling

Validation Constraints
----------------------

Z-Rad validates masks before extraction:

* ``3D`` extraction requires a minimum bounding-box extent and voxel count
* ``2D`` and ``2.5D`` extraction validate slices individually

These checks are important because many texture matrices are undefined or
unstable for extremely small masks.

Outputs
-------

After running ``extract_features()``, the results are exposed through the
``features_`` dictionary. Z-Rad also records metadata such as bounding-box
origin, voxel count, and the number of bins used for discretization.

For a task-oriented walkthrough, see :doc:`../examples/gui_radiomics`.
