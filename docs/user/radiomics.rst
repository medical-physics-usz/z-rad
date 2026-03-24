Radiomics Extraction
====================

Overview
--------

The ``Radiomics`` class computes morphological, intensity, histogram, and
texture features from an image-mask pair. The extracted feature set depends on
the aggregation dimensionality and discretization choices.

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
