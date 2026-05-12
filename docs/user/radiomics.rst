GUI Radiomics Extraction
========================

Overview
--------

The ``Radiomics`` class computes morphological, intensity, histogram, texture,
and optional IVH features from prepared ``RoiData``. Preprocessing now owns
intensity-mask construction, re-segmentation, texture discretization, and IVH
preparation. ``Radiomics`` only consumes those prepared fields.

.. figure:: ../images/Rad_tab.png
   :alt: Z-Rad radiomics tab
   :width: 900

   Radiomics extraction tab in the GUI.

Main Controls
-------------

The radiomics workflow is organized around the following GUI sections. The
numbering below matches the annotated screenshots used for this workflow.

``(1)`` Upper workflow section
   The upper part of the radiomics tab follows the same layout as the
   preprocessing and filtering tabs. You use it to select the input directory,
   output directory, thread count, imaging modality, and the folders that
   should be processed.

``(2)`` ``Data Type``
   Select whether the input dataset is DICOM or NIfTI. As in preprocessing,
   this choice determines which data-type-specific fields become visible for
   image and mask selection.

``(3)`` ``Intensity Range``
   Restricts the analyzed voxel intensities to a user-defined interval. This is
   useful when radiomic features should only be computed within a selected
   signal range. In the GUI backend this range is applied before texture
   discretization. The lower bound is used as the fixed-bin-size origin when
   fixed bin size discretization is selected. For practical guidance, see
   :doc:`resegmentation_guidelines`.

``(4)`` ``Outlier Removal``
   Removes extreme voxel values based on a selected number of standard
   deviations calculated from the current valid intensity mask. If an intensity
   range is also configured, range re-segmentation is applied first. This can
   suppress unusually high or low intensities, but it can also create holes in
   the effective region of interest.

``(5)`` ``Texture Aggregation Method``
   Defines how texture matrices are computed and merged. The GUI supports
   ``2D``, ``2.5D``, and ``3D`` strategies, with merging or averaging rules
   depending on the selected option.

``(6)`` ``Discretization``
   Controls how image intensities are discretized before texture feature
   computation. The GUI prepares ``RoiData.texture_discretized_image`` using
   either fixed bin size or fixed bin number discretization. Fixed bin size
   requires an intensity range so the lower bound can be used as a stable bin
   origin. For practical guidance, see :doc:`discretization_guidelines`.

``(7)`` ``RUN``
   Starts radiomics extraction with the currently selected configuration.

Feature Families
----------------

The implementation includes features from these groups:

* morphology
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
first-order, histogram, GLCM, GLRLM, GLSZM, and GLDZM features.

**Note:** Intensity-volume histogram features and the computationally
expensive Moran's I and Geary's C measures remain API-only.

Validation Constraints
----------------------

Z-Rad validates masks before extraction:

* For ``3D`` extraction, the mask must contain at least ``27`` valid voxels,
  and the bounding box of the nonzero mask region must be at least ``3``
  voxels wide in every dimension.
* For ``2D`` and ``2.5D`` extraction, Z-Rad validates each slice
  independently. A slice is discarded if it contains fewer than ``9`` valid
  voxels or if its nonzero bounding box is smaller than ``3`` voxels in either
  in-plane dimension.
* If no slice satisfies these ``2D`` or ``2.5D`` requirements, radiomics
  extraction is aborted for that mask.

These checks are important because many texture matrices are undefined or
unstable for extremely small masks.

Outputs
-------

After running ``(7)``, Z-Rad writes radiomics tables as ``.csv`` files to the
selected output directory. Each row begins with case and mask metadata, then
continues with the extracted radiomic features.

The output includes metadata such as:

* patient or case identifier
* mask identifier
* bounding-box metadata
* voxel count
* number of bins used for discretization

At the API level, ``Radiomics.extract_features(roi_data=...)`` returns the
extracted values directly as a dictionary. If ``families`` is omitted, Z-Rad
extracts all feature families available from the prepared ``RoiData``. Summary
fields such as bounding-box size, voxel count, and discretized-bin count are
opt-in via ``include_metadata=True``.

Practical Notes
---------------

* The upper workflow section ``(1)`` uses the same dataset-selection logic as
  preprocessing and filtering.
* The choice of extraction parameters should be kept consistent across all analyzed cases.
* Intensity-volume histogram features and computationally expensive measures
  such as Moran's I and Geary's C are not exposed in the GUI and remain
  available through the API only.
* API users should prepare ``RoiData.texture_discretized_image`` before
  requesting histogram or texture families.
* API users should prepare both ``RoiData.ivh_intensity_image`` and
  ``RoiData.ivh_axis`` before requesting IVH features.
* If you extract features from a filtered NIfTI image, the GUI expects both the
  original NIfTI image and the filtered NIfTI image to be provided.

For a task-oriented walkthrough, see :doc:`../examples/gui_radiomics`.
