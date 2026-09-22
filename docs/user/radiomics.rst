GUI radiomics extraction
========================

Use the Radiomics tab to extract features from images and region-of-interest
(ROI) masks and save the results to a CSV file. Start with the images and masks
prepared for your analysis; see :doc:`gui_quickstart` for a complete workflow.

.. figure:: ../images/Rad_tab.png
   :alt: Z-Rad radiomics tab
   :width: 900

   Radiomics extraction tab in the GUI.

Main controls
-------------

The numbers below match the annotated screenshot.

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
   Keep only voxels within the selected intensity interval for intensity
   and texture analysis. This restriction is applied before discretization
   and leaves the mask used for morphology unchanged. For fixed bin size
   discretization, the lower bound becomes the bin origin. See
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
   computation. Choose ``Bin Size`` for a fixed intensity width or
   ``Number of Bins`` to divide each ROI's intensity range into a fixed number
   of bins. Fixed bin size requires an intensity range to define the bin
   origin. See :doc:`discretization_guidelines`.

``(7)`` ``RUN``
   Starts radiomics extraction with the currently selected configuration.

Choose texture aggregation
--------------------------

The dimension controls whether texture neighbourhoods stay within slices or
extend through the volume:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Dimension
     - Texture calculation
     - How slices are combined
   * - ``2D``
     - Calculate texture within each slice.
     - Combine the resulting feature values across slices.
   * - ``2.5D``
     - Calculate texture within each slice.
     - Merge matrices across slices before calculating features.
   * - ``3D``
     - Calculate texture across the volume, including between slices.
     - Use the volume's texture matrices.

For directional features such as GLCM and GLRLM (defined below), ``averaged``
calculates features for each direction and averages the values; ``merged``
combines matrices before calculating features. ``2D, slice-merged`` merges
directions within each slice, while ``2.5D, direction-merged`` merges slices
for each direction. Other texture families use their own dimension-specific
aggregation rules.

For 2D extraction, ``Slice Averaging`` offers ``Mean``, ``Weighted Mean``
(weighted by ROI voxel count), and ``Median``. Keep the dimension, aggregation,
and slice-averaging settings consistent across cases and record them with the
results.

Feature families
----------------

The GUI extracts the supported feature families for the selected image and
settings:

* morphology
* local intensity
* intensity statistics
* intensity histogram
* grey level co-occurrence matrix (GLCM)
* grey level run length matrix (GLRLM)
* grey level size zone matrix (GLSZM)
* grey level distance zone matrix (GLDZM)
* neighbourhood grey tone difference matrix (NGTDM)
* neighbouring grey level dependence matrix (NGLDM)

Intensity-volume histogram (IVH) features are available through the Python API;
see :doc:`api_workflows`.

Validation constraints
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

After clicking ``RUN``, open ``radiomics.csv`` in the selected output
directory. Each row begins with case and mask metadata, then
continues with the extracted radiomic features.

The output includes metadata such as:

* patient or case identifier
* mask identifier
* bounding-box metadata
* voxel count
* number of bins used for discretization

Extraction from filtered images
-------------------------------

For NIfTI input, provide both the original image in ``NIfTI Image`` and the
filtered image in ``NIfTI Filtered Image``, together with the masks. All files
must be in the corresponding case folder, and names are entered without file
extensions. See :doc:`gui_quickstart` for the folder layout.

For a configuration example, see :doc:`../examples/gui_radiomics`. For missing
results or rejected settings, see :doc:`troubleshooting`.
