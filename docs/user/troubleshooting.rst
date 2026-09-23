Troubleshooting
===============

When a run produces missing or unexpected results, start with the completion
message and the log for that run. Check the affected case and mask before
rerunning the full dataset.

Find the logs
-------------

GUI logs are timestamped ``.log`` files. Their location depends on how you
start Z-Rad:

* From source: ``logs/`` in the directory where you launched the application
  (normally the repository root).
* Packaged Windows application: ``%LOCALAPPDATA%\Z-Rad\logs``. If
  ``LOCALAPPDATA`` is unavailable, Z-Rad uses ``%APPDATA%\Z-Rad\logs``.
* Packaged macOS application: ``~/Library/Application Support/Z-Rad/logs``.

Open the log matching the run time and look for the affected case or mask
name and any error messages. For Python batch workflows, inspect
``result.errors`` as shown in :doc:`api_batch`.

A case or mask is missing from the output
-----------------------------------------

* Check that ``Input Directory`` contains case subfolders and that the folder
  selection includes the affected case. Numeric ranges require integer folder
  names; leave both range and list fields empty to process every subfolder.
* For NIfTI, check that the configured image and mask names exist in each case
  folder. Enter names without extensions in the GUI.
* For DICOM, check the selected modality and structure names. A requested
  structure must exist in the case's RTSTRUCT or binary SEG file.

Missing masks or structures can be skipped while other cases continue. Compare
the output identifiers with your expected case and mask list. See
:doc:`data_structure` for complete input requirements and :doc:`results` for
checking output completeness. In batch preprocessing and radiomics, inspect
``skipped_structures`` as well as ``result.errors``; a case may succeed for
some masks and skip others.

The image and mask do not align
-------------------------------

Open the case in :doc:`visualization` and inspect the mask overlay in all three
views. Check voxel spacing, origin, direction, and image dimensions; matching
array shapes alone do not establish alignment. Use a mask belonging to the
source image and resample the image and mask to the same physical grid.
For Python grid alignment, see :doc:`api_workflows`.

A mask is empty or too small for extraction
-------------------------------------------

Check the original mask, the resampled mask, and any intensity restrictions.
Resampling, range re-segmentation, or outlier removal can reduce the number of
valid voxels. See :ref:`extraction-mask-requirements` for minimum voxel counts and
bounding-box dimensions, including per-slice checks for volumetric images
with 2D texture aggregation.

Confirm that the settings match your protocol before changing them. If no
valid ROI remains, correct the input mask or exclude that case from extraction.

Discretization or aggregation settings are rejected
---------------------------------------------------

* Select one discretization method and enter a positive bin size or bin count.
* ``Bin Size`` also requires ``Intensity Range``; its lower bound defines the
  bin origin. See :doc:`discretization_guidelines`.
* Choose a supported dimension and aggregation combination from the GUI menu.
  Python users can find the corresponding options in
  :doc:`../reference/radiomics`.
* In Python, build the intensity mask and run re-segmentation before preparing
  texture or intensity-volume histogram (IVH) images. Prepare the fields needed
  by the requested feature families; see :doc:`api_workflows`.

The GUI cannot start or load data
---------------------------------

When running from source, activate the environment where Z-Rad is installed
and launch ``python main.py`` from the repository root. Review terminal errors
if the application closes before it creates a log.

If the application opens but cannot load a case, check the input path, data
type, modality, and filenames. For installation choices and the macOS release
app's Gatekeeper warning, see :doc:`installation`.
