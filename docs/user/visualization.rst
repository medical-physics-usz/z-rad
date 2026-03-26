GUI Visualization
=================

Overview
--------

The visualization tab is used to load image studies and inspect the image and
mask geometry before or after preprocessing, filtering, and radiomics runs.

.. figure:: ../images/Visual_tab.png
   :alt: Z-Rad visualization tab
   :width: 900

   Visualization tab in the GUI.

Main Controls
-------------

The visualization workflow is organized around the following GUI sections. The
numbering below matches the annotated screenshots used for this workflow.

``(1)`` Upper workflow section
   The upper part of the visualization tab follows the same dataset-selection
   pattern as preprocessing, except that no output directory is required. Use
   this section to select the input directory, imaging modality, thread count,
   and the folders that should be opened.

``(2)`` ``RUN``
   Loads the selected images and opens the dedicated visualization window.

Viewer Window
-------------

.. figure:: ../images/Visual_window.png
   :alt: Z-Rad visualization window
   :width: 800

   Visualization window for image and mask inspection.

The dedicated viewer is organized into the following sections:

``(2.1)`` Projection panes
   The upper part of the viewer displays three orthogonal projections. These
   views can be scrolled and zoomed, and double-clicking one of them expands it
   to full-screen for closer inspection.

``(2.2)`` Windowing controls
   Use the windowing controls to select a comfortable intensity display range
   for the loaded image.

``(2.3)`` Mask visibility controls
   Masks can be hidden individually or all at once with the
   ``Hide All Masks`` control.

``(2.4)`` Image information panel
   This section reports important metadata about the loaded image, including
   the current folder name, image shape, voxel spacing, the cursor position,
   and the intensity value at the current voxel.

``(2.5)`` Navigation controls
   Use these controls to move through the loaded images and slices.

Practical Notes
---------------

* The upper workflow section ``(1)`` follows the same folder-selection logic as
  preprocessing, so you can open a full dataset, a numeric range of folders, or
  an explicit folder list.
* This tab is useful for quick inspection before running a batch workflow.
* The projection panes ``(2.1)`` are useful for confirming that images and
  masks are spatially aligned before filtering or radiomics extraction.
* Windowing in ``(2.2)`` can substantially improve visual assessment,
  especially for modalities or datasets with a wide intensity range.
* The visualization workflow is particularly useful for spotting obvious
  data-quality issues before launching preprocessing or feature extraction.
