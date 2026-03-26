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

Loading Data
------------

The upper part of the visualization tab follows the same dataset-selection
pattern as preprocessing, but no output directory is required. After the input
path and case range are configured, pressing ``RUN`` opens the dedicated image
viewer window.

Viewer Window
-------------

.. figure:: ../images/Visual_window.png
   :alt: Z-Rad visualization window
   :width: 800

   Visualization window for image and mask inspection.

The viewer provides:

* three synchronized orthogonal projections
* mouse-wheel scrolling and zooming through the image
* double-click full-screen expansion of a selected projection
* windowing controls for comfortable intensity display
* per-mask visibility toggles and a global hide-all action
* image metadata including case name, array shape, voxel spacing, cursor
  coordinates, and the intensity at the current voxel

Practical Use
-------------

The visualization workflow is useful for:

* confirming that images and masks are aligned before feature extraction
* checking the effect of preprocessing or filtering on the anatomy of interest
* identifying obvious data-quality issues before launching a batch run
