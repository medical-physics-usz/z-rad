GUI Quickstart
==============

.. figure:: ../images/zrad_screenshot.png
   :alt: Z-Rad graphical interface
   :width: 700

   Z-Rad main window.

Launch The Application
----------------------

Start the GUI from the repository root:

.. code-block:: bash

   python main.py

On Windows, you can also start the packaged executable attached to each
release.

The application opens three main tabs:

* ``Preprocessing``
* ``Filtering``
* ``Radiomics``
* ``Visualization``

Typical Workflow
----------------

This is the typical GUI workflow for one study:

1. Load an image and, when required, a region-of-interest mask.
2. In ``Preprocessing``, select the imaging modality and configure resampling
   or format conversion parameters.
3. If you need a transformed image, move to ``Filtering`` and choose the filter
   family and its parameters.
4. In ``Radiomics``, configure aggregation, discretization, and any intensity
   restrictions.
5. Inspect images and masks in the visualization window when needed.
6. Save the settings, run the analysis, and review the log output for the
   applied parameters.

Input And Output
----------------

Z-Rad supports workflows based on:

* DICOM image series
* NIfTI images
* ROI masks aligned to the source image

The GUI writes logs that document the executed processing steps and parameter
choices. This is the primary record for troubleshooting and reproducibility.

For the required input layout, see :doc:`data_structure`.

For tab-by-tab walkthroughs, continue with :doc:`preprocessing`,
:doc:`filtering`, :doc:`radiomics`, and :doc:`visualization`.
