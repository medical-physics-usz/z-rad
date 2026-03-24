GUI Quickstart
==============

Launch The Application
----------------------

Start the GUI from the repository root:

.. code-block:: bash

   python main.py

The application opens three main tabs:

* ``Preprocessing``
* ``Filtering``
* ``Radiomics``

Typical Workflow
----------------

1. Load an image and, when required, a region-of-interest mask.
2. Configure preprocessing or format conversion parameters.
3. Optionally apply one of the supported image filters.
4. Configure radiomics extraction parameters.
5. Save the settings and run the analysis.

Input And Output
----------------

Z-Rad supports workflows based on:

* DICOM image series
* NIfTI images
* ROI masks aligned to the source image

The GUI writes logs that document the executed processing steps and parameter
choices. This is the primary record for troubleshooting and reproducibility.
