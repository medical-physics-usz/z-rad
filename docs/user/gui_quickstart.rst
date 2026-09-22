GUI quickstart
==============

This walkthrough takes a dataset from preprocessing to a radiomics CSV file.
Install Z-Rad using :doc:`installation`, then open the packaged application or
run ``python main.py`` from the repository root in your Python environment.

.. figure:: ../images/zrad_screenshot.png
   :alt: Z-Rad graphical interface
   :width: 700

   Z-Rad main window.

The application has four tabs: ``Preprocessing``, ``Filtering``, ``Radiomics``,
and ``Visualization``. Each tab reads files from its selected input directory.
After a processing step, select its output directory as the input for the next
step.

Prepare the dataset
-------------------

For this example, use CT images with a region-of-interest (ROI) mask named
``GTV-1``. Organize the NIfTI files with one folder per case:

.. code-block:: text

   study/
   └── input/
       ├── case_01/
       │   ├── phantom.nii.gz
       │   └── GTV-1.nii.gz
       └── case_02/
           ├── phantom.nii.gz
           └── GTV-1.nii.gz

Use the same filenames in every case folder. Enter NIfTI names in the GUI
without ``.nii`` or ``.nii.gz``. For DICOM input and other layouts, see
:doc:`data_structure`.

Preprocess the images and masks
-------------------------------

1. Open ``Preprocessing`` and select ``study/input`` as ``Input Directory``.
2. Set ``Output Directory`` to ``study/preprocessed``. Leave the folder-range
   and folder-list fields empty to process every case.
3. Select CT and NIfTI, enter ``phantom`` as ``NIfTI Image`` and ``GTV-1`` as
   ``NIfTI Masks``.
4. Set the resampling resolution, dimension, and interpolation methods required
   by your analysis protocol. See :doc:`preprocessing` for the controls.
5. Click ``RUN`` and review the completion message and logs for skipped or
   failed cases.

For each processed case, the output contains ``image.nii.gz`` and
``GTV-1.nii.gz``. Open ``study/preprocessed`` in :doc:`visualization` and inspect
the image and mask alignment before extraction.

Extract features
----------------

1. Open ``Radiomics``. Select ``study/preprocessed`` as ``Input Directory``
   and ``study/results`` as ``Output Directory``.
2. Select CT and NIfTI. Enter ``image`` as ``NIfTI Image`` and ``GTV-1`` as
   ``NIfTI Masks``. Leave ``NIfTI Filtered Image`` empty for this workflow.
3. Choose the texture aggregation and discretization settings for your protocol.
   Configure ``Intensity Range`` and ``Outlier Removal`` if required.
   See :doc:`radiomics` for an explanation of these choices.
4. Save the configuration with ``File -> Save Input`` or ``Ctrl+S``, then click
   ``RUN``.
5. Open ``study/results/radiomics.csv``. Check the case and mask identifiers and
   review the logs for any missing results. See :doc:`troubleshooting` if a
   case or mask was skipped.

Keep the saved configuration and logs with the results so you can reproduce
the run. Save the settings for each processing tab you use.

Add filtering when needed
-------------------------

To extract features from a filtered image, run :doc:`filtering` after
preprocessing. Select ``study/preprocessed`` as input, enter ``image`` as the
NIfTI image name, and save the filtered output to ``study/filtered``.

Filtering writes a filtered image into each case folder; it does not copy the
original image or masks. Copy each filtered image into the matching case folder
under ``study/preprocessed`` so extraction can read all three files together:

.. code-block:: text

   study/preprocessed/case_01/
   ├── image.nii.gz
   ├── GTV-1.nii.gz
   └── <filter-output-name>.nii.gz

In ``Radiomics``, keep ``NIfTI Image`` set to ``image`` and enter the actual
filtered filename, without the extension, in ``NIfTI Filtered Image``. Use a
separate results directory to keep this extraction distinct from the
unfiltered run.
