GUI quickstart
==============

This walkthrough takes the bundled CT phantom from preprocessing to a
radiomics CSV file.
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

Use the bundled CT phantom for a first run. The steps below use one case and
fixed demonstration settings; choose settings for your own study separately.
No Python installation is needed if you use the packaged application.

1. Download ``ibsi_ct_radiomics_phantom.zip`` from the repository's
   `phantom archive page <https://github.com/medical-physics-usz/z-rad/blob/master/tests/data/ibsi_ct_radiomics_phantom.zip>`_
   using the download button. If you already have a source checkout, use the
   same archive in ``tests/data/``.
2. Extract the ZIP and open ``ibsi_ct_radiomics_phantom/nifti``.
3. Create a working folder named ``study`` with an ``input/case_01`` subfolder.
   Copy ``image/phantom.nii.gz`` into that case folder. Copy
   ``mask/mask.nii.gz`` there too and rename it to ``GTV-1.nii.gz``.
   Keep the ``.nii.gz`` files compressed.

The resulting layout is:

.. code-block:: text

   study/
   └── input/
       └── case_01/
           ├── phantom.nii.gz
           └── GTV-1.nii.gz

See the `dataset attribution and license terms
<https://github.com/medical-physics-usz/z-rad/blob/master/tests/data/README.md>`_
before reusing or redistributing the phantom. Enter NIfTI names in the GUI
without ``.nii`` or ``.nii.gz``. For DICOM input and datasets with multiple
cases, see :doc:`data_structure`.

Preprocess the images and masks
-------------------------------

1. Open ``Preprocessing`` and select ``study/input`` as ``Input Directory``.
2. Set ``Output Directory`` to ``study/preprocessed``. Leave the folder-range
   and folder-list fields empty to process every case.
3. Select CT and NIfTI, enter ``phantom`` as ``NIfTI Image`` and ``GTV-1`` as
   ``NIfTI Masks``.
4. Set ``Threads`` to ``1``, ``Resample Resolution`` to ``2`` mm, and
   ``Resample Dimension`` to ``3D``. Select ``Linear`` for both image and mask
   interpolation and set the mask interpolation threshold to ``0.5``.
   Leave ``Mask Union`` unchecked.
5. Save these preprocessing settings with ``File -> Save Input`` or ``Ctrl+S``
   to a file named ``preprocessing.json`` in ``study``. Click ``RUN`` and
   wait for the completion message. Check the run log for one processed case,
   zero skipped cases, and zero failed cases. See :doc:`troubleshooting` for
   log locations.

For each processed case, the output contains ``image.nii.gz`` and
``GTV-1.nii.gz``. Open ``study/preprocessed`` in :doc:`visualization` and inspect
the image and mask alignment before extraction.

Extract features
----------------

1. Open ``Radiomics``. Select ``study/preprocessed`` as ``Input Directory``
   and ``study/results`` as ``Output Directory``.
2. Select CT and NIfTI. Enter ``image`` as ``NIfTI Image`` and ``GTV-1`` as
   ``NIfTI Masks``. Leave ``NIfTI Filtered Image`` empty for this workflow.
3. Set ``Threads`` to ``1`` and leave the folder-range and folder-list fields
   empty. Select ``3D, averaged`` and ``Number of Bins``, then enter ``32``.
   Leave ``Intensity Range`` and ``Outlier Removal`` unchecked.
4. Save the radiomics settings to a separate ``radiomics.json`` file in
   ``study`` with ``File -> Save Input`` or ``Ctrl+S``, then click ``RUN``.
5. Open ``study/results/radiomics.csv``. It should contain one data row with
   ``pat_id`` equal to ``case_01``, ``mask_id`` equal to ``GTV-1``, and
   ``stat_mean`` approximately ``-48.93`` HU. The ``no_bins`` value is ``31``:
   it counts occupied grey levels, which can be fewer than the requested 32.
   See :doc:`results` for column definitions and :doc:`troubleshooting` if the
   row is missing.

Keep both saved configurations and the logs with the results. This mean differs
from the :doc:`api_quickstart` because this workflow resamples the image and
mask first. For your own studies, use :doc:`extraction_concepts` to choose
aggregation and :doc:`discretization_guidelines` to choose bin settings.

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
