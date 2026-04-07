Expected Data Structure
=======================

Z-Rad expects the input directory to contain one subfolder per case. In the
GUI, this is the directory selected with ``Input Directory``. Each case folder
is then processed either by explicit folder list, by numeric start and stop
range, or by scanning all subfolders in the input directory.

Supported Input Types
---------------------

Z-Rad supports:

* Windows, macOS, and Linux workflows
* DICOM and NIfTI input data
* CT, MRI, PET, mammography, and RTDOSE imaging modalities

Basic Directory Layout
----------------------

To process a dataset with case folders such as ``folder_1``, ``folder_2``, and
``folder_n``, organize the input directory like this:

.. code-block:: text

   main_path/
   └── data_folder/
       ├── folder_1/
       ├── folder_2/
       ├── ...
       └── folder_n/

Here, ``data_folder`` is the directory you select as the input dataset. Each
case folder contains the image data to be processed for one study or patient.

Recommended Layout For Multiple Modalities
------------------------------------------

If you process multiple imaging modalities or data collections in parallel, use
a consistent layout for each modality:

.. code-block:: text

   main_path/
   ├── PET/
   │   ├── folder_1/
   │   ├── folder_2/
   │   ├── ...
   │   └── folder_n/
   └── CT/
       ├── folder_1/
       ├── folder_2/
       ├── ...
       └── folder_n/

In this setup, you would select either ``PET`` or ``CT`` as the GUI input
directory, depending on the workflow you want to run.

DICOM Folder Contents
---------------------

For DICOM workflows, each case folder should contain:

* exactly one image series for the selected modality
* zero or one RTSTRUCT file when ROI-based processing is required

In practice, a DICOM case folder is expected to look like this:

.. code-block:: text

   data_folder/
   └── folder_1/
       ├── image_slice_001.dcm
       ├── image_slice_002.dcm
       ├── ...
       └── structures.dcm

Notes:

* Z-Rad reads the image series directly from the case folder.
* For preprocessing, if multiple RTSTRUCT files are present, the first detected
  file is used.
* For radiomics extraction, ROI-based DICOM workflows assume that an RTSTRUCT
  file is available in the case folder.

NIfTI Folder Contents
---------------------

For NIfTI workflows, each case folder should contain the image and mask files
that Z-Rad should process together.

In practice, a NIfTI case folder is expected to look like this:

.. code-block:: text

   data_folder/
   └── folder_1/
       ├── phantom.nii.gz
       ├── GTV-1.nii.gz
       ├── liver.nii.gz
       └── filtered_image.nii.gz

Notes:

* The GUI expects image and mask names without file extensions.
* Z-Rad accepts both ``.nii.gz`` and ``.nii`` files.
* The image filename entered in the GUI must exist in every case folder that is
  processed.
* The same naming convention should be used across all case folders so the
  configured image and mask names resolve consistently.
* Optional filtered images used in radiomics workflows must follow the same
  naming convention across folders.

Folder Selection In The GUI
---------------------------

The GUI can process case folders in three ways:

* ``Start Folder`` and ``Stop Folder`` for numerically named folders
* ``List of Folders`` for an explicit comma-separated list
* all subfolders in the selected input directory when no folder filter is set

If you use numeric start and stop selection, the case folders must have integer
names such as ``1``, ``2``, and ``15``.
