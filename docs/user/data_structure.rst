Expected data structure
=======================

GUI and Python batch workflows expect one subfolder per case inside the input
directory. In the GUI, select this parent folder with ``Input Directory``.
The single-image Python API can read files directly and does not require this
folder layout.

Supported input types
---------------------

Z-Rad supports:

* DICOM and NIfTI input data
* CT, MRI, PET, mammography, ultrasound, and RTDOSE imaging modalities

Basic directory layout
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

Recommended layout for multiple modalities
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

DICOM folder contents
---------------------

For DICOM workflows, each case folder should contain:

* one image series for the selected modality
* an RTSTRUCT or DICOM SEG file when ROI-based processing is required

Example DICOM case folder:

.. code-block:: text

   data_folder/
   └── folder_1/
       ├── image_slice_001.dcm
       ├── image_slice_002.dcm
       ├── ...
       └── structures.dcm

* Z-Rad reads the image series directly from the case folder.
* If both RTSTRUCT and SEG objects are present, the first detected RTSTRUCT is
  used; otherwise, the first detected SEG is used.
* DICOM SEG support is limited to BINARY objects; fractional and label-map
  segmentations are not supported.
* Ultrasound input must be a single DICOM file with ``PixelSpacing`` and
  ``SliceThickness`` metadata.
* For SEG input, enter the segment's ``SegmentLabel`` as the structure name.
  The segmentation must reference the source image series.

NIfTI folder contents
---------------------

For NIfTI workflows, each case folder should contain the image and mask files
that Z-Rad should process together.

Example NIfTI case folder:

.. code-block:: text

   data_folder/
   └── folder_1/
       ├── phantom.nii.gz
       ├── GTV-1.nii.gz
       ├── liver.nii.gz
       └── filtered_image.nii.gz

* The GUI expects image and mask names without file extensions.
* Z-Rad accepts both ``.nii.gz`` and ``.nii`` files.
* The image filename entered in the GUI must exist in every case folder that is
  processed.
* Use consistent image, mask, and optional filtered-image names across cases.
  A requested mask that is missing from a case is skipped.
* For extraction from a filtered image, keep the original image, filtered
  image, and masks together as shown in :doc:`gui_quickstart`.

Folder selection in the GUI
---------------------------

The GUI can process case folders in three ways:

* ``Start Folder`` and ``Stop Folder`` for numerically named folders
* ``List of Folders`` for an explicit comma-separated list
* all subfolders in the selected input directory when no folder filter is set

If you use numeric start and stop selection, the case folders must have integer
names such as ``1``, ``2``, and ``15``.
