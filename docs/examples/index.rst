Examples
========

These worked examples show common GUI and Python API workflows.

GUI examples
------------

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: Preprocessing in GUI
      :link: gui_preprocessing
      :link-type: doc
      :img-top: ../images/prepr_tab.png

      DICOM and NIfTI preprocessing examples, including representative
      resampling settings for each workflow.

   .. grid-item-card:: Filtering in GUI
      :link: gui_filtering
      :link-type: doc
      :img-top: ../images/Filt_tab.png

      Mean, LoG, Riesz-transformed LoG, separable wavelet, and Simoncelli
      filtering controls, with example configurations and result comparisons.

   .. grid-item-card:: Radiomics extraction in GUI
      :link: gui_radiomics
      :link-type: doc
      :img-top: ../images/Rad_tab.png

      A representative radiomics extraction setup showing the main parameter
      choices used in the GUI.

Python API examples
-------------------

The snippets use example paths for images, masks, and output files. Replace
them with your own paths after :doc:`installing Z-Rad <../user/installation>`.
The single-image examples can use the bundled IBSI CT phantom; read its
`attribution and license terms
<https://github.com/medical-physics-usz/z-rad/blob/master/tests/data/README.md>`_
before reusing or redistributing the data.

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: Preprocess NIfTI
      :link: api_preprocess_nifti
      :link-type: doc

      Resample a CT image and ROI mask to a shared isotropic grid.

   .. grid-item-card:: Preprocess DICOM and RTSTRUCT
      :link: api_preprocess_dicom
      :link-type: doc

      Load a DICOM series and structure, then resample slice-wise.

   .. grid-item-card:: Compare filters
      :link: api_compare_filters
      :link-type: doc

      Save Mean, LoG, and wavelet responses from the same image.

   .. grid-item-card:: Extract radiomics from one ROI
      :link: api_extract_radiomics
      :link-type: doc

      Prepare an intensity mask and extract statistics and GLCM features.

   .. grid-item-card:: Inspect texture matrices
      :link: api_texture_matrices
      :link-type: doc

      Build matrices from NumPy arrays and Images and inspect extraction inputs.

   .. grid-item-card:: Compare original and filtered radiomics
      :link: api_filtered_radiomics
      :link-type: doc

      Use the same ROI to compare original and filtered feature images.

   .. grid-item-card:: Process case folders
      :link: api_batch_cases
      :link-type: doc

      Preprocess a case, extract a CSV, and inspect batch results.

.. toctree::
   :maxdepth: 1
   :hidden:

   gui_preprocessing
   gui_filtering
   gui_radiomics
   api_preprocess_nifti
   api_preprocess_dicom
   api_compare_filters
   api_extract_radiomics
   api_texture_matrices
   api_filtered_radiomics
   api_batch_cases
