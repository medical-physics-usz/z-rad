GUI Filtering
=============

Overview
--------

The filtering layer applies optional image transforms before radiomics feature
extraction. Filters are created through ``create_filter(...)`` and resolved to
concrete implementations in the filtering package, grouped by family.

.. figure:: ../images/Filt_tab.png
   :alt: Z-Rad filtering tab
   :width: 900

   Filtering tab in the GUI.

Main Controls
-------------

The filtering workflow is organized around the following GUI sections. The
numbering below matches the annotated screenshots used for this workflow.

``(1)`` Upper workflow section
   The upper part of the filtering tab mirrors the preprocessing tab. You use
   it to select the input directory, output directory, thread count, imaging
   modality, and the folders that should be processed. Unlike preprocessing, no
   mask selection is required because filtering is applied to images only.

``(2)`` ``Filter Type``
   Select the image transform to apply. The GUI currently supports:

   * Mean
   * Laplacian of Gaussian
   * Gabor
   * Laws kernels
   * Wavelets

   Once a filter is selected, Z-Rad displays the corresponding
   filter-specific parameters.

``(3)`` ``RUN``
   Starts the filtering process. The filtered images are written to the
   selected output directory.

Filter Parameters
-----------------

After a filter is selected in ``(2)``, the parameter fields shown in the GUI
depend on the chosen filter family.

Most filters use some combination of:

* ``padding_type``: ``constant``, ``nearest``, ``wrap``, or ``reflect``
* ``dimensionality``: ``2D`` or ``3D``
* physical scale parameters such as ``sigma_mm`` or ``lambda_mm``

The available filter families are:

* Mean
* Laplacian of Gaussian
* Gabor
* Laws kernels
* Wavelets, including Daubechies 2, Daubechies 3, first-order Coiflet, and
  Haar filters

Wavelet filtering additionally requires:

* ``wavelet_type``
* ``response_map``
* ``decomposition_level``
* optional rotation invariance

The implementation follows IBSI II definitions, so physical scales, response
maps, decomposition levels, and rotation-invariance settings should be chosen
consistently with the downstream analysis protocol.

Practical Notes
---------------

* The upper section ``(1)`` uses the same dataset-selection logic as
  preprocessing, including support for numeric folder ranges and explicit
  folder lists.
* Filtering is optional. You can extract radiomics features directly from a
  resampled image if your analysis does not require a transformed image.
* The filter parameters shown after selecting ``(2)`` depend entirely on the
  chosen filter family.
* Laplacian-of-Gaussian filtering derives the working resolution from the input
  image spacing.
* Input configurations can be saved from the GUI and loaded again for repeated
  experiments, which is useful when comparing several filter settings.
* If you are comparing multiple filter families, keep the preprocessing and
  radiomics settings fixed so that the impact of the filtering step remains
  interpretable.

For a task-oriented walkthrough, see :doc:`../examples/gui_filtering`.
