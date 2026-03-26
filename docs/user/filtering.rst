GUI Filtering
=============

Overview
--------

The filtering layer applies optional image transforms before radiomics feature
extraction. Filters are configured through the ``Filtering`` class and resolved
to concrete implementations in ``zrad.filtering.filtering_definitions``.

.. figure:: ../images/Filt_tab.png
   :alt: Z-Rad filtering tab
   :width: 900

   Filtering tab in the GUI.

GUI Workflow
------------

The upper part of the filtering tab mirrors preprocessing, except that mask
selection is not required because filtering is applied to images only.

The filter configuration area lets users choose among:

* Mean
* Laplacian of Gaussian
* Gabor
* Laws kernels
* Wavelets

After the filter type is selected, Z-Rad displays the filter-specific
parameters. The implementation follows the IBSI II definitions, so physical
scales, response maps, decomposition levels, and rotation-invariance options
should be chosen consistently with the downstream analysis protocol.

Supported Filters
-----------------

Most filters use some combination of:

* ``padding_type``: ``constant``, ``nearest``, ``wrap``, or ``reflect``
* ``dimensionality``: ``2D`` or ``3D``
* physical scale parameters such as ``sigma_mm`` or ``lambda_mm``

Wavelet filtering additionally requires:

* ``wavelet_type``
* ``response_map``
* ``decomposition_level``
* optional rotation invariance

Practical Notes
---------------

* Laplacian-of-Gaussian filtering derives the working resolution from the input
  image spacing.
* Z-Rad internally reorders array axes when applying filters and restores the
  original layout before returning a new ``Image`` object.
* Filtering is optional. You can extract radiomics features directly from a
  resampled image if your analysis does not require a transformed image.
* Input configurations can be saved from the GUI and loaded again for repeated
  experiments.

For a task-oriented walkthrough, see :doc:`../examples/gui_filtering`.
