Filtering
=========

Overview
--------

The filtering layer applies optional image transforms before radiomics feature
extraction. Filters are configured through the ``Filtering`` class and resolved
to concrete implementations in ``zrad.filtering.filtering_definitions``.

Supported Filters
-----------------

Z-Rad currently supports:

* Mean
* Laplacian of Gaussian
* Laws Kernels
* Gabor
* Wavelets

Shared Concepts
---------------

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
