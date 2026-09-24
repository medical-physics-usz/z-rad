Radiomics extraction in GUI
===========================

This example shows a representative radiomics extraction setup for the GUI
workflow.

Example configuration
---------------------

.. figure:: ../images/Rad_example.png
   :alt: Example radiomics extraction configuration
   :width: 700

   Example radiomics configuration in the GUI.

The archived configuration extracts radiomics from ``phantom.nii.gz`` inside
the mask ``GTV-1.nii.gz`` with:

* no filtered image
* no outlier removal
* intensity range ``-400`` to ``400``
* ``3D`` averaged texture aggregation
* bin size ``32``

The ``32`` bin size sets texture discretization. It does not set the IVH
interval. With CT selected, the GUI extracts IVH features directly from
retained HU values at ``1`` HU steps, using the configured ``-400`` to ``400``
intensity range.

See also :doc:`../user/radiomics` for the full radiomics guide.
