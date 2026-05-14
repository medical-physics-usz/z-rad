Preprocessing in GUI
====================

This example shows two common preprocessing setups in the GUI: one for DICOM
input and one for NIfTI input.

DICOM Example
-------------

.. figure:: ../images/prepr_B.png
   :alt: Example DICOM preprocessing configuration
   :width: 700

   Example DICOM preprocessing setup.

This configuration corresponds to:

* structure ``GTV-1``
* slice-wise ``2D`` resampling
* linear interpolation to ``2 x 2`` mm in-plane spacing
* linear ROI interpolation with a threshold of ``0.5``

NIfTI Example
-------------

.. figure:: ../images/prepr_E.png
   :alt: Example NIfTI preprocessing configuration
   :width: 700

   Example NIfTI preprocessing setup.

This configuration corresponds to:

* image ``phantom.nii.gz``
* mask ``GTV-1.nii.gz``
* full ``3D`` resampling
* B-spline image interpolation to ``2 x 2 x 2`` mm
* linear ROI interpolation with a threshold of ``0.5``

See also :doc:`../user/preprocessing` for the full preprocessing guide.
