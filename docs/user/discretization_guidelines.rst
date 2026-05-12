Image Discretization Guidelines
===============================

Overview
--------

Image discretization, also called quantization, converts image intensities into
grey-level bins. It is mainly used before histogram and texture feature
calculation.

Discretization has two main purposes:

* it makes texture matrix calculation easier and faster
* it reduces small intensity fluctuations by grouping similar values together

Discretized grey levels start at ``1``. They do not start at ``0``. This is
important because some texture feature definitions do not allow grey level
``0``.

Feature Families
----------------

These feature families require discretization:

* intensity histogram features
* intensity-volume histogram features
* GLCM
* GLRLM
* GLSZM
* GLDZM
* NGTDM
* NGLDM

These feature families do not require discretization:

* morphology
* local intensity
* intensity-based statistics

Fixed Bin Number
----------------

Fixed bin number discretization divides the observed ROI intensity range into a
selected number of bins, ``N_g``.

The minimum and maximum intensity values are taken from the ROI. Intensities
are then mapped to bins from ``1`` to ``N_g``.

Main effect:

* each ROI is normalized to the same number of bins
* two lesions with different absolute intensity ranges can both be mapped to,
  for example, ``32`` bins

Advantages:

* useful when intensity units are arbitrary
* often suitable for raw MRI
* often suitable for filtered images
* makes feature values easier to compare when absolute intensity has no fixed
  physical meaning

Disadvantages:

* it weakens the link between grey levels and physical intensity values
* in PET, bin ``10`` may represent different SUV ranges in different patients
* this can be undesirable when absolute intensity values are important

Fixed Bin Size
--------------

Fixed bin size discretization uses a selected bin width. New bins are created
with this width.

Examples:

* ``25`` HU
* ``0.5`` SUV
* ``5`` arbitrary intensity units, but only if this is justified

Main effect:

* the bin width has the same intensity meaning across patients
* physical contrast differences are preserved

Advantages:

* well suited to calibrated units
* useful for CT when using HU
* useful for PET when using SUV
* preserves absolute intensity differences between patients or lesions

Disadvantages:

* not recommended when intensities are arbitrary
* needs a meaningful bin width
* should use a consistent lower bin origin when possible

For fixed bin size discretization, the lower bin origin should ideally be fixed
across all samples. A common choice is the lower bound of the re-segmentation
range.

Examples:

* CT: use a lower origin such as ``-500`` HU if this matches the protocol
* PET: use a lower origin such as ``0`` SUV

Using a fixed origin improves comparability. A patient-specific origin can make
bins harder to compare across cases.

Choosing a Method
-----------------

The best discretization method depends on the image units and the
re-segmentation strategy.

.. list-table::
   :header-rows: 1

   * - Intensity units
     - Re-segmentation range
     - Fixed bin number
     - Fixed bin size
   * - Calibrated units, such as CT HU or PET SUV
     - ``[a, b]``
     - Recommended
     - Recommended
   * - Calibrated units
     - ``[a, infinity)``
     - Recommended
     - Recommended
   * - Calibrated units
     - none
     - Recommended
     - Not recommended
   * - Arbitrary units, such as raw MRI
     - none
     - Recommended
     - Not recommended

The main rule is simple:

* fixed bin number can be used even when there is no fixed physical intensity
  range
* fixed bin size needs a meaningful intensity scale and a consistent bin origin

Practical Guidance
------------------

For CT and PET:

* range re-segmentation can be physically meaningful
* fixed bin size is often suitable because HU and SUV are calibrated units
* fixed bin number is also allowed, but it normalizes away absolute intensity
  scale

For raw MRI:

* global range re-segmentation is usually not meaningful unless intensities
  have been standardized
* fixed bin size is generally not recommended
* fixed bin number is usually the safer choice

For filtered images:

* intensities often have arbitrary or transformed units
* fixed bin number is usually more appropriate
* fixed bin size should be avoided unless the filtered scale has a clear and
  reproducible meaning

Other Methods
-------------

Other discretization methods exist, such as histogram equalization and
Lloyd-Max discretization. IBSI mentions these methods, but they are not the
main standard approaches in this section.

What to Report
--------------

Report the discretization settings so the analysis can be reproduced.

Include:

* the discretization method: fixed bin number or fixed bin size
* the number of bins, if fixed bin number was used
* the bin width, if fixed bin size was used
* the lower bin origin, if fixed bin size was used
* the image modality and intensity units
* the reason for the chosen method, especially for CT, PET, MRI, and filtered
  images

Main Point
----------

Re-segmentation decides which voxels are included in the analysis.
Discretization decides how the retained voxel intensities are converted into
grey-level bins before histogram or texture feature calculation.
