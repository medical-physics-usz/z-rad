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

Intensity-volume histogram features need special care. IVH features are based
on a cumulative intensity-volume curve. This curve should use a fine and
ordered intensity axis. See `IVH-Specific Discretization`_ for details.

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

In Z-Rad's high-level ``Radiomics`` workflow, fixed bin size therefore requires
an ``intensity_range``. Its lower bound is used as the bin origin. Use fixed bin
number when no stable lower bound is available.

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
* choose either fixed bin number or fixed bin size, not both

IVH-Specific Discretization
---------------------------

Intensity-volume histogram, or IVH, features use discretization differently
from texture features.

Texture features often use a small number of grey levels, such as ``16``,
``32``, ``64``, or ``128``. IVH should usually be finer. IVH features describe
how the volume fraction changes over the intensity range. They need a detailed
and ordered intensity axis.

Do not choose ``32`` bins for IVH only because texture features use ``32``
bins. This is usually too coarse for IVH.

What IVH Measures
~~~~~~~~~~~~~~~~~

For each discretized intensity level, IVH measures the fraction of ROI voxels
with intensity at least that level.

Two quantities are used:

.. list-table::
   :header-rows: 1

   * - Quantity
     - Meaning
   * - Volume fraction
     - Fraction of ROI voxels with intensity at least the selected level.
   * - Intensity fraction
     - Position of the selected intensity level within the full IVH intensity
       range.

Examples of IVH features include:

* ``V10`` and ``V90``: volume fractions at ``10%`` and ``90%`` of the intensity
  range
* ``I10`` and ``I90``: intensities at ``10%`` and ``90%`` of the volume range
* ``V10 - V90``: difference between low- and high-intensity volume fractions
* ``I10 - I90``: difference between intensities at two volume fractions

IBSI also defines an IVH area-under-the-curve feature. It has no reference
values and should not be used for reproducible analyses.

Recommended IVH Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended IVH strategy depends on the image intensity type.

.. list-table::
   :header-rows: 1

   * - Image type
     - Example
     - IVH strategy
     - Typical parameter
   * - Discrete calibrated units
     - CT HU
     - Use original intensities directly.
     - ``1`` HU steps
   * - Continuous calibrated units
     - PET SUV
     - Use fixed bin size discretization.
     - small bin width, such as ``0.1`` SUV
   * - Arbitrary units
     - raw MRI
     - Use fixed bin number discretization.
     - ``N_g = 1000``

For CT IVH features:

* use HU values directly
* use a physically justified re-segmentation range
* use an IVH interval of ``1`` HU
* report the exact re-segmentation range

Example:

* re-segmentation range: ``[-500, 400]`` HU
* IVH range: ``[-500, 400]`` HU
* IVH interval: ``1`` HU

For PET IVH features:

* use fixed bin size discretization
* use a small SUV bin width, such as ``0.1`` SUV
* use the re-segmentation bounds as the IVH intensity range when available
* use bin-centre intensities for the IVH axis

Example:

* re-segmentation range: ``[0, 20]`` SUV
* IVH bin width: ``0.1`` SUV
* IVH axis after bin-centre conversion: ``0.05, 0.15, 0.25, ..., 19.95`` SUV

For MRI or other arbitrary units:

* use fixed bin number discretization with ``N_g = 1000``
* avoid fixed bin size unless intensities have been standardized or calibrated
* construct the IVH over the discretized range ``[1, 1000]``

Relationship to Re-segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For IVH, the re-segmentation range also helps define the IVH intensity range.
It is not a separate IVH-only re-segmentation step.

Examples:

* CT re-segmentation ``[-500, 400]`` HU gives IVH range ``[-500, 400]`` HU
* PET re-segmentation ``[0, 20]`` SUV with bin width ``0.1`` SUV gives IVH
  axis ``0.05`` to ``19.95`` SUV after bin-centre conversion

This matters because IVH features such as ``V10`` and ``V90`` depend on the
chosen intensity range.

For example, ``V10`` does not mean volume above intensity value ``10``. It means
volume above the intensity that corresponds to ``10%`` of the selected IVH
range.

Example:

* range ``[0, 20]`` SUV: ``10%`` intensity fraction is ``2`` SUV
* range ``[2, 12]`` SUV: ``10%`` intensity fraction is ``3`` SUV

The same lesion can therefore have different ``V10`` and ``V90`` values if the
IVH range changes.

What to Avoid for IVH
~~~~~~~~~~~~~~~~~~~~~

Avoid these settings unless there is a clear reason:

* using the same coarse ``32`` or ``64`` bins for IVH that are used for texture
  features
* using fixed bin size for raw MRI IVH when the intensity scale is arbitrary
* letting each CT or PET ROI define its own IVH range when a physical
  re-segmentation range is available

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
* for IVH features, the IVH-specific discretization method
* for IVH features, the IVH intensity range
* for IVH features, the IVH interval or bin width
* for IVH features, whether the range came from re-segmentation or from the ROI
  intensity values
* the image modality and intensity units
* the reason for the chosen method, especially for CT, PET, MRI, and filtered
  images

Main Point
----------

Re-segmentation decides which voxels are included in the analysis.
Discretization decides how the retained voxel intensities are converted into
grey-level bins before histogram or texture feature calculation.
