Python filtering
================

Filters return a transformed image for subsequent inspection or feature
extraction. Use a concrete filter class when writing Python code.

Concrete filters expose a small, consistent API:

* configure the filter in the constructor
* call ``apply(image)`` to return a filtered ``Image``
* call ``apply(roi_data)`` inside a preprocessing ``Pipeline`` to set
  ``roi_data.filtered_image``

.. code-block:: python

   from zrad.filtering import Mean
   from zrad.image import Image

   image = Image.from_nifti("path/to/image.nii.gz")

   image_filter = Mean(
       padding_type="reflect",
       support=3,
       dimensionality="3D",
   )

   filtered_image = image_filter.apply(image)

For dynamic workflows, use ``create_filter(...)`` when the filter type and
parameters come from a GUI form or saved configuration:

.. code-block:: python

   from zrad.filtering import create_filter

   image_filter = create_filter(
       filtering_method="Mean",
       padding_type="reflect",
       support=3,
       dimensionality="3D",
   )

Riesz and Simoncelli filters
----------------------------

``RieszLoG`` composes a Laplacian-of-Gaussian response with a normalized Riesz
transform. The Riesz multi-index follows physical axis order and must match the
selected dimensionality. A structure-tensor scale can be supplied for locally
aligned, pure second-order 3D responses.

.. code-block:: python

   from zrad.filtering import RieszLoG, Simoncelli
   from zrad.image import Image

   image = Image.from_nifti("path/to/image.nii.gz")

   riesz_log = RieszLoG(
       padding_type="reflect",
       sigma_mm=1.5,
       cutoff=4.0,
       dimensionality="3D",
       riesz_order=(2, 0, 0),
       structure_tensor_sigma_mm=1.0,
   )

   simoncelli = Simoncelli(
       padding_type="wrap",
       decomposition_level=2,
       dimensionality="3D",
       riesz_order=(1, 0, 0),
   )

   riesz_log_image = riesz_log.apply(image)
   simoncelli_image = simoncelli.apply(image)

Omit ``riesz_order`` from ``Simoncelli`` to obtain its isotropic band-pass
response. Simoncelli filtering supports ``nearest`` padding and periodic
padding (``wrap``; ``periodic`` is accepted as an alias).


For all filter parameters, see :doc:`../reference/filtering`. To use a filter
as part of feature extraction, see :doc:`api_workflows`.
