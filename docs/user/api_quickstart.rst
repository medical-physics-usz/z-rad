API Quickstart
==============

The Python API mirrors the main GUI workflows through three public classes:

* ``Preprocessing``
* ``Filtering``
* ``Radiomics``

Minimal Example
---------------

.. code-block:: python

   from zrad.preprocessing.preprocessing import Preprocessing
   from zrad.filtering.filtering import Filtering
   from zrad.radiomics.radiomics import Radiomics

   # image and mask are zrad.image.Image instances loaded by your workflow

   prep = Preprocessing(
       input_imaging_modality="CT",
       resample_resolution=1.0,
       resample_dimension="3D",
       interpolation_method="Linear",
       interpolation_threshold=0.5,
   )

   resampled_image = prep.resample(image, image_type="image")
   resampled_mask = prep.resample(mask, image_type="mask")

   filt = Filtering(
       "Laplacian of Gaussian",
       padding_type="reflect",
       sigma_mm=1.0,
       cutoff=4.0,
       dimensionality="3D",
   )
   filtered_image = filt.apply_filter(resampled_image)

   rad = Radiomics(
       aggr_dim="3D",
       aggr_method="AVER",
       number_of_bins=32,
   )
   rad.extract_features(filtered_image, resampled_mask)
   features = rad.features_

Data Model Expectations
-----------------------

The processing classes operate on ``zrad.image.Image`` instances. In practice,
that means your workflow needs to provide:

* voxel data as a NumPy array
* image spacing
* image origin
* image direction
* image shape

For correct feature extraction, images and masks must refer to the same spatial
frame and voxel grid.
