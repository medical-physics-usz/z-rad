API Quickstart
==============

The Python API mirrors the main GUI workflows through public preprocessing,
filtering, and radiomics interfaces:

* ``Preprocessing``
* concrete filters created via ``create_filter(...)``
* ``Radiomics``

Recommended Workflow
--------------------

The typical Python workflow is:

1. Load the image and mask into ``zrad.image.Image`` objects with aligned
   geometry.
2. Resample them with ``Preprocessing`` so image and mask share the intended
   voxel spacing.
3. Apply a configured filter if the experiment requires a filtered representation.
4. Run ``Radiomics.extract_features()`` and collect ``features_`` for storage
   in a table or downstream analysis pipeline.
5. Keep the exact preprocessing, filtering, and discretization settings next
   to the extracted features so the run remains reproducible.

Minimal Example
---------------

.. code-block:: python

   from zrad.preprocessing.preprocessing import Preprocessing
   from zrad.filtering import create_filter
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

   filt = create_filter(
       filtering_method="Laplacian of Gaussian",
       padding_type="reflect",
       sigma_mm=1.0,
       cutoff=4.0,
       dimensionality="3D",
   )
   filtered_image = filt.apply(resampled_image)

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

For parameter details, combine this quickstart with the dedicated user-guide
pages and the API reference.
