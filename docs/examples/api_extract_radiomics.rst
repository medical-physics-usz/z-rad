Extract radiomics from one ROI
==============================

Build an intensity mask, retain CT intensities from -400 to 400 HU, discretize
the retained values into 32 bins, and extract intensity statistics and GLCM
features. These parameters are illustrative; choose and record settings for your
study using :doc:`../user/resegmentation_guidelines` and
:doc:`../user/discretization_guidelines`.

.. code-block:: python

   from zrad.image import Image
   from zrad.preprocessing import IntensityMaskBuilder, Resegmenter, RoiData, TextureDiscretizer
   from zrad.radiomics import Radiomics

   image = Image.from_nifti("path/to/phantom.nii.gz")
   mask = Image.from_nifti_mask("path/to/mask.nii.gz", reference=image)

   # Keep original CT values inside the ROI and NaN outside it.
   roi = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))
   roi = Resegmenter(intensity_range=(-400, 400)).apply(roi)
   roi = TextureDiscretizer(number_of_bins=32).apply(roi)

   # Request only the feature families needed for this analysis.
   features = Radiomics(aggr_dim="3D", aggr_method="AVER").extract_features(
       roi_data=roi, families=["intensity_statistics", "glcm"], include_metadata=True
   )
   print(f"Mean intensity: {features['stat_mean']:.2f} HU")
   print(f"GLCM contrast: {features['cm_contrast_3D_avg']:.3f}")
   print(f"ROI voxels: {features['no_voxels']}; occupied bins: {features['no_bins']}")

With the bundled phantom, the output includes approximately ``25.07 HU`` for
mean intensity and ``9.567`` for GLCM contrast. ``no_voxels`` describes the
morphological ROI; it can exceed the number of voxels retained after intensity
resegmentation. See :doc:`../user/results` for feature and metadata meanings.
