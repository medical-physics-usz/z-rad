Compare original and filtered-image radiomics
=============================================

Extract the same features from an original CT image and from its 3D Mean-filtered
version. Both runs use the same morphological ROI. Each feature image is
discretized separately into 32 bins, so the GLCM values reflect both filtering
and per-image binning.

.. code-block:: python

   from zrad.filtering import Mean
   from zrad.image import Image
   from zrad.preprocessing import IntensityMaskBuilder, RoiData, TextureDiscretizer
   from zrad.radiomics import Radiomics

   image = Image.from_nifti("path/to/phantom.nii.gz")
   mask = Image.from_nifti_mask("path/to/mask.nii.gz", reference=image)
   results = {}

   for label, use_filter in (("Original", False), ("Mean filtered", True)):
       roi = RoiData(image=image, morphological_mask=mask)
       if use_filter:
           # The filter sets roi.filtered_image and leaves the original image and ROI in place.
           roi = Mean(padding_type="reflect", support=3, dimensionality="3D").apply(roi)
       roi = IntensityMaskBuilder().apply(roi)
       roi = TextureDiscretizer(number_of_bins=32).apply(roi)
       results[label] = Radiomics(aggr_dim="3D", aggr_method="AVER").extract_features(
           roi_data=roi,
           features=["stat_mean", "cm_contrast_3D_avg"],
           include_metadata=True,
       )

   for label, features in results.items():
       print(label)
       print(f"  Mean: {features['stat_mean']:.2f} HU")
       print(f"  GLCM contrast: {features['cm_contrast_3D_avg']:.3f}")
       print(f"  ROI voxels: {features['no_voxels']}")

``no_voxels`` is the same in both runs because the anatomical ROI is the same.
Mean smoothing retains HU units, but its feature values describe the smoothed
image. See :doc:`../user/api_workflows` for how ``RoiData`` carries the
original and filtered images.
