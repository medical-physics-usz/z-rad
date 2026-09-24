Preprocess a NIfTI image and mask
=================================

Load a CT image and its binary ROI mask, then resample both to 2 mm isotropic
spacing. B-spline interpolation handles image intensities; linear interpolation
followed by a 0.5 threshold keeps the mask binary. Replace the paths and choose
the settings for your analysis protocol; see :doc:`../user/resampling_guidelines`.

.. code-block:: python

   from zrad.image import Image
   from zrad.preprocessing import ImageResampler, MaskResampler, RoiData

   # Load the mask on the image grid so their physical geometry matches.
   image = Image.from_nifti("path/to/phantom.nii.gz")
   mask = Image.from_nifti_mask("path/to/mask.nii.gz", reference=image)
   roi = RoiData(image=image, morphological_mask=mask)

   # Spacing is in millimetres and (x, y, z) order.
   resolution = (2.0, 2.0, 2.0)
   roi = ImageResampler(resolution, method="bspline").apply(roi)
   roi = MaskResampler(
       resolution, method="linear", partial_volume_threshold=0.5
   ).apply(roi)

   # Confirm that the resampled image and mask share a voxel grid.
   assert tuple(roi.image.shape) == tuple(roi.morphological_mask.shape)
   assert tuple(roi.image.spacing) == tuple(roi.morphological_mask.spacing)

   # Z-Rad creates the output folder when saving these files.
   roi.image.save_as_nifti("output/image.nii.gz")
   roi.morphological_mask.save_as_nifti("output/mask.nii.gz")
   print(f"Spacing (x, y, z): {tuple(roi.image.spacing)} mm")

The two files in ``output/`` are ready for subsequent ROI preparation and
feature extraction; see :doc:`../user/api_workflows`.
