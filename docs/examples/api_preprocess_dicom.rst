Preprocess DICOM and RTSTRUCT
=============================

Load a CT DICOM series and its RTSTRUCT ``GTV-1`` ROI. Resample the image and
mask to 2 mm in-plane spacing while retaining the original slice spacing. Use
the matching series directory, RTSTRUCT path, and structure name for your data;
see :doc:`../user/data_structure` and :doc:`../user/resampling_guidelines`.

.. code-block:: python

   from zrad.image import Image
   from zrad.preprocessing import ImageResampler, MaskResampler, RoiData

   # The RTSTRUCT mask is converted onto the CT image grid.
   image = Image.from_dicom("path/to/dicom_series", modality="CT")
   mask = Image.from_dicom_mask(
       "path/to/rtstruct.dcm", "GTV-1", reference=image
   )
   roi = RoiData(image=image, morphological_mask=mask)

   # Keep the original through-plane spacing for slice-wise resampling.
   resolution = (2.0, 2.0, float(image.spacing[2]))
   roi = ImageResampler(resolution, method="linear").apply(roi)
   roi = MaskResampler(
       resolution, method="linear", partial_volume_threshold=0.5
   ).apply(roi)

   assert tuple(roi.image.shape) == tuple(roi.morphological_mask.shape)
   assert tuple(roi.image.spacing) == tuple(roi.morphological_mask.spacing)
   roi.image.save_as_nifti("output/image.nii.gz")
   roi.morphological_mask.save_as_nifti("output/GTV-1.nii.gz")
   print(f"Spacing (x, y, z): {tuple(roi.image.spacing)} mm")

The exported CT and mask share a voxel grid. With the bundled phantom, the
reported spacing is ``(2.0, 2.0, 3.0)`` mm.
