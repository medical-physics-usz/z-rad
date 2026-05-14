import SimpleITK as sitk


def read_nifti_image(image_path):
    """Read a NIfTI image as a SimpleITK image."""
    sitk_reader = sitk.ImageFileReader()
    sitk_reader.SetImageIO("NiftiImageIO")
    sitk_reader.SetFileName(image_path)
    return sitk_reader.Execute()


def read_nifti_mask(mask_path, reference_image):
    """Read a NIfTI mask and align it to a reference SimpleITK image."""
    mask = read_nifti_image(mask_path)

    if mask.GetOrigin() != reference_image.GetOrigin() or mask.GetDirection() != reference_image.GetDirection():
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(sitk.Transform())
        mask = resampler.Execute(mask)

    return mask
