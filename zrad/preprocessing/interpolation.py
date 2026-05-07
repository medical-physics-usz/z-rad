import SimpleITK as sitk
import numpy as np

from ..image import Image


class Resampler:
    """Resample images and masks onto a target voxel grid."""

    def __init__(self, input_imaging_modality, resample_resolution, resample_dimension, interpolation_method,
                 interpolation_threshold=None):
        if not (isinstance(resample_resolution, (float, int)) and resample_resolution > 0):
            raise ValueError(f'Resample resolution {resample_resolution} must be a positive int or float.')
        if resample_dimension not in ['3D', '2D']:
            raise ValueError(f"Resample dimension '{resample_dimension}' is not '2D' or '3D'.")

        self.input_imaging_modality = input_imaging_modality
        self.resample_resolution = resample_resolution
        self.resample_dimension = resample_dimension
        self.interpolation_method = interpolation_method
        self.interpolation_threshold = interpolation_threshold

    def get_params(self):
        """Return resampling parameters mapped to their configured values."""
        return {
            'input_imaging_modality': self.input_imaging_modality,
            'resample_resolution': self.resample_resolution,
            'resample_dimension': self.resample_dimension,
            'interpolation_method': self.interpolation_method,
            'interpolation_threshold': self.interpolation_threshold,
        }

    @staticmethod
    def _get_interpolator(interpolation_method):
        interpolator_mapping = {
            'Linear': sitk.sitkLinear,
            'NN': sitk.sitkNearestNeighbor,
            'BSpline': sitk.sitkBSpline,
            'Gaussian': sitk.sitkGaussian,
        }
        if interpolation_method not in interpolator_mapping:
            raise ValueError(f"Interpolation method '{interpolation_method}' is not supported.")
        return interpolator_mapping[interpolation_method]

    @staticmethod
    def _calculate_resampled_origin(initial_shape, initial_spacing, resulted_spacing, initial_origin, axis=0):
        n_a = float(initial_shape[axis])
        s_a = initial_spacing[axis]
        s_b = resulted_spacing[axis]
        n_b = np.ceil((n_a * s_a) / s_b)
        return initial_origin[axis] + (s_a * (n_a - 1) - s_b * (n_b - 1)) / 2

    def apply(self, image, image_type):
        """Resample an image or mask and return a new image."""
        if not isinstance(image, Image):
            raise TypeError(f"Expected Image, got {type(image)}.")

        if self.resample_dimension == '3D':
            output_spacing = [self.resample_resolution] * 3
        elif self.resample_dimension == '2D':
            output_spacing = [self.resample_resolution] * 2 + [image.spacing[2]]
        else:
            raise ValueError(f"Resample dimension '{self.resample_dimension}' is not supported.")

        output_origin = [
            self._calculate_resampled_origin(image.shape, image.spacing, output_spacing, image.origin, axis)
            for axis in range(3)
        ]
        output_shape = np.ceil((np.array(image.shape) * (np.array(image.spacing) / np.array(output_spacing)))).astype(
            int)

        sitk_image = sitk.GetImageFromArray(image.array)
        sitk_image.SetSpacing(image.spacing)
        sitk_image.SetOrigin(image.origin)
        sitk_image.SetDirection(image.direction)

        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetOutputSpacing(output_spacing)
        resample_filter.SetOutputOrigin(output_origin)
        resample_filter.SetOutputDirection(image.direction)
        resample_filter.SetSize(output_shape.tolist())
        resample_filter.SetOutputPixelType(sitk.sitkFloat64)
        resample_filter.SetInterpolator(self._get_interpolator(self.interpolation_method))

        resampled_sitk_image = resample_filter.Execute(sitk_image)
        if image_type == 'image':
            resampled_sitk_image = self._process_resampled_image(resampled_sitk_image)
        elif image_type == 'mask':
            resampled_sitk_image = self._process_resampled_mask(resampled_sitk_image)
        else:
            raise ValueError(f"Image type '{image_type}' is not supported.")

        return Image(
            array=sitk.GetArrayFromImage(resampled_sitk_image),
            origin=output_origin,
            spacing=output_spacing,
            direction=image.direction,
            shape=output_shape,
        )

    def _process_resampled_image(self, resampled_sitk_image):
        if self.input_imaging_modality == 'CT':
            resampled_sitk_image = sitk.Round(resampled_sitk_image)
            return sitk.Cast(resampled_sitk_image, sitk.sitkInt16)
        if self.input_imaging_modality in ['MR', 'PT']:
            return sitk.Cast(resampled_sitk_image, sitk.sitkFloat64)
        return resampled_sitk_image

    def _process_resampled_mask(self, resampled_sitk_image):
        mask_array = sitk.GetArrayFromImage(resampled_sitk_image)
        threshold = 0.5 if self.interpolation_threshold is None else self.interpolation_threshold
        mask_array = np.where(mask_array >= threshold, 1, 0).astype(np.int16)
        return sitk.GetImageFromArray(mask_array)
