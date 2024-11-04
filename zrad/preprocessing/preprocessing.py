import sys

import SimpleITK as sitk
import numpy as np

from ..image import Image
from ..toolbox_logic import handle_uncaught_exception

sys.excepthook = handle_uncaught_exception


class Preprocessing:

    def __init__(self, input_imaging_modality,
                 resample_resolution=None, resample_dimension=None,
                 interpolation_method=None, interpolation_threshold=None):
        self.input_imaging_modality = input_imaging_modality

        if not (isinstance(resample_resolution, (float, int)) and resample_resolution > 0):
            raise ValueError(f'Resample resolution {resample_resolution} must be a positive int or float.')
        if resample_dimension not in ['3D', '2D']:
            raise ValueError(f"Resample dimension '{resample_dimension}' is not '2D' or '3D'.")

        self.resample_resolution = resample_resolution
        self.resample_dimension = resample_dimension
        self.interpolation_method = interpolation_method
        self.interpolation_threshold = interpolation_threshold

    @staticmethod
    def get_interpolator(interpolation_method):
        """Returns the appropriate SimpleITK interpolator based on the method name."""
        interpolator_mapping = {
            'Linear': sitk.sitkLinear,
            'NN': sitk.sitkNearestNeighbor,
            'BSpline': sitk.sitkBSpline,
            'Gaussian': sitk.sitkGaussian
        }
        if interpolation_method not in interpolator_mapping:
            raise ValueError(f"Interpolation method '{interpolation_method}' is not supported.")
        return interpolator_mapping[interpolation_method]

    @staticmethod
    def calculate_resampled_origin(initial_shape, initial_spacing, resulted_spacing, initial_origin, axis=0):
        """Calculates the resampled image origin for a given axis based on spacing and shape."""
        n_a = float(initial_shape[axis])
        s_a = initial_spacing[axis]
        s_b = resulted_spacing[axis]
        n_b = np.ceil((n_a * s_a) / s_b)
        x_b = initial_origin[axis] + (s_a * (n_a - 1) - s_b * (n_b - 1)) / 2
        return x_b

    def resample(self, image, image_type):
        """Resamples the given image or mask according to the specified resolution and interpolation method."""

        # Calculate output spacing
        if self.resample_dimension == '3D':
            output_spacing = [self.resample_resolution] * 3
        elif self.resample_dimension == '2D':  # '2D' case
            output_spacing = [self.resample_resolution] * 2 + [image.spacing[2]]
        else:
            raise ValueError(f"Resample dimension '{self.resample_dimension}' is not supported.")

        # Calculate output origin
        output_origin = [
            self.calculate_resampled_origin(image.shape, image.spacing, output_spacing, image.origin, axis)
            for axis in range(3)
        ]

        # Calculate output shape
        output_shape = np.ceil((np.array(image.shape) * (np.array(image.spacing) / np.array(output_spacing)))).astype(
            int)

        # Set up SimpleITK image
        sitk_image = sitk.GetImageFromArray(image.array)
        sitk_image.SetSpacing(image.spacing)
        sitk_image.SetOrigin(image.origin)
        sitk_image.SetDirection(image.direction)

        # Configure resampling filter
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetOutputSpacing(output_spacing)
        resample_filter.SetOutputOrigin(output_origin)
        resample_filter.SetOutputDirection(image.direction)
        resample_filter.SetSize(output_shape.tolist())
        resample_filter.SetOutputPixelType(sitk.sitkFloat64)
        resample_filter.SetInterpolator(self.get_interpolator(self.interpolation_method))

        # Execute resampling
        resampled_sitk_image = resample_filter.Execute(sitk_image)

        # Process resampled image or mask
        if image_type == 'image':
            resampled_sitk_image = self.process_resampled_image(resampled_sitk_image)
        elif image_type == 'mask':
            resampled_sitk_image = self.process_resampled_mask(resampled_sitk_image)

        resampled_array = sitk.GetArrayFromImage(resampled_sitk_image)

        # Create a new Image instance with resampled data
        return Image(array=resampled_array, origin=output_origin, spacing=output_spacing,
                     direction=image.direction, shape=output_shape)

    def process_resampled_image(self, resampled_sitk_image):
        """Processes resampled image according to modality-specific requirements."""
        if self.input_imaging_modality == 'CT':
            resampled_sitk_image = sitk.Round(resampled_sitk_image)
            resampled_sitk_image = sitk.Cast(resampled_sitk_image, sitk.sitkInt16)
        elif self.input_imaging_modality in ['MR', 'PT']:
            resampled_sitk_image = sitk.Cast(resampled_sitk_image, sitk.sitkFloat64)
        return resampled_sitk_image

    def process_resampled_mask(self, resampled_sitk_image):
        """Processes resampled mask using thresholding."""
        mask_array = sitk.GetArrayFromImage(resampled_sitk_image)
        mask_array = np.where(mask_array >= self.interpolation_threshold, 1, 0).astype(np.int16)
        return sitk.GetImageFromArray(mask_array)
