import SimpleITK as sitk
import numpy as np

from ..image import Image
from .roi import RoiData


def _normalize_resolution(resolution):
    if isinstance(resolution, (float, int)):
        if resolution <= 0:
            raise ValueError(f"Resolution {resolution} must be positive.")
        return [float(resolution)] * 3

    try:
        values = [float(value) for value in resolution]
    except (TypeError, ValueError):
        raise ValueError("Resolution must be a positive number or a sequence of three positive numbers.")

    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("Resolution must be a positive number or a sequence of three positive numbers.")
    return values


def _normalize_interpolation_method(method):
    normalized = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    method_mapping = {
        "linear": "Linear",
        "trilinear": "Linear",
        "nn": "NN",
        "nearest": "NN",
        "nearest_neighbor": "NN",
        "nearest_neighbour": "NN",
        "bspline": "BSpline",
        "b_spline": "BSpline",
        "cubic_spline": "BSpline",
        "tricubic_spline": "BSpline",
        "tricubic": "BSpline",
        "gaussian": "Gaussian",
    }
    if normalized not in method_mapping:
        raise ValueError(f"Interpolation method '{method}' is not supported.")
    return method_mapping[normalized]


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


def _calculate_resampled_origin(initial_shape, initial_spacing, resulted_spacing, initial_origin, axis=0):
    n_a = float(initial_shape[axis])
    s_a = initial_spacing[axis]
    s_b = resulted_spacing[axis]
    n_b = np.ceil((n_a * s_a) / s_b)
    return initial_origin[axis] + (s_a * (n_a - 1) - s_b * (n_b - 1)) / 2


class ImageResampler:
    """Resample an image or ``RoiData.image`` onto a target voxel grid."""

    def __init__(self, resolution, method='linear', intensity_rounding=None):
        self.resolution = tuple(_normalize_resolution(resolution))
        self.method = method
        self.intensity_rounding = intensity_rounding

    def get_params(self):
        """Return image-resampling parameters mapped to their configured values."""
        return {
            'resolution': self.resolution,
            'method': self.method,
            'intensity_rounding': self.intensity_rounding,
        }

    def apply(self, data):
        """Return a resampled image or ROI data with a resampled image."""
        if isinstance(data, RoiData):
            return RoiData(
                image=self._resample(data.image),
                filtered_image=data.filtered_image,
                morphological_mask=data.morphological_mask,
                intensity_mask=None,
            )
        if isinstance(data, Image):
            return self._resample(data)
        raise TypeError(f"Expected Image or RoiData, got {type(data)}.")

    def _resample(self, image):
        result = _resample_image(
            image=image,
            output_spacing=self.resolution,
            interpolation_method=_normalize_interpolation_method(self.method),
        )
        if self.intensity_rounding is None:
            return result
        if self.intensity_rounding == 'nearest_integer':
            result.array = np.rint(result.array).astype(np.int16)
            return result
        raise ValueError(f"Intensity rounding '{self.intensity_rounding}' is not supported.")


class MaskResampler:
    """Resample a mask or ``RoiData.morphological_mask`` onto a target voxel grid."""

    def __init__(self, resolution, method='nearest_neighbor', partial_volume_threshold=0.5):
        self.resolution = tuple(_normalize_resolution(resolution))
        self.method = method
        self.partial_volume_threshold = 0.5 if partial_volume_threshold is None else partial_volume_threshold

    def get_params(self):
        """Return mask-resampling parameters mapped to their configured values."""
        return {
            'resolution': self.resolution,
            'method': self.method,
            'partial_volume_threshold': self.partial_volume_threshold,
        }

    def apply(self, data):
        """Return a resampled mask or ROI data with a resampled morphological mask."""
        if isinstance(data, RoiData):
            if data.morphological_mask is None:
                raise ValueError("MaskResampler requires RoiData.morphological_mask.")
            return RoiData(
                image=data.image,
                filtered_image=data.filtered_image,
                morphological_mask=self._resample(data.morphological_mask),
                intensity_mask=None,
            )
        if isinstance(data, Image):
            return self._resample(data)
        raise TypeError(f"Expected Image or RoiData, got {type(data)}.")

    def _resample(self, mask):
        result = _resample_image(
            image=mask,
            output_spacing=self.resolution,
            interpolation_method=_normalize_interpolation_method(self.method),
        )
        result.array = np.where(result.array >= self.partial_volume_threshold, 1, 0).astype(np.int16)
        return result


def _resample_image(image, output_spacing, interpolation_method):
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image)}.")

    output_origin = [
        _calculate_resampled_origin(image.shape, image.spacing, output_spacing, image.origin, axis)
        for axis in range(3)
    ]
    output_shape = np.ceil((np.array(image.shape) * (np.array(image.spacing) / np.array(output_spacing)))).astype(int)

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
    resample_filter.SetInterpolator(_get_interpolator(interpolation_method))

    resampled_sitk_image = resample_filter.Execute(sitk_image)
    return Image(
        array=sitk.GetArrayFromImage(resampled_sitk_image),
        origin=output_origin,
        spacing=list(output_spacing),
        direction=image.direction,
        shape=output_shape,
    )
