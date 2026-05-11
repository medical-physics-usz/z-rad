import numpy as np

from ..image import Image
from .roi import RoiData


class RangeResegmenter:
    """Remove ROI voxels outside a configured intensity range."""

    def __init__(self, intensity_range):
        self.intensity_range = intensity_range

    def get_params(self):
        """Return re-segmentation parameters mapped to their configured values."""
        return {
            'intensity_range': self.intensity_range,
        }

    def apply(self, roi_data, reference_image):
        """Apply range re-segmentation to the intensity mask."""
        if self.intensity_range is None:
            return roi_data

        lower, upper = self.intensity_range
        range_mask = np.where(
            (reference_image.array <= upper) & (reference_image.array >= lower),
            1,
            0,
        )
        intensity_mask = roi_data.intensity_mask
        return RoiData(
            image=roi_data.image,
            filtered_image=roi_data.filtered_image,
            morphological_mask=roi_data.morphological_mask,
            intensity_mask=Image(
                array=np.where((range_mask > 0) & (~np.isnan(intensity_mask.array)), intensity_mask.array, np.nan),
                origin=intensity_mask.origin,
                spacing=intensity_mask.spacing,
                direction=intensity_mask.direction,
                shape=intensity_mask.shape,
            ),
        )


class OutlierResegmenter:
    """Remove ROI voxels outside a mean-centered standard-deviation range."""

    def __init__(self, outlier_range):
        self.outlier_range = outlier_range

    def get_params(self):
        """Return re-segmentation parameters mapped to their configured values."""
        return {
            'outlier_range': self.outlier_range,
        }

    def apply(self, roi_data, reference_image):
        """Apply outlier re-segmentation to the intensity mask."""
        if self.outlier_range is None or not str(self.outlier_range).strip().replace('.', '').isdigit():
            return roi_data

        outlier_range = float(self.outlier_range)
        morphological_array = roi_data.morphological_mask.array
        flattened_image = np.where(morphological_array > 0, reference_image.array, np.nan).ravel()
        valid_values = flattened_image[~np.isnan(flattened_image)]
        mean = np.mean(valid_values)
        std = np.std(valid_values)
        outlier_mask = np.where(
            (reference_image.array <= mean + outlier_range * std)
            & (reference_image.array >= mean - outlier_range * std),
            1,
            0,
        )
        intensity_mask = roi_data.intensity_mask
        return RoiData(
            image=roi_data.image,
            filtered_image=roi_data.filtered_image,
            morphological_mask=roi_data.morphological_mask,
            intensity_mask=Image(
                array=np.where((outlier_mask > 0) & (~np.isnan(intensity_mask.array)), intensity_mask.array, np.nan),
                origin=intensity_mask.origin,
                spacing=intensity_mask.spacing,
                direction=intensity_mask.direction,
                shape=intensity_mask.shape,
            ),
        )


class Resegmenter:
    """Apply range and outlier re-segmentation in sequence."""

    def __init__(self, intensity_range=None, outlier_range=None):
        self.intensity_range = intensity_range
        self.outlier_range = outlier_range

    def get_params(self):
        """Return re-segmentation parameters mapped to their configured values."""
        return {
            'intensity_range': self.intensity_range,
            'outlier_range': self.outlier_range,
        }

    def apply(self, roi_data, reference_image):
        """Return re-segmented ROI masks."""
        roi_data = RangeResegmenter(self.intensity_range).apply(roi_data, reference_image)
        return OutlierResegmenter(self.outlier_range).apply(roi_data, reference_image)
