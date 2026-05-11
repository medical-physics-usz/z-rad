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

    def apply(self, roi_data):
        """Apply range re-segmentation to the intensity mask."""
        if self.intensity_range is None:
            return roi_data

        lower, upper = self.intensity_range
        range_mask = np.where(
            (roi_data.image.array <= upper) & (roi_data.image.array >= lower),
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

    def apply(self, roi_data):
        """Apply outlier re-segmentation to the intensity mask."""
        if self.outlier_range is None or not str(self.outlier_range).strip().replace('.', '').isdigit():
            return roi_data

        outlier_range = float(self.outlier_range)
        morphological_array = roi_data.morphological_mask.array
        flattened_image = np.where(morphological_array > 0, roi_data.image.array, np.nan).ravel()
        valid_values = flattened_image[~np.isnan(flattened_image)]
        mean = np.mean(valid_values)
        std = np.std(valid_values)
        outlier_mask = np.where(
            (roi_data.image.array <= mean + outlier_range * std)
            & (roi_data.image.array >= mean - outlier_range * std),
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
    """Apply range and outlier re-segmentation to ``RoiData.intensity_mask``.

    Re-segmentation removes voxels from the intensity ROI by replacing excluded
    voxels with ``NaN``. The current implementation evaluates range and outlier
    criteria on ``RoiData.image`` and applies them to the existing
    ``RoiData.intensity_mask``. If both criteria are configured, range
    re-segmentation is applied first and outlier re-segmentation second.
    """

    def __init__(self, intensity_range=None, outlier_range=None):
        self.intensity_range = intensity_range
        self.outlier_range = outlier_range

    def get_params(self):
        """Return re-segmentation parameters mapped to their configured values."""
        return {
            'intensity_range': self.intensity_range,
            'outlier_range': self.outlier_range,
        }

    def apply(self, roi_data):
        """Return ROI data with an updated intensity mask.

        Parameters
        ----------
        roi_data : RoiData
            ROI data containing ``image``, ``morphological_mask``, and
            ``intensity_mask``. The intensity mask is usually created with
            ``IntensityMaskBuilder`` before re-segmentation.

        Returns
        -------
        roi_data : RoiData
            New ROI data with ``intensity_mask`` updated by the configured
            range and outlier criteria.
        """
        roi_data = RangeResegmenter(self.intensity_range).apply(roi_data)
        return OutlierResegmenter(self.outlier_range).apply(roi_data)
