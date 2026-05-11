import numpy as np

from ..image import Image
from .roi import RoiData


class RangeResegmenter:
    """Remove ROI voxels outside a configured intensity range.

    Range re-segmentation restricts the intensity ROI to voxels whose original
    image intensities lie within user-defined absolute bounds. Excluded voxels
    are represented as ``NaN`` in the intensity mask.

    Parameters
    ----------
    intensity_range : tuple[float, float] or None
        Inclusive lower and upper intensity limits as ``(lower, upper)``. Voxels
        outside this range are removed from ``RoiData.intensity_mask`` by
        replacing them with ``NaN``. If ``None``, range re-segmentation is
        skipped.

    """

    def __init__(self, intensity_range):
        self.intensity_range = intensity_range

    def get_params(self):
        """Return re-segmentation parameters mapped to their configured values.

        Returns
        -------
        params : dict
            Dictionary containing ``intensity_range``.
        """
        return {
            'intensity_range': self.intensity_range,
        }

    def apply(self, roi_data):
        """Apply range re-segmentation to the intensity mask.

        Parameters
        ----------
        roi_data : RoiData
            ROI data with an existing ``intensity_mask``.

        Returns
        -------
        roi_data : RoiData
            ROI data with voxels outside ``intensity_range`` removed from the
            intensity mask.
        """
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
    """Remove ROI voxels outside a mean-centered standard-deviation range.

    Outlier re-segmentation excludes voxels whose original image intensity is
    outside a symmetric interval around the ROI mean. The interval width is
    controlled by a standard-deviation multiplier.

    Parameters
    ----------
    outlier_range : float, str, or None
        Number of standard deviations around the ROI mean to retain. Values
        outside ``mean +/- outlier_range * std`` are removed from
        ``RoiData.intensity_mask`` by replacing them with ``NaN``. If ``None``
        or a non-numeric string is supplied, outlier re-segmentation is skipped.

    """

    def __init__(self, outlier_range):
        self.outlier_range = outlier_range

    def get_params(self):
        """Return re-segmentation parameters mapped to their configured values.

        Returns
        -------
        params : dict
            Dictionary containing ``outlier_range``.
        """
        return {
            'outlier_range': self.outlier_range,
        }

    def apply(self, roi_data):
        """Apply outlier re-segmentation to the intensity mask.

        Parameters
        ----------
        roi_data : RoiData
            ROI data with ``morphological_mask`` and ``intensity_mask``.

        Returns
        -------
        roi_data : RoiData
            ROI data with statistical outliers removed from the intensity mask.
        """
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

    Parameters
    ----------
    intensity_range : tuple[float, float] or None, optional
        Inclusive lower and upper intensity limits as ``(lower, upper)``. If
        ``None``, range re-segmentation is skipped.
    outlier_range : float, str, or None, optional
        Number of standard deviations around the ROI mean to retain. If
        ``None`` or a non-numeric string is supplied, outlier re-segmentation is
        skipped.

    """

    def __init__(self, intensity_range=None, outlier_range=None):
        self.intensity_range = intensity_range
        self.outlier_range = outlier_range

    def get_params(self):
        """Return re-segmentation parameters mapped to their configured values.

        Returns
        -------
        params : dict
            Dictionary containing ``intensity_range`` and ``outlier_range``.
        """
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
