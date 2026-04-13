import numpy as np
from scipy.ndimage import convolve
from scipy.stats import iqr

from ..exceptions import DataStructureError
from ..image import Image
from .base import BaseFeatureGroup

class LocalIntensityFeatures:
    """Local intensity peak features derived from the ROI and surrounding image.

    Parameters
    ----------
    image : np.ndarray
        Full intensity image used to evaluate spherical neighborhood means.
    masked_image : np.ndarray
        ROI-masked version of ``image`` where voxels outside the ROI are
        encoded as ``NaN``.
    spacing : sequence of float
        Physical voxel spacing used to convert the IBSI neighborhood radius to
        image coordinates.

    Notes
    -----
    This class implements the local and global intensity peak features from the
    IBSI local intensity family.
    """

    def __init__(self, image, masked_image, spacing):
        self.array_image = image
        self.array_masked_image = masked_image
        self.spacing = spacing

    def _calc_local_intensity_peak(self):
        radius_mm = 6.2
        max_intensity = np.nanmax(self.array_masked_image)
        max_voxels = np.argwhere(self.array_masked_image == max_intensity)
        highest_peak = []
        for voxel in max_voxels:
            distances = np.sqrt(
                ((np.indices(self.array_masked_image.shape).T * self.spacing - voxel * self.spacing) ** 2).sum(axis=3))
            sphere_mask = (distances <= radius_mm)
            selected_voxels = self.array_image[sphere_mask.T]
            mean_intensity = np.mean(selected_voxels)
            highest_peak.append(mean_intensity)
        return max(highest_peak)

    def _calc_global_intensity_peak(self):
        radius_mm = 6.2
        spacing = np.array(self.spacing)
        half_sizes = np.ceil(radius_mm / spacing).astype(int)
        grid_ranges = [np.arange(-hs, hs + 1) for hs in half_sizes]
        zz, yy, xx = np.meshgrid(grid_ranges[0], grid_ranges[1], grid_ranges[2], indexing='ij')
        distances = np.sqrt((zz * spacing[0]) ** 2 +
                            (yy * spacing[1]) ** 2 +
                            (xx * spacing[2]) ** 2)
        spherical_mask = distances <= radius_mm
        n_s = np.sum(spherical_mask)
        if n_s == 0:
            raise DataStructureError(f"Ns is zero in global int. mask.")
        kernel = spherical_mask.astype(float) / n_s
        local_means = convolve(self.array_image, kernel, mode='constant', cval=0.0)
        roi_mask = ~np.isnan(self.array_masked_image)
        return np.max(local_means[roi_mask])

    def calculate_local_intensity_features(self):
        return {
            'loc_peak_loc': self._calc_local_intensity_peak(),
            'loc_peak_glob': self._calc_global_intensity_peak(),
        }


class _IntensityFeatureCalculator:
    @staticmethod
    def _valid_values(array):
        x = np.asarray(array)
        return x[~np.isnan(x)]

    @classmethod
    def _skewness(cls, array):
        x = cls._valid_values(array)
        mu = np.mean(x)
        diff = x - mu
        v2 = np.mean(diff ** 2)
        if v2 == 0:
            return 0.0
        m3 = np.mean(diff ** 3)
        return m3 / (v2 ** 1.5)

    @classmethod
    def _kurtosis(cls, array):
        x = cls._valid_values(array)
        mu = np.mean(x)
        diff = x - mu
        v2 = np.mean(diff ** 2)
        if v2 == 0:
            return 0.0
        m4 = np.mean(diff ** 4)
        return (m4 / (v2 ** 2)) - 3

    @staticmethod
    def _robust_mean_abs_deviation(array):
        trimmed = np.array(array, copy=True)
        p10 = np.nanpercentile(trimmed, 10)
        p90 = np.nanpercentile(trimmed, 90)
        trimmed[(trimmed < p10) | (trimmed > p90)] = np.nan
        return np.nanmean(np.absolute(trimmed - np.nanmean(trimmed)))

    @staticmethod
    def _variation_coefficient(array):
        denum = np.nanmean(array)
        if denum == 0:
            return 1_000_000
        return np.nanstd(array) / denum

    @staticmethod
    def _quartile_coefficient_dispersion(array):
        p25 = np.nanpercentile(array, 25)
        p75 = np.nanpercentile(array, 75)
        denum = p75 + p25
        if denum == 0:
            return 1_000_000
        return (p75 - p25) / denum

    @staticmethod
    def _histogram_mode(array):
        values, counts = np.unique(array[~np.isnan(array)], return_counts=True)
        max_count_index = np.argmax(counts)
        return values[max_count_index]

    @staticmethod
    def _histogram_probabilities(array):
        _, counts = np.unique(array[~np.isnan(array)], return_counts=True)
        sum_counts = np.sum(counts)
        if sum_counts == 0:
            raise DataStructureError(f"Sum of counts is zero.")
        return counts / sum_counts

    @classmethod
    def _histogram_entropy(cls, array):
        probabilities = cls._histogram_probabilities(array)
        return (-1) * np.sum(probabilities * np.log2(probabilities))

    @classmethod
    def _histogram_uniformity(cls, array):
        probabilities = cls._histogram_probabilities(array)
        return np.sum(probabilities * probabilities)

    @staticmethod
    def _histogram_gradient(array):
        values, counts = np.unique(array[~np.isnan(array)], return_counts=True)
        if len(counts) <= 1:
            raise DataStructureError(f"Not enough bins to calculate gradient.")
        gradient = np.gradient(counts)
        return values, gradient


class IntensityStatisticsFeatures(_IntensityFeatureCalculator):
    """First-order statistics for continuous ROI intensities."""

    def __init__(self, array):
        self.array = array

    def calculate_intensity_statistics_features(self):
        return {
            'stat_mean': np.nanmean(self.array),
            'stat_var': np.nanstd(self.array) ** 2,
            'stat_skew': self._skewness(self.array),
            'stat_kurt': self._kurtosis(self.array),
            'stat_median': np.nanmedian(self.array),
            'stat_min': np.nanmin(self.array),
            'stat_p10': np.nanpercentile(self.array, 10),
            'stat_p90': np.nanpercentile(self.array, 90),
            'stat_max': np.nanmax(self.array),
            'stat_iqr': iqr(self.array, nan_policy='omit'),
            'stat_range': np.nanmax(self.array) - np.nanmin(self.array),
            'stat_mad': np.nanmean(np.absolute(self.array - np.nanmean(self.array))),
            'stat_rmad': self._robust_mean_abs_deviation(self.array),
            'stat_medad': np.nanmean(np.absolute(self.array - np.nanmedian(self.array))),
            'stat_cov': self._variation_coefficient(self.array),
            'stat_qcod': self._quartile_coefficient_dispersion(self.array),
            'stat_energy': np.nansum(self.array ** 2),
            'stat_rms': np.sqrt(np.nanmean(self.array ** 2)),
        }


class IntensityHistogramFeatures(_IntensityFeatureCalculator):
    """Histogram-based first-order statistics for discretized ROI intensities."""

    def __init__(self, array):
        self.array = array

    def calculate_intensity_histogram_features(self):
        values, gradient = self._histogram_gradient(self.array)
        return {
            'ih_mean': np.nanmean(self.array),
            'ih_var': np.nanstd(self.array) ** 2,
            'ih_skew': self._skewness(self.array),
            'ih_kurt': self._kurtosis(self.array),
            'ih_median': np.nanmedian(self.array),
            'ih_min': np.nanmin(self.array),
            'ih_p10': np.nanpercentile(self.array, 10),
            'ih_p90': np.nanpercentile(self.array, 90),
            'ih_max': np.nanmax(self.array),
            'ih_mode': self._histogram_mode(self.array),
            'ih_iqr': iqr(self.array, nan_policy='omit'),
            'ih_range': np.nanmax(self.array) - np.nanmin(self.array),
            'ih_mad': np.nanmean(np.absolute(self.array - np.nanmean(self.array))),
            'ih_rmad': self._robust_mean_abs_deviation(self.array),
            'ih_medad': np.nanmean(np.absolute(self.array - np.nanmedian(self.array))),
            'ih_cov': self._variation_coefficient(self.array),
            'ih_qcod': self._quartile_coefficient_dispersion(self.array),
            'ih_entropy': self._histogram_entropy(self.array),
            'ih_uniformity': self._histogram_uniformity(self.array),
            'ih_max_grad': np.max(gradient),
            'ih_max_grad_g': values[np.argmax(gradient)],
            'ih_min_grad': np.min(gradient),
            'ih_min_grad_g': values[np.argmin(gradient)],
        }


class IntensityVolumeHistogramFeatures:
    """Intensity-volume histogram features computed from discretized intensities.

    Parameters
    ----------
    array : np.ndarray
        Input intensity array with ROI voxels retained and non-ROI voxels set
        to ``NaN``.
    min_intensity : int or float
        Lower bound of the discretized intensity range.
    max_intensity : int or float
        Upper bound of the discretized intensity range.
    discr : int or float, default=1
        Discretization step used to sample the intensity-volume histogram.

    Notes
    -----
    The constructor precomputes the fractional volume and fractional intensity
    curves so that the IBSI IVH summary features can be queried directly.
    """
    def __init__(self, array, min_intensity, max_intensity, discr=1):
        # Flatten array and remove NaN values
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.valid_values = array.ravel()[~np.isnan(array.ravel())]
        # Create a discretized list of intensities using the given step size
        self.intensities = np.arange(min_intensity, max_intensity + discr, discr)
        self.fractional_volumes = np.zeros(len(self.intensities))
        self.intensity_fractions = np.zeros(len(self.intensities))
        # Copy the discretized intensities (optional, kept for clarity)
        self.intensity = np.copy(self.intensities)

        self._fractions()

    def _fractions(self):
        for idx, intensity_value in enumerate(self.intensities):
            len_valid_vals = len(self.valid_values)
            if len_valid_vals == 0:
                raise DataStructureError(f"No valid values in fractions.")
            self.fractional_volumes[idx] = 1 - np.sum(self.valid_values < intensity_value) / len_valid_vals
            intensity_diff = self.max_intensity - self.min_intensity
            if intensity_diff == 0:
                raise DataStructureError(f"Intensity range is zero.")
            self.intensity_fractions[idx] = (intensity_value - self.min_intensity) / intensity_diff

    def _calc_volume_at_intensity_fraction(self, x):
        valid_indices = np.where(self.intensity_fractions > x / 100)
        return np.max(self.fractional_volumes[valid_indices])

    def _calc_intensity_at_volume_fraction(self, x):
        return np.min(self.intensity[self.fractional_volumes <= x / 100])

    def _calc_volume_fraction_diff_intensity_fractions(self):
        return self._calc_volume_at_intensity_fraction(10) - self._calc_volume_at_intensity_fraction(90)

    def _calc_intensity_fraction_diff_volume_fractions(self):
        return self._calc_intensity_at_volume_fraction(10) - self._calc_intensity_at_volume_fraction(90)

    def calculate_ivh_features(self):
        return {
            'ivh_v10': self._calc_volume_at_intensity_fraction(10),
            'ivh_v90': self._calc_volume_at_intensity_fraction(90),
            'ivh_i10': self._calc_intensity_at_volume_fraction(10),
            'ivh_i90': self._calc_intensity_at_volume_fraction(90),
            'ivh_diff_v10_v90': self._calc_volume_fraction_diff_intensity_fractions(),
            'ivh_diff_i10_i90': self._calc_intensity_fraction_diff_volume_fractions(),
        }



LOCAL_INTENSITY_FEATURE_NAMES = (
    'loc_peak_loc',
    'loc_peak_glob',
)


INTENSITY_STATISTICS_FEATURE_NAMES = (
    'stat_mean',
    'stat_var',
    'stat_skew',
    'stat_kurt',
    'stat_median',
    'stat_min',
    'stat_p10',
    'stat_p90',
    'stat_max',
    'stat_iqr',
    'stat_range',
    'stat_mad',
    'stat_rmad',
    'stat_medad',
    'stat_cov',
    'stat_qcod',
    'stat_energy',
    'stat_rms',
)


INTENSITY_HISTOGRAM_FEATURE_NAMES = (
    'ih_mean',
    'ih_var',
    'ih_skew',
    'ih_kurt',
    'ih_median',
    'ih_min',
    'ih_p10',
    'ih_p90',
    'ih_max',
    'ih_mode',
    'ih_iqr',
    'ih_range',
    'ih_mad',
    'ih_rmad',
    'ih_medad',
    'ih_cov',
    'ih_qcod',
    'ih_entropy',
    'ih_uniformity',
    'ih_max_grad',
    'ih_max_grad_g',
    'ih_min_grad',
    'ih_min_grad_g',
)


IVH_FEATURE_NAMES = (
    'ivh_v10',
    'ivh_v90',
    'ivh_i10',
    'ivh_i90',
    'ivh_diff_v10_v90',
    'ivh_diff_i10_i90',
)


class LocalIntensityFeatureGroup(BaseFeatureGroup):
    family = 'local_intensity'
    requirements = frozenset({'base_masks'})

    def supports(self, context):
        return True

    def output_names(self, context):
        return LOCAL_INTENSITY_FEATURE_NAMES

    def feature_aliases(self, context):
        return {name: name for name in self.output_names(context)}

    def calculate(self, context, prepared_data):
        masks = prepared_data.require_base_masks()
        local = LocalIntensityFeatures(
            context.feature_image.array,
            masks.intensity_mask.array,
            context.feature_image.spacing[::-1],
        )
        return local.calculate_local_intensity_features()


class IntensityStatisticsFeatureGroup(BaseFeatureGroup):
    family = 'intensity_statistics'
    requirements = frozenset({'base_masks'})

    def supports(self, context):
        return True

    def output_names(self, context):
        return INTENSITY_STATISTICS_FEATURE_NAMES

    def feature_aliases(self, context):
        return {name: name for name in self.output_names(context)}

    def calculate(self, context, prepared_data):
        intensity = prepared_data.require_base_masks().intensity_mask.array
        stats = IntensityStatisticsFeatures(intensity)
        return stats.calculate_intensity_statistics_features()


class IntensityHistogramFeatureGroup(BaseFeatureGroup):
    family = 'intensity_histogram'
    requirements = frozenset({'analysis_masks', 'discretized_intensity_image'})

    def supports(self, context):
        return True

    def output_names(self, context):
        return INTENSITY_HISTOGRAM_FEATURE_NAMES

    def feature_aliases(self, context):
        return {name: name for name in self.output_names(context)}

    def calculate(self, context, prepared_data):
        intensity = prepared_data.require_discretized_intensity_image().array
        stats = IntensityHistogramFeatures(intensity)
        return stats.calculate_intensity_histogram_features()


class IVHFeatureGroup(BaseFeatureGroup):
    family = 'ivh'
    requirements = frozenset({'analysis_masks', 'ivh_intensity_image'})

    def supports(self, context):
        return True

    def default_enabled(self, context):
        return context.calc_ivh_features

    def output_names(self, context):
        return IVH_FEATURE_NAMES

    def feature_aliases(self, context):
        return {name: name for name in self.output_names(context)}

    def calculate(self, context, prepared_data):
        image = prepared_data.require_ivh_intensity_image()

        if context.ivh_number_of_bins is not None:
            ivh = IntensityVolumeHistogramFeatures(
                image.array,
                np.nanmin(image.array),
                np.nanmax(image.array),
            )
        elif context.ivh_bin_size is not None:
            if context.intensity_range is not None:
                min_val = context.intensity_range[0] + 0.5 * context.ivh_bin_size
                max_val = context.intensity_range[1] - 0.5 * context.ivh_bin_size
            else:
                min_val = np.nanmin(image.array) + 0.5 * context.ivh_bin_size
                max_val = np.nanmax(image.array) - 0.5 * context.ivh_bin_size
            ivh = IntensityVolumeHistogramFeatures(
                image.array,
                min_val,
                max_val,
                context.ivh_bin_size,
            )
        else:
            ivh = IntensityVolumeHistogramFeatures(
                image.array,
                np.nanmin(image.array),
                np.nanmax(image.array),
            )
        return ivh.calculate_ivh_features()


def build_ivh_mask(context, intensity_mask, *, bin_number_discretize, bin_size_discretize):
    ivh_mask = intensity_mask.copy()

    if context.ivh_bin_size is not None:
        if context.intensity_range is not None:
            min_val = context.intensity_range[0]
        else:
            min_val = np.nanmin(ivh_mask.array)
        ivh_mask = Image(
            array=min_val + (bin_size_discretize(ivh_mask, min_val, context.ivh_bin_size).array - 0.5) * context.ivh_bin_size,
            origin=ivh_mask.origin,
            spacing=ivh_mask.spacing,
            direction=ivh_mask.direction,
            shape=ivh_mask.shape,
        )

    if context.ivh_number_of_bins is not None:
        return bin_number_discretize(ivh_mask, context.ivh_number_of_bins)

    return ivh_mask
