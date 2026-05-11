import numpy as np
from scipy.ndimage import convolve
from scipy.stats import iqr

from ..exceptions import DataStructureError
from .base import BaseFeatureGroup

class LocalIntensityFeatures:
    """Local intensity peak features derived from the ROI and surrounding image.

    Local and global intensity peaks are calculated from spherical
    neighbourhoods around high-intensity ROI voxels. The neighbourhood radius is
    defined in millimetres, so image spacing is required.

    Parameters
    ----------
    spacing : sequence of float
        Physical voxel spacing used to convert the IBSI neighborhood radius to
        image coordinates.
    """

    def __init__(self, spacing):
        self.spacing = spacing

    def get_params(self):
        """Return the configuration parameters of this local intensity calculator.

        Returns
        -------
        dict
            Parameter names mapped to their configured values.
        """
        return {
            'spacing': self.spacing,
        }

    def get_feature_names(self):
        """Return the local intensity feature names produced by this calculator.

        Returns
        -------
        list of str
            Feature names defined for the local intensity family.
        """
        return list(LOCAL_INTENSITY_FEATURE_NAMES)

    def _calc_local_intensity_peak(self, image, masked_image):
        radius_mm = 6.2
        max_intensity = np.nanmax(masked_image)
        max_voxels = np.argwhere(masked_image == max_intensity)
        highest_peak = []
        for voxel in max_voxels:
            distances = np.sqrt(
                ((np.indices(masked_image.shape).T * self.spacing - voxel * self.spacing) ** 2).sum(axis=3))
            sphere_mask = (distances <= radius_mm)
            selected_voxels = image[sphere_mask.T]
            mean_intensity = np.mean(selected_voxels)
            highest_peak.append(mean_intensity)
        return max(highest_peak)

    def _calc_global_intensity_peak(self, image, masked_image):
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
        local_means = convolve(image, kernel, mode='constant', cval=0.0)
        roi_mask = ~np.isnan(masked_image)
        return np.max(local_means[roi_mask])

    def calculate_features(self, image_array, masked_image_array):
        """Calculate local intensity features for prepared image arrays.

        Parameters
        ----------
        image_array : numpy.ndarray
            Full prepared intensity image used to evaluate local neighborhoods.
        masked_image_array : numpy.ndarray
            ROI-masked version of ``image_array`` with voxels outside the ROI
            set to ``NaN``.

        Returns
        -------
        dict
            Mapping of local intensity feature names to calculated values.
        """
        image_array = np.asarray(image_array)
        masked_image_array = np.asarray(masked_image_array)
        return {
            'loc_peak_loc': self._calc_local_intensity_peak(image_array, masked_image_array),
            'loc_peak_glob': self._calc_global_intensity_peak(image_array, masked_image_array),
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
    """First-order statistics for continuous ROI intensities.

    This feature family summarizes non-discretized intensity values inside the
    ROI, including location, dispersion, percentiles, energy, and robust
    deviation measures.

    This class has no constructor parameters.
    """

    def get_params(self):
        """Return the configuration parameters of this intensity statistics calculator.

        Returns
        -------
        dict
            Parameter names mapped to their configured values.
        """
        return {}

    def get_feature_names(self):
        """Return the intensity statistics feature names produced by this calculator.

        Returns
        -------
        list of str
            Feature names defined for the intensity statistics family.
        """
        return list(INTENSITY_STATISTICS_FEATURE_NAMES)

    def calculate_features(self, intensity_array):
        """Calculate first-order statistics for a prepared continuous intensity array.

        Parameters
        ----------
        intensity_array : numpy.ndarray
            Prepared ROI intensity array with voxels outside the ROI set to
            ``NaN``.

        Returns
        -------
        dict
            Mapping of intensity statistics feature names to calculated values.
        """
        intensity_array = np.asarray(intensity_array)
        return {
            'stat_mean': np.nanmean(intensity_array),
            'stat_var': np.nanstd(intensity_array) ** 2,
            'stat_skew': self._skewness(intensity_array),
            'stat_kurt': self._kurtosis(intensity_array),
            'stat_median': np.nanmedian(intensity_array),
            'stat_min': np.nanmin(intensity_array),
            'stat_p10': np.nanpercentile(intensity_array, 10),
            'stat_p90': np.nanpercentile(intensity_array, 90),
            'stat_max': np.nanmax(intensity_array),
            'stat_iqr': iqr(intensity_array, nan_policy='omit'),
            'stat_range': np.nanmax(intensity_array) - np.nanmin(intensity_array),
            'stat_mad': np.nanmean(np.absolute(intensity_array - np.nanmean(intensity_array))),
            'stat_rmad': self._robust_mean_abs_deviation(intensity_array),
            'stat_medad': np.nanmean(np.absolute(intensity_array - np.nanmedian(intensity_array))),
            'stat_cov': self._variation_coefficient(intensity_array),
            'stat_qcod': self._quartile_coefficient_dispersion(intensity_array),
            'stat_energy': np.nansum(intensity_array ** 2),
            'stat_rms': np.sqrt(np.nanmean(intensity_array ** 2)),
        }


class IntensityHistogramFeatures(_IntensityFeatureCalculator):
    """Histogram-based first-order statistics for discretized ROI intensities.

    This feature family applies first-order summary statistics to discretized
    grey levels and adds histogram entropy, uniformity, and gradient measures.

    This class has no constructor parameters.
    """

    def get_params(self):
        """Return the configuration parameters of this intensity histogram calculator.

        Returns
        -------
        dict
            Parameter names mapped to their configured values.
        """
        return {}

    def get_feature_names(self):
        """Return the intensity histogram feature names produced by this calculator.

        Returns
        -------
        list of str
            Feature names defined for the intensity histogram family.
        """
        return list(INTENSITY_HISTOGRAM_FEATURE_NAMES)

    def calculate_features(self, discretized_image_array):
        """Calculate histogram-based statistics for a prepared discretized intensity array.

        Parameters
        ----------
        discretized_image_array : numpy.ndarray
            Prepared discretized intensity array with voxels outside the ROI set
            to ``NaN``.

        Returns
        -------
        dict
            Mapping of intensity histogram feature names to calculated values.
        """
        discretized_image_array = np.asarray(discretized_image_array)
        values, gradient = self._histogram_gradient(discretized_image_array)
        return {
            'ih_mean': np.nanmean(discretized_image_array),
            'ih_var': np.nanstd(discretized_image_array) ** 2,
            'ih_skew': self._skewness(discretized_image_array),
            'ih_kurt': self._kurtosis(discretized_image_array),
            'ih_median': np.nanmedian(discretized_image_array),
            'ih_min': np.nanmin(discretized_image_array),
            'ih_p10': np.nanpercentile(discretized_image_array, 10),
            'ih_p90': np.nanpercentile(discretized_image_array, 90),
            'ih_max': np.nanmax(discretized_image_array),
            'ih_mode': self._histogram_mode(discretized_image_array),
            'ih_iqr': iqr(discretized_image_array, nan_policy='omit'),
            'ih_range': np.nanmax(discretized_image_array) - np.nanmin(discretized_image_array),
            'ih_mad': np.nanmean(np.absolute(discretized_image_array - np.nanmean(discretized_image_array))),
            'ih_rmad': self._robust_mean_abs_deviation(discretized_image_array),
            'ih_medad': np.nanmean(np.absolute(discretized_image_array - np.nanmedian(discretized_image_array))),
            'ih_cov': self._variation_coefficient(discretized_image_array),
            'ih_qcod': self._quartile_coefficient_dispersion(discretized_image_array),
            'ih_entropy': self._histogram_entropy(discretized_image_array),
            'ih_uniformity': self._histogram_uniformity(discretized_image_array),
            'ih_max_grad': np.max(gradient),
            'ih_max_grad_g': values[np.argmax(gradient)],
            'ih_min_grad': np.min(gradient),
            'ih_min_grad_g': values[np.argmin(gradient)],
        }


class IntensityVolumeHistogramFeatures:
    """Intensity-volume histogram features computed from discretized intensities.

    IVH features quantify the relationship between intensity thresholds and the
    fraction of ROI volume above those thresholds. They are commonly used for
    dose or intensity-volume summaries.

    Parameters
    ----------
    min_intensity : int or float
        Lower bound of the discretized intensity range.
    max_intensity : int or float
        Upper bound of the discretized intensity range.
    discr : int or float, default=1
        Discretization step used to sample the intensity-volume histogram.

    """
    def __init__(self, min_intensity, max_intensity, discr=1):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.discr = discr

    def get_params(self):
        """Return the configuration parameters of this IVH calculator.

        Returns
        -------
        dict
            Parameter names mapped to their configured values.
        """
        return {
            'min_intensity': self.min_intensity,
            'max_intensity': self.max_intensity,
            'discr': self.discr,
        }

    def get_feature_names(self):
        """Return the IVH feature names produced by this calculator.

        Returns
        -------
        list of str
            Feature names defined for the IVH family.
        """
        return list(IVH_FEATURE_NAMES)

    def _fractions(self, array):
        valid_values = array.ravel()[~np.isnan(array.ravel())]
        intensities = np.arange(self.min_intensity, self.max_intensity + self.discr, self.discr)
        fractional_volumes = np.zeros(len(intensities))
        intensity_fractions = np.zeros(len(intensities))
        for idx, intensity_value in enumerate(intensities):
            len_valid_vals = len(valid_values)
            if len_valid_vals == 0:
                raise DataStructureError(f"No valid values in fractions.")
            fractional_volumes[idx] = 1 - np.sum(valid_values < intensity_value) / len_valid_vals
            intensity_diff = self.max_intensity - self.min_intensity
            if intensity_diff == 0:
                raise DataStructureError(f"Intensity range is zero.")
            intensity_fractions[idx] = (intensity_value - self.min_intensity) / intensity_diff
        return intensities, fractional_volumes, intensity_fractions

    @staticmethod
    def _calc_volume_at_intensity_fraction(fractional_volumes, intensity_fractions, x):
        valid_indices = np.where(intensity_fractions > x / 100)
        return np.max(fractional_volumes[valid_indices])

    @staticmethod
    def _calc_intensity_at_volume_fraction(intensities, fractional_volumes, x):
        return np.min(intensities[fractional_volumes <= x / 100])

    def calculate_features(self, ivh_image_array):
        """Calculate IVH features for a prepared IVH intensity array.

        Parameters
        ----------
        ivh_image_array : numpy.ndarray
            Prepared intensity array used to compute the intensity-volume
            histogram.

        Returns
        -------
        dict
            Mapping of IVH feature names to calculated values.
        """
        ivh_image_array = np.asarray(ivh_image_array)
        intensities, fractional_volumes, intensity_fractions = self._fractions(ivh_image_array)
        return {
            'ivh_v10': self._calc_volume_at_intensity_fraction(fractional_volumes, intensity_fractions, 10),
            'ivh_v90': self._calc_volume_at_intensity_fraction(fractional_volumes, intensity_fractions, 90),
            'ivh_i10': self._calc_intensity_at_volume_fraction(intensities, fractional_volumes, 10),
            'ivh_i90': self._calc_intensity_at_volume_fraction(intensities, fractional_volumes, 90),
            'ivh_diff_v10_v90': (
                self._calc_volume_at_intensity_fraction(fractional_volumes, intensity_fractions, 10)
                - self._calc_volume_at_intensity_fraction(fractional_volumes, intensity_fractions, 90)
            ),
            'ivh_diff_i10_i90': (
                self._calc_intensity_at_volume_fraction(intensities, fractional_volumes, 10)
                - self._calc_intensity_at_volume_fraction(intensities, fractional_volumes, 90)
            ),
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
            context.feature_image.spacing[::-1],
        )
        return local.calculate_features(
            context.feature_image.array,
            masks.intensity_mask.array,
        )


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
        stats = IntensityStatisticsFeatures()
        return stats.calculate_features(intensity)


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
        stats = IntensityHistogramFeatures()
        return stats.calculate_features(intensity)


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
                min_val,
                max_val,
                context.ivh_bin_size,
            )
        else:
            ivh = IntensityVolumeHistogramFeatures(
                np.nanmin(image.array),
                np.nanmax(image.array),
            )
        return ivh.calculate_features(image.array)
