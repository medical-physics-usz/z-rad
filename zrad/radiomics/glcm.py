import numpy as np

from ..exceptions import DataStructureError
from .base import BaseFeatureGroup
from .texture_aggregation import format_cm_rlm_feature_names
from .texture_matrices import TextureMatrixMixin

GLCM_FEATURE_NAMES = (
    'cm_joint_max',
    'cm_joint_avg',
    'cm_joint_var',
    'cm_joint_entr',
    'cm_diff_avg',
    'cm_diff_var',
    'cm_diff_entr',
    'cm_sum_avg',
    'cm_sum_var',
    'cm_sum_entr',
    'cm_energy',
    'cm_contrast',
    'cm_dissimilarity',
    'cm_inv_diff',
    'cm_inv_diff_norm',
    'cm_inv_diff_mom',
    'cm_inv_diff_mom_norm',
    'cm_inv_var',
    'cm_corr',
    'cm_auto_corr',
    'cm_clust_tend',
    'cm_clust_shade',
    'cm_clust_prom',
    'cm_info_corr1',
    'cm_info_corr2',
)


class GLCM(TextureMatrixMixin):
    """Grey level co-occurrence matrix features.

    GLCM features summarize how often pairs of discretized grey levels occur at
    fixed neighbour offsets. The class supports IBSI-style 2D, 2.5D, and 3D
    directional aggregation.

    Parameters
    ----------
    aggr_dim : {"2D", "2.5D", "3D"}
        Spatial dimensionality used to build co-occurrence matrices.
    aggr_method : {"MERG", "AVER", "SLICE_MERG", "DIR_MERG"}
        Strategy used to combine matrices across directions and slices.
    slice_weight : bool, default=False
        Weight slice-wise averages by slice ROI voxel count.
    slice_median : bool, default=False
        Aggregate slice-wise values by median instead of mean.
    """

    matrix_family = 'glcm'

    def __init__(self, aggr_dim, aggr_method, slice_weight=False, slice_median=False):
        self.aggr_dim = aggr_dim
        self.aggr_method = aggr_method
        self.slice_weight = slice_weight
        self.slice_median = slice_median

    def get_params(self):
        """Return the configuration parameters of this GLCM calculator.

        Returns
        -------
        dict
            Parameter names mapped to their configured values.
        """
        return {
            'aggr_dim': self.aggr_dim,
            'aggr_method': self.aggr_method,
            'slice_weight': self.slice_weight,
            'slice_median': self.slice_median,
        }

    def get_feature_names(self):
        """Return the GLCM feature names produced by this calculator.

        Returns
        -------
        list of str
            Feature names defined for the GLCM family.
        """
        return list(GLCM_FEATURE_NAMES)

    def calculate_features(self, discretized_image_array):
        """Calculate GLCM features for a prepared discretized intensity array.

        Parameters
        ----------
        discretized_image_array : numpy.ndarray
            Legacy (x, y, z) array. Use calculate_matrices for NumPy (z, y, x)
            or 2D (row, column) inputs. Prepared discretized intensity array with ROI voxels represented by
            integer grey levels and voxels outside the ROI set to ``NaN``.

        Returns
        -------
        dict
            Mapping of GLCM feature names to calculated values.
        """
        return self._legacy_features(discretized_image_array)

    @staticmethod
    def _calc_p_minus(matrix):
        n_g = matrix.shape[0]
        p_minus = np.zeros(n_g)
        for k in range(n_g):
            mask = np.abs(np.subtract.outer(np.arange(n_g), np.arange(n_g))) == k
            p_minus[k] = matrix[mask].sum()
        return p_minus

    @staticmethod
    def _calc_p_plus(matrix):
        n_g = matrix.shape[0]
        p_plus = np.zeros(2 * n_g - 1)
        for k in range(2 * n_g - 1):
            mask = np.add.outer(np.arange(n_g), np.arange(n_g)) == k
            p_plus[k] = matrix[mask].sum()
        return p_plus

    @staticmethod
    def _calc_mu_i_and_sigma_i(matrix):
        p_i = np.sum(matrix, axis=0)
        indices = np.arange(len(p_i))
        mu_i = np.sum(p_i * indices)
        sigma_i = np.sqrt(np.sum(((indices - mu_i) ** 2) * p_i))
        return mu_i, sigma_i

    @classmethod
    def _calc_correlation(cls, matrix):
        i, j = np.indices(matrix.shape)
        mu_i, sigma_i = cls._calc_mu_i_and_sigma_i(matrix)
        if sigma_i == 0:
            raise DataStructureError('Sigma_i in correlation is zero.')
        return (np.sum(matrix * i * j) - mu_i**2) / sigma_i**2

    @classmethod
    def _calc_cluster_tendency_shade_prominence(cls, matrix, power):
        mu_i, _ = cls._calc_mu_i_and_sigma_i(matrix)
        i, j = np.indices(matrix.shape)
        return np.sum((i + j - 2 * mu_i) ** power * matrix)

    @staticmethod
    def _calc_information_correlation_1(matrix):
        non_zero_mask = matrix != 0
        hxy = (-1) * np.sum(matrix[non_zero_mask] * np.log2(matrix[non_zero_mask]))
        p_i = np.sum(matrix, axis=0)
        non_zero_mask_p_i = p_i != 0
        hx = (-1) * np.sum(p_i[non_zero_mask_p_i] * np.log2(p_i[non_zero_mask_p_i]))

        hxy_1 = 0
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if p_i[i] != 0 and p_i[j] != 0:
                    hxy_1 += matrix[i][j] * np.log2(p_i[i] * p_i[j])
        hxy_1 *= -1
        if hx == 0:
            raise DataStructureError('hx in information correlation 1 is zero.')
        return (hxy - hxy_1) / hx

    @staticmethod
    def _calc_information_correlation_2(matrix):
        non_zero_mask = matrix != 0
        hxy = (-1) * np.sum(matrix[non_zero_mask] * np.log2(matrix[non_zero_mask]))
        p_i = np.sum(matrix, axis=0)

        hxy_2 = 0
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if p_i[i] != 0 and p_i[j] != 0:
                    hxy_2 += p_i[i] * p_i[j] * np.log2(p_i[i] * p_i[j])
        hxy_2 *= -1
        return np.sqrt(1 - np.exp(-2 * (hxy_2 - hxy)))

    @staticmethod
    def _calc_joint_average(matrix):
        i, _ = np.indices(matrix.shape)
        return np.sum(matrix * i)

    @staticmethod
    def _calc_joint_var(matrix, mu):
        i, _ = np.indices(matrix.shape)
        return np.sum(matrix * (i - mu) ** 2)

    @staticmethod
    def _calc_joint_entropy(matrix):
        non_zero_mask = matrix != 0
        return (-1) * np.sum(matrix[non_zero_mask] * np.log2(matrix[non_zero_mask]))

    @staticmethod
    def _calc_diff_average(p_minus):
        k = np.indices(p_minus.shape)
        return np.sum(p_minus * k)

    @staticmethod
    def _calc_dif_var(p_minus, mu):
        k = np.indices(p_minus.shape)
        return np.sum(p_minus * (k - mu) ** 2)

    @staticmethod
    def _calc_diff_entropy(p_minus):
        non_zero_mask = p_minus != 0
        return (-1) * np.sum(p_minus[non_zero_mask] * np.log2(p_minus[non_zero_mask]))

    @staticmethod
    def _calc_sum_average(p_plus):
        k = np.indices(p_plus.shape)
        return np.sum(p_plus * k)

    @staticmethod
    def _calc_sum_var(p_plus, mu):
        k = np.indices(p_plus.shape)
        return np.sum(p_plus * (k - mu) ** 2)

    @staticmethod
    def _calc_sum_entropy(p_plus):
        non_zero_mask = p_plus != 0
        return (-1) * np.sum(p_plus[non_zero_mask] * np.log2(p_plus[non_zero_mask]))

    @staticmethod
    def _calc_second_moment(matrix):
        return np.sum(matrix * matrix)

    @staticmethod
    def _calc_contrast(matrix):
        i, j = np.indices(matrix.shape)
        return np.sum(matrix * (i - j) ** 2)

    @staticmethod
    def _calc_dissimilarity(matrix):
        i, j = np.indices(matrix.shape)
        return np.sum(matrix * abs(i - j))

    @staticmethod
    def _calc_inverse_diff(matrix):
        i, j = np.indices(matrix.shape)
        return np.sum(matrix / (1 + abs(i - j)))

    @staticmethod
    def _calc_norm_inv_diff(matrix):
        n_g = len(matrix) - 1
        i, j = np.indices(matrix.shape)
        if n_g == 0:
            raise DataStructureError('n_g in calc_norm_inv_diff is zero.')
        return np.sum(matrix / (1 + abs(i - j) / n_g))

    @staticmethod
    def _calc_inv_diff_moment(p_minus):
        k = np.indices(p_minus.shape)
        return np.sum(p_minus / (1 + k**2))

    @staticmethod
    def _calc_norm_inv_diff_moment(p_minus):
        k = np.indices(p_minus.shape)
        n_g = len(p_minus) - 1
        if n_g == 0:
            raise DataStructureError('n_g in calc_norm_inv_diff_moment is zero.')
        return np.sum(p_minus / (1 + (k / n_g) ** 2))

    @staticmethod
    def _calc_inv_variance(p_minus):
        k = np.indices(p_minus.shape)
        non_zero_mask = k != 0
        return np.sum(p_minus[1:] / (k[non_zero_mask] ** 2))

    @staticmethod
    def _calc_autocor(matrix):
        i, j = np.indices(matrix.shape)
        return np.sum(matrix * i * j)

    @staticmethod
    def _normalize_matrix(matrix, context_name):
        norm = np.sum(matrix)
        if norm == 0:
            raise DataStructureError(f'Denominator is zero in {context_name}.')
        return matrix / norm

    @classmethod
    def _feature_values(cls, matrix):
        joint_average = cls._calc_joint_average(matrix)
        p_minus = cls._calc_p_minus(matrix)
        diff_average = cls._calc_diff_average(p_minus)
        p_plus = cls._calc_p_plus(matrix)
        sum_average = cls._calc_sum_average(p_plus)

        return {
            'cm_joint_max': np.max(matrix),
            'cm_joint_avg': joint_average,
            'cm_joint_var': cls._calc_joint_var(matrix, joint_average),
            'cm_joint_entr': cls._calc_joint_entropy(matrix),
            'cm_diff_avg': diff_average,
            'cm_diff_var': cls._calc_dif_var(p_minus, diff_average),
            'cm_diff_entr': cls._calc_diff_entropy(p_minus),
            'cm_sum_avg': sum_average,
            'cm_sum_var': cls._calc_sum_var(p_plus, sum_average),
            'cm_sum_entr': cls._calc_sum_entropy(p_plus),
            'cm_energy': cls._calc_second_moment(matrix),
            'cm_contrast': cls._calc_contrast(matrix),
            'cm_dissimilarity': cls._calc_dissimilarity(matrix),
            'cm_inv_diff': cls._calc_inverse_diff(matrix),
            'cm_inv_diff_norm': cls._calc_norm_inv_diff(matrix),
            'cm_inv_diff_mom': cls._calc_inv_diff_moment(p_minus),
            'cm_inv_diff_mom_norm': cls._calc_norm_inv_diff_moment(p_minus),
            'cm_inv_var': cls._calc_inv_variance(p_minus),
            'cm_corr': cls._calc_correlation(matrix),
            'cm_auto_corr': cls._calc_autocor(matrix),
            'cm_clust_tend': cls._calc_cluster_tendency_shade_prominence(matrix, 2),
            'cm_clust_shade': cls._calc_cluster_tendency_shade_prominence(matrix, 3),
            'cm_clust_prom': cls._calc_cluster_tendency_shade_prominence(matrix, 4),
            'cm_info_corr1': cls._calc_information_correlation_1(matrix),
            'cm_info_corr2': cls._calc_information_correlation_2(matrix),
        }


class GLCMFeatureGroup(BaseFeatureGroup):
    family = 'glcm'
    requirements = frozenset({'analysis_masks', 'discretized_intensity_image'})

    def supports(self, context):
        return context.roi_data.texture_discretized_image is not None

    def output_names(self, context):
        return format_cm_rlm_feature_names(GLCM_FEATURE_NAMES, context.aggr_dim, context.aggr_method)

    def feature_aliases(self, context):
        output_names = self.output_names(context)
        aliases = {name: name for name in output_names}
        aliases.update(dict(zip(GLCM_FEATURE_NAMES, output_names)))
        return aliases

    def calculate(self, context, prepared_data):
        from .texture_extraction import calculate_texture_family

        return calculate_texture_family(self, context, prepared_data)[0]
