import numpy as np

from ..exceptions import DataStructureError
from .base import BaseFeatureGroup
from .texture_aggregation import format_cm_rlm_feature_names

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


class GLCM:
    """Gray level co-occurrence matrix features.

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
            Prepared discretized intensity array with ROI voxels represented by
            integer gray levels and voxels outside the ROI set to ``NaN``.

        Returns
        -------
        dict
            Mapping of GLCM feature names to calculated values.
        """
        discretized_image_array = np.asarray(discretized_image_array)
        lvl = int(np.nanmax(discretized_image_array) + 1)
        tot_no_of_roi_voxels = int(np.sum(~np.isnan(discretized_image_array)))

        if self.aggr_dim == '3D':
            glcm_3d_matrices = self._calc_3d_matrices(discretized_image_array, lvl)
            if self.aggr_method == 'AVER':
                return self._calc_3d_averaged_glcm_features(glcm_3d_matrices)
            if self.aggr_method == 'MERG':
                return self._calc_3d_merged_glcm_features(glcm_3d_matrices)
        else:
            glcm_2d_matrices, slice_no_of_roi_voxels = self._calc_2d_matrices(discretized_image_array, lvl)
            if self.aggr_method == 'DIR_MERG':
                return self._calc_2_5d_direction_merged_glcm_features(glcm_2d_matrices)
            if self.aggr_method == 'MERG':
                return self._calc_2_5d_merged_glcm_features(glcm_2d_matrices)
            if self.aggr_method == 'AVER':
                return self._calc_2d_averaged_glcm_features(
                    glcm_2d_matrices,
                    slice_no_of_roi_voxels,
                    tot_no_of_roi_voxels,
                )
            if self.aggr_method == 'SLICE_MERG':
                return self._calc_2d_slice_merged_glcm_features(
                    glcm_2d_matrices,
                    slice_no_of_roi_voxels,
                    tot_no_of_roi_voxels,
                )
        raise DataStructureError(
            f'Unsupported GLCM aggregation: aggr_dim={self.aggr_dim}, aggr_method={self.aggr_method}.'
        )

    @staticmethod
    def _calc_2d_matrices(image, lvl):
        def calc_2d_glcm_slice(image_slice, direction):
            dx, dy, *_ = direction
            rows, cols = image_slice.shape
            glcm_slice = np.zeros((lvl, lvl), dtype=int)
            nan_mask = np.isnan(image_slice)

            valid_i = np.arange(rows - dx) if dx >= 0 else np.arange(-dx, rows)
            valid_j = np.arange(cols - dy) if dy >= 0 else np.arange(-dy, cols)
            i_grid, j_grid = np.meshgrid(valid_i, valid_j, indexing='ij')

            row_pixels = image_slice[i_grid, j_grid]
            col_pixels = image_slice[i_grid + dx, j_grid + dy]
            valid_pairs = ~nan_mask[i_grid, j_grid] & ~nan_mask[i_grid + dx, j_grid + dy]
            np.add.at(
                glcm_slice,
                (row_pixels[valid_pairs].astype(int), col_pixels[valid_pairs].astype(int)),
                1,
            )
            return glcm_slice

        glcm_2d_matrices = []
        slice_no_of_roi_voxels = []
        for z_index in range(image.shape[2]):
            if np.all(np.isnan(image[:, :, z_index])):
                continue
            slice_no_of_roi_voxels.append(int(np.sum(~np.isnan(image[:, :, z_index]))))
            z_slice_matrices = []
            for direction_2d in ([1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0]):
                glcm = calc_2d_glcm_slice(image[:, :, z_index], direction_2d)
                z_slice_matrices.append(glcm + glcm.T)
            glcm_2d_matrices.append(z_slice_matrices)

        return np.array(glcm_2d_matrices), np.array(slice_no_of_roi_voxels, dtype=float)

    @staticmethod
    def _calc_3d_matrices(image, lvl):
        glcm_3d_matrices = []
        for direction_3d in (
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
            [0, 1, -1],
            [1, 0, 1],
            [1, 0, -1],
            [1, 1, 0],
            [1, -1, 0],
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
        ):
            co_matrix = np.zeros((lvl, lvl), dtype=np.float64)
            depth, height, width = image.shape
            min_i = max(0, -direction_3d[2])
            min_y = max(0, -direction_3d[1])
            min_x = max(0, -direction_3d[0])
            max_i = min(depth, depth - direction_3d[2])
            max_y = min(height, height - direction_3d[1])
            max_x = min(width, width - direction_3d[0])

            arr1 = image[min_i:max_i, min_y:max_y, min_x:max_x]
            arr2 = image[
                min_i + direction_3d[2] : max_i + direction_3d[2],
                min_y + direction_3d[1] : max_y + direction_3d[1],
                min_x + direction_3d[0] : max_x + direction_3d[0],
            ]
            not_nan_mask = np.logical_and(~np.isnan(arr1), ~np.isnan(arr2))
            y_cm_values = arr1[not_nan_mask].astype(int)
            x_cm_values = arr2[not_nan_mask].astype(int)

            if y_cm_values.size:
                flat_indices = y_cm_values * lvl + x_cm_values
                reverse_flat_indices = x_cm_values * lvl + y_cm_values
                co_matrix += np.bincount(
                    np.concatenate((flat_indices, reverse_flat_indices)),
                    minlength=lvl * lvl,
                ).reshape(lvl, lvl)
            glcm_3d_matrices.append(co_matrix)

        return np.array(glcm_3d_matrices)

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

    def _aggregate_feature_dicts(self, feature_dicts, weights=None):
        if not feature_dicts:
            raise DataStructureError('No GLCM matrices available for aggregation.')
        if self.slice_median:
            if weights is not None:
                raise DataStructureError('Weighted median is not supported for GLCM aggregation.')
            return {
                feature_name: float(np.median([values[feature_name] for values in feature_dicts]))
                for feature_name in GLCM_FEATURE_NAMES
            }
        return {
            feature_name: float(np.average([values[feature_name] for values in feature_dicts], weights=weights))
            for feature_name in GLCM_FEATURE_NAMES
        }

    def _calc_2d_averaged_glcm_features(self, glcm_2d_matrices, slice_no_of_roi_voxels, tot_no_of_roi_voxels):
        feature_dicts = []
        weights = []
        for slice_index in range(glcm_2d_matrices.shape[0]):
            for direction_index in range(glcm_2d_matrices.shape[1]):
                glcm_slice = self._normalize_matrix(
                    glcm_2d_matrices[slice_index][direction_index],
                    'calc_2d_averaged_glcm_features',
                )
                feature_dicts.append(self._feature_values(glcm_slice))
                if self.slice_weight:
                    if tot_no_of_roi_voxels == 0:
                        raise DataStructureError('tot_no_of_roi_voxels in calc_2d_averaged_glcm_features is zero.')
                    weights.append(slice_no_of_roi_voxels[slice_index] / tot_no_of_roi_voxels)
                else:
                    weights.append(1.0)
        return self._aggregate_feature_dicts(
            feature_dicts,
            None if self.slice_median and not self.slice_weight else weights,
        )

    def _calc_2d_slice_merged_glcm_features(self, glcm_2d_matrices, slice_no_of_roi_voxels, tot_no_of_roi_voxels):
        averaged_glcm = np.sum(glcm_2d_matrices, axis=1)
        feature_dicts = []
        weights = []
        for slice_index in range(averaged_glcm.shape[0]):
            glcm_slice = self._normalize_matrix(
                averaged_glcm[slice_index],
                'calc_2d_slice_merged_glcm_features',
            )
            feature_dicts.append(self._feature_values(glcm_slice))
            if self.slice_weight:
                if tot_no_of_roi_voxels == 0:
                    raise DataStructureError('tot_no_of_roi_voxels in calc_2d_slice_merged_glcm_features is zero.')
                weights.append(slice_no_of_roi_voxels[slice_index] / tot_no_of_roi_voxels)
            else:
                weights.append(1.0)
        return self._aggregate_feature_dicts(
            feature_dicts,
            None if self.slice_median and not self.slice_weight else weights,
        )

    def _calc_2_5d_merged_glcm_features(self, glcm_2d_matrices):
        glcm = self._normalize_matrix(
            np.sum(np.sum(glcm_2d_matrices, axis=1), axis=0),
            'calc_2_5d_merged_glcm_features',
        )
        return self._feature_values(glcm)

    def _calc_2_5d_direction_merged_glcm_features(self, glcm_2d_matrices):
        averaged_glcm = np.sum(glcm_2d_matrices, axis=0)
        feature_dicts = []
        for direction_index in range(averaged_glcm.shape[0]):
            direction_matrix = self._normalize_matrix(
                averaged_glcm[direction_index],
                'calc_2_5d_direction_merged_glcm_features',
            )
            feature_dicts.append(self._feature_values(direction_matrix))
        return self._aggregate_feature_dicts(feature_dicts)

    def _calc_3d_averaged_glcm_features(self, glcm_3d_matrices):
        feature_dicts = [
            self._feature_values(self._normalize_matrix(matrix, 'calc_3d_averaged_glcm_features'))
            for matrix in glcm_3d_matrices
        ]
        return self._aggregate_feature_dicts(feature_dicts)

    def _calc_3d_merged_glcm_features(self, glcm_3d_matrices):
        matrix = self._normalize_matrix(
            np.sum(glcm_3d_matrices, axis=0),
            'calc_3d_merged_glcm_features',
        )
        return self._feature_values(matrix)


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
        glcm = GLCM(
            aggr_dim=context.aggr_dim,
            aggr_method=context.aggr_method,
            slice_weight=context.slice_weighting,
            slice_median=context.slice_median,
        )
        feature_values = glcm.calculate_features(prepared_data.require_discretized_intensity_image().array.T)
        return {
            output_name: feature_values[base_name]
            for output_name, base_name in zip(self.output_names(context), GLCM_FEATURE_NAMES)
        }
