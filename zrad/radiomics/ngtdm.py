import numpy as np
from scipy.ndimage import convolve

from ..exceptions import DataStructureError
from .base import BaseFeatureGroup
from .texture_aggregation import format_texture_feature_names
from .texture_matrices import TextureMatrixMixin

NGTDM_FEATURE_NAMES = (
    'ngt_coarseness',
    'ngt_contrast',
    'ngt_busyness',
    'ngt_complexity',
    'ngt_strength',
)


class NGTDM(TextureMatrixMixin):
    """Neighbouring grey tone difference matrix features.

    NGTDM features compare each discretized grey level with the average grey
    level in its local neighbourhood. They quantify coarseness, contrast,
    busyness, complexity, and strength.

    Parameters
    ----------
    aggr_dim : {"2D", "2.5D", "3D"}
        Spatial dimensionality used to build neighbouring difference matrices.
    slice_weight : bool, default=False
        Weight 2D slice-wise averages by slice ROI voxel count.
    slice_median : bool, default=False
        Aggregate 2D slice-wise values by median instead of mean.
    """

    matrix_family = 'ngtdm'

    def __init__(self, aggr_dim, slice_weight=False, slice_median=False):
        self.aggr_dim = aggr_dim
        self.slice_weight = slice_weight
        self.slice_median = slice_median

    def get_params(self):
        """Return the configuration parameters of this NGTDM calculator.

        Returns
        -------
        dict
            Parameter names mapped to their configured values.
        """
        return {
            'aggr_dim': self.aggr_dim,
            'slice_weight': self.slice_weight,
            'slice_median': self.slice_median,
        }

    def get_feature_names(self):
        """Return the NGTDM feature names produced by this calculator.

        Returns
        -------
        list of str
            Feature names defined for the NGTDM family.
        """
        return list(NGTDM_FEATURE_NAMES)

    @staticmethod
    def _calc_3d_matrix(image, lvl):
        valid = ~np.isnan(image)
        img_filled = np.where(valid, image, 0.0)
        kernel = np.ones((3, 3, 3), dtype=np.int8)
        kernel[1, 1, 1] = 0

        neighbor_sum = convolve(img_filled, kernel, mode='constant', cval=0.0)
        neighbor_count = convolve(valid.astype(np.int8), kernel, mode='constant', cval=0)
        ngtdm = np.zeros((lvl, 2), dtype=np.float64)

        for gray_level in range(lvl):
            mask_lvl = image == gray_level
            mask_good = mask_lvl & (neighbor_count > 0)
            n_i = np.count_nonzero(mask_good)
            if n_i > 0:
                mean_nb = neighbor_sum[mask_good] / neighbor_count[mask_good]
                s_i = np.sum(np.abs(gray_level - mean_nb))
            else:
                s_i = 0.0
            ngtdm[gray_level, 0] = n_i
            ngtdm[gray_level, 1] = s_i

        return ngtdm

    @staticmethod
    def _calc_2d_matrices(image, lvl):
        kernel2d = np.ones((3, 3), dtype=np.int8)
        kernel2d[1, 1] = 0
        range_z = np.unique(np.where(~np.isnan(image))[2])

        slice_matrices = []
        slice_voxel_counts = []

        for z_index in range_z:
            z_slice = image[:, :, z_index]
            valid = ~np.isnan(z_slice)
            roi_voxel_count = int(valid.sum())
            if roi_voxel_count == 0:
                continue
            slice_voxel_counts.append(roi_voxel_count)
            filled = np.where(valid, z_slice, 0.0)
            neighbor_sum = convolve(filled, kernel2d, mode='constant', cval=0.0)
            neighbor_count = convolve(valid.astype(np.int8), kernel2d, mode='constant', cval=0)

            ngtdm_slice = np.zeros((lvl, 2), dtype=np.float64)
            for gray_level in range(lvl):
                mask = (z_slice == gray_level) & (neighbor_count > 0)
                n_i = mask.sum()
                if n_i > 0:
                    mean_nb = neighbor_sum[mask] / neighbor_count[mask]
                    s_i = np.abs(gray_level - mean_nb).sum()
                else:
                    s_i = 0.0
                ngtdm_slice[gray_level, 0] = n_i
                ngtdm_slice[gray_level, 1] = s_i

            slice_matrices.append(ngtdm_slice)

        return np.array(slice_matrices), np.array(slice_voxel_counts, dtype=float)

    @staticmethod
    def _calc_coarseness(matrix):
        num = np.sum(matrix[:, 0])
        denum = np.sum(matrix[:, 0] * matrix[:, 1])
        return 1_000_000 if denum == 0 else num / denum

    @staticmethod
    def _calc_contrast(matrix):
        n = np.sum(matrix[:, 0])
        if n == 0:
            raise DataStructureError(' Denominator is zero in calc_contrast.')
        n_g = np.sum(matrix[:, 0] != 0)
        s_1 = 0.0
        s_2 = np.sum(matrix[:, 1])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[0]):
                s_1 += (matrix[i, 0] * matrix[j, 0] * (i - j) ** 2) / (n**2)
        denum = n_g * (n_g - 1) * np.sum(matrix[:, 0])
        return 0 if denum == 0 else (s_1 * s_2) / denum

    @staticmethod
    def _calc_busyness(matrix):
        n = np.sum(matrix[:, 0])
        if n == 0:
            raise DataStructureError(' Denominator is zero in calc_busyness.')
        num = 0.0
        denum = 0.0
        for i in range(matrix.shape[0]):
            num += (matrix[i, 0] * matrix[i, 1]) / n
            for j in range(matrix.shape[0]):
                if matrix[i, 0] != 0 and matrix[j, 0] != 0:
                    denum += abs(i * matrix[i, 0] - j * matrix[j, 0]) / n
        return 0 if denum == 0 else num / denum

    @staticmethod
    def _calc_complexity(matrix):
        n = np.sum(matrix[:, 0])
        if n == 0:
            return 0
        sum_compl = 0.0
        for i in range(matrix.shape[0]):
            p_i, s_i = matrix[i, 0], matrix[i, 1]
            if p_i == 0:
                continue
            for j in range(matrix.shape[0]):
                p_j, s_j = matrix[j, 0], matrix[j, 1]
                if p_j == 0:
                    continue
                num = (p_i * s_i + p_j * s_j) * abs(i - j) / n
                den = (p_i + p_j) / n
                sum_compl += num / den
        return sum_compl / n

    @staticmethod
    def _calc_strength(matrix):
        n = np.sum(matrix[:, 0])
        if n == 0:
            raise DataStructureError(' Denominator is zero in calc_strength.')
        num = 0.0
        denum = np.sum(matrix[:, 1])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[0]):
                if matrix[i, 0] != 0 and matrix[j, 0] != 0:
                    num += ((matrix[i, 0] + matrix[j, 0]) * (i - j) ** 2) / n
        return 0 if denum == 0 else num / denum

    @classmethod
    def _matrix_feature_values(cls, matrix):
        return {
            'ngt_coarseness': cls._calc_coarseness(matrix),
            'ngt_contrast': cls._calc_contrast(matrix),
            'ngt_busyness': cls._calc_busyness(matrix),
            'ngt_complexity': cls._calc_complexity(matrix),
            'ngt_strength': cls._calc_strength(matrix),
        }

    def calculate_features(self, discretized_image_array):
        """Calculate NGTDM features for a prepared discretized intensity array.

        Parameters
        ----------
        discretized_image_array : numpy.ndarray
            Legacy (x, y, z) array. Use calculate_matrices for NumPy (z, y, x)
            or 2D (row, column) inputs. Prepared discretized intensity array with voxels outside the ROI set
            to ``NaN``.

        Returns
        -------
        dict
            Mapping of NGTDM feature names to calculated values.
        """
        return self._legacy_features(discretized_image_array)


class NGTDMFeatureGroup(BaseFeatureGroup):
    family = 'ngtdm'
    requirements = frozenset({'analysis_masks', 'discretized_intensity_image'})

    def supports(self, context):
        return context.roi_data.texture_discretized_image is not None

    def output_names(self, context):
        return format_texture_feature_names(NGTDM_FEATURE_NAMES, context.aggr_dim)

    def feature_aliases(self, context):
        output_names = self.output_names(context)
        aliases = {name: name for name in output_names}
        aliases.update(dict(zip(NGTDM_FEATURE_NAMES, output_names)))
        return aliases

    def calculate(self, context, prepared_data):
        from .texture_extraction import calculate_texture_family

        return calculate_texture_family(self, context, prepared_data)[0]
