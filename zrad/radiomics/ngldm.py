import numpy as np
from scipy.ndimage import convolve

from ..exceptions import DataStructureError
from .base import BaseFeatureGroup
from .texture_aggregation import format_texture_feature_names
from .texture_base import NGLDM_ATTRIBUTE_NAMES, TextureFeatureBase


NGLDM_FEATURE_NAMES = (
    'ngl_lde',
    'ngl_hde',
    'ngl_lgce',
    'ngl_hgce',
    'ngl_ldlge',
    'ngl_ldhge',
    'ngl_hdlge',
    'ngl_hdhge',
    'ngl_glnu',
    'ngl_glnu_norm',
    'ngl_dcnu',
    'ngl_dcnu_norm',
    'ngl_dc_perc',
    'ngl_gl_var',
    'ngl_dc_var',
    'ngl_dc_entr',
    'ngl_dc_energy',
)


class NGLDM(TextureFeatureBase):
    """Neighbouring gray level dependence matrix features."""

    def __init__(self, aggr_dim, slice_weight=False, slice_median=False):
        super().__init__(slice_weight=slice_weight, slice_median=slice_median)
        self.aggr_dim = aggr_dim

    def get_params(self):
        """Return the configuration parameters of this NGLDM calculator.

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
        """Return the NGLDM feature names produced by this calculator.

        Returns
        -------
        list of str
            Feature names defined for the NGLDM family.
        """
        return list(NGLDM_FEATURE_NAMES)

    @staticmethod
    def _map_feature_names(values):
        return dict(zip(NGLDM_FEATURE_NAMES, [values[name] for name in NGLDM_ATTRIBUTE_NAMES]))

    @staticmethod
    def _calc_3d_matrix(image, lvl):
        ngldm = np.zeros((lvl, 27), dtype=np.int64)
        valid_mask = ~np.isnan(image)
        kernel = np.ones((3, 3, 3), dtype=int)
        kernel[1, 1, 1] = 0

        for gray_level in range(lvl):
            matrix = ((image == gray_level) & valid_mask).astype(np.int64)
            if np.sum(matrix) == 0:
                continue
            neighbor_counts = convolve(matrix, kernel, mode='constant', cval=0)
            counts = neighbor_counts[matrix.astype(bool)]
            if counts.size:
                bincounts = np.bincount(counts, minlength=27)
                ngldm[gray_level, :len(bincounts)] += bincounts

        return ngldm

    @staticmethod
    def _calc_2d_matrices(image, lvl):
        ngldm_2d_matrices = []
        roi_voxel_counts = []
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]

        def calc_ngldm_slice(array):
            padded = np.pad(array, pad_width=1, mode='constant', constant_values=np.nan)
            center = padded[1:-1, 1:-1]
            neighbor_count = np.zeros_like(center, dtype=int)

            for dx, dy in offsets:
                neighbor = padded[
                    1 + dx: 1 + dx + center.shape[0],
                    1 + dy: 1 + dy + center.shape[1],
                ]
                neighbor_count += neighbor == center

            ngldm = np.zeros((lvl, 9), dtype=np.int64)
            valid = ~np.isnan(center)
            intensities = center[valid].astype(int)
            counts = neighbor_count[valid]
            np.add.at(ngldm, (intensities, counts), 1)
            return ngldm

        for z_idx in range(image.shape[2]):
            slice_ = image[:, :, z_idx]
            roi_voxel_count = int(np.count_nonzero(~np.isnan(slice_)))
            if roi_voxel_count == 0:
                continue
            roi_voxel_counts.append(roi_voxel_count)
            ngldm_2d_matrices.append(calc_ngldm_slice(slice_))

        return np.array(ngldm_2d_matrices, dtype=np.int64), np.array(roi_voxel_counts, dtype=float)

    def _calc_2d_features(self, matrices, roi_voxel_counts, total_roi_voxels):
        feature_dicts = []
        weights = []
        for slice_index, matrix in enumerate(matrices):
            if self.slice_weight:
                if total_roi_voxels == 0:
                    raise DataStructureError(' Denominator is zero in calc_2d_ngldm_features.')
                weights.append(roi_voxel_counts[slice_index] / total_roi_voxels)
            else:
                weights.append(1.0)
            feature_dicts.append(
                self._matrix_feature_values(matrix, roi_voxel_counts[slice_index], include_energy=True)
            )
        return self._aggregate_feature_dicts(
            feature_dicts,
            None if self.slice_median else weights,
            include_energy=True,
        )

    def _calc_2_5d_features(self, matrices, roi_voxel_counts):
        matrix = np.sum(matrices, axis=0)
        return self._matrix_feature_values(matrix, np.sum(roi_voxel_counts), include_energy=True)

    def _calc_3d_features(self, matrix, total_roi_voxels):
        return self._matrix_feature_values(matrix, total_roi_voxels, include_energy=True)

    def calculate_features(self, discretized_image_array):
        """Calculate NGLDM features for a prepared discretized intensity array.

        Parameters
        ----------
        discretized_image_array : numpy.ndarray
            Prepared discretized intensity array with voxels outside the ROI set
            to ``NaN``.

        Returns
        -------
        dict
            Mapping of NGLDM feature names to calculated values.
        """
        discretized_image_array = np.asarray(discretized_image_array)
        lvl = int(np.nanmax(discretized_image_array) + 1)
        total_roi_voxels = int(np.sum(~np.isnan(discretized_image_array)))

        if self.aggr_dim == '3D':
            values = self._calc_3d_features(self._calc_3d_matrix(discretized_image_array, lvl), total_roi_voxels)
        else:
            matrices, roi_voxel_counts = self._calc_2d_matrices(discretized_image_array, lvl)
            if self.aggr_dim == '2.5D':
                values = self._calc_2_5d_features(matrices, roi_voxel_counts)
            else:
                values = self._calc_2d_features(matrices, roi_voxel_counts, total_roi_voxels)
        return self._map_feature_names(values)


class NGLDMFeatureGroup(BaseFeatureGroup):
    family = 'ngldm'
    requirements = frozenset({'analysis_masks', 'discretized_intensity_image'})

    def supports(self, context):
        return True

    def output_names(self, context):
        return format_texture_feature_names(NGLDM_FEATURE_NAMES, context.aggr_dim)

    def feature_aliases(self, context):
        output_names = self.output_names(context)
        aliases = {name: name for name in output_names}
        aliases.update(dict(zip(NGLDM_FEATURE_NAMES, output_names)))
        return aliases

    def calculate(self, context, prepared_data):
        ngldm = NGLDM(
            aggr_dim=context.aggr_dim,
            slice_weight=context.slice_weighting,
            slice_median=context.slice_median,
        )
        feature_values = ngldm.calculate_features(
            prepared_data.require_discretized_intensity_image().array.T
        )
        return {
            output_name: feature_values[base_name]
            for output_name, base_name in zip(self.output_names(context), NGLDM_FEATURE_NAMES)
        }
