import numpy as np

from ..exceptions import DataStructureError
from .base import BaseFeatureGroup
from .texture_aggregation import format_texture_feature_names
from .texture_base import TEXTURE_ATTRIBUTE_NAMES, ZoneMatrixFeatureBase


GLSZM_FEATURE_NAMES = (
    'szm_sze',
    'szm_lze',
    'szm_lgze',
    'szm_hgze',
    'szm_szlge',
    'szm_szhge',
    'szm_lzlge',
    'szm_lzhge',
    'szm_glnu',
    'szm_glnu_norm',
    'szm_zsnu',
    'szm_zsnu_norm',
    'szm_z_perc',
    'szm_gl_var',
    'szm_zs_var',
    'szm_zs_entr',
)


class GLSZM(ZoneMatrixFeatureBase):
    """Gray level size zone matrix features."""

    def __init__(self, aggr_dim, slice_weight=False, slice_median=False):
        super().__init__(slice_weight=slice_weight, slice_median=slice_median)
        self.aggr_dim = aggr_dim

    def get_params(self):
        """Return the configuration parameters of this GLSZM calculator.

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
        """Return the GLSZM feature names produced by this calculator.

        Returns
        -------
        list of str
            Feature names defined for the GLSZM family.
        """
        return list(GLSZM_FEATURE_NAMES)

    @staticmethod
    def _map_feature_names(values):
        return dict(zip(GLSZM_FEATURE_NAMES, [values[name] for name in TEXTURE_ATTRIBUTE_NAMES]))

    def _calc_2d_features(self, matrices, roi_voxel_counts, total_roi_voxels):
        feature_dicts = []
        weights = []
        for slice_index, matrix in enumerate(matrices):
            if self.slice_weight:
                if total_roi_voxels == 0:
                    raise DataStructureError(' Denominator is zero in calc_2d_glszm_features.')
                weights.append(roi_voxel_counts[slice_index] / total_roi_voxels)
            else:
                weights.append(1.0)
            feature_dicts.append(self._matrix_feature_values(matrix, roi_voxel_counts[slice_index]))
        return self._aggregate_feature_dicts(feature_dicts, None if self.slice_median else weights)

    def _calc_2_5d_features(self, matrices, roi_voxel_counts):
        matrix = np.sum(matrices, axis=0)
        return self._matrix_feature_values(matrix, np.sum(roi_voxel_counts))

    def _calc_3d_features(self, matrix, total_roi_voxels):
        return self._matrix_feature_values(matrix, total_roi_voxels)

    def calculate_features(self, discretized_image_array):
        """Calculate GLSZM features for prepared discretized intensities.

        Parameters
        ----------
        discretized_image_array : numpy.ndarray
            Prepared discretized intensity array with voxels outside the ROI set
            to ``NaN``.

        Returns
        -------
        dict
            Mapping of GLSZM feature names to calculated values.
        """
        discretized_image_array = np.asarray(discretized_image_array)
        lvl = int(np.nanmax(discretized_image_array) + 1)

        if self.aggr_dim == '3D':
            glszm_matrix, total_roi_voxels = self._calc_glsz_3d_matrix(discretized_image_array, lvl)
            return self._map_feature_names(self._calc_3d_features(glszm_matrix, total_roi_voxels))

        glszm_matrices, roi_voxel_counts = self._calc_glsz_2d_matrices(discretized_image_array, lvl)
        total_roi_voxels = np.sum(roi_voxel_counts)
        if self.aggr_dim == '2.5D':
            values = self._calc_2_5d_features(glszm_matrices, roi_voxel_counts)
        else:
            values = self._calc_2d_features(glszm_matrices, roi_voxel_counts, total_roi_voxels)
        return self._map_feature_names(values)


class GLSZMFeatureGroup(BaseFeatureGroup):
    family = 'glszm'
    requirements = frozenset({'analysis_masks', 'discretized_intensity_image'})

    def supports(self, context):
        return True

    def output_names(self, context):
        return format_texture_feature_names(GLSZM_FEATURE_NAMES, context.aggr_dim)

    def feature_aliases(self, context):
        output_names = self.output_names(context)
        aliases = {name: name for name in output_names}
        aliases.update(dict(zip(GLSZM_FEATURE_NAMES, output_names)))
        return aliases

    def calculate(self, context, prepared_data):
        glszm = GLSZM(
            aggr_dim=context.aggr_dim,
            slice_weight=context.slice_weighting,
            slice_median=context.slice_median,
        )
        feature_values = glszm.calculate_features(
            prepared_data.require_discretized_intensity_image().array.T
        )
        return {
            output_name: feature_values[base_name]
            for output_name, base_name in zip(self.output_names(context), GLSZM_FEATURE_NAMES)
        }
