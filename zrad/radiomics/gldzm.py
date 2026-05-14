import numpy as np

from ..exceptions import DataStructureError
from .base import BaseFeatureGroup
from .texture_aggregation import format_texture_feature_names
from .texture_base import TEXTURE_ATTRIBUTE_NAMES, ZoneMatrixFeatureBase


GLDZM_FEATURE_NAMES = (
    'dzm_sde',
    'dzm_lde',
    'dzm_lgze',
    'dzm_hgze',
    'dzm_sdlge',
    'dzm_sdhge',
    'dzm_ldlge',
    'dzm_ldhge',
    'dzm_glnu',
    'dzm_glnu_norm',
    'dzm_zdnu',
    'dzm_zdnu_norm',
    'dzm_z_perc',
    'dzm_gl_var',
    'dzm_zd_var',
    'dzm_zd_entr',
)


class GLDZM(ZoneMatrixFeatureBase):
    """Gray level distance zone matrix features.

    GLDZM features describe connected grey-level zones together with their
    distance from the ROI border. They summarize how grey-level zones are
    distributed from boundary-adjacent to deeper ROI regions.

    Parameters
    ----------
    aggr_dim : {"2D", "2.5D", "3D"}
        Spatial dimensionality used to define zones and border distances.
    slice_weight : bool, default=False
        Weight 2D slice-wise averages by slice ROI voxel count.
    slice_median : bool, default=False
        Aggregate 2D slice-wise values by median instead of mean.
    """

    def __init__(self, aggr_dim, slice_weight=False, slice_median=False):
        super().__init__(slice_weight=slice_weight, slice_median=slice_median)
        self.aggr_dim = aggr_dim

    def get_params(self):
        """Return the configuration parameters of this GLDZM calculator.

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
        """Return the GLDZM feature names produced by this calculator.

        Returns
        -------
        list of str
            Feature names defined for the GLDZM family.
        """
        return list(GLDZM_FEATURE_NAMES)

    @staticmethod
    def _map_feature_names(values):
        return dict(zip(GLDZM_FEATURE_NAMES, [values[name] for name in TEXTURE_ATTRIBUTE_NAMES]))

    def _calc_2d_features(self, matrices, roi_voxel_counts, total_roi_voxels):
        feature_dicts = []
        weights = []
        for slice_index, matrix in enumerate(matrices):
            if self.slice_weight:
                if total_roi_voxels == 0:
                    raise DataStructureError(' Denominator is zero in calc_2d_gldzm_features.')
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

    def calculate_features(self, discretized_image_array, mask_array):
        """Calculate GLDZM features for prepared discretized intensities and a morphology mask.

        Parameters
        ----------
        discretized_image_array : numpy.ndarray
            Prepared discretized intensity array with voxels outside the ROI set
            to ``NaN``.
        mask_array : numpy.ndarray
            Morphological ROI mask aligned with ``discretized_image_array``.

        Returns
        -------
        dict
            Mapping of GLDZM feature names to calculated values.
        """
        discretized_image_array = np.asarray(discretized_image_array)
        mask_array = np.asarray(mask_array)
        lvl = int(np.nanmax(discretized_image_array) + 1)

        if self.aggr_dim == '3D':
            gldzm_matrix, total_roi_voxels = self._calc_gldz_3d_matrix(discretized_image_array, mask_array, lvl)
            return self._map_feature_names(self._calc_3d_features(gldzm_matrix, total_roi_voxels))

        gldzm_matrices, roi_voxel_counts = self._calc_gldz_2d_matrices(discretized_image_array, mask_array, lvl)
        total_roi_voxels = np.sum(roi_voxel_counts)
        if self.aggr_dim == '2.5D':
            values = self._calc_2_5d_features(gldzm_matrices, roi_voxel_counts)
        else:
            values = self._calc_2d_features(gldzm_matrices, roi_voxel_counts, total_roi_voxels)
        return self._map_feature_names(values)


class GLDZMFeatureGroup(BaseFeatureGroup):
    family = 'gldzm'
    requirements = frozenset({'analysis_masks', 'discretized_intensity_image'})

    def supports(self, context):
        return context.roi_data.texture_discretized_image is not None

    def output_names(self, context):
        return format_texture_feature_names(GLDZM_FEATURE_NAMES, context.aggr_dim)

    def feature_aliases(self, context):
        output_names = self.output_names(context)
        aliases = {name: name for name in output_names}
        aliases.update(dict(zip(GLDZM_FEATURE_NAMES, output_names)))
        return aliases

    def calculate(self, context, prepared_data):
        gldzm = GLDZM(
            aggr_dim=context.aggr_dim,
            slice_weight=context.slice_weighting,
            slice_median=context.slice_median,
        )
        feature_values = gldzm.calculate_features(
            prepared_data.require_discretized_intensity_image().array.T,
            prepared_data.require_analysis_masks().morphological_mask.array.T,
        )
        return {
            output_name: feature_values[base_name]
            for output_name, base_name in zip(self.output_names(context), GLDZM_FEATURE_NAMES)
        }
