import numpy as np

from ..exceptions import DataStructureError
from .base import BaseFeatureGroup
from .texture_aggregation import format_texture_feature_names
from .texture_base import ZoneMatrixFeatureBase, extract_texture_values


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
    def calc_2d_gldzm_features(self):
        weights = []
        for i, matrix in enumerate(self.gldzm_2D_matrices):
            weight = 1
            if self.slice_weight:
                if self.tot_no_of_roi_voxels == 0:
                    raise DataStructureError(" Denominator is zero in calc_2d_gldzm_features.")
                weight = self.no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
            weights.append(weight)
            self._append_feature_values(self._matrix_feature_values(matrix, self.no_of_roi_voxels[i]))
        self._aggregate_feature_lists(weights)

    def calc_2_5d_gldzm_features(self):
        matrix = np.sum(self.gldzm_2D_matrices, axis=0)
        self._set_feature_values(self._matrix_feature_values(matrix, np.sum(self.no_of_roi_voxels)))

    def calc_3d_gldzm_features(self):
        self._set_feature_values(self._matrix_feature_values(self.gldzm_3D_matrix, self.tot_no_of_roi_voxels))


class GLDZMFeatureGroup(BaseFeatureGroup):
    family = 'gldzm'
    requirements = frozenset({'analysis_masks', 'discretized_intensity_image'})

    def supports(self, context):
        return True

    def output_names(self, context):
        return format_texture_feature_names(GLDZM_FEATURE_NAMES, context.aggr_dim)

    def feature_aliases(self, context):
        output_names = self.output_names(context)
        aliases = {name: name for name in output_names}
        aliases.update(dict(zip(GLDZM_FEATURE_NAMES, output_names)))
        return aliases

    def calculate(self, context, prepared_data):
        gldzm = GLDZM(
            prepared_data.require_discretized_intensity_image().array.T,
            slice_weight=context.slice_weighting,
            slice_median=context.slice_median,
        )
        morph_mask = prepared_data.require_analysis_masks().morphological_mask.array.T

        if context.aggr_dim == '3D':
            gldzm.calc_glsz_gldz_3d_matrices(morph_mask)
            gldzm.calc_3d_gldzm_features()
        else:
            gldzm.calc_glsz_gldz_2d_matrices(morph_mask)
            if context.aggr_dim == '2.5D':
                gldzm.calc_2_5d_gldzm_features()
            else:
                gldzm.calc_2d_gldzm_features()

        return dict(zip(self.output_names(context), extract_texture_values(gldzm)))
