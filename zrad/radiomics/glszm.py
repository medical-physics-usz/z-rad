import numpy as np

from ..exceptions import DataStructureError
from .base import BaseFeatureGroup
from .texture_aggregation import format_texture_feature_names
from .texture_base import ZoneMatrixFeatureBase, extract_texture_values


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
    def calc_2d_glszm_features(self):
        weights = []
        for i, matrix in enumerate(self.glszm_2D_matrices):
            weight = 1
            if self.slice_weight:
                if self.tot_no_of_roi_voxels == 0:
                    raise DataStructureError(" Denominator is zero in calc_2d_glszm_features.")
                weight = self.no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
            weights.append(weight)
            self._append_feature_values(self._matrix_feature_values(matrix, self.no_of_roi_voxels[i]))
        self._aggregate_feature_lists(weights)

    def calc_2_5d_glszm_features(self):
        matrix = np.sum(self.glszm_2D_matrices, axis=0)
        self._set_feature_values(self._matrix_feature_values(matrix, np.sum(self.no_of_roi_voxels)))

    def calc_3d_glszm_features(self):
        self._set_feature_values(self._matrix_feature_values(self.glszm_3D_matrix, self.tot_no_of_roi_voxels))


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
            prepared_data.require_discretized_intensity_image().array.T,
            slice_weight=context.slice_weighting,
            slice_median=context.slice_median,
        )
        morph_mask = prepared_data.require_analysis_masks().morphological_mask.array.T

        if context.aggr_dim == '3D':
            glszm.calc_glsz_gldz_3d_matrices(morph_mask)
            glszm.calc_3d_glszm_features()
        else:
            glszm.calc_glsz_gldz_2d_matrices(morph_mask)
            if context.aggr_dim == '2.5D':
                glszm.calc_2_5d_glszm_features()
            else:
                glszm.calc_2d_glszm_features()

        return dict(zip(self.output_names(context), extract_texture_values(glszm)))
