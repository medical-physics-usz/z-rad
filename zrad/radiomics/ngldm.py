import numpy as np
from scipy.ndimage import convolve

from ..exceptions import DataStructureError
from .base import BaseFeatureGroup
from .texture_aggregation import format_texture_feature_names
from .texture_base import TextureFeatureBase, extract_ngldm_values


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
    def calc_ngld_3d_matrix(self):
        ngldm = np.zeros((self.lvl, 27), dtype=np.int64)
        valid_mask = ~np.isnan(self.image)
        kernel = np.ones((3, 3, 3), dtype=int)
        kernel[1, 1, 1] = 0

        for lvl in range(self.lvl):
            matrix = ((self.image == lvl) & valid_mask).astype(np.int64)
            if np.sum(matrix) == 0:
                continue
            neighbor_counts = convolve(matrix, kernel, mode='constant', cval=0)
            counts = neighbor_counts[matrix.astype(bool)]
            if counts.size:
                bincounts = np.bincount(counts, minlength=27)
                ngldm[lvl, :len(bincounts)] += bincounts

        self.ngldm_3D_matrix = ngldm

    def calc_ngld_2d_matrices(self):
        self.ngldm_2d_matrices = []
        self.no_of_roi_voxels = []
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

            ngldm = np.zeros((self.lvl, 9), dtype=int)
            valid = ~np.isnan(center)
            intensities = center[valid].astype(int)
            counts = neighbor_count[valid]
            np.add.at(ngldm, (intensities, counts), 1)
            return ngldm

        for z_idx in range(self.image.shape[2]):
            slice_ = self.image[:, :, z_idx]
            if np.any(~np.isnan(slice_)):
                self.no_of_roi_voxels.append(np.count_nonzero(~np.isnan(slice_)))
                self.ngldm_2d_matrices.append(calc_ngldm_slice(slice_))

        self.ngldm_2d_matrices = np.array(self.ngldm_2d_matrices, dtype=np.int64)

    def calc_2d_ngldm_features(self):
        weights = []
        for i, matrix in enumerate(self.ngldm_2d_matrices):
            weight = 1
            if self.slice_weight:
                if self.tot_no_of_roi_voxels == 0:
                    raise DataStructureError(" Denominator is zero in calc_2d_ngldm_features.")
                weight = self.no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
            weights.append(weight)
            self._append_feature_values(
                self._matrix_feature_values(matrix, self.no_of_roi_voxels[i], include_energy=True)
            )

        self._aggregate_feature_lists(weights, include_energy=True)

    def calc_2_5d_ngldm_features(self):
        matrix = np.sum(self.ngldm_2d_matrices, axis=0)
        self._set_feature_values(
            self._matrix_feature_values(matrix, np.sum(self.no_of_roi_voxels), include_energy=True)
        )

    def calc_3d_ngldm_features(self):
        self._set_feature_values(
            self._matrix_feature_values(self.ngldm_3D_matrix, self.tot_no_of_roi_voxels, include_energy=True)
        )


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
            prepared_data.require_discretized_intensity_image().array.T,
            slice_weight=context.slice_weighting,
            slice_median=context.slice_median,
        )
        if context.aggr_dim == '3D':
            ngldm.calc_ngld_3d_matrix()
            ngldm.calc_3d_ngldm_features()
        elif context.aggr_dim == '2.5D':
            ngldm.calc_ngld_2d_matrices()
            ngldm.calc_2_5d_ngldm_features()
        else:
            ngldm.calc_ngld_2d_matrices()
            ngldm.calc_2d_ngldm_features()

        return dict(zip(self.output_names(context), extract_ngldm_values(ngldm)))
