import numpy as np

from ..exceptions import DataStructureError
from .base import BaseFeatureGroup
from .texture_aggregation import format_cm_rlm_feature_names
from .texture_base import TextureFeatureBase, extract_texture_values


GLRLM_FEATURE_NAMES = (
    'rlm_sre',
    'rlm_lre',
    'rlm_lgre',
    'rlm_hgre',
    'rlm_srlge',
    'rlm_srhge',
    'rlm_lrlge',
    'rlm_lrhge',
    'rlm_glnu',
    'rlm_glnu_norm',
    'rlm_rlnu',
    'rlm_rlnu_norm',
    'rlm_r_perc',
    'rlm_gl_var',
    'rlm_rl_var',
    'rlm_rl_entr',
)


class GLRLM(TextureFeatureBase):
    def rle_1d(self, arr, lvl, rlm):
        valid_idx = np.where(~np.isnan(arr))[0]
        if valid_idx.size == 0:
            return
        splits = np.where(np.diff(valid_idx) != 1)[0] + 1
        segments = np.split(valid_idx, splits)

        for seg in segments:
            seg_vals = arr[seg]
            n = seg_vals.size
            if n == 0:
                continue
            diff = np.diff(seg_vals)
            run_breaks = np.where(diff != 0)[0] + 1
            run_starts = np.concatenate(([0], run_breaks))
            run_ends = np.concatenate((run_breaks, [n]))
            run_lengths = run_ends - run_starts

            for start, run_len in zip(run_starts, run_lengths):
                if run_len - 1 < rlm.shape[1]:
                    gray = int(seg_vals[start])
                    rlm[gray, run_len - 1] += 1

    def process_horizontal(self, z_slice, lvl):
        rows, cols = z_slice.shape
        rlm = np.zeros((lvl, max(rows, cols)), dtype=int)
        for i in range(rows):
            self.rle_1d(z_slice[i, :], lvl, rlm)
        return rlm

    def process_vertical(self, z_slice, lvl):
        rows, cols = z_slice.shape
        rlm = np.zeros((lvl, max(rows, cols)), dtype=int)
        for j in range(cols):
            self.rle_1d(z_slice[:, j], lvl, rlm)
        return rlm

    def process_diagonal(self, z_slice, lvl):
        rows, cols = z_slice.shape
        rlm = np.zeros((lvl, max(rows, cols)), dtype=int)
        for offset in range(-rows + 1, cols):
            self.rle_1d(np.diagonal(z_slice, offset=offset), lvl, rlm)
        return rlm

    def process_antidiagonal(self, z_slice, lvl):
        rows, cols = z_slice.shape
        rlm = np.zeros((lvl, max(rows, cols)), dtype=int)
        flipped = np.fliplr(z_slice)
        for offset in range(-rows + 1, cols):
            self.rle_1d(np.diagonal(flipped, offset=offset), lvl, rlm)
        return rlm

    def calc_glrl_2d_matrices(self):
        direction_funcs = [
            self.process_horizontal,
            self.process_vertical,
            self.process_diagonal,
            self.process_antidiagonal,
        ]

        glrlm_2d_matrices = []
        no_of_roi_voxels = []
        for z_slice_index in self.range_z:
            z_slice = self.image[:, :, z_slice_index]
            no_of_roi_voxels.append(np.count_nonzero(~np.isnan(z_slice)))
            slice_rlms = [func(z_slice, self.lvl) for func in direction_funcs]
            glrlm_2d_matrices.append(slice_rlms)

        self.glrlm_2D_matrices = np.array(glrlm_2d_matrices, dtype=np.int64)
        self.no_of_roi_voxels = no_of_roi_voxels

    def calc_glrl_3d_matrix(self):
        x, y, z = self.image.shape
        directions = np.array([
            (0, 0, 1), (0, 1, -1), (0, 1, 0),
            (0, 1, 1), (1, -1, -1), (1, -1, 0),
            (1, -1, 1), (1, 0, -1), (1, 0, 0),
            (1, 0, 1), (1, 1, -1), (1, 1, 0),
            (1, 1, 1),
        ])

        max_dim = max(x, y, z)
        self.glrlm_3D_matrix = np.zeros((len(directions), self.lvl, max_dim), dtype=np.int64)
        nan_mask = np.isnan(self.image)

        for d_idx, (dx, dy, dz) in enumerate(directions):
            rlm = np.zeros((self.lvl, max_dim), dtype=np.int64)
            visited = np.zeros((x, y, z), dtype=bool)
            i_idx, j_idx, k_idx = np.where(~nan_mask)

            for i, j, k in zip(i_idx, j_idx, k_idx):
                if visited[i, j, k]:
                    continue

                gr_lvl = int(self.image[i, j, k])
                run_len = 1
                visited[i, j, k] = True
                new_i, new_j, new_k = i + dx, j + dy, k + dz
                while (
                    0 <= new_i < x and 0 <= new_j < y and 0 <= new_k < z
                    and self.image[new_i, new_j, new_k] == gr_lvl
                    and not visited[new_i, new_j, new_k]
                    and not nan_mask[new_i, new_j, new_k]
                ):
                    visited[new_i, new_j, new_k] = True
                    run_len += 1
                    new_i += dx
                    new_j += dy
                    new_k += dz

                rlm[gr_lvl, run_len - 1] += 1

            self.glrlm_3D_matrix[d_idx] = rlm

        self.glrlm_3D_matrix = self.glrlm_3D_matrix.astype(np.int64)

    def calc_2d_averaged_glrlm_features(self):
        weights = []
        for i in range(self.glrlm_2D_matrices.shape[0]):
            for matrix in self.glrlm_2D_matrices[i]:
                weight = 1
                if self.slice_weight:
                    if self.tot_no_of_roi_voxels == 0:
                        raise DataStructureError(" Denominator is zero in calc_2d_averaged_glrlm_features.")
                    weight = self.no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
                weights.append(weight)
                self._append_feature_values(self._matrix_feature_values(matrix, self.no_of_roi_voxels[i]))

        self._aggregate_feature_lists(weights)

    def calc_2d_slice_merged_glrlm_features(self):
        number_of_directions = self.glrlm_2D_matrices.shape[1]
        if number_of_directions == 0:
            raise DataStructureError(" Denominator is zero in calc_2d_slice_merged_glrlm_features. ")

        averaged_matrices = np.sum(self.glrlm_2D_matrices, axis=1)
        weights = []
        for i, matrix in enumerate(averaged_matrices):
            weight = 1
            if self.slice_weight:
                if self.tot_no_of_roi_voxels == 0:
                    raise DataStructureError(" Denominator is zero in calc_2d_slice_merged_glrlm_features. ")
                weight = self.no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
            weights.append(weight)
            self._append_feature_values(
                self._matrix_feature_values(matrix, self.no_of_roi_voxels[i] * number_of_directions)
            )

        self._aggregate_feature_lists(weights)

    def calc_2_5d_merged_glrlm_features(self):
        number_of_directions = self.glrlm_2D_matrices.shape[1]
        if number_of_directions == 0:
            raise DataStructureError(" Denominator is zero in calc_2_5d_merged_glrlm_features.")
        matrix = np.sum(np.sum(self.glrlm_2D_matrices, axis=1), axis=0)
        self._set_feature_values(
            self._matrix_feature_values(matrix, np.sum(self.no_of_roi_voxels) * number_of_directions)
        )

    def calc_2_5d_direction_merged_glrlm_features(self):
        number_of_directions = self.glrlm_2D_matrices.shape[1]
        if number_of_directions == 0:
            raise DataStructureError(" Denominator is zero in calc_2_5d_direction_merged_glrlm_features.")
        averaged_glrlm = np.sum(self.glrlm_2D_matrices, axis=0)
        values = [
            self._matrix_feature_values(matrix, np.sum(self.no_of_roi_voxels))
            for matrix in averaged_glrlm
        ]
        self._set_feature_values(self._average_feature_values(values))

    def calc_3d_averaged_glrlm_features(self):
        number_of_directions = self.glrlm_3D_matrix.shape[0]
        if number_of_directions == 0:
            raise DataStructureError(" Denominator is zero in calc_3d_averaged_glrlm_features.")
        values = [
            self._matrix_feature_values(matrix, self.tot_no_of_roi_voxels)
            for matrix in self.glrlm_3D_matrix
        ]
        self._set_feature_values(self._average_feature_values(values))

    def calc_3d_merged_glrlm_features(self):
        number_of_directions = self.glrlm_3D_matrix.shape[0]
        if number_of_directions == 0:
            raise DataStructureError(" Denominator is zero in calc_3d_merged_glrlm_features.")
        matrix = np.sum(self.glrlm_3D_matrix, axis=0)
        self._set_feature_values(
            self._matrix_feature_values(matrix, self.tot_no_of_roi_voxels * number_of_directions)
        )


class GLRLMFeatureGroup(BaseFeatureGroup):
    family = 'glrlm'
    requirements = frozenset({'analysis_masks', 'discretized_intensity_image'})

    def supports(self, context):
        return True

    def output_names(self, context):
        return format_cm_rlm_feature_names(GLRLM_FEATURE_NAMES, context.aggr_dim, context.aggr_method)

    def feature_aliases(self, context):
        output_names = self.output_names(context)
        aliases = {name: name for name in output_names}
        aliases.update(dict(zip(GLRLM_FEATURE_NAMES, output_names)))
        return aliases

    def calculate(self, context, prepared_data):
        glrlm = GLRLM(
            prepared_data.require_discretized_intensity_image().array.T,
            slice_weight=context.slice_weighting,
            slice_median=context.slice_median,
        )
        if context.aggr_dim == '3D':
            glrlm.calc_glrl_3d_matrix()
            if context.aggr_method == 'AVER':
                glrlm.calc_3d_averaged_glrlm_features()
            elif context.aggr_method == 'MERG':
                glrlm.calc_3d_merged_glrlm_features()
        else:
            glrlm.calc_glrl_2d_matrices()
            if context.aggr_method == 'DIR_MERG':
                glrlm.calc_2_5d_direction_merged_glrlm_features()
            elif context.aggr_method == 'MERG':
                glrlm.calc_2_5d_merged_glrlm_features()
            elif context.aggr_method == 'AVER':
                glrlm.calc_2d_averaged_glrlm_features()
            elif context.aggr_method == 'SLICE_MERG':
                glrlm.calc_2d_slice_merged_glrlm_features()

        return dict(zip(self.output_names(context), extract_texture_values(glrlm)))
