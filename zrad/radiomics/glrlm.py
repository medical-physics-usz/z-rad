import numpy as np

from ..exceptions import DataStructureError
from .base import BaseFeatureGroup
from .texture_aggregation import format_cm_rlm_feature_names
from .texture_base import TEXTURE_ATTRIBUTE_NAMES, TextureFeatureBase, crop_to_valid_bbox

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
    """Gray level run length matrix features.

    GLRLM features describe contiguous runs of equal discretized grey level
    along predefined directions. They capture coarse versus fine texture and
    low- versus high-grey-level run patterns.

    Parameters
    ----------
    aggr_dim : {"2D", "2.5D", "3D"}
        Spatial dimensionality used to build run length matrices.
    aggr_method : {"MERG", "AVER", "SLICE_MERG", "DIR_MERG"}
        Strategy used to combine matrices across directions and slices.
    slice_weight : bool, default=False
        Weight slice-wise averages by slice ROI voxel count.
    slice_median : bool, default=False
        Aggregate slice-wise values by median instead of mean.
    """

    def __init__(self, aggr_dim, aggr_method, slice_weight=False, slice_median=False):
        super().__init__(slice_weight=slice_weight, slice_median=slice_median)
        self.aggr_dim = aggr_dim
        self.aggr_method = aggr_method

    def get_params(self):
        """Return the configuration parameters of this GLRLM calculator.

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
        """Return the GLRLM feature names produced by this calculator.

        Returns
        -------
        list of str
            Feature names defined for the GLRLM family.
        """
        return list(GLRLM_FEATURE_NAMES)

    @staticmethod
    def _map_feature_names(values):
        return dict(zip(GLRLM_FEATURE_NAMES, [values[name] for name in TEXTURE_ATTRIBUTE_NAMES]))

    @staticmethod
    def _rle_1d(arr, lvl, rlm):
        valid_idx = np.where(~np.isnan(arr))[0]
        if valid_idx.size == 0:
            return
        splits = np.where(np.diff(valid_idx) != 1)[0] + 1
        segments = np.split(valid_idx, splits)

        for seg in segments:
            seg_vals = arr[seg]
            if seg_vals.size == 0:
                continue
            diff = np.diff(seg_vals)
            run_breaks = np.where(diff != 0)[0] + 1
            run_starts = np.concatenate(([0], run_breaks))
            run_ends = np.concatenate((run_breaks, [seg_vals.size]))
            run_lengths = run_ends - run_starts

            for start, run_len in zip(run_starts, run_lengths):
                if run_len - 1 < rlm.shape[1]:
                    gray = int(seg_vals[start])
                    rlm[gray, run_len - 1] += 1

    @classmethod
    def _process_horizontal(cls, z_slice, lvl):
        rows, cols = z_slice.shape
        rlm = np.zeros((lvl, max(rows, cols)), dtype=np.int64)
        for i in range(rows):
            cls._rle_1d(z_slice[i, :], lvl, rlm)
        return rlm

    @classmethod
    def _process_vertical(cls, z_slice, lvl):
        rows, cols = z_slice.shape
        rlm = np.zeros((lvl, max(rows, cols)), dtype=np.int64)
        for j in range(cols):
            cls._rle_1d(z_slice[:, j], lvl, rlm)
        return rlm

    @classmethod
    def _process_diagonal(cls, z_slice, lvl):
        rows, cols = z_slice.shape
        rlm = np.zeros((lvl, max(rows, cols)), dtype=np.int64)
        for offset in range(-rows + 1, cols):
            cls._rle_1d(np.diagonal(z_slice, offset=offset), lvl, rlm)
        return rlm

    @classmethod
    def _process_antidiagonal(cls, z_slice, lvl):
        rows, cols = z_slice.shape
        rlm = np.zeros((lvl, max(rows, cols)), dtype=np.int64)
        flipped = np.fliplr(z_slice)
        for offset in range(-rows + 1, cols):
            cls._rle_1d(np.diagonal(flipped, offset=offset), lvl, rlm)
        return rlm

    @classmethod
    def _calc_2d_matrices(cls, image, lvl):
        direction_funcs = (
            cls._process_horizontal,
            cls._process_vertical,
            cls._process_diagonal,
            cls._process_antidiagonal,
        )
        range_z = np.unique(np.where(~np.isnan(image))[2])
        glrlm_2d_matrices = []
        roi_voxel_counts = []

        for z_slice_index in range_z:
            z_slice = image[:, :, z_slice_index]
            roi_voxel_count = int(np.count_nonzero(~np.isnan(z_slice)))
            if roi_voxel_count == 0:
                continue
            roi_voxel_counts.append(roi_voxel_count)
            glrlm_2d_matrices.append([func(z_slice, lvl) for func in direction_funcs])

        return np.array(glrlm_2d_matrices, dtype=np.int64), np.array(roi_voxel_counts, dtype=float)

    @staticmethod
    def _same_neighbor_mask(image, valid_mask, direction):
        same_neighbor = np.zeros(image.shape, dtype=bool)
        current_slices = [slice(None)] * image.ndim
        neighbor_slices = [slice(None)] * image.ndim

        for axis, delta in enumerate(direction):
            if delta > 0:
                current_slices[axis] = slice(1, None)
                neighbor_slices[axis] = slice(None, -1)
            elif delta < 0:
                current_slices[axis] = slice(None, -1)
                neighbor_slices[axis] = slice(1, None)

        current_slices = tuple(current_slices)
        neighbor_slices = tuple(neighbor_slices)
        same_neighbor[current_slices] = (
            valid_mask[current_slices]
            & valid_mask[neighbor_slices]
            & (image[current_slices] == image[neighbor_slices])
        )
        return same_neighbor

    @staticmethod
    def _line_ids_and_positions(coords, shape, direction):
        distances = []
        for axis, delta in enumerate(direction):
            if delta > 0:
                distances.append(coords[axis])
            elif delta < 0:
                distances.append(shape[axis] - 1 - coords[axis])
        positions = np.minimum.reduce(distances)
        line_start_coords = [coords[axis] - positions * direction[axis] for axis in range(len(shape))]
        line_ids = np.ravel_multi_index(line_start_coords, shape)
        return line_ids, positions

    @classmethod
    def _rlm_for_direction(cls, image, valid_mask, direction, lvl, max_dim):
        same_previous = cls._same_neighbor_mask(image, valid_mask, direction)
        same_next = cls._same_neighbor_mask(image, valid_mask, tuple(-delta for delta in direction))
        run_start_coords = np.where(valid_mask & ~same_previous)
        run_end_coords = np.where(valid_mask & ~same_next)

        if run_start_coords[0].size == 0:
            return np.zeros((lvl, max_dim), dtype=np.int64)

        start_line_ids, start_positions = cls._line_ids_and_positions(run_start_coords, image.shape, direction)
        end_line_ids, end_positions = cls._line_ids_and_positions(run_end_coords, image.shape, direction)
        start_order = np.lexsort((start_positions, start_line_ids))
        end_order = np.lexsort((end_positions, end_line_ids))

        run_lengths = end_positions[end_order] - start_positions[start_order] + 1
        gray_levels = image[tuple(coord[start_order] for coord in run_start_coords)].astype(int)
        flat_indices = gray_levels * max_dim + run_lengths - 1
        return np.bincount(flat_indices, minlength=lvl * max_dim).reshape(lvl, max_dim)

    @classmethod
    def _calc_3d_matrices(cls, image, lvl):
        image = crop_to_valid_bbox(image)
        directions = (
            (0, 0, 1),
            (0, 1, -1),
            (0, 1, 0),
            (0, 1, 1),
            (1, -1, -1),
            (1, -1, 0),
            (1, -1, 1),
            (1, 0, -1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, -1),
            (1, 1, 0),
            (1, 1, 1),
        )

        max_dim = max(image.shape)
        valid_mask = ~np.isnan(image)
        glrlm_3d_matrix = np.zeros((len(directions), lvl, max_dim), dtype=np.int64)

        for d_idx, direction in enumerate(directions):
            glrlm_3d_matrix[d_idx] = cls._rlm_for_direction(image, valid_mask, direction, lvl, max_dim)

        return glrlm_3d_matrix

    def _calc_2d_averaged_features(self, matrices, roi_voxel_counts, total_roi_voxels):
        feature_dicts = []
        weights = []
        for slice_index in range(matrices.shape[0]):
            for matrix in matrices[slice_index]:
                if self.slice_weight:
                    if total_roi_voxels == 0:
                        raise DataStructureError(' Denominator is zero in calc_2d_averaged_glrlm_features.')
                    weights.append(roi_voxel_counts[slice_index] / total_roi_voxels)
                else:
                    weights.append(1.0)
                feature_dicts.append(self._matrix_feature_values(matrix, roi_voxel_counts[slice_index]))
        return self._aggregate_feature_dicts(feature_dicts, None if self.slice_median else weights)

    def _calc_2d_slice_merged_features(self, matrices, roi_voxel_counts, total_roi_voxels):
        number_of_directions = matrices.shape[1]
        if number_of_directions == 0:
            raise DataStructureError(' Denominator is zero in calc_2d_slice_merged_glrlm_features.')
        averaged_matrices = np.sum(matrices, axis=1)
        feature_dicts = []
        weights = []
        for slice_index, matrix in enumerate(averaged_matrices):
            if self.slice_weight:
                if total_roi_voxels == 0:
                    raise DataStructureError(' Denominator is zero in calc_2d_slice_merged_glrlm_features.')
                weights.append(roi_voxel_counts[slice_index] / total_roi_voxels)
            else:
                weights.append(1.0)
            feature_dicts.append(
                self._matrix_feature_values(matrix, roi_voxel_counts[slice_index] * number_of_directions)
            )
        return self._aggregate_feature_dicts(feature_dicts, None if self.slice_median else weights)

    def _calc_2_5d_merged_features(self, matrices, roi_voxel_counts):
        number_of_directions = matrices.shape[1]
        if number_of_directions == 0:
            raise DataStructureError(' Denominator is zero in calc_2_5d_merged_glrlm_features.')
        matrix = np.sum(np.sum(matrices, axis=1), axis=0)
        return self._matrix_feature_values(matrix, np.sum(roi_voxel_counts) * number_of_directions)

    def _calc_2_5d_direction_merged_features(self, matrices, roi_voxel_counts):
        number_of_directions = matrices.shape[1]
        if number_of_directions == 0:
            raise DataStructureError(' Denominator is zero in calc_2_5d_direction_merged_glrlm_features.')
        averaged_glrlm = np.sum(matrices, axis=0)
        values = [self._matrix_feature_values(matrix, np.sum(roi_voxel_counts)) for matrix in averaged_glrlm]
        return self._mean_feature_dicts(values)

    def _calc_3d_averaged_features(self, matrices, total_roi_voxels):
        if matrices.shape[0] == 0:
            raise DataStructureError(' Denominator is zero in calc_3d_averaged_glrlm_features.')
        values = [self._matrix_feature_values(matrix, total_roi_voxels) for matrix in matrices]
        return self._mean_feature_dicts(values)

    def _calc_3d_merged_features(self, matrices, total_roi_voxels):
        number_of_directions = matrices.shape[0]
        if number_of_directions == 0:
            raise DataStructureError(' Denominator is zero in calc_3d_merged_glrlm_features.')
        matrix = np.sum(matrices, axis=0)
        return self._matrix_feature_values(matrix, total_roi_voxels * number_of_directions)

    def calculate_features(self, discretized_image_array):
        """Calculate GLRLM features for a prepared discretized intensity array.

        Parameters
        ----------
        discretized_image_array : numpy.ndarray
            Prepared discretized intensity array with ROI voxels represented by
            integer gray levels and voxels outside the ROI set to ``NaN``.

        Returns
        -------
        dict
            Mapping of GLRLM feature names to calculated values.
        """
        discretized_image_array = np.asarray(discretized_image_array)
        lvl = int(np.nanmax(discretized_image_array) + 1)
        total_roi_voxels = int(np.sum(~np.isnan(discretized_image_array)))

        if self.aggr_dim == '3D':
            matrices = self._calc_3d_matrices(discretized_image_array, lvl)
            values = (
                self._calc_3d_averaged_features(matrices, total_roi_voxels)
                if self.aggr_method == 'AVER'
                else self._calc_3d_merged_features(matrices, total_roi_voxels)
            )
            return self._map_feature_names(values)

        matrices, roi_voxel_counts = self._calc_2d_matrices(discretized_image_array, lvl)
        if self.aggr_method == 'DIR_MERG':
            values = self._calc_2_5d_direction_merged_features(matrices, roi_voxel_counts)
        elif self.aggr_method == 'MERG':
            values = self._calc_2_5d_merged_features(matrices, roi_voxel_counts)
        elif self.aggr_method == 'AVER':
            values = self._calc_2d_averaged_features(matrices, roi_voxel_counts, total_roi_voxels)
        elif self.aggr_method == 'SLICE_MERG':
            values = self._calc_2d_slice_merged_features(matrices, roi_voxel_counts, total_roi_voxels)
        else:
            raise DataStructureError(
                f'Unsupported GLRLM aggregation: aggr_dim={self.aggr_dim}, aggr_method={self.aggr_method}.'
            )
        return self._map_feature_names(values)


class GLRLMFeatureGroup(BaseFeatureGroup):
    family = 'glrlm'
    requirements = frozenset({'analysis_masks', 'discretized_intensity_image'})

    def supports(self, context):
        return context.roi_data.texture_discretized_image is not None

    def output_names(self, context):
        return format_cm_rlm_feature_names(GLRLM_FEATURE_NAMES, context.aggr_dim, context.aggr_method)

    def feature_aliases(self, context):
        output_names = self.output_names(context)
        aliases = {name: name for name in output_names}
        aliases.update(dict(zip(GLRLM_FEATURE_NAMES, output_names)))
        return aliases

    def calculate(self, context, prepared_data):
        glrlm = GLRLM(
            aggr_dim=context.aggr_dim,
            aggr_method=context.aggr_method,
            slice_weight=context.slice_weighting,
            slice_median=context.slice_median,
        )
        feature_values = glrlm.calculate_features(prepared_data.require_discretized_intensity_image().array.T)
        return {
            output_name: feature_values[base_name]
            for output_name, base_name in zip(self.output_names(context), GLRLM_FEATURE_NAMES)
        }
