import numpy as np
from scipy.ndimage import distance_transform_cdt, label, minimum

from ..exceptions import DataStructureError

TEXTURE_ATTRIBUTE_NAMES = (
    'short_runs_emphasis',
    'long_runs_emphasis',
    'low_grey_level_run_emphasis',
    'high_gr_lvl_emphasis',
    'short_low_gr_lvl_emphasis',
    'short_high_gr_lvl_emphasis',
    'long_low_gr_lvl_emphasis',
    'long_high_gr_lvl_emphasis',
    'non_uniformity',
    'norm_non_uniformity',
    'length_non_uniformity',
    'norm_length_non_uniformity',
    'percentage',
    'gr_lvl_var',
    'length_var',
    'entropy',
)


NGLDM_ATTRIBUTE_NAMES = TEXTURE_ATTRIBUTE_NAMES + ('energy',)


class TextureFeatureBase:
    """Shared feature formulas and aggregation helpers for texture matrices."""

    def __init__(self, slice_weight=False, slice_median=False):
        self.slice_weight = slice_weight
        self.slice_median = slice_median

    @staticmethod
    def _feature_names(include_energy=False):
        return NGLDM_ATTRIBUTE_NAMES if include_energy else TEXTURE_ATTRIBUTE_NAMES

    def _matrix_feature_values(self, matrix, voxel_count, *, include_energy=False):
        values = {
            'short_runs_emphasis': self._calc_short_emphasis(matrix),
            'long_runs_emphasis': self._calc_long_emphasis(matrix),
            'low_grey_level_run_emphasis': self._calc_low_gr_lvl_emphasis(matrix),
            'high_gr_lvl_emphasis': self._calc_high_gr_lvl_emphasis(matrix),
            'short_low_gr_lvl_emphasis': self._calc_short_low_gr_lvl_emphasis(matrix),
            'short_high_gr_lvl_emphasis': self._calc_short_high_gr_lvl_emphasis(matrix),
            'long_low_gr_lvl_emphasis': self._calc_long_low_gr_lvl_emphasis(matrix),
            'long_high_gr_lvl_emphasis': self._calc_long_high_gr_lvl_emphasis(matrix),
            'non_uniformity': self._calc_non_uniformity(matrix),
            'norm_non_uniformity': self._calc_norm_non_uniformity(matrix),
            'length_non_uniformity': self._calc_length_non_uniformity(matrix),
            'norm_length_non_uniformity': self._calc_norm_length_non_uniformity(matrix),
            'percentage': self._calc_percentage(matrix, voxel_count),
            'gr_lvl_var': self._calc_gr_lvl_var(matrix),
            'length_var': self._calc_length_var(matrix),
            'entropy': self._calc_entropy(matrix),
        }
        if include_energy:
            values['energy'] = self._calc_energy(matrix)
        return values

    @staticmethod
    def _mean_feature_dicts(feature_dicts):
        if not feature_dicts:
            raise DataStructureError('No feature values were computed for aggregation.')
        names = tuple(feature_dicts[0].keys())
        return {name: float(np.mean([values[name] for values in feature_dicts])) for name in names}

    def _aggregate_feature_dicts(self, feature_dicts, weights=None, *, include_energy=False):
        names = self._feature_names(include_energy=include_energy)
        if not feature_dicts:
            raise DataStructureError('No feature values were computed for aggregation.')
        if self.slice_median:
            if self.slice_weight and weights is not None:
                raise DataStructureError('Weighted median is not supported for texture aggregation.')
            return {name: float(np.median([values[name] for values in feature_dicts])) for name in names}
        return {name: float(np.average([values[name] for values in feature_dicts], weights=weights)) for name in names}

    @staticmethod
    def _calc_short_emphasis(matrix):
        n_s = np.sum(matrix)
        _, j = np.indices(matrix.shape)
        if n_s == 0:
            raise DataStructureError(' Denominator is zero in calc_short_emphasis.')
        return np.sum(matrix / (j + 1) ** 2) / n_s

    @staticmethod
    def _calc_long_emphasis(matrix):
        n_s = np.sum(matrix)
        _, j = np.indices(matrix.shape)
        if n_s == 0:
            raise DataStructureError(' Denominator is zero in calc_long_emphasis.')
        return np.sum(matrix * (j + 1) ** 2) / n_s

    @staticmethod
    def _calc_low_gr_lvl_emphasis(matrix):
        n_s = np.sum(matrix)
        i, _ = np.indices(matrix.shape)
        mask = i != 0
        if n_s == 0:
            raise DataStructureError(' Denominator is zero in calc_low_gr_lvl_emphasis.')
        return np.sum(matrix[mask] / (i[mask]) ** 2) / n_s

    @staticmethod
    def _calc_high_gr_lvl_emphasis(matrix):
        n_s = np.sum(matrix)
        i, _ = np.indices(matrix.shape)
        if n_s == 0:
            raise DataStructureError(' Denominator is zero in calc_high_gr_lvl_emphasis.')
        return np.sum(matrix * i**2) / n_s

    @staticmethod
    def _calc_short_low_gr_lvl_emphasis(matrix):
        n_s = np.sum(matrix)
        i, j = np.indices(matrix.shape)
        mask = i != 0
        if n_s == 0:
            raise DataStructureError(' Denominator is zero in calc_short_low_gr_lvl_emphasis.')
        return np.sum((matrix[mask] / (i[mask] ** 2)) / ((j[mask] + 1) ** 2)) / n_s

    @staticmethod
    def _calc_short_high_gr_lvl_emphasis(matrix):
        n_s = np.sum(matrix)
        i, j = np.indices(matrix.shape)
        if n_s == 0:
            raise DataStructureError(' Denominator is zero in calc_short_high_gr_lvl_emphasis.')
        return np.sum((i**2 * matrix) / ((j + 1) ** 2)) / n_s

    @staticmethod
    def _calc_long_low_gr_lvl_emphasis(matrix):
        n_s = np.sum(matrix)
        i, j = np.indices(matrix.shape)
        mask = i != 0
        if n_s == 0:
            raise DataStructureError(' Denominator is zero in calc_long_low_gr_lvl_emphasis.')
        return np.sum((matrix[mask] * (j[mask] + 1) ** 2) / (i[mask] ** 2)) / n_s

    @staticmethod
    def _calc_long_high_gr_lvl_emphasis(matrix):
        n_s = np.sum(matrix)
        i, j = np.indices(matrix.shape)
        if n_s == 0:
            raise DataStructureError(' Denominator is zero in calc_long_high_gr_lvl_emphasis.')
        return np.sum(matrix * (j + 1) ** 2 * i**2) / n_s

    @staticmethod
    def _calc_non_uniformity(matrix):
        n_s = np.sum(matrix)
        if n_s == 0:
            raise DataStructureError(' Denominator is zero in calc_non_uniformity.')
        return np.sum(np.sum(matrix, axis=1) ** 2) / n_s

    @staticmethod
    def _calc_norm_non_uniformity(matrix):
        n_s = np.sum(matrix)
        if n_s == 0:
            raise DataStructureError(' Denominator is zero in calc_norm_non_uniformity.')
        return np.sum(np.sum(matrix, axis=1) ** 2) / n_s**2

    @staticmethod
    def _calc_length_non_uniformity(matrix):
        n_s = np.sum(matrix)
        if n_s == 0:
            raise DataStructureError(' Denominator is zero in calc_length_non_uniformity.')
        return np.sum(np.sum(matrix, axis=0) ** 2) / n_s

    @staticmethod
    def _calc_norm_length_non_uniformity(matrix):
        n_s = np.sum(matrix)
        if n_s == 0:
            raise DataStructureError(' Denominator is zero in calc_norm_length_non_uniformity.')
        return np.sum(np.sum(matrix, axis=0) ** 2) / n_s**2

    @staticmethod
    def _calc_percentage(matrix, voxel_count):
        n_s = np.sum(matrix)
        if voxel_count == 0:
            raise DataStructureError(' Denominator is zero in calc_percentage.')
        return n_s / voxel_count

    @staticmethod
    def _calc_gr_lvl_var(matrix):
        n_s = np.sum(matrix)
        i, _ = np.indices(matrix.shape)
        if n_s == 0:
            raise DataStructureError(' Denominator is zero in calc_gr_lvl_var.')
        mu = np.sum(matrix * i / n_s)
        return np.sum((i - mu) ** 2 * (matrix / n_s))

    @staticmethod
    def _calc_length_var(matrix):
        n_s = np.sum(matrix)
        _, j = np.indices(matrix.shape)
        if n_s == 0:
            raise DataStructureError(' Denominator is zero in calc_length_var.')
        mu = np.sum(matrix * j / n_s)
        return np.sum((j - mu) ** 2 * (matrix / n_s))

    @staticmethod
    def _calc_entropy(matrix):
        n_s = np.sum(matrix)
        mask = matrix != 0
        if n_s == 0:
            raise DataStructureError(' Denominator is zero in calc_entropy.')
        return np.sum((matrix[mask] / n_s) * np.log2(matrix[mask] / n_s)) * (-1)

    @staticmethod
    def _calc_energy(matrix):
        n_s = np.sum(matrix)
        mask = matrix != 0
        if n_s == 0:
            raise DataStructureError(' Denominator is zero in calc_energy.')
        return np.sum((matrix[mask] / n_s) ** 2)


class ZoneMatrixFeatureBase(TextureFeatureBase):
    """Shared zone-matrix builders for GLSZM and GLDZM."""

    @staticmethod
    def _range_indices(image):
        x_indices, y_indices, z_indices = np.where(~np.isnan(image))
        return np.unique(x_indices), np.unique(y_indices), np.unique(z_indices)

    @classmethod
    def _calc_glsz_3d_matrix(cls, image, lvl):
        image = np.asarray(image)
        flattened_array = image.flatten()
        _, counts = np.unique(flattened_array[~np.isnan(flattened_array)], return_counts=True)
        max_region_size = int(np.max(counts))
        range_x, range_y, range_z = cls._range_indices(image)

        glszm = np.zeros((lvl, max_region_size), dtype=np.int64)

        def find_connected_region_3d(start, intensity, visited):
            stack = [start]
            size = 0
            x_max, y_max, z_max = image.shape

            while stack:
                x, y, z = stack.pop()
                if 0 <= x < x_max and 0 <= y < y_max and 0 <= z < z_max:
                    if visited[x, y, z] == 0 and image[x, y, z] == intensity:
                        visited[x, y, z] = 1
                        size += 1
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                for dz in [-1, 0, 1]:
                                    if dx == 0 and dy == 0 and dz == 0:
                                        continue
                                    nx, ny, nz = x + dx, y + dy, z + dz
                                    if 0 <= nx < x_max and 0 <= ny < y_max and 0 <= nz < z_max:
                                        if visited[nx, ny, nz] == 0 and image[nx, ny, nz] == intensity:
                                            stack.append((nx, ny, nz))

            return size

        visited = np.zeros_like(image, dtype=int)
        for x in range_x:
            for y in range_y:
                for z in range_z:
                    if visited[x, y, z] == 0 and not np.isnan(image[x, y, z]):
                        intensity = int(image[x, y, z])
                        size = find_connected_region_3d((x, y, z), intensity, visited)
                        if size > 0:
                            glszm[intensity, size - 1] += 1

        return glszm, int(np.sum(~np.isnan(image)))

    @classmethod
    def _calc_glsz_2d_matrices(cls, image, lvl):
        image = np.asarray(image)
        _, _, range_z = cls._range_indices(image)
        max_region_size_list = []
        for z_idx in range_z:
            z_slice = image[:, :, z_idx]
            valid = ~np.isnan(z_slice)
            if np.any(valid):
                counts = np.bincount(z_slice[valid].astype(int))
                if counts.size:
                    max_region_size_list.append(counts.max())
        max_region_size = max(max_region_size_list) if max_region_size_list else 0

        glszm_matrices = []
        roi_voxels = []
        structure = np.ones((3, 3), dtype=int)

        for z_idx in range_z:
            z_slice = image[:, :, z_idx]
            roi_voxel_count = int(np.sum(~np.isnan(z_slice)))
            if roi_voxel_count == 0:
                continue
            roi_voxels.append(roi_voxel_count)
            glszm = np.zeros((lvl, max_region_size), dtype=np.int64)

            for intensity in range(lvl):
                comp_mask = z_slice == intensity
                if not np.any(comp_mask):
                    continue
                labeled, num_features = label(comp_mask, structure=structure)
                if num_features == 0:
                    continue
                sizes = np.bincount(labeled.ravel())[1:]

                unique_sizes, counts_sizes = np.unique(sizes, return_counts=True)
                for size, count in zip(unique_sizes, counts_sizes):
                    if size - 1 < glszm.shape[1]:
                        glszm[intensity, size - 1] += count

            glszm_matrices.append(glszm)

        return np.array(glszm_matrices), np.array(roi_voxels, dtype=float)

    @classmethod
    def _calc_gldz_3d_matrix(cls, image, mask, lvl):
        image = np.asarray(image)
        max_distance = int(np.max(image.shape))
        range_x, range_y, range_z = cls._range_indices(image)

        def calc_dist_map_3d(image_orig):
            image_copy = image_orig.copy()
            larger_array = np.zeros((image_copy.shape[0] + 2, image_copy.shape[1] + 2, image_copy.shape[2] + 2))
            larger_array[1:-1, 1:-1, 1:-1] = image_copy
            return distance_transform_cdt(larger_array, metric='taxicab')[1:-1, 1:-1, 1:-1].astype(float)

        dist_map = calc_dist_map_3d(mask)
        gldzm = np.zeros((lvl, max_distance), dtype=np.int64)

        def find_connected_region_3d(start, intensity, visited):
            stack = [start]
            size = 0
            min_dist = np.inf
            x_max, y_max, z_max = image.shape

            while stack:
                x, y, z = stack.pop()
                if 0 <= x < x_max and 0 <= y < y_max and 0 <= z < z_max:
                    if visited[x, y, z] == 0 and image[x, y, z] == intensity:
                        visited[x, y, z] = 1
                        size += 1
                        min_dist = min(min_dist, dist_map[x, y, z])
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                for dz in [-1, 0, 1]:
                                    if dx == 0 and dy == 0 and dz == 0:
                                        continue
                                    nx, ny, nz = x + dx, y + dy, z + dz
                                    if 0 <= nx < x_max and 0 <= ny < y_max and 0 <= nz < z_max:
                                        if visited[nx, ny, nz] == 0 and image[nx, ny, nz] == intensity:
                                            stack.append((nx, ny, nz))

            return size, min_dist

        visited = np.zeros_like(image, dtype=int)
        for x in range_x:
            for y in range_y:
                for z in range_z:
                    if visited[x, y, z] == 0 and not np.isnan(image[x, y, z]):
                        intensity = int(image[x, y, z])
                        size, min_dist = find_connected_region_3d((x, y, z), intensity, visited)
                        if size > 0:
                            gldzm[intensity, int(min_dist) - 1] += 1

        return gldzm, int(np.sum(~np.isnan(image)))

    @classmethod
    def _calc_gldz_2d_matrices(cls, image, mask, lvl):
        image = np.asarray(image)
        _, _, range_z = cls._range_indices(image)

        def calc_dist_map_2d(image_orig):
            larger_array = np.zeros((image_orig.shape[0] + 2, image_orig.shape[1] + 2))
            larger_array[1:-1, 1:-1] = image_orig
            return distance_transform_cdt(larger_array, metric='taxicab')[1:-1, 1:-1].astype(float)

        gldzm_matrices = []
        roi_voxels = []
        structure = np.ones((3, 3), dtype=int)

        for z_idx in range_z:
            z_slice = image[:, :, z_idx]
            z_mask = mask[:, :, z_idx]
            roi_voxel_count = int(np.sum(~np.isnan(z_slice)))
            if roi_voxel_count == 0:
                continue
            roi_voxels.append(roi_voxel_count)
            dist_map = calc_dist_map_2d(z_mask)
            gldzm = np.zeros((lvl, np.max(image.shape)), dtype=np.int64)

            for intensity in range(lvl):
                comp_mask = z_slice == intensity
                if not np.any(comp_mask):
                    continue
                labeled, num_features = label(comp_mask, structure=structure)
                if num_features == 0:
                    continue
                min_dists = minimum(dist_map, labeled, index=np.arange(1, num_features + 1))

                min_dists_int = min_dists.astype(int)
                unique_dists, counts_dists = np.unique(min_dists_int, return_counts=True)
                for dist, count in zip(unique_dists, counts_dists):
                    if dist - 1 < gldzm.shape[1]:
                        gldzm[intensity, dist - 1] += count

            gldzm_matrices.append(gldzm)

        return np.array(gldzm_matrices), np.array(roi_voxels, dtype=float)
