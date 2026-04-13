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
    """Shared setup and feature formulas for run-, zone-, and dependence matrices."""

    def __init__(self, image, slice_weight=False, slice_median=False):
        self.image = image
        self.lvl = int(np.nanmax(self.image) + 1)
        self.tot_no_of_roi_voxels = np.sum(~np.isnan(image))
        self.slice_weight = slice_weight
        self.slice_median = slice_median

        x_indices, y_indices, z_indices = np.where(~np.isnan(self.image))
        self.range_x = np.unique(x_indices)
        self.range_y = np.unique(y_indices)
        self.range_z = np.unique(z_indices)

        self.reset_fields()

    def reset_fields(self):
        for name in NGLDM_ATTRIBUTE_NAMES:
            setattr(self, name, 0)
            setattr(self, f'{name}_list', [])

    def _attribute_names(self, include_energy=False):
        return NGLDM_ATTRIBUTE_NAMES if include_energy else TEXTURE_ATTRIBUTE_NAMES

    def _matrix_feature_values(self, matrix, voxel_count, *, include_energy=False):
        values = {
            'short_runs_emphasis': self.calc_short_emphasis(matrix),
            'long_runs_emphasis': self.calc_long_emphasis(matrix),
            'low_grey_level_run_emphasis': self.calc_low_gr_lvl_emphasis(matrix),
            'high_gr_lvl_emphasis': self.calc_high_gr_lvl_emphasis(matrix),
            'short_low_gr_lvl_emphasis': self.calc_short_low_gr_lvl_emphasis(matrix),
            'short_high_gr_lvl_emphasis': self.calc_short_high_gr_lvl_emphasis(matrix),
            'long_low_gr_lvl_emphasis': self.calc_long_low_gr_lvl_emphasis(matrix),
            'long_high_gr_lvl_emphasis': self.calc_long_high_gr_lvl_emphasis(matrix),
            'non_uniformity': self.calc_non_uniformity(matrix),
            'norm_non_uniformity': self.calc_norm_non_uniformity(matrix),
            'length_non_uniformity': self.calc_length_non_uniformity(matrix),
            'norm_length_non_uniformity': self.calc_norm_length_non_uniformity(matrix),
            'percentage': self.calc_percentage(matrix, voxel_count),
            'gr_lvl_var': self.calc_gr_lvl_var(matrix),
            'length_var': self.calc_length_var(matrix),
            'entropy': self.calc_entropy(matrix),
        }
        if include_energy:
            values['energy'] = self.calc_energy(matrix)
        return values

    def _append_feature_values(self, values):
        for name, value in values.items():
            getattr(self, f'{name}_list').append(value)

    def _set_feature_values(self, values):
        for name, value in values.items():
            setattr(self, name, value)

    def _average_feature_values(self, values_list):
        if not values_list:
            raise DataStructureError("No feature values were computed for aggregation.")
        names = tuple(values_list[0].keys())
        return {
            name: float(np.mean([values[name] for values in values_list]))
            for name in names
        }

    def _aggregate_feature_lists(self, weights, *, include_energy=False):
        names = self._attribute_names(include_energy=include_energy)
        if self.slice_median and not self.slice_weight:
            for name in names:
                setattr(self, name, np.median(getattr(self, f'{name}_list')))
        elif not self.slice_median:
            for name in names:
                setattr(
                    self,
                    name,
                    np.average(getattr(self, f'{name}_list'), weights=weights),
                )

    def calc_short_emphasis(self, matrix):
        n_s = np.sum(matrix)
        _, j = np.indices(matrix.shape)
        if n_s == 0:
            raise DataStructureError(" Denominator is zero in calc_short_emphasis.")
        return np.sum(matrix / (j + 1) ** 2) / n_s

    def calc_long_emphasis(self, matrix):
        n_s = np.sum(matrix)
        _, j = np.indices(matrix.shape)
        if n_s == 0:
            raise DataStructureError(" Denominator is zero in calc_long_emphasis.")
        return np.sum(matrix * (j + 1) ** 2) / n_s

    def calc_low_gr_lvl_emphasis(self, matrix):
        n_s = np.sum(matrix)
        i, _ = np.indices(matrix.shape)
        mask = i != 0
        if n_s == 0:
            raise DataStructureError(" Denominator is zero in calc_low_gr_lvl_emphasis.")
        return np.sum(matrix[mask] / (i[mask]) ** 2) / n_s

    def calc_high_gr_lvl_emphasis(self, matrix):
        n_s = np.sum(matrix)
        i, _ = np.indices(matrix.shape)
        if n_s == 0:
            raise DataStructureError(" Denominator is zero in calc_high_gr_lvl_emphasis.")
        return np.sum(matrix * i ** 2) / n_s

    def calc_short_low_gr_lvl_emphasis(self, matrix):
        n_s = np.sum(matrix)
        i, j = np.indices(matrix.shape)
        mask = i != 0
        if np.any(i[mask] == 0):
            raise DataStructureError(" Denominator is zero in calc_short_low_gr_lvl_emphasis.")
        matrix_j = matrix[mask] / (i[mask] ** 2)
        if n_s == 0:
            raise DataStructureError(" Denominator is zero in calc_short_low_gr_lvl_emphasis.")
        return np.sum(matrix_j / ((j[mask] + 1) ** 2)) / n_s

    def calc_short_high_gr_lvl_emphasis(self, matrix):
        n_s = np.sum(matrix)
        i, j = np.indices(matrix.shape)
        if n_s == 0:
            raise DataStructureError(" Denominator is zero in calc_short_high_gr_lvl_emphasis.")
        return np.sum((i ** 2 * matrix) / ((j + 1)) ** 2) / n_s

    def calc_long_low_gr_lvl_emphasis(self, matrix):
        n_s = np.sum(matrix)
        i, j = np.indices(matrix.shape)
        mask = i != 0
        if n_s == 0:
            raise DataStructureError(" Denominator is zero in calc_long_low_gr_lvl_emphasis.")
        return np.sum((matrix[mask] * (j[mask] + 1) ** 2) / (i[mask]) ** 2) / n_s

    def calc_long_high_gr_lvl_emphasis(self, matrix):
        n_s = np.sum(matrix)
        i, j = np.indices(matrix.shape)
        if n_s == 0:
            raise DataStructureError(" Denominator is zero in calc_long_high_gr_lvl_emphasis.")
        return np.sum(matrix * (j + 1) ** 2 * i ** 2) / n_s

    def calc_non_uniformity(self, matrix):
        n_s = np.sum(matrix)
        if n_s == 0:
            raise DataStructureError(" Denominator is zero in calc_non_uniformity.")
        return np.sum(np.sum(matrix, axis=1) ** 2) / n_s

    def calc_norm_non_uniformity(self, matrix):
        n_s = np.sum(matrix)
        if n_s == 0:
            raise DataStructureError(" Denominator is zero in calc_norm_non_uniformity.")
        return np.sum(np.sum(matrix, axis=1) ** 2) / n_s ** 2

    def calc_length_non_uniformity(self, matrix):
        n_s = np.sum(matrix)
        if n_s == 0:
            raise DataStructureError(" Denominator is zero in calc_length_non_uniformity.")
        return np.sum(np.sum(matrix, axis=0) ** 2) / n_s

    def calc_norm_length_non_uniformity(self, matrix):
        n_s = np.sum(matrix)
        if n_s == 0:
            raise DataStructureError(" Denominator is zero in calc_norm_length_non_uniformity.")
        return np.sum(np.sum(matrix, axis=0) ** 2) / n_s ** 2

    def calc_percentage(self, matrix, voxel_count):
        n_s = np.sum(matrix)
        if voxel_count == 0:
            raise DataStructureError(" Denominator is zero in calc_percentage.")
        return n_s / voxel_count

    def calc_gr_lvl_var(self, matrix):
        n_s = np.sum(matrix)
        i, _ = np.indices(matrix.shape)
        if n_s == 0:
            raise DataStructureError(" Denominator is zero in calc_gr_lvl_var.")
        mu = np.sum(matrix * i / n_s)
        return np.sum((i - mu) ** 2 * (matrix / n_s))

    def calc_length_var(self, matrix):
        n_s = np.sum(matrix)
        _, j = np.indices(matrix.shape)
        if n_s == 0:
            raise DataStructureError(" Denominator is zero in calc_length_var.")
        mu = np.sum(matrix * j / n_s)
        return np.sum((j - mu) ** 2 * (matrix / n_s))

    def calc_entropy(self, matrix):
        n_s = np.sum(matrix)
        mask = matrix != 0
        if n_s == 0:
            raise DataStructureError(" Denominator is zero in calc_entropy.")
        return np.sum((matrix[mask] / n_s) * np.log2(matrix[mask] / n_s)) * (-1)

    def calc_energy(self, matrix):
        n_s = np.sum(matrix)
        mask = matrix != 0
        if n_s == 0:
            raise DataStructureError(" Denominator is zero in calc_energy.")
        return np.sum((matrix[mask] / n_s) ** 2)


class ZoneMatrixFeatureBase(TextureFeatureBase):
    """Shared zone-matrix builders for GLSZM and GLDZM."""

    def calc_glsz_gldz_3d_matrices(self, mask):
        flattened_array = self.image.flatten()
        _, counts = np.unique(flattened_array[~np.isnan(flattened_array)], return_counts=True)
        max_region_size = np.max(counts)

        def calc_dist_map_3d(image_orig):
            image = image_orig.copy()
            larger_array = np.zeros((image.shape[0] + 2, image.shape[1] + 2, image.shape[2] + 2))
            larger_array[1:-1, 1:-1, 1:-1] = image
            return distance_transform_cdt(larger_array, metric='taxicab')[1:-1, 1:-1, 1:-1].astype(float)

        dist_map = calc_dist_map_3d(mask)
        glszm = np.zeros((self.lvl, max_region_size), dtype=int)
        gldzm = np.zeros((self.lvl, np.max(self.image.shape)), dtype=int)

        def find_connected_region_3d(start, intensity):
            stack = [start]
            size = 0
            min_dist = np.inf
            x_max, y_max, z_max = self.image.shape

            while stack:
                x, y, z = stack.pop()
                if 0 <= x < x_max and 0 <= y < y_max and 0 <= z < z_max:
                    if visited[x, y, z] == 0 and self.image[x, y, z] == intensity:
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
                                        if visited[nx, ny, nz] == 0 and self.image[nx, ny, nz] == intensity:
                                            stack.append((nx, ny, nz))

            return size, min_dist

        visited = np.zeros_like(self.image, dtype=int)
        for x in self.range_x:
            for y in self.range_y:
                for z in self.range_z:
                    if visited[x, y, z] == 0 and not np.isnan(self.image[x, y, z]):
                        intensity = int(self.image[x, y, z])
                        size, min_dist = find_connected_region_3d((x, y, z), intensity)
                        if size > 0:
                            glszm[intensity, size - 1] += 1
                            gldzm[intensity, int(min_dist) - 1] += 1

        self.glszm_3D_matrix = glszm.astype(np.int64)
        self.gldzm_3D_matrix = gldzm.astype(np.int64)

    def calc_glsz_gldz_2d_matrices(self, mask):
        max_region_size_list = []
        for z_idx in self.range_z:
            z_slice = self.image[:, :, z_idx]
            valid = ~np.isnan(z_slice)
            if np.any(valid):
                counts = np.bincount(z_slice[valid].astype(int))
                if counts.size:
                    max_region_size_list.append(counts.max())
        max_region_size = max(max_region_size_list) if max_region_size_list else 0

        def calc_dist_map_2d(image_orig):
            larger_array = np.zeros((image_orig.shape[0] + 2, image_orig.shape[1] + 2))
            larger_array[1:-1, 1:-1] = image_orig
            return distance_transform_cdt(larger_array, metric='taxicab')[1:-1, 1:-1].astype(float)

        glszm_matrices = []
        gldzm_matrices = []
        roi_voxels = []
        structure = np.ones((3, 3), dtype=int)

        for z_idx in self.range_z:
            z_slice = self.image[:, :, z_idx]
            z_mask = mask[:, :, z_idx]
            roi_voxels.append(np.sum(~np.isnan(z_slice)))
            dist_map = calc_dist_map_2d(z_mask)
            glszm = np.zeros((self.lvl, max_region_size), dtype=int)
            gldzm = np.zeros((self.lvl, np.max(self.image.shape)), dtype=int)

            for intensity in range(self.lvl):
                comp_mask = z_slice == intensity
                if not np.any(comp_mask):
                    continue
                labeled, num_features = label(comp_mask, structure=structure)
                if num_features == 0:
                    continue
                sizes = np.bincount(labeled.ravel())[1:]
                min_dists = minimum(dist_map, labeled, index=np.arange(1, num_features + 1))

                unique_sizes, counts_sizes = np.unique(sizes, return_counts=True)
                for size, count in zip(unique_sizes, counts_sizes):
                    if size - 1 < glszm.shape[1]:
                        glszm[intensity, size - 1] += count

                min_dists_int = min_dists.astype(int)
                unique_dists, counts_dists = np.unique(min_dists_int, return_counts=True)
                for dist, count in zip(unique_dists, counts_dists):
                    if dist - 1 < gldzm.shape[1]:
                        gldzm[intensity, dist - 1] += count

            glszm_matrices.append(glszm.astype(np.int64))
            gldzm_matrices.append(gldzm.astype(np.int64))

        self.glszm_2D_matrices = np.array(glszm_matrices)
        self.gldzm_2D_matrices = np.array(gldzm_matrices)
        self.no_of_roi_voxels = roi_voxels


def extract_texture_values(calculator):
    return [getattr(calculator, name) for name in TEXTURE_ATTRIBUTE_NAMES]


def extract_ngldm_values(calculator):
    return [getattr(calculator, name) for name in NGLDM_ATTRIBUTE_NAMES]
