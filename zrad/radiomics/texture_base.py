import numpy as np
from scipy.ndimage import distance_transform_cdt, label, minimum

from ..exceptions import DataStructureError


def valid_bbox(image):
    """Return slices for the smallest subarray containing non-NaN voxels."""
    valid_coords = np.where(~np.isnan(image))
    if valid_coords[0].size == 0:
        return None
    return tuple(slice(int(coords.min()), int(coords.max()) + 1) for coords in valid_coords)


def crop_to_valid_bbox(image):
    """Return the smallest subarray containing non-NaN voxels."""
    bbox = valid_bbox(image)
    if bbox is None:
        return image
    return image[bbox]


def crop_to_valid_bbox_pair(image, paired_array):
    """Crop an image and aligned array to the image's non-NaN bounding box."""
    bbox = valid_bbox(image)
    if bbox is None:
        return image, paired_array
    return image[bbox], paired_array[bbox]

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
        matrix = np.asarray(matrix)
        n_s = np.sum(matrix)
        if n_s == 0:
            raise DataStructureError(' Denominator is zero in matrix feature calculation.')
        if voxel_count == 0:
            raise DataStructureError(' Denominator is zero in calc_percentage.')

        i = np.arange(matrix.shape[0], dtype=float)[:, np.newaxis]
        j = np.arange(matrix.shape[1], dtype=float)[np.newaxis, :]
        run_lengths = j + 1.0
        gray_nonzero = i != 0
        row_sums = np.sum(matrix, axis=1)
        col_sums = np.sum(matrix, axis=0)
        probabilities = matrix / n_s
        nz_probabilities = probabilities[matrix != 0]

        low_gray = np.divide(matrix, i**2, out=np.zeros_like(matrix, dtype=float), where=gray_nonzero)
        mu_gray = np.sum(matrix * i / n_s)
        mu_length = np.sum(matrix * j / n_s)

        values = {
            'short_runs_emphasis': np.sum(matrix / run_lengths**2) / n_s,
            'long_runs_emphasis': np.sum(matrix * run_lengths**2) / n_s,
            'low_grey_level_run_emphasis': np.sum(low_gray) / n_s,
            'high_gr_lvl_emphasis': np.sum(matrix * i**2) / n_s,
            'short_low_gr_lvl_emphasis': np.sum(low_gray / run_lengths**2) / n_s,
            'short_high_gr_lvl_emphasis': np.sum((i**2 * matrix) / run_lengths**2) / n_s,
            'long_low_gr_lvl_emphasis': np.sum(low_gray * run_lengths**2) / n_s,
            'long_high_gr_lvl_emphasis': np.sum(matrix * run_lengths**2 * i**2) / n_s,
            'non_uniformity': np.sum(row_sums**2) / n_s,
            'norm_non_uniformity': np.sum(row_sums**2) / n_s**2,
            'length_non_uniformity': np.sum(col_sums**2) / n_s,
            'norm_length_non_uniformity': np.sum(col_sums**2) / n_s**2,
            'percentage': n_s / voxel_count,
            'gr_lvl_var': np.sum((i - mu_gray) ** 2 * probabilities),
            'length_var': np.sum((j - mu_length) ** 2 * probabilities),
            'entropy': np.sum(nz_probabilities * np.log2(nz_probabilities)) * (-1),
        }
        if include_energy:
            values['energy'] = np.sum(nz_probabilities**2)
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
        image = crop_to_valid_bbox(np.asarray(image))
        valid = ~np.isnan(image)
        if not np.any(valid):
            return np.zeros((lvl, 0), dtype=np.int64), 0

        valid_values = image[valid].astype(int)
        counts = np.bincount(valid_values, minlength=lvl)
        max_region_size = int(np.max(counts))
        glszm = np.zeros((lvl, max_region_size), dtype=np.int64)
        structure = np.ones((3, 3, 3), dtype=int)

        for intensity in np.flatnonzero(counts):
            labeled, num_features = label(image == intensity, structure=structure)
            if num_features == 0:
                continue
            sizes = np.bincount(labeled.ravel())[1:]
            unique_sizes, size_counts = np.unique(sizes, return_counts=True)
            glszm[intensity, unique_sizes - 1] += size_counts

        return glszm, int(valid_values.size)

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
        mask = np.asarray(mask)
        valid_coords = np.where(~np.isnan(image))
        if valid_coords[0].size:
            bbox = tuple(slice(int(coords.min()), int(coords.max()) + 1) for coords in valid_coords)
            image = image[bbox]
            mask = mask[bbox]
        valid = ~np.isnan(image)
        if not np.any(valid):
            return np.zeros((lvl, 0), dtype=np.int64), 0

        def calc_dist_map_3d(image_orig):
            image_copy = image_orig.copy()
            larger_array = np.zeros((image_copy.shape[0] + 2, image_copy.shape[1] + 2, image_copy.shape[2] + 2))
            larger_array[1:-1, 1:-1, 1:-1] = image_copy
            return distance_transform_cdt(larger_array, metric='taxicab')[1:-1, 1:-1, 1:-1].astype(float)

        dist_map = calc_dist_map_3d(mask)
        gldzm = np.zeros((lvl, int(np.max(image.shape))), dtype=np.int64)
        structure = np.ones((3, 3, 3), dtype=int)
        valid_values = image[valid].astype(int)

        for intensity in np.unique(valid_values):
            labeled, num_features = label(image == intensity, structure=structure)
            if num_features == 0:
                continue
            min_dists = np.full(num_features + 1, np.inf)
            component_labels = labeled.ravel()
            component_mask = component_labels != 0
            np.minimum.at(min_dists, component_labels[component_mask], dist_map.ravel()[component_mask])
            unique_dists, dist_counts = np.unique(min_dists[1:].astype(int), return_counts=True)
            valid_dists = (unique_dists > 0) & (unique_dists <= gldzm.shape[1])
            gldzm[intensity, unique_dists[valid_dists] - 1] += dist_counts[valid_dists]

        return gldzm, int(valid_values.size)

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
                min_dists = np.full(num_features + 1, np.inf)
                component_labels = labeled.ravel()
                component_mask = component_labels != 0
                np.minimum.at(min_dists, component_labels[component_mask], dist_map.ravel()[component_mask])
                unique_dists, counts_dists = np.unique(min_dists[1:].astype(int), return_counts=True)
                valid_dists = (unique_dists > 0) & (unique_dists <= gldzm.shape[1])
                gldzm[intensity, unique_dists[valid_dists] - 1] += counts_dists[valid_dists]

            gldzm_matrices.append(gldzm)

        return np.array(gldzm_matrices), np.array(roi_voxels, dtype=float)
