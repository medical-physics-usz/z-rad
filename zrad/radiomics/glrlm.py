import numpy as np

from .base import BaseFeatureGroup
from .texture_aggregation import format_cm_rlm_feature_names
from .texture_base import TEXTURE_ATTRIBUTE_NAMES, TextureFeatureBase
from .texture_matrices import TextureMatrixMixin

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


class GLRLM(TextureMatrixMixin, TextureFeatureBase):
    """Grey level run length matrix features.

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

    matrix_family = 'glrlm'

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
            valid_mask[current_slices] & valid_mask[neighbor_slices] & (image[current_slices] == image[neighbor_slices])
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

    def calculate_features(self, discretized_image_array):
        """Calculate GLRLM features for a prepared discretized intensity array.

        Parameters
        ----------
        discretized_image_array : numpy.ndarray
            Legacy (x, y, z) array. Use calculate_matrices for NumPy (z, y, x)
            or 2D (row, column) inputs. Prepared discretized intensity array with ROI voxels represented by
            integer grey levels and voxels outside the ROI set to ``NaN``.

        Returns
        -------
        dict
            Mapping of GLRLM feature names to calculated values.
        """
        return self._legacy_features(discretized_image_array)


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
        from .texture_extraction import calculate_texture_family

        return calculate_texture_family(self, context, prepared_data)[0]
