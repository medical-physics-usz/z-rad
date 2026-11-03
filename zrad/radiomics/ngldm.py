import numpy as np

from .base import BaseFeatureGroup
from .texture_aggregation import format_texture_feature_names
from .texture_base import NGLDM_ATTRIBUTE_NAMES, TextureFeatureBase
from .texture_matrices import TextureMatrixMixin

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


class NGLDM(TextureMatrixMixin, TextureFeatureBase):
    """Neighbouring grey level dependence matrix features.

    NGLDM features count neighbouring voxels that depend on the centre voxel's
    discretized grey level. They describe local homogeneity, dependence counts,
    and grey-level emphasis patterns.

    Parameters
    ----------
    aggr_dim : {"2D", "2.5D", "3D"}
        Spatial dimensionality used to count neighbouring dependencies.
    slice_weight : bool, default=False
        Weight 2D slice-wise averages by slice ROI voxel count.
    slice_median : bool, default=False
        Aggregate 2D slice-wise values by median instead of mean.
    """

    matrix_family = 'ngldm'

    def __init__(self, aggr_dim, slice_weight=False, slice_median=False):
        super().__init__(slice_weight=slice_weight, slice_median=slice_median)
        self.aggr_dim = aggr_dim

    def get_params(self):
        """Return the configuration parameters of this NGLDM calculator.

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
        """Return the NGLDM feature names produced by this calculator.

        Returns
        -------
        list of str
            Feature names defined for the NGLDM family.
        """
        return list(NGLDM_FEATURE_NAMES)

    @staticmethod
    def _map_feature_names(values):
        return dict(zip(NGLDM_FEATURE_NAMES, [values[name] for name in NGLDM_ATTRIBUTE_NAMES]))

    @staticmethod
    def _calc_3d_matrix(image, lvl):
        padded = np.pad(image, pad_width=1, mode='constant', constant_values=np.nan)
        center = padded[1:-1, 1:-1, 1:-1]
        neighbor_count = np.zeros_like(center, dtype=np.int64)

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = padded[
                        1 + dx : 1 + dx + center.shape[0],
                        1 + dy : 1 + dy + center.shape[1],
                        1 + dz : 1 + dz + center.shape[2],
                    ]
                    neighbor_count += neighbor == center

        ngldm = np.zeros((lvl, 27), dtype=np.int64)
        valid = ~np.isnan(center)
        intensities = center[valid].astype(int)
        counts = neighbor_count[valid]
        np.add.at(ngldm, (intensities, counts), 1)
        return ngldm

    @staticmethod
    def _calc_2d_matrices(image, lvl):
        ngldm_2d_matrices = []
        roi_voxel_counts = []
        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        def calc_ngldm_slice(array):
            padded = np.pad(array, pad_width=1, mode='constant', constant_values=np.nan)
            center = padded[1:-1, 1:-1]
            neighbor_count = np.zeros_like(center, dtype=int)

            for dx, dy in offsets:
                neighbor = padded[
                    1 + dx : 1 + dx + center.shape[0],
                    1 + dy : 1 + dy + center.shape[1],
                ]
                neighbor_count += neighbor == center

            ngldm = np.zeros((lvl, 9), dtype=np.int64)
            valid = ~np.isnan(center)
            intensities = center[valid].astype(int)
            counts = neighbor_count[valid]
            np.add.at(ngldm, (intensities, counts), 1)
            return ngldm

        for z_idx in range(image.shape[2]):
            slice_ = image[:, :, z_idx]
            roi_voxel_count = int(np.count_nonzero(~np.isnan(slice_)))
            if roi_voxel_count == 0:
                continue
            roi_voxel_counts.append(roi_voxel_count)
            ngldm_2d_matrices.append(calc_ngldm_slice(slice_))

        return np.array(ngldm_2d_matrices, dtype=np.int64), np.array(roi_voxel_counts, dtype=float)

    def calculate_features(self, discretized_image_array):
        """Calculate NGLDM features for a prepared discretized intensity array.

        Parameters
        ----------
        discretized_image_array : numpy.ndarray
            Legacy (x, y, z) array. Use calculate_matrices for NumPy (z, y, x)
            or 2D (row, column) inputs. Prepared discretized intensity array with voxels outside the ROI set
            to ``NaN``.

        Returns
        -------
        dict
            Mapping of NGLDM feature names to calculated values.
        """
        return self._legacy_features(discretized_image_array)


class NGLDMFeatureGroup(BaseFeatureGroup):
    family = 'ngldm'
    requirements = frozenset({'analysis_masks', 'discretized_intensity_image'})

    def supports(self, context):
        return context.roi_data.texture_discretized_image is not None

    def output_names(self, context):
        return format_texture_feature_names(NGLDM_FEATURE_NAMES, context.aggr_dim)

    def feature_aliases(self, context):
        output_names = self.output_names(context)
        aliases = {name: name for name in output_names}
        aliases.update(dict(zip(NGLDM_FEATURE_NAMES, output_names)))
        return aliases

    def calculate(self, context, prepared_data):
        from .texture_extraction import calculate_texture_family

        return calculate_texture_family(self, context, prepared_data)[0]
