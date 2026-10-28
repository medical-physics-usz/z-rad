from .base import BaseFeatureGroup
from .texture_aggregation import format_texture_feature_names
from .texture_base import TEXTURE_ATTRIBUTE_NAMES, ZoneMatrixFeatureBase
from .texture_matrices import TextureMatrixMixin

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


class GLDZM(TextureMatrixMixin, ZoneMatrixFeatureBase):
    """Grey level distance zone matrix features.

    GLDZM features describe connected grey-level zones together with their
    distance from the ROI border. They summarize how grey-level zones are
    distributed from boundary-adjacent to deeper ROI regions.

    Parameters
    ----------
    aggr_dim : {"2D", "2.5D", "3D"}
        Spatial dimensionality used to define zones and border distances.
    slice_weight : bool, default=False
        Weight 2D slice-wise averages by slice ROI voxel count.
    slice_median : bool, default=False
        Aggregate 2D slice-wise values by median instead of mean.
    """

    matrix_family = 'gldzm'

    def __init__(self, aggr_dim, slice_weight=False, slice_median=False):
        super().__init__(slice_weight=slice_weight, slice_median=slice_median)
        self.aggr_dim = aggr_dim

    def get_params(self):
        """Return the configuration parameters of this GLDZM calculator.

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
        """Return the GLDZM feature names produced by this calculator.

        Returns
        -------
        list of str
            Feature names defined for the GLDZM family.
        """
        return list(GLDZM_FEATURE_NAMES)

    @staticmethod
    def _map_feature_names(values):
        return dict(zip(GLDZM_FEATURE_NAMES, [values[name] for name in TEXTURE_ATTRIBUTE_NAMES]))

    def calculate_features(self, discretized_image_array, mask_array):
        """Calculate GLDZM features for prepared discretized intensities and a morphology mask.

        Parameters
        ----------
        discretized_image_array : numpy.ndarray
            Legacy (x, y, z) array. Use calculate_matrices for NumPy (z, y, x)
            or 2D (row, column) inputs. Prepared discretized intensity array with voxels outside the ROI set
            to ``NaN``.
        mask_array : numpy.ndarray
            Morphological ROI mask aligned with ``discretized_image_array``.

        Returns
        -------
        dict
            Mapping of GLDZM feature names to calculated values.
        """
        return self._legacy_features(discretized_image_array, mask_array)


class GLDZMFeatureGroup(BaseFeatureGroup):
    family = 'gldzm'
    requirements = frozenset({'analysis_masks', 'discretized_intensity_image'})

    def supports(self, context):
        return context.roi_data.texture_discretized_image is not None

    def output_names(self, context):
        return format_texture_feature_names(GLDZM_FEATURE_NAMES, context.aggr_dim)

    def feature_aliases(self, context):
        output_names = self.output_names(context)
        aliases = {name: name for name in output_names}
        aliases.update(dict(zip(GLDZM_FEATURE_NAMES, output_names)))
        return aliases

    def calculate(self, context, prepared_data):
        from .texture_extraction import calculate_texture_family

        return calculate_texture_family(self, context, prepared_data)[0]
