from .base import BaseFeatureGroup
from .texture_aggregation import format_texture_feature_names
from .texture_base import TEXTURE_ATTRIBUTE_NAMES, ZoneMatrixFeatureBase
from .texture_matrices import TextureMatrixMixin

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


class GLSZM(TextureMatrixMixin, ZoneMatrixFeatureBase):
    """Grey level size zone matrix features.

    GLSZM features describe connected zones of equal discretized grey level and
    their sizes. They quantify small versus large zones and low- versus
    high-grey-level zone patterns.

    Parameters
    ----------
    aggr_dim : {"2D", "2.5D", "3D"}
        Spatial dimensionality used to define connected zones.
    slice_weight : bool, default=False
        Weight 2D slice-wise averages by slice ROI voxel count.
    slice_median : bool, default=False
        Aggregate 2D slice-wise values by median instead of mean.
    """

    matrix_family = 'glszm'

    def __init__(self, aggr_dim, slice_weight=False, slice_median=False):
        super().__init__(slice_weight=slice_weight, slice_median=slice_median)
        self.aggr_dim = aggr_dim

    def get_params(self):
        """Return the configuration parameters of this GLSZM calculator.

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
        """Return the GLSZM feature names produced by this calculator.

        Returns
        -------
        list of str
            Feature names defined for the GLSZM family.
        """
        return list(GLSZM_FEATURE_NAMES)

    @staticmethod
    def _map_feature_names(values):
        return dict(zip(GLSZM_FEATURE_NAMES, [values[name] for name in TEXTURE_ATTRIBUTE_NAMES]))

    def calculate_features(self, discretized_image_array):
        """Calculate GLSZM features for prepared discretized intensities.

        Parameters
        ----------
        discretized_image_array : numpy.ndarray
            Legacy (x, y, z) array. Use calculate_matrices for NumPy (z, y, x)
            or 2D (row, column) inputs. Prepared discretized intensity array with voxels outside the ROI set
            to ``NaN``.

        Returns
        -------
        dict
            Mapping of GLSZM feature names to calculated values.
        """
        return self._legacy_features(discretized_image_array)


class GLSZMFeatureGroup(BaseFeatureGroup):
    family = 'glszm'
    requirements = frozenset({'discretized_intensity_image'})

    def supports(self, context):
        return context.roi_data.texture_discretized_image is not None

    def output_names(self, context):
        return format_texture_feature_names(GLSZM_FEATURE_NAMES, context.aggr_dim)

    def feature_aliases(self, context):
        output_names = self.output_names(context)
        aliases = {name: name for name in output_names}
        aliases.update(dict(zip(GLSZM_FEATURE_NAMES, output_names)))
        return aliases

    def calculate(self, context, prepared_data):
        from .texture_extraction import calculate_texture_family

        return calculate_texture_family(self, context, prepared_data)[0]
