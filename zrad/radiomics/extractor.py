from .extraction_context import ExtractionContext
from .extraction_preparation import build_extraction_metadata, prepare_extraction_data
from .feature_registry import resolve_groups


class Radiomics:
    """Extract radiomics features from an image and mask pair."""

    def __init__(
        self,
        aggr_dim='3D',
        aggr_method='AVER',
        intensity_range=None,
        outlier_range=None,
        number_of_bins=None,
        bin_size=None,
        calc_ivh_features=False,
        ivh_number_of_bins=None,
        ivh_bin_size=None,
        calc_morph_moran_i_and_geary_c_features=False,
        slice_weighting=False,
        slice_median=False,
    ):
        if slice_weighting and slice_median:
            raise ValueError('Only one slice median averaging is not supported with weighting strategy.')

        if aggr_dim not in ['2D', '2.5D', '3D']:
            raise ValueError(f"Wrong aggregation dim {aggr_dim}. Available '2D', '2.5D', and '3D'.")

        if aggr_method not in ['MERG', 'AVER', 'SLICE_MERG', 'DIR_MERG']:
            raise ValueError(
                f"Wrong aggregation dim {aggr_method}. Available 'MERG', 'AVER', 'SLICE_MERG', and 'DIR_MERG'."
            )

        self.aggr_dim = aggr_dim
        self.aggr_method = aggr_method
        self.intensity_range = intensity_range
        self.outlier_range = outlier_range
        self.number_of_bins = number_of_bins
        self.bin_size = bin_size
        self.calc_ivh_features = calc_ivh_features
        self.ivh_number_of_bins = ivh_number_of_bins
        self.ivh_bin_size = ivh_bin_size
        self.calc_morph_moran_i_and_geary_c_features = calc_morph_moran_i_and_geary_c_features
        self.slice_weighting = slice_weighting
        self.slice_median = slice_median

    def extract_features(self, image, mask, filtered_image=None, families=None, features=None, include_metadata=False):
        """Run feature extraction and return a flat feature dictionary."""
        context = self._build_context(image=image, mask=mask, filtered_image=filtered_image)
        groups, selected_features = resolve_groups(context, families=families, features=features)
        prepared_data = prepare_extraction_data(
            context=context,
            groups=groups,
            include_metadata=include_metadata,
        )

        extracted = {}
        for group in groups:
            extracted.update(group.calculate(context, prepared_data))

        if selected_features is not None:
            extracted = {name: extracted[name] for name in selected_features}
        if include_metadata:
            extracted.update(build_extraction_metadata(prepared_data))

        return extracted

    def _build_context(self, image, mask, filtered_image=None):
        return ExtractionContext(
            image=image,
            mask=mask,
            filtered_image=filtered_image,
            aggr_dim=self.aggr_dim,
            aggr_method=self.aggr_method,
            intensity_range=self.intensity_range,
            outlier_range=self.outlier_range,
            number_of_bins=self.number_of_bins,
            bin_size=self.bin_size,
            calc_ivh_features=self.calc_ivh_features,
            ivh_number_of_bins=self.ivh_number_of_bins,
            ivh_bin_size=self.ivh_bin_size,
            calc_morph_moran_i_and_geary_c_features=self.calc_morph_moran_i_and_geary_c_features,
            slice_weighting=self.slice_weighting,
            slice_median=self.slice_median,
        )
