from ..preprocessing import IntensityMaskBuilder, RoiData
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

    def extract_features(
        self,
        image=None,
        mask=None,
        filtered_image=None,
        families=None,
        features=None,
        include_metadata=False,
        roi_data=None,
    ):
        """Run feature extraction and return a flat feature dictionary."""
        context = self._build_context(
            image=image,
            mask=mask,
            filtered_image=filtered_image,
            roi_data=roi_data,
        )
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

    def _build_context(self, image=None, mask=None, filtered_image=None, roi_data=None):
        if roi_data is not None:
            if image is not None or mask is not None or filtered_image is not None:
                raise ValueError("Pass either roi_data or image/mask inputs, not both.")
            return self._build_context_from_roi_data(roi_data)

        if image is None or mask is None:
            raise ValueError("Either roi_data or both image and mask must be provided.")

        return ExtractionContext(
            roi_data=IntensityMaskBuilder().apply(RoiData(
                image=image,
                filtered_image=filtered_image,
                morphological_mask=mask,
            )),
            resegment_roi_data=True,
            is_slice_2d_image=image.shape[2] == 1,
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

    def _build_context_from_roi_data(self, roi_data):
        self._validate_roi_data(roi_data)
        return ExtractionContext(
            roi_data=roi_data,
            resegment_roi_data=False,
            is_slice_2d_image=roi_data.image.shape[2] == 1,
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

    @staticmethod
    def _validate_roi_data(roi_data):
        if not isinstance(roi_data, RoiData):
            raise TypeError("roi_data must be an instance of zrad.preprocessing.RoiData.")
        required_fields = {
            "image": roi_data.image,
            "morphological_mask": roi_data.morphological_mask,
            "intensity_mask": roi_data.intensity_mask,
        }
        missing = [name for name, value in required_fields.items() if value is None]
        if missing:
            raise ValueError(f"roi_data is missing required field(s): {', '.join(missing)}.")
