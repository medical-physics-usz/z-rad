from .extraction_context import ExtractionContext
from .extraction_preparation import build_extraction_metadata, prepare_extraction_data
from .feature_registry import resolve_groups
from ..preprocessing import RoiData


class Radiomics:
    """Extract radiomics features from prepared ROI data.

    ``Radiomics`` consumes a fully prepared ``RoiData`` instance. Preprocessing
    steps are responsible for building the intensity mask, applying
    re-segmentation, and preparing texture or IVH intensity images before
    extraction.

    Parameters
    ----------
    aggr_dim : {"2D", "2.5D", "3D"}, default="3D"
        Spatial aggregation dimensionality for texture features.
    aggr_method : {"MERG", "AVER", "SLICE_MERG", "DIR_MERG"}, default="AVER"
        Texture aggregation strategy across directions and slices.
    slice_weighting : bool, default=False
        Weight 2D slice-wise texture averages by slice ROI size.
    slice_median : bool, default=False
        Aggregate 2D slice-wise texture values by median instead of mean.
    """

    def __init__(
        self,
        aggr_dim='3D',
        aggr_method='AVER',
        slice_weighting=False,
        slice_median=False,
    ):
        if slice_weighting and slice_median:
            raise ValueError('Slice median averaging is not supported with weighting strategy.')

        if aggr_dim not in ['2D', '2.5D', '3D']:
            raise ValueError(f"Wrong aggregation dim {aggr_dim}. Available '2D', '2.5D', and '3D'.")

        if aggr_method not in ['MERG', 'AVER', 'SLICE_MERG', 'DIR_MERG']:
            raise ValueError(
                f"Wrong aggregation method {aggr_method}. Available 'MERG', 'AVER', 'SLICE_MERG', and 'DIR_MERG'."
            )

        self.aggr_dim = aggr_dim
        self.aggr_method = aggr_method
        self.slice_weighting = slice_weighting
        self.slice_median = slice_median

    def extract_features(
        self,
        roi_data=None,
        families=None,
        features=None,
        include_metadata=False,
    ):
        """Run radiomics feature extraction.

        Parameters
        ----------
        roi_data : RoiData
            Prepared ROI data containing at least ``image``,
            ``morphological_mask``, and ``intensity_mask``. Texture and IVH
            feature families additionally require their corresponding prepared
            fields on ``RoiData``.
        families : list of str, optional
            Feature families to extract. If omitted, all families available
            from the prepared ``RoiData`` are extracted.
        features : list of str, optional
            Individual feature names to extract. Names may come from one or
            more feature families.
        include_metadata : bool, default=False
            If ``True``, append extraction metadata to the returned dictionary.

        Returns
        -------
        features : dict
            Flat dictionary mapping feature names to calculated values.
        """
        context = self._build_context(roi_data)
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

    def _build_context(self, roi_data):
        self._validate_roi_data(roi_data)
        return ExtractionContext(
            roi_data=roi_data,
            is_slice_2d_image=roi_data.image.shape[2] == 1,
            aggr_dim=self.aggr_dim,
            aggr_method=self.aggr_method,
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
