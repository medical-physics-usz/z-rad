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

    Supported feature families are:

    * ``"morphology"``
    * ``"local_intensity"``
    * ``"intensity_statistics"``
    * ``"intensity_histogram"``
    * ``"glcm"``
    * ``"glrlm"``
    * ``"glszm"``
    * ``"gldzm"``
    * ``"ngtdm"``
    * ``"ngldm"``
    * ``"ivh"``
    * ``"morphology_correlation"``

    ``"morphology_correlation"`` contains Moran's I and Geary's C. It is
    supported for 3D ROIs but is not extracted by default because it can be
    computationally expensive.

    Parameters
    ----------
    aggr_dim : {"2D", "2.5D", "3D"}, default="3D"
        Spatial aggregation dimensionality for texture features. This affects
        GLCM, GLRLM, GLSZM, GLDZM, NGTDM, and NGLDM feature names and values.
    aggr_method : {"MERG", "AVER", "SLICE_MERG", "DIR_MERG"}, default="AVER"
        Texture aggregation strategy across directions and slices. This is used
        by GLCM and GLRLM features.
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
        families : str or sequence of str, optional
            Feature families to extract. Supported names are:
            ``"morphology"``, ``"local_intensity"``,
            ``"intensity_statistics"``, ``"intensity_histogram"``,
            ``"glcm"``, ``"glrlm"``, ``"glszm"``, ``"gldzm"``,
            ``"ngtdm"``, ``"ngldm"``, ``"ivh"``, and
            ``"morphology_correlation"``.

            If omitted, all default-enabled families supported by the prepared
            ``RoiData`` are extracted. Use ``"all"`` to extract every
            supported family, including ``"morphology_correlation"``.
            Repeated family names are ignored.
        features : str or sequence of str, optional
            Individual feature names to extract. Names may come from one or
            more feature families. Use either ``families`` or ``features``,
            not both. For texture features, either configured output names or
            base feature names can be supplied. Base names are mapped to the
            configured output names for the current aggregation settings.
        include_metadata : bool, default=False
            If ``True``, append extraction metadata to the returned dictionary.
            Metadata currently includes the minimum bounding-box side length,
            voxel count, and number of discretized texture bins.

        Returns
        -------
        features : dict
            Flat dictionary mapping feature names to calculated values.

        Raises
        ------
        TypeError
            If ``roi_data`` is not a ``RoiData`` instance.
        ValueError
            If required ROI fields are missing, if an unknown family or feature
            is requested, or if both ``families`` and ``features`` are set.
        DataStructureError
            If a requested family is not supported for the current image shape
            or prepared ROI data.

        Notes
        -----
        Feature availability depends on the prepared ``RoiData``:

        * ``"morphology"`` and ``"morphology_correlation"`` require a 3D ROI.
        * ``"intensity_histogram"`` and texture families require
          ``texture_discretized_image``.
        * ``"ivh"`` requires ``ivh_intensity_image`` and IVH discretization
          metadata.
        * ``"local_intensity"`` and ``"intensity_statistics"`` use the
          non-discretized intensity mask.
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
