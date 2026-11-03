import numpy as np

from ..preprocessing import RoiData
from .extraction_context import ExtractionContext
from .extraction_preparation import build_extraction_metadata, prepare_extraction_data
from .feature_registry import resolve_groups
from .texture_extraction import ExtractionResult, build_texture_family, calculate_texture_family
from .texture_matrices import TEXTURE_FAMILIES


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

    ``"morphology"`` includes Moran's I and Geary's C for 3D ROIs, including
    default extraction. Both share an exact hybrid FFT or blocked calculation.
    They can also be selected by their individual feature names.

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
        *,
        texture_options=None,
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
            ``"ngtdm"``, ``"ngldm"``, and ``"ivh"``.

            If omitted, all default-enabled families supported by the prepared
            ``RoiData`` are extracted. Use ``"all"`` to extract every
            supported family. Morphology includes Moran's I and Geary's C.
            Repeated family selections are calculated once.
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

        texture_options : dict, optional
            Per-family matrix options, e.g. {"glcm": {"directions": [(0, 0, 1)]}}.
            Offsets use Image.array order (z, y, x). Asymmetric GLCM feature
            extraction is not supported.

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

        * ``"morphology"`` requires a 3D ROI.
        * ``"intensity_histogram"`` and texture families require
          ``texture_discretized_image``.
        * ``"ivh"`` requires ``ivh_intensity_image`` and IVH discretization
          metadata.
        * ``"local_intensity"`` and ``"intensity_statistics"`` use the
          non-discretized intensity mask.
        """
        return self._extract(roi_data, families, features, include_metadata, texture_options, False)

    def calculate_texture_matrices(self, roi_data, families=None, *, texture_options=None):
        """Build unmerged texture matrices from the effective extraction ROI.

        Defaults to all six texture families. Applies the same ROI validation
        and prepared discretization as extract_features, without evaluating any
        features. Returns a family-to-TextureMatrixCollection mapping. Unlike
        feature extraction, GLCM symmetric=False is supported here. For tiny
        arrays without extraction geometry checks, use a family calculator's
        calculate_matrices method instead.
        """
        context = self._build_context(roi_data)
        if families is None or (isinstance(families, str) and families == 'all'):
            families = TEXTURE_FAMILIES
        groups, _ = resolve_groups(context, families=families)
        if any(group.family not in TEXTURE_FAMILIES for group in groups):
            raise ValueError('calculate_texture_matrices accepts texture families only.')
        options = dict(texture_options or {})
        if set(options) - {group.family for group in groups}:
            raise ValueError('Texture options must refer to selected families.')
        prepared = prepare_extraction_data(context, groups)
        return {
            group.family: build_texture_family(group, context, prepared, options=options.get(group.family))[1]
            for group in groups
        }

    def extract_with_details(
        self,
        roi_data=None,
        families=None,
        features=None,
        include_metadata=False,
        *,
        retain_matrices="all",
        texture_options=None,
    ):
        """Extract features and retain the exact texture calculation trace.

        Returns an ExtractionResult with features, per-family texture traces,
        effective discretized_image and morphological_mask arrays, and indices
        of slices removed by ROI validation. Traces contain raw matrices, exact
        formula inputs, per-matrix features, merge provenance and reduction
        weights. ``retain_matrices`` currently supports only ``"all"``; ordinary
        ``extract_features`` does not retain matrices after returning.

        Family/feature selection and texture_options match extract_features.
        """
        if retain_matrices != 'all':
            raise ValueError('retain_matrices currently supports only "all".')
        return self._extract(roi_data, families, features, include_metadata, texture_options, True)

    def _extract(self, roi_data, families, features, include_metadata, texture_options, retain):
        context = self._build_context(roi_data)
        groups, selected_features = resolve_groups(context, families=families, features=features)
        prepared_data = prepare_extraction_data(
            context=context,
            groups=groups,
            include_metadata=include_metadata,
        )

        texture_options = dict(texture_options or {})
        selected_families = {group.family for group in groups}
        invalid_options = set(texture_options) - (selected_families & set(TEXTURE_FAMILIES))
        if invalid_options:
            raise ValueError(f'Texture options refer to unselected or unsupported families: {sorted(invalid_options)}.')
        extracted = {}
        traces = {}
        for group in groups:
            if group.family in TEXTURE_FAMILIES:
                values, trace = calculate_texture_family(
                    group,
                    context,
                    prepared_data,
                    retain=retain,
                    options=texture_options.get(group.family),
                )
                extracted.update(values)
                if retain:
                    traces[group.family] = trace
            elif selected_features is None:
                extracted.update(group.calculate(context, prepared_data))
            else:
                group_features = [name for name in selected_features if name in group.output_names(context)]
                extracted.update(group.calculate_selected(context, prepared_data, group_features))

        if selected_features is not None:
            extracted = {name: extracted[name] for name in selected_features}
        if include_metadata:
            extracted.update(build_extraction_metadata(prepared_data))

        if retain:
            masks = prepared_data.analysis_masks
            morphology = None if masks is None else masks.morphological_mask.array
            discretized = prepared_data.discretized_intensity_image
            excluded = ()
            if morphology is not None:
                before = np.any(roi_data.morphological_mask.array > 0, axis=(1, 2))
                after = np.any(morphology > 0, axis=(1, 2))
                excluded = tuple(int(i) for i in np.flatnonzero(before & ~after))
            return ExtractionResult(
                extracted, traces, None if discretized is None else discretized.array, morphology, excluded
            )
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
