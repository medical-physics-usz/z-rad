import numpy as np

from ..preprocessing import IntensityMaskBuilder, RoiData
from .extraction_context import ExtractionContext
from .extraction_preparation import build_extraction_metadata, prepare_extraction_data
from .feature_registry import resolve_groups


class Radiomics:
    """Extract radiomics features from an image and ROI mask.

    This workflow class prepares ROI masks, optional re-segmentation,
    discretization, and IBSI-style feature aggregation before calculating the
    selected feature families.

    Parameters
    ----------
    aggr_dim : {"2D", "2.5D", "3D"}, default="3D"
        Spatial aggregation dimensionality for texture features.
    aggr_method : {"MERG", "AVER", "SLICE_MERG", "DIR_MERG"}, default="AVER"
        Texture aggregation strategy across directions and slices.
    intensity_range : tuple[float, float] or None, optional
        Absolute intensity limits used for range re-segmentation. Required when
        fixed-bin-size discretization is configured.
    outlier_range : float, str, or None, optional
        Standard-deviation multiplier used for outlier re-segmentation.
    number_of_bins : int or None, optional
        Number of bins for fixed-bin-number discretization. Mutually exclusive
        with ``bin_size``.
    bin_size : float or None, optional
        Bin width for fixed-bin-size discretization. Mutually exclusive with
        ``number_of_bins`` and requires ``intensity_range`` so its lower bound
        can be used as a stable bin origin.
    calc_ivh_features : bool, default=False
        If true, include intensity-volume histogram features. When no
        IVH-specific discretization is supplied, IVH uses retained intensity
        values directly and uses ``intensity_range`` as the IVH range when
        available.
    ivh_number_of_bins : int or None, optional
        Number of bins for IVH discretization. Mutually exclusive with
        ``ivh_bin_size``. If neither IVH discretization parameter is supplied,
        IVH features use the retained intensity values directly.
    ivh_bin_size : float or None, optional
        Bin width for IVH discretization. Mutually exclusive with
        ``ivh_number_of_bins`` and requires ``intensity_range``.
    calc_morph_moran_i_and_geary_c_features : bool, default=False
        If true, include morphology correlation features.
    slice_weighting : bool, default=False
        Weight 2D slice-wise texture averages by slice ROI size.
    slice_median : bool, default=False
        Aggregate 2D slice-wise texture values by median instead of mean.
    """

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

        self._validate_resegmentation(intensity_range, outlier_range)
        self._validate_discretization(
            number_of_bins=number_of_bins,
            bin_size=bin_size,
            intensity_range=intensity_range,
            label='radiomics',
        )
        self._validate_ivh_discretization(
            ivh_number_of_bins=ivh_number_of_bins,
            ivh_bin_size=ivh_bin_size,
            intensity_range=intensity_range,
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
        """Run radiomics feature extraction.

        Parameters
        ----------
        image : Image, optional
            Original image used when ``roi_data`` is not provided.
        mask : Image, optional
            Morphological ROI mask used when ``roi_data`` is not provided.
        filtered_image : Image, optional
            Filtered image used for intensity-based feature calculation when
            ``roi_data`` is not provided.
        families : list of str, optional
            Feature families to extract. If omitted, all registered families are
            extracted.
        features : list of str, optional
            Individual feature names to extract. Names may come from one or
            more feature families.
        include_metadata : bool, default=False
            If ``True``, append extraction metadata to the returned dictionary.
        roi_data : RoiData, optional
            Preprocessed ROI data. When this is provided, it must contain
            ``image``, ``morphological_mask``, and ``intensity_mask``. Passing
            ``roi_data`` disables the automatic intensity-mask building and
            re-segmentation path used for ``image``/``mask`` inputs.

        Returns
        -------
        features : dict
            Flat dictionary mapping feature names to calculated values.
        """
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

    @staticmethod
    def _validate_resegmentation(intensity_range, outlier_range):
        if intensity_range is not None:
            if (
                not isinstance(intensity_range, (list, tuple))
                or len(intensity_range) != 2
                or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in intensity_range)
            ):
                raise ValueError("intensity_range must be a two-value numeric sequence.")
            lower, upper = intensity_range
            if not np.isfinite(lower) or np.isnan(upper) or lower > upper:
                raise ValueError("intensity_range must have a finite lower bound and lower <= upper.")

        if outlier_range is not None:
            try:
                outlier_range = float(outlier_range)
            except (TypeError, ValueError) as exc:
                raise ValueError("outlier_range must be a positive number.") from exc
            if not np.isfinite(outlier_range) or outlier_range <= 0:
                raise ValueError("outlier_range must be a positive number.")

    @staticmethod
    def _validate_discretization(number_of_bins, bin_size, intensity_range, label):
        if number_of_bins is not None and bin_size is not None:
            raise ValueError(f"Specify only one of {label} number_of_bins or bin_size.")
        if number_of_bins is not None and (
            not isinstance(number_of_bins, int) or isinstance(number_of_bins, bool) or number_of_bins <= 0
        ):
            raise ValueError(f"{label} number_of_bins must be a positive integer.")
        if bin_size is not None:
            if not isinstance(bin_size, (int, float)) or isinstance(bin_size, bool) or bin_size <= 0:
                raise ValueError(f"{label} bin_size must be a positive number.")
            if intensity_range is None:
                raise ValueError(f"{label} bin_size requires intensity_range to define a stable lower anchor.")

    @classmethod
    def _validate_ivh_discretization(cls, ivh_number_of_bins, ivh_bin_size, intensity_range):
        cls._validate_discretization(
            number_of_bins=ivh_number_of_bins,
            bin_size=ivh_bin_size,
            intensity_range=intensity_range,
            label='IVH',
        )
