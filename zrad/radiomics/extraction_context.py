from dataclasses import dataclass

from ..image import Image
from ..preprocessing import RoiMasks


@dataclass(frozen=True)
class ExtractionContext:
    """Immutable extraction inputs and configuration."""

    image: Image
    mask: Image
    filtered_image: Image | None
    aggr_dim: str
    aggr_method: str
    intensity_range: tuple[float, float] | None
    outlier_range: float | None
    number_of_bins: int | None
    bin_size: float | None
    calc_ivh_features: bool
    ivh_number_of_bins: int | None
    ivh_bin_size: float | None
    calc_morph_moran_i_and_geary_c_features: bool
    slice_weighting: bool
    slice_median: bool

    @property
    def feature_image(self) -> Image:
        return self.filtered_image if self.filtered_image is not None else self.image

    @property
    def is_slice_2d(self) -> bool:
        return self.image.shape[2] == 1


@dataclass
class PreparedExtractionData:
    """Prepared intermediate data shared across feature groups."""

    base_masks: RoiMasks | None = None
    analysis_masks: RoiMasks | None = None
    discretized_intensity_image: Image | None = None
    ivh_intensity_image: Image | None = None

    def require_base_masks(self) -> RoiMasks:
        if self.base_masks is None:
            raise RuntimeError('Base masks were not prepared for this extraction.')
        return self.base_masks

    def require_analysis_masks(self) -> RoiMasks:
        if self.analysis_masks is None:
            raise RuntimeError('Analysis masks were not prepared for this extraction.')
        return self.analysis_masks

    def require_discretized_intensity_image(self) -> Image:
        if self.discretized_intensity_image is None:
            raise RuntimeError('Discretized intensity image was not prepared for this extraction.')
        return self.discretized_intensity_image

    def require_ivh_intensity_image(self) -> Image:
        if self.ivh_intensity_image is None:
            raise RuntimeError('IVH intensity image was not prepared for this extraction.')
        return self.ivh_intensity_image
