from dataclasses import dataclass

from ..image import Image
from ..preprocessing import RoiData


@dataclass(frozen=True)
class ExtractionContext:
    """Immutable extraction inputs and configuration."""

    roi_data: RoiData
    is_slice_2d_image: bool
    aggr_dim: str
    aggr_method: str
    slice_weighting: bool
    slice_median: bool

    @property
    def image(self) -> Image:
        return self.roi_data.image

    @property
    def mask(self) -> Image:
        return self.roi_data.morphological_mask

    @property
    def filtered_image(self) -> Image | None:
        return self.roi_data.filtered_image

    @property
    def feature_image(self) -> Image:
        return self.roi_data.feature_image

    @property
    def is_slice_2d(self) -> bool:
        return self.is_slice_2d_image


@dataclass
class PreparedExtractionData:
    """Prepared intermediate data shared across feature groups."""

    base_masks: RoiData | None = None
    analysis_masks: RoiData | None = None
    discretized_intensity_image: Image | None = None
    ivh_intensity_image: Image | None = None
    ivh_min_intensity: float | None = None
    ivh_max_intensity: float | None = None
    ivh_discretization_step: float = 1

    def require_base_masks(self) -> RoiData:
        if self.base_masks is None:
            raise RuntimeError('Base masks were not prepared for this extraction.')
        return self.base_masks

    def require_analysis_masks(self) -> RoiData:
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

    def require_ivh_parameters(self):
        if self.ivh_min_intensity is None or self.ivh_max_intensity is None:
            raise RuntimeError('IVH intensity range was not prepared for this extraction.')
        return self.ivh_min_intensity, self.ivh_max_intensity, self.ivh_discretization_step
