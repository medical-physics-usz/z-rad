import numpy as np

from ..image import Image
from .masks import RoiMasks


class RoiExtractor:
    """Extract ROI masks from an image and binary ROI mask."""

    def get_params(self):
        """Return ROI extraction parameters mapped to their configured values."""
        return {}

    def apply(self, image, mask):
        """Return morphological and intensity masks for ``image`` inside ``mask``."""
        morphological_mask = mask.copy()
        morphological_mask.array = morphological_mask.array.astype(np.int8)

        intensity_mask = mask.copy()
        intensity_mask.array = np.where(mask.array > 0, image.array, np.nan)
        return RoiMasks(
            morphological_mask=morphological_mask,
            intensity_mask=intensity_mask,
        )


class IntensityRoiExtractor:
    """Apply an ROI mask to an image and return an intensity image with NaNs outside the ROI."""

    def get_params(self):
        """Return ROI extraction parameters mapped to their configured values."""
        return {}

    def apply(self, image, mask):
        """Return a copy of ``image`` with voxels outside ``mask`` set to ``NaN``."""
        return Image(
            array=np.where(mask.array > 0, image.array, np.nan),
            origin=image.origin,
            spacing=image.spacing,
            direction=image.direction,
            shape=image.shape,
        )
