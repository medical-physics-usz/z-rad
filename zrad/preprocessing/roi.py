from dataclasses import dataclass

import numpy as np

from ..image import Image


@dataclass
class RoiData:
    """Current image and ROI masks used for feature calculation."""

    image: Image
    filtered_image: Image | None = None
    morphological_mask: Image | None = None
    intensity_mask: Image | None = None

    @property
    def feature_image(self):
        """Return the image used for intensity-based feature calculation."""
        return self.filtered_image if self.filtered_image is not None else self.image


class RoiMaskBuilder:
    """Build morphological and intensity ROI masks from an image and binary ROI mask."""

    def get_params(self):
        """Return ROI mask-building parameters mapped to their configured values."""
        return {}

    def apply(self, image, mask, filtered_image=None):
        """Return morphological and intensity masks for ``image`` inside ``mask``."""
        morphological_mask = mask.copy()
        morphological_mask.array = morphological_mask.array.astype(np.int8)

        feature_image = filtered_image if filtered_image is not None else image
        intensity_mask = mask.copy()
        intensity_mask.array = np.where(mask.array > 0, feature_image.array, np.nan)
        return RoiData(
            image=image,
            filtered_image=filtered_image,
            morphological_mask=morphological_mask,
            intensity_mask=intensity_mask,
        )


class IntensityRoiMaskBuilder:
    """Apply an ROI mask to an image and return an intensity image with NaNs outside the ROI."""

    def get_params(self):
        """Return ROI mask-building parameters mapped to their configured values."""
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


class RoiCropper:
    """Crop aligned images and masks to the ROI bounding box."""

    def __init__(self, padding=0):
        self.padding = padding

    def get_params(self):
        """Return ROI cropping parameters mapped to their configured values."""
        return {
            'padding': self.padding,
        }

    def apply(self, roi_data):
        """Return ROI data cropped to the morphological ROI bounding box."""
        if roi_data.morphological_mask is None or roi_data.intensity_mask is None:
            raise ValueError("RoiCropper requires RoiData with morphological and intensity masks.")

        bbox_slices = self._bounding_box_slices(roi_data.morphological_mask.array)
        return RoiData(
            image=self._crop_image(roi_data.image, bbox_slices),
            filtered_image=(
                None
                if roi_data.filtered_image is None
                else self._crop_image(roi_data.filtered_image, bbox_slices)
            ),
            morphological_mask=self._crop_image(roi_data.morphological_mask, bbox_slices),
            intensity_mask=self._crop_image(roi_data.intensity_mask, bbox_slices),
        )

    def _bounding_box_slices(self, mask_array):
        coords = np.argwhere(mask_array > 0)
        if coords.size == 0:
            raise ValueError("Cannot crop an empty ROI mask.")

        padding = self._normalize_padding(mask_array.ndim)
        starts = np.maximum(coords.min(axis=0) - padding, 0)
        stops = np.minimum(coords.max(axis=0) + padding + 1, mask_array.shape)
        return tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))

    def _normalize_padding(self, ndim):
        if isinstance(self.padding, int):
            return np.repeat(self.padding, ndim)

        padding = np.asarray(self.padding, dtype=int)
        if padding.size != ndim:
            raise ValueError(f"Padding must be an int or contain {ndim} values.")
        return padding

    def _crop_image(self, image, bbox_slices):
        cropped_array = image.array[bbox_slices]
        return Image(
            array=cropped_array,
            origin=self._cropped_origin(image, bbox_slices),
            spacing=image.spacing,
            direction=image.direction,
            shape=tuple(cropped_array.shape[::-1]),
        )

    @staticmethod
    def _cropped_origin(image, bbox_slices):
        if image.origin is None or image.spacing is None or image.direction is None:
            return image.origin

        array_starts = np.array([bbox_slice.start for bbox_slice in bbox_slices], dtype=float)
        physical_starts = np.array(
            [
                array_starts[2] * image.spacing[0],
                array_starts[1] * image.spacing[1],
                array_starts[0] * image.spacing[2],
            ]
        )
        direction = np.asarray(image.direction, dtype=float).reshape(3, 3)
        return tuple(np.asarray(image.origin, dtype=float) + direction @ physical_starts)
