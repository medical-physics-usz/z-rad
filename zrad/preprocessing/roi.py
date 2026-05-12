from dataclasses import dataclass

import numpy as np

from ..image import Image


@dataclass
class RoiData:
    """Current image and ROI masks used for feature calculation.

    ``RoiData`` groups the image and ROI representations that move through a
    preprocessing pipeline. The morphological mask defines the anatomical ROI;
    the intensity mask stores feature-image values inside that ROI and ``NaN``
    outside it. Derived discretization fields are cleared by preprocessing
    steps that change the image, mask, or valid intensity population.

    Parameters
    ----------
    image : Image
        Original image used as the spatial reference for ROI processing.
    filtered_image : Image, optional
        Filtered image used for intensity-based feature calculation. If this is
        ``None``, ``image`` is used as the feature image.
    morphological_mask : Image, optional
        Binary ROI mask used for morphology-based feature calculation and as
        the base mask for intensity ROI construction.
    intensity_mask : Image, optional
        Image containing feature-image intensities inside the ROI and ``NaN``
        outside it. This mask is used by intensity-based feature families.
    intensity_range : tuple[float, float], optional
        Absolute re-segmentation range applied to ``intensity_mask``. The lower
        bound is reused as the fixed-bin-size discretization anchor.
    texture_discretized_image : Image, optional
        Discretized intensity image used by intensity histogram and texture
        feature families.
    ivh_intensity_image : Image, optional
        Intensity image prepared for intensity-volume histogram features.
    ivh_discretization_method : str, optional
        IVH preparation method used for ``ivh_intensity_image``.
    ivh_discretization_step : float, optional
        Intensity interval used to sample IVH features.
    """

    image: Image
    filtered_image: Image | None = None
    morphological_mask: Image | None = None
    intensity_mask: Image | None = None
    intensity_range: tuple[float, float] | None = None
    texture_discretized_image: Image | None = None
    ivh_intensity_image: Image | None = None
    ivh_discretization_method: str | None = None
    ivh_discretization_step: float | None = None

    @property
    def feature_image(self):
        """Return the image used for intensity-based feature calculation.

        Returns
        -------
        image : Image
            ``filtered_image`` when present; otherwise the original ``image``.
        """
        return self.filtered_image if self.filtered_image is not None else self.image


class IntensityMaskBuilder:
    """Build the intensity ROI image used by intensity-based feature families.

    The builder keeps the morphological mask binary, selects the current
    feature image, writes an ``intensity_mask`` whose voxels outside the ROI
    are set to ``NaN``, and clears any prepared texture or IVH images. This
    mirrors the IBSI distinction between morphology and intensity masks.

    This class has no constructor parameters.
    """

    def get_params(self):
        """Return intensity-mask-building parameters.

        Returns
        -------
        params : dict
            Empty dictionary because this class has no constructor parameters.
        """
        return {}

    def apply(self, roi_data):
        """Return ROI data with ``intensity_mask`` built from the feature image.

        Parameters
        ----------
        roi_data : RoiData
            ROI data containing ``image`` and ``morphological_mask``. If
            ``filtered_image`` is present, its voxel values are used inside the
            intensity ROI.

        Returns
        -------
        roi_data : RoiData
            New ROI data with a binary ``morphological_mask`` and an
            ``intensity_mask`` image containing feature-image values inside the
            ROI and ``NaN`` outside it.
        """
        if roi_data.morphological_mask is None:
            raise ValueError("IntensityMaskBuilder requires RoiData.morphological_mask.")

        morphological_mask = roi_data.morphological_mask.copy()
        morphological_mask.array = morphological_mask.array.astype(np.int8)

        feature_image = roi_data.feature_image
        intensity_mask = morphological_mask.copy()
        intensity_mask.array = np.where(morphological_mask.array > 0, feature_image.array, np.nan)
        return RoiData(
            image=roi_data.image,
            filtered_image=roi_data.filtered_image,
            morphological_mask=morphological_mask,
            intensity_mask=intensity_mask,
        )


class RoiCropper:
    """Crop aligned images and masks to the ROI bounding box.

    Cropping reduces the working image domain to the non-zero morphological ROI
    plus optional padding. Image geometry metadata is updated so the cropped
    image remains located correctly in physical space. Prepared texture and IVH
    images are cropped with the same bounding box, and re-segmentation and IVH
    metadata are preserved.

    Parameters
    ----------
    padding : int or sequence of int, optional
        Number of voxels to keep around the ROI bounding box. A single integer
        applies the same padding to all array axes. A sequence must contain one
        value per array axis in ``(z, y, x)`` order. The default is ``0``.

    """

    def __init__(self, padding=0):
        self.padding = padding

    def get_params(self):
        """Return ROI cropping parameters mapped to their configured values.

        Returns
        -------
        params : dict
            Dictionary containing ``padding``.
        """
        return {
            'padding': self.padding,
        }

    def apply(self, roi_data):
        """Return ROI data cropped to the morphological ROI bounding box.

        Parameters
        ----------
        roi_data : RoiData
            ROI data with ``morphological_mask`` and ``intensity_mask``.

        Returns
        -------
        roi_data : RoiData
            New ROI data with all available images cropped to the same bounding
            box.
        """
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
            intensity_range=roi_data.intensity_range,
            texture_discretized_image=(
                None
                if roi_data.texture_discretized_image is None
                else self._crop_image(roi_data.texture_discretized_image, bbox_slices)
            ),
            ivh_intensity_image=(
                None
                if roi_data.ivh_intensity_image is None
                else self._crop_image(roi_data.ivh_intensity_image, bbox_slices)
            ),
            ivh_discretization_method=roi_data.ivh_discretization_method,
            ivh_discretization_step=roi_data.ivh_discretization_step,
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
