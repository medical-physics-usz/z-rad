import numpy as np

from ..image import Image
from ..preprocessing.roi import RoiData


class BaseFilter:
    """Common image-oriented API for all concrete filters."""

    def __init__(self, filtering_method, **filtering_params):
        self.filtering_method = filtering_method
        self.filtering_params = filtering_params

    def get_params(self):
        """Return filter parameters mapped to their configured values.

        Returns
        -------
        params : dict
            Constructor parameters stored by the filter instance.
        """
        return dict(self.filtering_params)

    def _prepare(self, image):
        """Hook for subclasses that need image metadata before filtering."""

    def _apply_array(self, img):
        raise NotImplementedError

    def apply(self, image):
        """Apply the filter to an image or set ``RoiData.filtered_image``.

        Parameters
        ----------
        image : Image or RoiData
            Input image to filter. If ``RoiData`` is supplied, filtering is
            applied to ``image.image`` and the result is stored as
            ``filtered_image`` in the returned ROI data. Existing intensity,
            texture, and IVH prepared fields are cleared.

        Returns
        -------
        filtered : Image or RoiData
            Filtered image, or ROI data with ``filtered_image`` updated.
        """
        if isinstance(image, RoiData):
            return RoiData(
                image=image.image,
                filtered_image=self.apply(image.image),
                morphological_mask=image.morphological_mask,
                intensity_mask=None,
            )
        if not isinstance(image, Image):
            raise TypeError(f"Expected Image or RoiData, got {type(image)}.")

        self._prepare(image)

        arr = image.array.astype(np.float64).transpose(1, 2, 0)
        try:
            filtered = self._apply_array(arr)
        except Exception as e:
            raise ValueError(f"Error applying filter: {e}")

        out_arr = filtered.transpose(2, 0, 1)
        return Image(
            array=out_arr,
            origin=image.origin,
            spacing=image.spacing,
            direction=image.direction,
            shape=image.shape
        )
