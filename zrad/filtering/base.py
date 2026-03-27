import sys

import numpy as np

from ..image import Image
from ..toolbox_logic import handle_uncaught_exception

sys.excepthook = handle_uncaught_exception


class BaseFilter:
    """Common image-oriented API for all concrete filters."""

    def __init__(self, filtering_method, **filtering_params):
        self.filtering_method = filtering_method
        self.filtering_params = filtering_params
        self.filter = self

    def _prepare(self, image):
        """Hook for subclasses that need image metadata before filtering."""

    def _apply_array(self, img):
        raise NotImplementedError

    def apply(self, image):
        """Apply the filter to an :class:`~zrad.image.Image` and return an image."""
        if not isinstance(image, Image):
            raise TypeError(f"Expected Image, got {type(image)}.")

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
