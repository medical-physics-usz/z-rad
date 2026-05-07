import numpy as np

from ..exceptions import DataStructureError
from ..image import Image


class FixedBinSizeDiscretizer:
    """Discretize intensities using a fixed bin size."""

    def __init__(self, bin_size, minimum=None):
        self.bin_size = bin_size
        self.minimum = minimum

    def get_params(self):
        """Return discretization parameters mapped to their configured values."""
        return {
            'bin_size': self.bin_size,
            'minimum': self.minimum,
        }

    def apply(self, image):
        """Return a discretized image."""
        minimum = np.nanmin(image.array) if self.minimum is None else self.minimum
        return Image(
            array=np.floor((image.array - minimum) / self.bin_size) + 1,
            origin=image.origin,
            spacing=image.spacing,
            direction=image.direction,
            shape=image.shape,
        )


class FixedBinNumberDiscretizer:
    """Discretize intensities using a fixed number of bins."""

    def __init__(self, number_of_bins):
        self.number_of_bins = number_of_bins

    def get_params(self):
        """Return discretization parameters mapped to their configured values."""
        return {
            'number_of_bins': self.number_of_bins,
        }

    def apply(self, image):
        """Return a discretized image."""
        minimum = np.nanmin(image.array)
        maximum = np.nanmax(image.array)
        return Image(
            array=np.where(
                image.array != maximum,
                np.floor(self.number_of_bins * (image.array - minimum) / (maximum - minimum)) + 1,
                self.number_of_bins,
            ),
            origin=image.origin,
            spacing=image.spacing,
            direction=image.direction,
            shape=image.shape,
        )


class IntensityVolumeHistogramDiscretizer:
    """Prepare an intensity image for intensity-volume histogram calculation."""

    def __init__(self, number_of_bins=None, bin_size=None, minimum=None):
        self.number_of_bins = number_of_bins
        self.bin_size = bin_size
        self.minimum = minimum

    def get_params(self):
        """Return discretization parameters mapped to their configured values."""
        return {
            'number_of_bins': self.number_of_bins,
            'bin_size': self.bin_size,
            'minimum': self.minimum,
        }

    def apply(self, image):
        """Return an IVH-prepared image."""
        result = image.copy()
        if self.bin_size is not None:
            minimum = np.nanmin(result.array) if self.minimum is None else self.minimum
            discretized = FixedBinSizeDiscretizer(self.bin_size, minimum).apply(result)
            result = Image(
                array=minimum + (discretized.array - 0.5) * self.bin_size,
                origin=result.origin,
                spacing=result.spacing,
                direction=result.direction,
                shape=result.shape,
            )
        if self.number_of_bins is not None:
            result = FixedBinNumberDiscretizer(self.number_of_bins).apply(result)
        return result


class ImageDiscretizer:
    """Choose fixed-bin-size or fixed-bin-number discretization from configuration."""

    def __init__(self, number_of_bins=None, bin_size=None, minimum=None):
        self.number_of_bins = number_of_bins
        self.bin_size = bin_size
        self.minimum = minimum

    def get_params(self):
        """Return discretization parameters mapped to their configured values."""
        return {
            'number_of_bins': self.number_of_bins,
            'bin_size': self.bin_size,
            'minimum': self.minimum,
        }

    def apply(self, image):
        """Return a discretized image, or a copy when no discretization is configured."""
        result = image.copy()
        if self.bin_size is not None:
            result = FixedBinSizeDiscretizer(self.bin_size, self.minimum).apply(result)
        if self.number_of_bins is not None:
            result = FixedBinNumberDiscretizer(self.number_of_bins).apply(result)
        return result


def count_bins(image):
    """Return the number of unique finite intensity values in an image."""
    valid_values = image.array[~np.isnan(image.array)]
    if valid_values.size == 0:
        raise DataStructureError("No valid values available for bin counting.")
    return int(np.unique(valid_values).size)
