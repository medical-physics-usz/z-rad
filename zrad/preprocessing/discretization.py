import numpy as np

from ..exceptions import DataStructureError
from ..image import Image


class FixedBinSizeDiscretizer:
    """Discretize intensities using a fixed bin size.

    Fixed-bin-size discretization maps finite intensities to integer grey
    levels using bins of constant physical intensity width. This is commonly
    used before texture feature calculation.

    Parameters
    ----------
    bin_size : float
        Width of each discretization bin in image intensity units.
    minimum : float or None, optional
        Intensity value used as the lower discretization anchor. If ``None``,
        the minimum finite value in the input image is used.

    """

    def __init__(self, bin_size, minimum=None):
        self.bin_size = bin_size
        self.minimum = minimum

    def get_params(self):
        """Return discretization parameters mapped to their configured values.

        Returns
        -------
        params : dict
            Dictionary containing ``bin_size`` and ``minimum``.
        """
        return {
            'bin_size': self.bin_size,
            'minimum': self.minimum,
        }

    def apply(self, image):
        """Return a discretized image.

        Parameters
        ----------
        image : Image
            Image with finite or ``NaN`` intensity values.

        Returns
        -------
        image : Image
            Image whose finite intensities are replaced by fixed-bin-size grey
            levels.
        """
        minimum = np.nanmin(image.array) if self.minimum is None else self.minimum
        return Image(
            array=np.floor((image.array - minimum) / self.bin_size) + 1,
            origin=image.origin,
            spacing=image.spacing,
            direction=image.direction,
            shape=image.shape,
        )


class FixedBinNumberDiscretizer:
    """Discretize intensities using a fixed number of bins.

    Fixed-bin-number discretization spans the finite intensity range with a
    configured number of grey levels. The maximum input intensity is assigned to
    the last bin.

    Parameters
    ----------
    number_of_bins : int
        Number of bins used to discretize the finite image intensities.

    """

    def __init__(self, number_of_bins):
        self.number_of_bins = number_of_bins

    def get_params(self):
        """Return discretization parameters mapped to their configured values.

        Returns
        -------
        params : dict
            Dictionary containing ``number_of_bins``.
        """
        return {
            'number_of_bins': self.number_of_bins,
        }

    def apply(self, image):
        """Return a discretized image.

        Parameters
        ----------
        image : Image
            Image with finite or ``NaN`` intensity values.

        Returns
        -------
        image : Image
            Image whose finite intensities are replaced by fixed-bin-number grey
            levels.
        """
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
    """Prepare an intensity image for intensity-volume histogram calculation.

    This helper applies the discretization pathway used before
    intensity-volume histogram features. Fixed-bin-size discretization converts
    bins to their centre intensities before optional fixed-bin-number
    discretization.

    Parameters
    ----------
    number_of_bins : int or None, optional
        Number of bins used for fixed-bin-number discretization. If ``None``,
        this step is skipped.
    bin_size : float or None, optional
        Width of each fixed-size intensity bin. If supplied, fixed-bin-size
        discretization is applied before optional fixed-bin-number
        discretization.
    minimum : float or None, optional
        Intensity value used as the lower anchor for fixed-bin-size
        discretization. If ``None``, the minimum finite value in the input
        image is used.

    """

    def __init__(self, number_of_bins=None, bin_size=None, minimum=None):
        self.number_of_bins = number_of_bins
        self.bin_size = bin_size
        self.minimum = minimum

    def get_params(self):
        """Return discretization parameters mapped to their configured values.

        Returns
        -------
        params : dict
            Dictionary containing ``number_of_bins``, ``bin_size``, and
            ``minimum``.
        """
        return {
            'number_of_bins': self.number_of_bins,
            'bin_size': self.bin_size,
            'minimum': self.minimum,
        }

    def apply(self, image):
        """Return an IVH-prepared image.

        Parameters
        ----------
        image : Image
            Intensity image to prepare for IVH feature calculation.

        Returns
        -------
        image : Image
            Image transformed according to the configured IVH discretization
            pathway.
        """
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
    """Choose fixed-bin-size or fixed-bin-number discretization from configuration.

    This convenience wrapper applies fixed-bin-size discretization, fixed-bin-
    number discretization, both in that order, or neither when no discretization
    parameters are configured.

    Parameters
    ----------
    number_of_bins : int or None, optional
        Number of bins used for fixed-bin-number discretization. If ``None``,
        this step is skipped.
    bin_size : float or None, optional
        Width of each fixed-size intensity bin. If supplied, fixed-bin-size
        discretization is applied before optional fixed-bin-number
        discretization.
    minimum : float or None, optional
        Intensity value used as the lower anchor for fixed-bin-size
        discretization. If ``None``, the minimum finite value in the input
        image is used.

    """

    def __init__(self, number_of_bins=None, bin_size=None, minimum=None):
        self.number_of_bins = number_of_bins
        self.bin_size = bin_size
        self.minimum = minimum

    def get_params(self):
        """Return discretization parameters mapped to their configured values.

        Returns
        -------
        params : dict
            Dictionary containing ``number_of_bins``, ``bin_size``, and
            ``minimum``.
        """
        return {
            'number_of_bins': self.number_of_bins,
            'bin_size': self.bin_size,
            'minimum': self.minimum,
        }

    def apply(self, image):
        """Return a discretized image, or a copy when no discretization is configured.

        Parameters
        ----------
        image : Image
            Image with finite or ``NaN`` intensity values.

        Returns
        -------
        image : Image
            Discretized image, or a copy of the input image when no
            discretization parameters are configured.
        """
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
