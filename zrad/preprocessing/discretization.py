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

    Exactly one discretization method must be configured: set ``bin_size`` for
    fixed-bin-size discretization or ``number_of_bins`` for fixed-bin-number
    discretization.

    Use fixed-bin-number discretization for arbitrary intensity scales such as
    raw MRI or many filtered images. Use fixed-bin-size discretization for
    calibrated units when a consistent lower anchor is defined for all samples,
    preferably the lower bound of the re-segmentation range. Fixed-bin-size is
    not recommended for arbitrary intensity scales without such an anchor.
    Report the chosen method and ``minimum`` value.

    Parameters
    ----------
    number_of_bins : int or None, optional
        Number of bins used for fixed-bin-number discretization. Mutually
        exclusive with ``bin_size``.
    bin_size : float or None, optional
        Width of each fixed-size intensity bin. Mutually exclusive with
        ``number_of_bins``.
    minimum : float or None, optional
        Intensity value used as the lower anchor for fixed-bin-size
        discretization. If ``None``, the minimum finite value in the input
        image is used.

    """

    def __init__(self, number_of_bins=None, bin_size=None, minimum=None):
        if (number_of_bins is None) == (bin_size is None):
            raise ValueError("Specify exactly one of number_of_bins or bin_size.")
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
        """Return a discretized image.

        Parameters
        ----------
        image : Image
            Image with finite or ``NaN`` intensity values.

        Returns
        -------
        image : Image
            Discretized image.
        """
        if self.bin_size is not None:
            return FixedBinSizeDiscretizer(self.bin_size, self.minimum).apply(image)
        return FixedBinNumberDiscretizer(self.number_of_bins).apply(image)


def count_bins(image):
    """Return the number of unique finite intensity values in an image."""
    valid_values = image.array[~np.isnan(image.array)]
    if valid_values.size == 0:
        raise DataStructureError("No valid values available for bin counting.")
    return int(np.unique(valid_values).size)
