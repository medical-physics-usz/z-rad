import numpy as np

from ..exceptions import DataStructureError
from ..image import Image
from .roi import IVHAxis, RoiData


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
        if not isinstance(bin_size, (int, float)) or isinstance(bin_size, bool) or bin_size <= 0:
            raise ValueError("bin_size must be a positive number.")
        if minimum is not None and (
            not isinstance(minimum, (int, float)) or isinstance(minimum, bool) or not np.isfinite(minimum)
        ):
            raise ValueError("minimum must be a finite number.")
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
        if not isinstance(number_of_bins, int) or isinstance(number_of_bins, bool) or number_of_bins <= 0:
            raise ValueError("number_of_bins must be a positive integer.")
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
        if maximum == minimum:
            return Image(
                array=np.where(np.isnan(image.array), np.nan, 1),
                origin=image.origin,
                spacing=image.spacing,
                direction=image.direction,
                shape=image.shape,
            )
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
    intensity-volume histogram features. Exactly one IVH discretization method
    must be configured. Fixed-bin-size discretization converts bins to their
    centre intensities. Direct IVH, where retained intensities are used without
    IVH-specific discretization, is handled by the radiomics preparation
    workflow rather than this helper.

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
        if number_of_bins is not None:
            FixedBinNumberDiscretizer(number_of_bins)
        if bin_size is not None:
            FixedBinSizeDiscretizer(bin_size, minimum)
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
        if self.bin_size is not None:
            minimum = np.nanmin(image.array) if self.minimum is None else self.minimum
            discretized = FixedBinSizeDiscretizer(self.bin_size, minimum).apply(image)
            return Image(
                array=minimum + (discretized.array - 0.5) * self.bin_size,
                origin=image.origin,
                spacing=image.spacing,
                direction=image.direction,
                shape=image.shape,
            )
        return FixedBinNumberDiscretizer(self.number_of_bins).apply(image)


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
        if number_of_bins is not None:
            FixedBinNumberDiscretizer(number_of_bins)
        if bin_size is not None:
            FixedBinSizeDiscretizer(bin_size, minimum)
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


class TextureDiscretizer:
    """Prepare the discretized image used by texture feature families.

    This pipeline step reads ``RoiData.intensity_mask`` and writes
    ``RoiData.texture_discretized_image``. Exactly one discretization method
    must be configured. Fixed-bin-size texture discretization requires a finite
    lower anchor, usually the lower bound of the re-segmentation range.

    Parameters
    ----------
    number_of_bins : int or None, optional
        Number of bins used for fixed-bin-number discretization. Mutually
        exclusive with ``bin_size``.
    bin_size : float or None, optional
        Width of each fixed-size intensity bin. Mutually exclusive with
        ``number_of_bins``.
    minimum : float or None, optional
        Lower anchor for fixed-bin-size discretization. Required when
        ``bin_size`` is configured.
    """

    def __init__(self, number_of_bins=None, bin_size=None, minimum=None):
        ImageDiscretizer(number_of_bins=number_of_bins, bin_size=bin_size, minimum=minimum)
        if bin_size is not None and minimum is None:
            raise ValueError("Texture bin_size requires a finite minimum lower anchor.")
        self.number_of_bins = number_of_bins
        self.bin_size = bin_size
        self.minimum = minimum

    def get_params(self):
        """Return texture discretization parameters."""
        return {
            'number_of_bins': self.number_of_bins,
            'bin_size': self.bin_size,
            'minimum': self.minimum,
        }

    def apply(self, roi_data):
        """Return ROI data with ``texture_discretized_image`` prepared."""
        if not isinstance(roi_data, RoiData):
            raise TypeError(f"Expected RoiData, got {type(roi_data)}.")
        if roi_data.intensity_mask is None:
            raise ValueError("TextureDiscretizer requires RoiData.intensity_mask.")

        texture_image = ImageDiscretizer(
            number_of_bins=self.number_of_bins,
            bin_size=self.bin_size,
            minimum=self.minimum,
        ).apply(roi_data.intensity_mask)
        return RoiData(
            image=roi_data.image,
            filtered_image=roi_data.filtered_image,
            morphological_mask=roi_data.morphological_mask,
            intensity_mask=roi_data.intensity_mask,
            texture_discretized_image=texture_image,
            ivh_intensity_image=roi_data.ivh_intensity_image,
            ivh_axis=roi_data.ivh_axis,
        )


class IVHIntensityPreparer:
    """Prepare the image and axis used by intensity-volume histogram features.

    This pipeline step reads ``RoiData.intensity_mask`` and writes both
    ``RoiData.ivh_intensity_image`` and ``RoiData.ivh_axis``.

    Parameters
    ----------
    method : {"direct", "fixed_bin_size", "fixed_bin_number"}
        IVH intensity-axis strategy.
    number_of_bins : int or None, optional
        Number of bins for fixed-bin-number IVH preparation. Required when
        ``method`` is ``"fixed_bin_number"``.
    bin_size : float or None, optional
        Bin width for fixed-bin-size IVH preparation. Required when ``method``
        is ``"fixed_bin_size"``.
    minimum : float or None, optional
        Lower IVH axis bound. Required for fixed-bin-size IVH and optional for
        direct IVH.
    maximum : float or None, optional
        Upper IVH axis bound. For direct IVH this sets the axis maximum. For
        fixed-bin-size IVH this sets the upper bin-edge reference; non-finite
        values fall back to the observed prepared-image maximum.
    """

    _METHODS = {'direct', 'fixed_bin_size', 'fixed_bin_number'}

    def __init__(self, method, number_of_bins=None, bin_size=None, minimum=None, maximum=None):
        if method not in self._METHODS:
            raise ValueError("method must be 'direct', 'fixed_bin_size', or 'fixed_bin_number'.")
        self._validate_axis_value(minimum, "minimum")
        self._validate_axis_value(maximum, "maximum", allow_infinite=True)

        if method == 'direct':
            if number_of_bins is not None or bin_size is not None:
                raise ValueError("direct IVH does not accept number_of_bins or bin_size.")
        elif method == 'fixed_bin_number':
            if bin_size is not None:
                raise ValueError("fixed_bin_number IVH does not accept bin_size.")
            if number_of_bins is None:
                raise ValueError("fixed_bin_number IVH requires number_of_bins.")
            FixedBinNumberDiscretizer(number_of_bins)
        else:
            if number_of_bins is not None:
                raise ValueError("fixed_bin_size IVH does not accept number_of_bins.")
            if bin_size is None:
                raise ValueError("fixed_bin_size IVH requires bin_size.")
            if minimum is None:
                raise ValueError("fixed_bin_size IVH requires a finite minimum lower anchor.")
            FixedBinSizeDiscretizer(bin_size, minimum)

        self.method = method
        self.number_of_bins = number_of_bins
        self.bin_size = bin_size
        self.minimum = minimum
        self.maximum = maximum

    @staticmethod
    def _validate_axis_value(value, label, allow_infinite=False):
        if value is None:
            return
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{label} must be a numeric value.")
        if np.isnan(value) or (not allow_infinite and not np.isfinite(value)):
            raise ValueError(f"{label} must be finite.")

    def get_params(self):
        """Return IVH preparation parameters."""
        return {
            'method': self.method,
            'number_of_bins': self.number_of_bins,
            'bin_size': self.bin_size,
            'minimum': self.minimum,
            'maximum': self.maximum,
        }

    def apply(self, roi_data):
        """Return ROI data with IVH image and axis prepared."""
        if not isinstance(roi_data, RoiData):
            raise TypeError(f"Expected RoiData, got {type(roi_data)}.")
        if roi_data.intensity_mask is None:
            raise ValueError("IVHIntensityPreparer requires RoiData.intensity_mask.")

        if self.method == 'direct':
            ivh_image = roi_data.intensity_mask.copy()
            minimum = np.nanmin(ivh_image.array) if self.minimum is None else self.minimum
            maximum = np.nanmax(ivh_image.array) if self.maximum is None or not np.isfinite(self.maximum) else self.maximum
            step = 1
        elif self.method == 'fixed_bin_number':
            ivh_image = IntensityVolumeHistogramDiscretizer(
                number_of_bins=self.number_of_bins,
            ).apply(roi_data.intensity_mask)
            minimum = np.nanmin(ivh_image.array)
            maximum = np.nanmax(ivh_image.array)
            step = 1
        else:
            ivh_image = IntensityVolumeHistogramDiscretizer(
                bin_size=self.bin_size,
                minimum=self.minimum,
            ).apply(roi_data.intensity_mask)
            minimum = self.minimum + 0.5 * self.bin_size
            maximum = np.nanmax(ivh_image.array)
            if self.maximum is not None and np.isfinite(self.maximum):
                maximum = self.maximum - 0.5 * self.bin_size
            step = self.bin_size

        return RoiData(
            image=roi_data.image,
            filtered_image=roi_data.filtered_image,
            morphological_mask=roi_data.morphological_mask,
            intensity_mask=roi_data.intensity_mask,
            texture_discretized_image=roi_data.texture_discretized_image,
            ivh_intensity_image=ivh_image,
            ivh_axis=IVHAxis(minimum=minimum, maximum=maximum, step=step),
        )


def count_bins(image):
    """Return the number of unique finite intensity values in an image."""
    valid_values = image.array[~np.isnan(image.array)]
    if valid_values.size == 0:
        raise DataStructureError("No valid values available for bin counting.")
    return int(np.unique(valid_values).size)
