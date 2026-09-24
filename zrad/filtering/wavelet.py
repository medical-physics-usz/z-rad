from math import factorial

import numpy as np
import pywt
from scipy import fft as sp_fft
from scipy import ndimage as ndi

from .base import BaseFilter


class Simoncelli(BaseFilter):
    """IBSI non-separable Simoncelli band-pass wavelet.

    The wavelet is evaluated directly in the Fourier domain.
    ``decomposition_level`` selects the B map; level one is the
    highest-frequency band. An optional Riesz multi-index applies the
    normalized higher-order Riesz transform to the B map.

    Parameters
    ----------
    padding_type : {"nearest", "wrap", "periodic"}
        Boundary handling. ``periodic`` is an alias for ``wrap``.
    decomposition_level : int
        One-based scale level of the B map.
    dimensionality : {"2D", "3D"}
        In 2D mode each slice is filtered independently.
    riesz_order : tuple of int, optional
        Non-negative Riesz multi-index in physical ``(x, y)`` or
        ``(x, y, z)`` axis order, for example ``(0, 2)`` or ``(0, 2, 0)``.
        Its length must match ``dimensionality``. Omitting it or supplying an
        all-zero index returns the isotropic B map.
    """

    def __init__(self, padding_type, decomposition_level, dimensionality='3D', riesz_order=None):
        if padding_type not in ('nearest', 'wrap', 'periodic'):
            raise ValueError("Simoncelli padding must be 'nearest', 'wrap', or 'periodic'.")
        if dimensionality not in ('2D', '3D'):
            raise ValueError("Simoncelli dimensionality must be '2D' or '3D'.")
        if not isinstance(decomposition_level, int) or isinstance(decomposition_level, bool) or decomposition_level < 1:
            raise ValueError('decomposition_level must be a positive integer.')

        dimensions = int(dimensionality[0])
        if riesz_order is not None:
            if len(riesz_order) != dimensions or any(
                not isinstance(order, (int, np.integer)) or isinstance(order, (bool, np.bool_)) or order < 0
                for order in riesz_order
            ):
                raise ValueError(f'riesz_order must contain {dimensions} non-negative integers.')
            riesz_order = tuple(int(order) for order in riesz_order)

        super().__init__(
            filtering_method='Simoncelli',
            padding_type='wrap' if padding_type == 'periodic' else padding_type,
            decomposition_level=decomposition_level,
            dimensionality=dimensionality,
            riesz_order=riesz_order,
        )
        self.decomposition_level = decomposition_level
        self.dimensionality = dimensionality
        self.padding_type = 'wrap' if padding_type == 'periodic' else padding_type
        self.riesz_order = riesz_order

    def _frequency_response(self, shape):
        frequency_axes = []
        for size in shape:
            frequencies = np.fft.ifftshift(np.linspace(-np.pi, np.pi, size))
            frequencies[0] = 0.0
            frequency_axes.append(frequencies)
        coordinates = np.meshgrid(*frequency_axes, indexing='ij', sparse=True)

        # 3. Calculate Euclidean radial distance
        radius = np.sqrt(sum(c**2 for c in coordinates))

        # 4. IBSI Nyquist cutoff frequency for scale j
        nyquist = np.pi / (2 ** (self.decomposition_level - 1))

        response = np.zeros(shape, dtype=np.float64)

        # 5. Mask support region: [nyquist/4, nyquist]
        support = (radius >= nyquist / 4.0) & (radius <= nyquist)

        # 6. Isotropic Simoncelli B-map continuous cosine profile
        response[support] = np.cos((np.pi / 2.0) * np.log2(2.0 * radius[support] / nyquist))

        # 7. Higher-Order Riesz Multi-Index Transform
        if self.riesz_order is not None and sum(self.riesz_order) > 0:
            total_order = sum(self.riesz_order)
            coefficient = np.sqrt(factorial(total_order) / np.prod([factorial(o) for o in self.riesz_order]))

            numerator = np.ones(shape, dtype=np.float64)
            array_order = (self.riesz_order[1], self.riesz_order[0], *self.riesz_order[2:])
            for c, order in zip(coordinates, array_order):
                numerator *= c**order

            riesz = np.zeros(shape, dtype=np.complex128)
            nonzero = radius > 0
            riesz[nonzero] = (
                ((-1j) ** total_order) * coefficient * numerator[nonzero] / (radius[nonzero] ** total_order)
            )
            response = response * riesz

        return response

    def _filter_periodic(self, image):
        spectrum = sp_fft.rfftn(image)
        response = self._periodic_half_spectrum_response(image.shape)
        spectrum *= response
        if self.riesz_order is not None and sum(self.riesz_order) > 0:
            total_order = sum(self.riesz_order)
            coefficient = np.sqrt(factorial(total_order) / np.prod([factorial(o) for o in self.riesz_order]))
            spectrum *= (-1j) ** total_order * coefficient
        return sp_fft.irfftn(spectrum, s=image.shape)

    def _periodic_half_spectrum_response(self, shape):
        """Return the Hermitian half-spectrum equivalent of the full response."""
        half_shape = (*shape[:-1], shape[-1] // 2 + 1)
        total_order = sum(self.riesz_order) if self.riesz_order is not None else 0
        array_order = (
            (self.riesz_order[1], self.riesz_order[0], *self.riesz_order[2:]) if total_order else (0,) * len(shape)
        )

        def scalar_response(mirrored):
            frequency_axes = []
            for axis, size in enumerate(shape):
                frequencies = np.fft.ifftshift(np.linspace(-np.pi, np.pi, size))
                frequencies[0] = 0.0
                count = half_shape[axis]
                indices = np.arange(count)
                if mirrored:
                    indices = (-indices) % size
                frequency_axes.append(frequencies[indices])
            coordinates = np.meshgrid(*frequency_axes, indexing='ij', sparse=True)

            response = np.zeros(half_shape, dtype=np.float64)
            for coordinate in coordinates:
                response += coordinate**2
            np.sqrt(response, out=response)

            nyquist = np.pi / (2 ** (self.decomposition_level - 1))
            support = (response >= nyquist / 4.0) & (response <= nyquist)
            values = response[support]
            denominator = values.copy() if total_order else None
            values *= 2.0 / nyquist
            np.log2(values, out=values)
            values *= np.pi / 2.0
            np.cos(values, out=values)
            if denominator is not None:
                np.power(denominator, total_order, out=denominator)
                values /= denominator

            response.fill(0.0)
            response[support] = values
            for coordinate, order in zip(coordinates, array_order):
                if order:
                    response *= coordinate**order
            return response

        response = scalar_response(mirrored=False)
        mirrored_response = scalar_response(mirrored=True)
        response += (-1) ** total_order * mirrored_response
        response *= 0.5
        return response

    def _filter_nearest(self, image):
        """Filter an edge-replicated extension for nearest padding.

        The explicit extension is needed for odd Riesz orders: those orders map
        cosine modes to sine (quadrature) modes and therefore cannot be
        synthesized by an inverse DCT alone.
        """
        padding = tuple((length // 2, length - length // 2) for length in image.shape)
        extended = np.pad(image, padding, mode='edge')

        spectrum = sp_fft.rfftn(extended)
        frequency_axes = [2.0 * np.pi * np.fft.fftfreq(length) for length in extended.shape[:-1]]
        last_axis = 2.0 * np.pi * np.fft.rfftfreq(extended.shape[-1])
        if extended.shape[-1] % 2 == 0:
            last_axis[-1] *= -1.0
        frequency_axes.append(last_axis)
        frequencies = np.meshgrid(*frequency_axes, indexing='ij', sparse=True)
        radius = np.zeros(spectrum.shape, dtype=np.float64)
        for frequency in frequencies:
            radius += frequency**2
        np.sqrt(radius, out=radius)

        nyquist = np.pi / (2 ** (self.decomposition_level - 1))
        support = (radius >= nyquist / 4.0) & (radius <= nyquist)
        radius[support] = np.cos((np.pi / 2.0) * np.log2(2.0 * radius[support] / nyquist))
        radius[~support] = 0.0
        spectrum *= radius

        if self.riesz_order is not None and sum(self.riesz_order) > 0:
            total_order = sum(self.riesz_order)
            coefficient = np.sqrt(factorial(total_order) / np.prod([factorial(o) for o in self.riesz_order]))
            array_order = (self.riesz_order[1], self.riesz_order[0], *self.riesz_order[2:])

            radius.fill(0.0)
            for frequency in frequencies:
                radius += frequency**2
            np.sqrt(radius, out=radius)
            np.power(radius, total_order, out=radius)
            radius[(0,) * extended.ndim] = np.inf

            spectrum *= (-1j) ** total_order * coefficient
            for frequency, order in zip(frequencies, array_order):
                if order:
                    spectrum *= frequency**order
            spectrum /= radius

            # An odd number of odd powers evaluated on Nyquist axes is
            # anti-Hermitian at that bin. The real part of the former full
            # inverse FFT discarded exactly these coefficients.
            cancel = np.zeros(spectrum.shape, dtype=bool)
            for axis, (length, order) in enumerate(zip(extended.shape, array_order)):
                if length % 2 == 0 and order % 2:
                    axis_shape = [1] * extended.ndim
                    axis_shape[axis] = spectrum.shape[axis]
                    cancel ^= np.arange(spectrum.shape[axis]).reshape(axis_shape) == length // 2
            spectrum[cancel] = 0.0

        result = sp_fft.irfftn(spectrum, s=extended.shape)
        crop = tuple(slice(before, before + length) for (before, _), length in zip(padding, image.shape))
        return result[crop].copy()

    def _filter(self, image):
        if self.padding_type == 'wrap':
            return self._filter_periodic(image)

        return self._filter_nearest(image)

    def _apply_array(self, img):
        if self.dimensionality == '3D':
            return self._filter(img)

        # 2D filtering mode
        if img.ndim == 2:
            return self._filter(img)

        # BaseFilter always supplies volumes in (y, x, z) order, so 2D
        # filtering must operate independently on planes along axis 2.
        return np.stack([self._filter(img[:, :, i]) for i in range(img.shape[2])], axis=2)


class Wavelets2D(BaseFilter):
    """2D separable wavelet filtering evaluated slice-wise.

    Each response map combines low-pass (``L``) and high-pass (``H``) wavelet
    kernels along the two in-plane axes. The result keeps the original image
    grid and is intended for downstream radiomics feature extraction.

    Parameters
    ----------
    wavelet_type : {"db3", "db2", "coif1", "haar"}
        Wavelet family used to obtain low- and high-pass filter kernels.
    padding_type : {"constant", "nearest", "wrap", "reflect"}
        Boundary handling mode used during convolution.
    response_map : {"LL", "HL", "LH", "HH"}
        Low/high-pass kernel combination for the two in-plane axes.
    decomposition_level : {1, 2}
        Wavelet decomposition level.
    rotation_invariance : bool, optional
        If true, average responses over four in-plane rotations.
    """

    def __init__(self, wavelet_type, padding_type, response_map, decomposition_level, rotation_invariance=False):
        super().__init__(
            filtering_method='Wavelets',
            wavelet_type=wavelet_type,
            padding_type=padding_type,
            response_map=response_map,
            decomposition_level=decomposition_level,
            rotation_invariance=rotation_invariance,
            dimensionality='2D',
        )

        self.dimensionality = '2D'

        if padding_type in ['constant', 'nearest', 'wrap', 'reflect']:
            self.padding_type = padding_type
        else:
            raise ValueError(
                f"Wrong padding type '{padding_type}'. "
                "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'."
            )

        if wavelet_type in ['db3', 'db2', 'coif1', 'haar']:
            self.wavelet_type = wavelet_type
        else:
            raise ValueError(
                f"Wrong wavelet type '{wavelet_type}'. Available wavelet types are: 'db3', 'db2', 'coif1', 'haar'."
            )

        if decomposition_level in [1, 2]:
            self.decomposition_level = decomposition_level
        else:
            raise ValueError(
                f"Wrong decomposition_level' {decomposition_level}'. "
                "Decomposition level should be integer. Available decomposition levels are: 1 and 2."
            )

        if response_map in ['LL', 'HL', 'LH', 'HH']:
            self.response_map = response_map
        else:
            raise ValueError(
                f"Wrong response_map' {response_map}'. Available response_maps are: 'LL', 'HL', 'LH', 'HH'."
            )

        if isinstance(rotation_invariance, bool):
            self.rotation_invariance = rotation_invariance
        else:
            raise ValueError(f"Rotation Invariance should be True or False but '{type(rotation_invariance)}' detected.")

    def _get_kernel(self, response, decomposition_level=1):
        if response == "L":
            kernel = pywt.Wavelet(name=self.wavelet_type).filter_bank[0]
        elif response == "H":
            kernel = pywt.Wavelet(name=self.wavelet_type).filter_bank[1]
        else:
            kernel = None
        if decomposition_level == 2:
            kernel = [[e, 0] for e in kernel]
            kernel = [item for sublist in kernel for item in sublist]
        return kernel

    def _filter(self, img, x_filter, y_filter):
        filtered_img = ndi.convolve1d(img, x_filter, axis=1, mode=self.padding_type)
        filtered_img = ndi.convolve1d(filtered_img, y_filter, axis=0, mode=self.padding_type)
        return filtered_img

    def _apply_array(self, img):
        if self.decomposition_level == 1:
            x_filter = self._get_kernel(self.response_map[0])
            y_filter = self._get_kernel(self.response_map[1])
            if self.rotation_invariance:
                final_image = np.zeros(img.shape)
                for i in range(img.shape[2]):
                    for k in range(4):
                        final_image[:, :, i] += np.rot90(
                            self._filter(np.rot90(img[:, :, i], k=k, axes=(0, 1)), x_filter, y_filter), k=k, axes=(1, 0)
                        )
                filtered_img = final_image / 4
            else:
                filtered_img = np.zeros(img.shape)
                for i in range(img.shape[2]):
                    filtered_img[:, :, i] = self._filter(img[:, :, i], x_filter, y_filter)
        elif self.decomposition_level == 2:
            l_filter = self._get_kernel("L")
            x_filter = self._get_kernel(self.response_map[0], decomposition_level=2)
            y_filter = self._get_kernel(self.response_map[1], decomposition_level=2)

            final_image = np.zeros(img.shape)
            for i in range(img.shape[2]):
                for k in range(4):
                    img_level0 = np.rot90(img[:, :, i], k=k, axes=(0, 1))
                    img_level1 = self._filter(img_level0, l_filter, l_filter)
                    img_level2 = self._filter(img_level1, x_filter, y_filter)
                    final_image[:, :, i] += np.rot90(img_level2, k=k, axes=(1, 0))
            filtered_img = final_image / 4

        return filtered_img


class Wavelets3D(BaseFilter):
    """3D separable wavelet filtering for volumetric response maps.

    Response maps combine low-pass (``L``) and high-pass (``H``) wavelet
    kernels along all three axes. Rotation-invariant mode averages over axis
    permutations and flips to reduce orientation dependence.

    Parameters
    ----------
    wavelet_type : {"db3", "db2", "coif1", "haar"}
        Wavelet family used to obtain low- and high-pass filter kernels.
    padding_type : {"constant", "nearest", "wrap", "reflect"}
        Boundary handling mode used during convolution.
    response_map : {"LLL", "LLH", "LHL", "HLL", "LHH", "HHL", "HLH", "HHH"}
        Low/high-pass kernel combination for the three axes.
    decomposition_level : {1, 2}
        Wavelet decomposition level.
    rotation_invariance : bool, optional
        If true, average responses over axis permutations and flips.
    """

    def __init__(self, wavelet_type, padding_type, response_map, decomposition_level, rotation_invariance=False):
        super().__init__(
            filtering_method='Wavelets',
            wavelet_type=wavelet_type,
            padding_type=padding_type,
            response_map=response_map,
            decomposition_level=decomposition_level,
            rotation_invariance=rotation_invariance,
            dimensionality='3D',
        )

        self.dimensionality = '3D'

        if padding_type in ['constant', 'nearest', 'wrap', 'reflect']:
            self.padding_type = padding_type
        else:
            raise ValueError(
                f"Wrong padding type '{padding_type}'. "
                "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'."
            )

        if wavelet_type in ['db3', 'db2', 'coif1', 'haar']:
            self.wavelet_type = wavelet_type
        else:
            raise ValueError(
                f"Wrong wavelet type '{wavelet_type}'. Available wavelet types are: 'db3', 'db2', 'coif1', 'haar'."
            )

        if decomposition_level in [1, 2]:
            self.decomposition_level = decomposition_level
        else:
            raise ValueError(
                f"Wrong decomposition_level' {decomposition_level}'. "
                "Decomposition level should be integer. Available decomposition levels are: 1 and 2."
            )

        if response_map in ['LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', 'HLH', 'HHH']:
            self.response_map = response_map
        else:
            raise ValueError(
                f"Wrong response_map' {response_map}'. "
                "Available response_maps are: 'LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', 'HLH', 'HHH'."
            )

        if isinstance(rotation_invariance, bool):
            self.rotation_invariance = rotation_invariance
        else:
            raise ValueError(f"Rotation Invariance should be True or False but '{type(rotation_invariance)}' detected.")

        self.pooling = None

    def _get_kernel(self, response, decomposition_level=1):
        if response == "L":
            kernel = pywt.Wavelet(name=self.wavelet_type).filter_bank[0]
        elif response == "H":
            kernel = pywt.Wavelet(name=self.wavelet_type).filter_bank[1]
        else:
            kernel = None
        if decomposition_level == 2:
            kernel = [[e, 0] for e in kernel]
            kernel = [item for sublist in kernel for item in sublist]
        return kernel

    def _filter(self, img, x_filter, y_filter, z_filter):
        filtered_img = ndi.convolve1d(img, x_filter, axis=1, mode=self.padding_type)
        filtered_img = ndi.convolve1d(filtered_img, y_filter, axis=0, mode=self.padding_type)
        filtered_img = ndi.convolve1d(filtered_img, z_filter, axis=2, mode=self.padding_type)
        return filtered_img

    def _apply_array(self, img):
        if self.decomposition_level == 1:
            x_filter = self._get_kernel(self.response_map[0])
            y_filter = self._get_kernel(self.response_map[1])
            z_filter = self._get_kernel(self.response_map[2])
            if self.rotation_invariance:
                final_image = np.zeros(img.shape)
                kernels_permutation = [
                    (x_filter, y_filter, z_filter),
                    (z_filter, x_filter, y_filter),
                    (y_filter, z_filter, x_filter),
                ]
                for kernels in kernels_permutation:
                    final_image += self._filter(img, kernels[0], kernels[1], kernels[2])
                    final_image += self._filter(img[::-1, :, :], kernels[0], kernels[1], kernels[2])[::-1, :, :]
                    final_image += self._filter(img[:, ::-1, :], kernels[0], kernels[1], kernels[2])[:, ::-1, :]
                    final_image += self._filter(img[:, :, ::-1], kernels[0], kernels[1], kernels[2])[:, :, ::-1]
                    final_image += self._filter(img[::-1, ::-1, :], kernels[0], kernels[1], kernels[2])[::-1, ::-1, :]
                    final_image += self._filter(img[::-1, :, ::-1], kernels[0], kernels[1], kernels[2])[::-1, :, ::-1]
                    final_image += self._filter(img[:, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[:, ::-1, ::-1]
                    final_image += self._filter(img[::-1, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[
                        ::-1, ::-1, ::-1
                    ]
                filtered_img = final_image / (8 * len(kernels_permutation))
            else:
                filtered_img = self._filter(img, x_filter, y_filter, z_filter)
        else:
            x_filter = self._get_kernel("L")
            y_filter = self._get_kernel("L")
            z_filter = self._get_kernel("L")
            kernels_permutation = [
                (x_filter, y_filter, z_filter),
                (z_filter, x_filter, y_filter),
                (y_filter, z_filter, x_filter),
            ]
            level1_responses = list()
            for kernels in kernels_permutation:
                level1_responses.append(self._filter(img, kernels[0], kernels[1], kernels[2]))
                level1_responses.append(self._filter(img[::-1, :, :], kernels[0], kernels[1], kernels[2])[::-1, :, :])
                level1_responses.append(self._filter(img[:, ::-1, :], kernels[0], kernels[1], kernels[2])[:, ::-1, :])
                level1_responses.append(self._filter(img[:, :, ::-1], kernels[0], kernels[1], kernels[2])[:, :, ::-1])
                level1_responses.append(
                    self._filter(img[::-1, ::-1, :], kernels[0], kernels[1], kernels[2])[::-1, ::-1, :]
                )
                level1_responses.append(
                    self._filter(img[::-1, :, ::-1], kernels[0], kernels[1], kernels[2])[::-1, :, ::-1]
                )
                level1_responses.append(
                    self._filter(img[:, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[:, ::-1, ::-1]
                )
                level1_responses.append(
                    self._filter(img[::-1, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[::-1, ::-1, ::-1]
                )

            x_filter = self._get_kernel(self.response_map[0], decomposition_level=2)
            y_filter = self._get_kernel(self.response_map[1], decomposition_level=2)
            z_filter = self._get_kernel(self.response_map[2], decomposition_level=2)
            final_image = np.zeros(img.shape)
            kernels_permutation = [
                (x_filter, y_filter, z_filter),
                (z_filter, x_filter, y_filter),
                (y_filter, z_filter, x_filter),
            ]
            for kernels in kernels_permutation:
                final_image += self._filter(level1_responses[0], kernels[0], kernels[1], kernels[2])
                final_image += self._filter(level1_responses[1][::-1, :, :], kernels[0], kernels[1], kernels[2])[
                    ::-1, :, :
                ]
                final_image += self._filter(level1_responses[2][:, ::-1, :], kernels[0], kernels[1], kernels[2])[
                    :, ::-1, :
                ]
                final_image += self._filter(level1_responses[3][:, :, ::-1], kernels[0], kernels[1], kernels[2])[
                    :, :, ::-1
                ]
                final_image += self._filter(level1_responses[4][::-1, ::-1, :], kernels[0], kernels[1], kernels[2])[
                    ::-1, ::-1, :
                ]
                final_image += self._filter(level1_responses[5][::-1, :, ::-1], kernels[0], kernels[1], kernels[2])[
                    ::-1, :, ::-1
                ]
                final_image += self._filter(level1_responses[6][:, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[
                    :, ::-1, ::-1
                ]
                final_image += self._filter(level1_responses[7][::-1, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[
                    ::-1, ::-1, ::-1
                ]
            filtered_img = final_image / (8 * len(kernels_permutation))
        return filtered_img
