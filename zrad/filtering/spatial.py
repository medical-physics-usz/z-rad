from functools import lru_cache
from itertools import permutations
from math import factorial

import cv2
import numpy as np
from scipy import fft as sp_fft
from scipy import ndimage as ndi

from .base import BaseFilter


class Mean(BaseFilter):
    """Mean filter for local intensity smoothing.

    The filter replaces each voxel by the average intensity in a square 2D or
    cubic 3D neighbourhood. Use it as a simple low-pass filter before feature
    extraction when local noise reduction is desired.

    Parameters
    ----------
    padding_type : {"constant", "nearest", "wrap", "reflect"}
        Boundary handling mode used during convolution.
    support : int
        Kernel side length in voxels.
    dimensionality : {"2D", "3D"}
        Apply the filter slice-wise in 2D or volumetrically in 3D.
    """

    def __init__(self, padding_type, support, dimensionality):
        super().__init__(
            filtering_method='Mean', padding_type=padding_type, support=support, dimensionality=dimensionality
        )

        if dimensionality in ['2D', '3D']:
            self.dimensionality = dimensionality
        else:
            raise ValueError(f"Wrong dimensionality '{dimensionality}'. Available dimensions '2D' and '3D'.")

        if isinstance(support, int):
            self.support = support
        else:
            raise ValueError(f"Support should be int but '{type(support)}' detected.")

        if padding_type in ['constant', 'nearest', 'wrap', 'reflect']:
            self.padding_type = padding_type
        else:
            raise ValueError(
                f"Wrong padding type '{padding_type}'. "
                "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'."
            )

    def _apply_array(self, img):
        if self.dimensionality == "2D":
            filt_mat = np.ones([self.support, self.support])
            filt_mat = filt_mat / np.prod(filt_mat.shape)
            filtered_img = np.ones(img.shape)
            for i in range(img.shape[2]):
                filtered_img[:, :, i] = ndi.convolve(input=img[:, :, i], weights=filt_mat, mode=self.padding_type)
        elif self.dimensionality == "3D":
            filt_mat = np.ones([self.support, self.support, self.support])
            filt_mat = filt_mat / np.prod(filt_mat.shape)
            filtered_img = ndi.convolve(input=img, weights=filt_mat, mode=self.padding_type)
        else:
            filtered_img = None
        return filtered_img


class LoG(BaseFilter):
    """Laplacian-of-Gaussian filter for blob and edge enhancement.

    The image is Gaussian-smoothed at a physical scale and then transformed
    with the Laplacian operator. This highlights intensity transitions and
    blob-like structures at the configured scale.

    Parameters
    ----------
    padding_type : {"constant", "nearest", "wrap", "reflect"}
        Boundary handling mode used during convolution.
    sigma_mm : float
        Gaussian standard deviation in millimetres.
    cutoff : float
        Kernel truncation radius in standard deviations.
    dimensionality : {"2D", "3D"}
        Apply the filter slice-wise in 2D or volumetrically in 3D.
    """

    def __init__(self, padding_type, sigma_mm, cutoff, dimensionality):
        super().__init__(
            filtering_method='Laplacian of Gaussian',
            padding_type=padding_type,
            sigma_mm=sigma_mm,
            cutoff=cutoff,
            dimensionality=dimensionality,
        )

        if dimensionality in ['2D', '3D']:
            self.dimensionality = dimensionality
        else:
            raise ValueError(f"Wrong dimensionality '{dimensionality}'. Available dimensions '2D' and '3D'.")

        if padding_type in ['constant', 'nearest', 'wrap', 'reflect']:
            self.padding_type = padding_type
        else:
            raise ValueError(
                f"Wrong padding type '{padding_type}'. "
                f"Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'."
            )

        if isinstance(sigma_mm, (int, float)):
            self.sigma_mm = sigma_mm
        else:
            raise ValueError(f'Sigma (in mm) should be int or float but {type(sigma_mm)} detected.')

        if isinstance(cutoff, (int, float)):
            self.cutoff = cutoff
        else:
            raise ValueError(f'Cutoff should be int or float but {type(cutoff)} detected.')

        self.padding_constant = 0.0
        self.res_mm = None

    def _prepare(self, image):
        try:
            self.res_mm = float(image.spacing[0])
        except (AttributeError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid image spacing data: {e}")

    def _apply_array(self, img):
        sigma = self.sigma_mm / self.res_mm
        if self.dimensionality == "3D":
            filtered_img = ndi.gaussian_laplace(
                img, sigma=sigma, mode=self.padding_type, cval=self.padding_constant, truncate=self.cutoff
            )
        elif self.dimensionality == "2D":
            filtered_img = np.nan * np.ones(img.shape)
            for i in range(img.shape[2]):
                filtered_img[:, :, i] = ndi.gaussian_laplace(
                    img[:, :, i], sigma=sigma, mode=self.padding_type, cval=self.padding_constant, truncate=self.cutoff
                )
        else:
            filtered_img = None
        return filtered_img


class RieszLoG(LoG):
    """Laplacian-of-Gaussian followed by a normalized Riesz transform.

    This is a composition of the spatial LoG filter and the Fourier-domain
    Riesz operator. A second-order
    response can optionally be steered along the local structure-tensor
    direction.

    Parameters
    ----------
    padding_type : {"constant", "nearest", "wrap", "reflect"}
        Boundary handling mode used by the LoG and Riesz operations.
    sigma_mm : float
        Gaussian standard deviation of the LoG filter in millimetres.
    cutoff : float
        LoG kernel truncation radius in standard deviations.
    dimensionality : {"2D", "3D"}
        Apply the composed filter slice-wise in 2D or volumetrically in 3D.
    riesz_order : tuple of int
        Non-negative Riesz multi-index in physical ``(x, y)`` or
        ``(x, y, z)`` axis order. Its length must match ``dimensionality`` and
        its total order must be positive.
    structure_tensor_sigma_mm : float, optional
        Gaussian scale in millimetres used to estimate the local structure
        tensor and steer the response. This is supported only for pure
        second-order 3D indices such as ``(2, 0, 0)``. If omitted, the Riesz
        response is evaluated along the fixed image axes.
    """

    def __init__(self, padding_type, sigma_mm, cutoff, dimensionality, riesz_order, structure_tensor_sigma_mm=None):
        super().__init__(padding_type, sigma_mm, cutoff, dimensionality)
        dimensions = int(dimensionality[0])
        if len(riesz_order) != dimensions or any(
            not isinstance(order, (int, np.integer)) or isinstance(order, (bool, np.bool_)) or order < 0
            for order in riesz_order
        ):
            raise ValueError(f'riesz_order must contain {dimensions} non-negative integers.')
        if sum(riesz_order) == 0:
            raise ValueError('riesz_order must have a positive total order.')
        if structure_tensor_sigma_mm is not None and (
            not isinstance(structure_tensor_sigma_mm, (int, float)) or structure_tensor_sigma_mm <= 0
        ):
            raise ValueError('structure_tensor_sigma_mm must be a positive number.')
        if structure_tensor_sigma_mm is not None and (dimensions != 3 or sum(riesz_order) != 2):
            raise ValueError('Structure-tensor alignment is supported for second-order 3D Riesz transforms only.')
        if structure_tensor_sigma_mm is not None and 2 not in riesz_order:
            raise ValueError(
                'Structure-tensor alignment supports pure second-order Riesz indices only; '
                'mixed-order indices have sign-ambiguous eigenvector steering.'
            )

        self.filtering_method = 'Riesz-transformed LoG'
        self.riesz_order = tuple(int(order) for order in riesz_order)
        self.structure_tensor_sigma_mm = structure_tensor_sigma_mm
        self.filtering_params.update(riesz_order=self.riesz_order, structure_tensor_sigma_mm=structure_tensor_sigma_mm)

    @staticmethod
    def _riesz_transform(image, order):
        total_order = sum(order)
        spectrum = sp_fft.rfftn(image)
        frequency_axes = [2.0 * np.pi * np.fft.fftfreq(size) for size in image.shape[:-1]]
        last_axis = 2.0 * np.pi * np.fft.rfftfreq(image.shape[-1])
        if image.shape[-1] % 2 == 0:
            last_axis[-1] *= -1.0
        frequency_axes.append(last_axis)
        coordinates = np.meshgrid(*frequency_axes, indexing='ij', sparse=True)
        radius = np.zeros(spectrum.shape, dtype=np.float64)
        for coordinate in coordinates:
            radius += coordinate**2
        np.sqrt(radius, out=radius)
        np.power(radius, total_order, out=radius)
        radius[(0,) * image.ndim] = np.inf

        coefficient = np.sqrt(factorial(total_order) / np.prod([factorial(value) for value in order]))
        spectrum *= (-1j) ** total_order * coefficient
        for coordinate, value in zip(coordinates, order):
            if value:
                spectrum *= coordinate**value
        spectrum /= radius

        # At self-conjugate Nyquist coordinates, an odd number of odd-axis
        # powers contributes only to the imaginary part of a full inverse FFT.
        cancel = np.zeros(spectrum.shape, dtype=bool)
        for axis, (size, value) in enumerate(zip(image.shape, order)):
            if size % 2 == 0 and value % 2:
                axis_shape = [1] * image.ndim
                axis_shape[axis] = spectrum.shape[axis]
                cancel ^= np.arange(spectrum.shape[axis]).reshape(axis_shape) == size // 2
        spectrum[cancel] = 0.0

        return sp_fft.irfftn(spectrum, s=image.shape)

    def _boundary_aware_riesz_transform(self, image, order):
        """Apply the Riesz transform without imposing unintended periodicity."""
        if self.padding_type == 'wrap':
            return self._riesz_transform(image, order)

        if self.padding_type in ('constant', 'nearest'):
            # These boundary modes do not have a cosine-transform
            # representation. Extend the domain according to the selected
            # mode before applying the periodic Riesz transform.
            padding = tuple((size // 2, size - size // 2) for size in image.shape)
            mode = 'edge' if self.padding_type == 'nearest' else 'constant'
            padded = np.pad(image, padding, mode=mode)
            transformed = self._riesz_transform(padded, order)
            crop = tuple(slice(before, before + size) for (before, _), size in zip(padding, image.shape))
            return transformed[crop]

        # A DCT represents the same even, non-periodic extension without
        # materializing a volume twice as large along every axis. Odd powers
        # map cosine modes into their sine/quadrature counterparts, while even
        # powers remain in the cosine basis.
        frequencies = np.meshgrid(*(np.pi * np.arange(size) / size for size in image.shape), indexing='ij', sparse=True)
        radius = np.sqrt(sum(frequency**2 for frequency in frequencies))
        total_order = sum(order)
        coefficient = np.sqrt(factorial(total_order) / np.prod([factorial(value) for value in order]))
        multiplier = np.ones(image.shape, dtype=np.float64)
        for frequency, value in zip(frequencies, order):
            multiplier *= frequency**value
        nonzero = radius > 0
        multiplier[nonzero] *= coefficient / radius[nonzero] ** total_order
        multiplier[~nonzero] = 0.0
        multiplier *= (-1) ** sum(value // 2 for value in order)

        coefficients = sp_fft.dctn(image, type=2, norm='ortho') * multiplier
        for axis, value in enumerate(order):
            if value % 2:
                shifted = np.zeros_like(coefficients)
                source = [slice(None)] * image.ndim
                destination = [slice(None)] * image.ndim
                source[axis] = slice(1, None)
                destination[axis] = slice(None, -1)
                shifted[tuple(destination)] = coefficients[tuple(source)]
                coefficients = shifted

        result = coefficients
        for axis, value in enumerate(order):
            transform = sp_fft.idst if value % 2 else sp_fft.idct
            result = transform(result, type=2, axis=axis, norm='ortho')
        return result

    def _smooth_tensor_component(self, component, sigma, row, column):
        if self.padding_type != 'reflect' or row == column:
            return ndi.gaussian_filter(component, sigma=sigma, mode=self.padding_type)

        # Cross-components are odd across the two Riesz axes and even across
        # the remaining axis. Extend only a kernel halo along each odd axis,
        # with a sign change at every half-sample reflection (also for small
        # images whose smoothing kernel spans multiple reflections).
        if sigma <= 1e-15:
            return component.copy()
        radius = int(4.0 * sigma + 0.5)
        for axis, size in enumerate(component.shape):
            if axis not in (row, column) or radius == 0:
                component = ndi.gaussian_filter1d(component, sigma=sigma, axis=axis, mode='reflect')
                continue
            positions = np.arange(-radius, size + radius)
            reflected = (positions // size) % 2 != 0
            indices = positions % size
            indices[reflected] = size - 1 - indices[reflected]
            extended = np.take(component, indices, axis=axis)
            axis_shape = [1] * component.ndim
            axis_shape[axis] = positions.size
            extended *= np.where(reflected, -1.0, 1.0).reshape(axis_shape)
            smoothed = ndi.gaussian_filter1d(extended, sigma=sigma, axis=axis, mode='constant')
            crop = [slice(None)] * component.ndim
            crop[axis] = slice(radius, radius + size)
            component = smoothed[tuple(crop)].copy()
        return component

    def _aligned_second_order_response(self, image, log_response, riesz_transform, crop=None):
        if crop is None:
            crop = (slice(None),) * image.ndim
        output_shape = image[crop].shape
        sigma = self.structure_tensor_sigma_mm / self.res_mm
        first_order_responses = []
        for axis in range(3):
            order = [0, 0, 0]
            order[axis] = 1
            first_order_responses.append(riesz_transform(image, order))

        # Smooth on the full domain, then retain only the final field of view.
        # Eigensystems and steering are voxel-local and need no exterior tail.
        tensor = np.empty(output_shape + (3, 3))
        for row in range(3):
            for column in range(row, 3):
                value = self._smooth_tensor_component(
                    first_order_responses[row] * first_order_responses[column],
                    sigma=sigma,
                    row=row,
                    column=column,
                )
                tensor[..., row, column] = value[crop]
                tensor[..., column, row] = value[crop]
        del value, first_order_responses
        eigenvectors = np.linalg.eigh(tensor)[1]
        del tensor
        rotation = np.swapaxes(eigenvectors[..., ::-1], -1, -2)

        target_axes = np.repeat(np.arange(3), self.riesz_order)
        first_direction = rotation[..., target_axes[0], :]
        second_direction = rotation[..., target_axes[1], :]
        target_coefficient = np.sqrt(factorial(2) / np.prod([factorial(order) for order in self.riesz_order]))

        response = np.zeros(output_shape, dtype=image.dtype)
        for row in range(3):
            order = [0, 0, 0]
            order[row] = 2
            response += (
                target_coefficient
                * first_direction[..., row]
                * second_direction[..., row]
                * riesz_transform(log_response, order)[crop]
            )
            for column in range(row + 1, 3):
                order = [0, 0, 0]
                order[row] = order[column] = 1
                response += (
                    target_coefficient
                    / np.sqrt(2.0)
                    * (
                        first_direction[..., row] * second_direction[..., column]
                        + first_direction[..., column] * second_direction[..., row]
                    )
                    * riesz_transform(log_response, order)[crop]
                )
        return response

    def _apply_composed(self, image, order):
        """Apply LoG and Riesz on one consistently bounded domain."""
        crop = None
        if self.padding_type in ('constant', 'nearest'):
            padding = tuple((size // 2, size - size // 2) for size in image.shape)
            mode = 'edge' if self.padding_type == 'nearest' else 'constant'
            domain = np.pad(image, padding, mode=mode)
            crop = tuple(slice(before, before + size) for (before, _), size in zip(padding, image.shape))
            riesz_transform = self._riesz_transform
        else:
            domain = image
            riesz_transform = self._boundary_aware_riesz_transform

        sigma = self.sigma_mm / self.res_mm
        log_response = ndi.gaussian_laplace(
            domain,
            sigma=sigma,
            mode=self.padding_type,
            cval=self.padding_constant,
            truncate=self.cutoff,
        )
        if self.structure_tensor_sigma_mm is not None:
            return self._aligned_second_order_response(domain, log_response, riesz_transform, crop=crop)
        response = riesz_transform(log_response, order)
        return response if crop is None else response[crop].copy()

    def _apply_array(self, img):
        # Image arrays are handled internally as (y, x, z), while the public
        # multi-index follows the physical image axes (x, y, z).
        order = (self.riesz_order[1], self.riesz_order[0], *self.riesz_order[2:])
        if self.dimensionality == '3D':
            return self._apply_composed(img, order)

        response = np.empty_like(img, dtype=np.result_type(img.dtype, np.float64))
        for index in range(img.shape[2]):
            response[:, :, index] = self._apply_composed(img[:, :, index], order)
        return response


class Laws(BaseFilter):
    """Laws-kernel texture filtering in 2D or 3D.

    Laws filters combine separable 1D kernels such as level, edge, spot, wave,
    and ripple operators to form texture response maps. Optional energy maps
    summarize absolute responses in a local neighbourhood.

    Parameters
    ----------
    response_map : str
        Kernel combination, for example ``"L5E5"`` in 2D or ``"L5E5S5"`` in
        3D. Supported kernel letters are ``L``, ``E``, ``S``, ``W``, and ``R``.
    padding_type : {"constant", "nearest", "wrap", "reflect"}
        Boundary handling mode used during convolution.
    distance : int
        Radius of the local averaging window used when ``energy_map`` is true.
    energy_map : bool
        If true, return a local mean absolute response map.
    dimensionality : {"2D", "3D"}
        Apply 2D or 3D Laws filtering.
    rotation_invariance : bool, optional
        If true, combine responses over axis permutations and flips.
    pooling : {"avg", "max", None}, optional
        Pooling rule for rotation-invariant responses.
    """

    def __init__(
        self, response_map, padding_type, distance, energy_map, dimensionality, rotation_invariance=False, pooling=None
    ):
        super().__init__(
            filtering_method='Laws Kernels',
            response_map=response_map,
            padding_type=padding_type,
            distance=distance,
            energy_map=energy_map,
            dimensionality=dimensionality,
            rotation_invariance=rotation_invariance,
            pooling=pooling,
        )

        if dimensionality in ['2D', '3D']:
            self.dimensionality = dimensionality
        else:
            raise ValueError(f"Wrong dimensionality '{dimensionality}'. Available dimensions '2D' and '3D'.")

        if padding_type in ['constant', 'nearest', 'wrap', 'reflect']:
            self.padding_type = padding_type
        else:
            raise ValueError(
                f"Wrong padding type '{padding_type}'. "
                "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'."
            )

        if isinstance(distance, int):
            self.distance = distance
        else:
            raise ValueError(f"Distance should be 'int' but '{type(distance)}' detected.")

        if isinstance(energy_map, bool):
            self.energy_map = energy_map
        else:
            raise ValueError('Energy map can be only True or False.')

        if isinstance(rotation_invariance, bool):
            self.rotation_invariance = rotation_invariance
        else:
            raise ValueError(f"Rotation Invariance should be True or False but '{type(rotation_invariance)}' detected.")

        self.response_map = response_map
        self.pooling = pooling

    def _get_kernel(self, l_type, support):
        if l_type == "L":
            if support == 3:
                return 1 / np.sqrt(6) * np.array([1, 2, 1])
            elif support == 5:
                return 1 / np.sqrt(70) * np.array([1, 4, 6, 4, 1])
        elif l_type == "E":
            if support == 3:
                return 1 / np.sqrt(2) * np.array([-1, 0, 1])
            elif support == 5:
                return 1 / np.sqrt(10) * np.array([-1, -2, 0, 2, 1])
        elif l_type == "S":
            if support == 3:
                return 1 / np.sqrt(6) * np.array([-1, 2, -1])
            elif support == 5:
                return 1 / np.sqrt(6) * np.array([-1, 0, 2, 0, -1])
        elif l_type == "W":
            if support == 5:
                return 1 / np.sqrt(10) * np.array([-1, 2, 0, -2, 1])
        elif l_type == "R":
            if support == 5:
                return 1 / np.sqrt(70) * np.array([1, -4, 6, -4, 1])

    def _get_response_maps(self):
        parts = [self.response_map[i : i + 2] for i in range(0, len(self.response_map), 2)]
        return [''.join(e) for e in permutations(parts)]

    def _filter(self, img, response_map):
        if self.dimensionality == "3D":
            x_filt = self._get_kernel(response_map[0], int(response_map[1]))
            y_filt = self._get_kernel(response_map[2], int(response_map[3]))
            z_filt = self._get_kernel(response_map[4], int(response_map[5]))

            filtered_img = ndi.convolve1d(img, x_filt, axis=1, mode=self.padding_type)
            filtered_img = ndi.convolve1d(filtered_img, y_filt, axis=0, mode=self.padding_type)
            filtered_img = ndi.convolve1d(filtered_img, z_filt, axis=2, mode=self.padding_type)
        elif self.dimensionality == "2D":
            x_filt = self._get_kernel(response_map[0], int(response_map[1]))
            y_filt = self._get_kernel(response_map[2], int(response_map[3]))

            filtered_img = ndi.convolve1d(img, x_filt, axis=1, mode=self.padding_type)
            filtered_img = ndi.convolve1d(filtered_img, y_filt, axis=0, mode=self.padding_type)
        else:
            filtered_img = None
        return filtered_img

    def _apply_array(self, img):
        final_image = None
        if self.rotation_invariance:
            response_maps = self._get_response_maps()

            if self.pooling == "avg":
                final_image = np.nan * np.ones(img.shape)
                for response_map in response_maps:
                    final_image += self._filter(img, response_map)
                    final_image += self._filter(img[::-1, :, :], response_map)[::-1, :, :]
                    final_image += self._filter(img[:, ::-1, :], response_map)[:, ::-1, :]
                    final_image += self._filter(img[:, :, ::-1], response_map)[:, :, ::-1]
                    final_image += self._filter(img[::-1, ::-1, :], response_map)[::-1, ::-1, :]
                    final_image += self._filter(img[::-1, :, ::-1], response_map)[::-1, :, ::-1]
                    final_image += self._filter(img[:, ::-1, ::-1], response_map)[:, ::-1, ::-1]
                    final_image += self._filter(img[::-1, ::-1, ::-1], response_map)[::-1, ::-1, ::-1]
                final_image = final_image / 24

            elif self.pooling == "max":
                final_image = -np.inf * np.ones(img.shape)
                for response_map in response_maps:
                    final_image = np.maximum(final_image, self._filter(img, response_map))
                    final_image = np.maximum(final_image, self._filter(img[::-1, :, :], response_map)[::-1, :, :])
                    final_image = np.maximum(final_image, self._filter(img[:, ::-1, :], response_map)[:, ::-1, :])
                    final_image = np.maximum(final_image, self._filter(img[:, :, ::-1], response_map)[:, :, ::-1])
                    final_image = np.maximum(final_image, self._filter(img[::-1, ::-1, :], response_map)[::-1, ::-1, :])
                    final_image = np.maximum(final_image, self._filter(img[::-1, :, ::-1], response_map)[::-1, :, ::-1])
                    final_image = np.maximum(final_image, self._filter(img[:, ::-1, ::-1], response_map)[:, ::-1, ::-1])
                    final_image = np.maximum(
                        final_image, self._filter(img[::-1, ::-1, ::-1], response_map)[::-1, ::-1, ::-1]
                    )
        else:
            final_image = self._filter(img, self.response_map)

        if self.energy_map:
            final_image = self._get_energy_map(final_image)

        return final_image

    def _get_energy_map(self, img):
        if self.dimensionality == "2D":
            filt_mat = np.ones([2 * self.distance + 1, 2 * self.distance + 1])
            filt_mat = filt_mat / np.prod(filt_mat.shape)
            energy_map = np.nan * np.ones(img.shape)
            for i in range(img.shape[2]):
                energy_map[:, :, i] = ndi.convolve(input=np.abs(img[:, :, i]), weights=filt_mat, mode='reflect')
        elif self.dimensionality == "3D":
            filt_mat = np.ones([2 * self.distance + 1, 2 * self.distance + 1, 2 * self.distance + 1])
            filt_mat = filt_mat / np.prod(filt_mat.shape)
            energy_map = ndi.convolve(input=np.abs(img), weights=filt_mat, mode='reflect')
        else:
            energy_map = None
        return energy_map


class Gabor(BaseFilter):
    """Gabor filtering with complex 2D kernels.

    Gabor filters measure oriented, frequency-selective texture. The
    implementation applies real and imaginary kernels slice-wise, returns their
    magnitude, and can average responses over orientations and orthogonal
    planes.

    Parameters
    ----------
    padding_type : {"constant", "nearest", "reflect", "mirror", "wrap"}
        Boundary handling mode used by OpenCV.
    res_mm : float
        Voxel spacing in millimetres used to convert physical scales to pixels.
    sigma_mm : float
        Gaussian envelope standard deviation in millimetres.
    lambda_mm : float
        Sinusoidal wavelength in millimetres.
    gamma : float
        Spatial aspect ratio of the Gabor kernel.
    theta : float
        Orientation angle in radians, or angular step when
        ``rotation_invariance`` is true.
    rotation_invariance : bool, optional
        If true, average responses over orientations from 0 to ``2*pi``.
    orthogonal_planes : bool, optional
        If true, also evaluate the three orthogonal slice planes.
    n_stds : float or None, optional
        Kernel size in standard deviations. If ``None``, seven standard
        deviations are used.
    """

    _PADDING_MAP = {
        'reflect': cv2.BORDER_REFLECT,
        'mirror': cv2.BORDER_REFLECT_101,
        'constant': cv2.BORDER_CONSTANT,
        'nearest': cv2.BORDER_REPLICATE,
        'wrap': cv2.BORDER_WRAP,
    }

    def __init__(
        self,
        padding_type: str,
        res_mm: float,
        sigma_mm: float,
        lambda_mm: float,
        gamma: float,
        theta: float,
        rotation_invariance: bool = False,
        orthogonal_planes: bool = False,
        n_stds: float = None,
    ):
        super().__init__(
            filtering_method='Gabor',
            padding_type=padding_type,
            res_mm=res_mm,
            sigma_mm=sigma_mm,
            lambda_mm=lambda_mm,
            gamma=gamma,
            theta=theta,
            rotation_invariance=rotation_invariance,
            orthogonal_planes=orthogonal_planes,
            n_stds=n_stds,
        )

        try:
            self._border = self._PADDING_MAP[padding_type]
        except KeyError:
            raise ValueError(f"padding_type must be one of {list(self._PADDING_MAP)}, got {padding_type!r}")
        self.rotation_invariance = rotation_invariance
        self.res_mm = res_mm
        self.theta = theta
        self.gamma = gamma
        self.lambda_mm = lambda_mm
        self.sigma_mm = sigma_mm
        self.padding_type = padding_type
        self.orthogonal_planes = orthogonal_planes
        self.n_stds = n_stds

    @lru_cache(maxsize=128)
    def _make_kernels(self, theta, ksize):
        if ksize % 2 == 0:
            ksize += 1
        kern_real = cv2.getGaborKernel(
            (ksize, ksize),
            self.sigma_mm / self.res_mm,
            theta,
            self.lambda_mm / self.res_mm,
            self.gamma,
            0,
            ktype=cv2.CV_32F,
        )
        kern_imag = cv2.getGaborKernel(
            (ksize, ksize),
            self.sigma_mm / self.res_mm,
            theta,
            self.lambda_mm / self.res_mm,
            self.gamma,
            np.pi / 2,
            ktype=cv2.CV_32F,
        )
        return kern_real, kern_imag

    def _filter(self, img, theta, plane2d=(0, 1)):
        axes = list(plane2d) + [i for i in range(3) if i not in plane2d]
        img_view = np.transpose(img, axes).astype(np.float32, copy=False)

        if self.n_stds is None:
            ksize = int(np.ceil(7 * (self.sigma_mm / self.res_mm))) | 1
        else:
            ksize = int(np.ceil(self.n_stds * (self.sigma_mm / self.res_mm))) | 1
        kern_r, kern_i = self._make_kernels(theta, ksize)

        out = np.empty_like(img_view)
        for z in range(img_view.shape[2]):
            slice_ = img_view[:, :, z]
            out_r = cv2.filter2D(slice_, -1, kern_r, borderType=self._border)
            out_i = cv2.filter2D(slice_, -1, kern_i, borderType=self._border)
            out[:, :, z] = np.hypot(out_r, out_i)

        return np.transpose(out, np.argsort(axes))

    def _apply_array(self, img):
        if self.rotation_invariance:
            thetas = np.arange(0, 2 * np.pi, self.theta, dtype=np.float32)
            planes = [(0, 1), (0, 2), (1, 2)] if self.orthogonal_planes else [(0, 1)]
            resp = [self._filter(img, th, pl) for th in thetas for pl in planes]
            return np.mean(resp, axis=0, dtype=np.float32)
        return self._filter(img, self.theta)
