from datetime import datetime
from functools import lru_cache
from itertools import permutations

import cv2
import numpy as np
from scipy import ndimage as ndi

from ..toolbox_logic import get_logger, close_all_loggers
from .base import BaseFilter


class Mean(BaseFilter):
    """Mean filter for 2D slice-wise or full 3D smoothing."""

    def __init__(self, padding_type, support, dimensionality):
        super().__init__(
            filtering_method='Mean',
            padding_type=padding_type,
            support=support,
            dimensionality=dimensionality
        )
        close_all_loggers()
        self.filter_logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.filter_logger = get_logger(self.filter_logger_date_time+"_Mean_filter")

        self.type = 'Mean'

        if dimensionality in ['2D', '3D']:
            self.dimensionality = dimensionality
        else:
            self.filter_logger.error(f"Wrong dimensionality '{dimensionality}'. Available dimensions '2D' and '3D'.")
            raise ValueError(f"Wrong dimensionality '{dimensionality}'. Available dimensions '2D' and '3D'.")

        if isinstance(support, int):
            self.support = support
        else:
            self.filter_logger.error(f"Support should be int but '{type(support)}' detected.")
            raise ValueError(f"Support should be int but '{type(support)}' detected.")

        if padding_type in ['constant', 'nearest', 'wrap', 'reflect']:
            self.padding_type = padding_type
        else:
            self.filter_logger.error(f"Wrong padding type '{padding_type}'. "
                                     "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")
            raise ValueError(f"Wrong padding type '{padding_type}'. "
                             "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")

        self.filter_logger.debug(f"Defined {dimensionality} mean filter with support of {support}, "
                                 f"and {padding_type} padding type.")

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
    """Laplacian-of-Gaussian filter for blob and edge enhancement."""

    def __init__(self, padding_type, sigma_mm, cutoff, dimensionality):
        super().__init__(
            filtering_method='Laplacian of Gaussian',
            padding_type=padding_type,
            sigma_mm=sigma_mm,
            cutoff=cutoff,
            dimensionality=dimensionality
        )

        close_all_loggers()
        self.filter_logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.filter_logger = get_logger(self.filter_logger_date_time+'_LoG_filter')

        self.type = 'Laplacian of Gaussian'

        if dimensionality in ['2D', '3D']:
            self.dimensionality = dimensionality
        else:
            self.filter_logger.error(f"Wrong dimensionality '{dimensionality}'. Available dimensions '2D' and '3D'.")
            raise ValueError(f"Wrong dimensionality '{dimensionality}'. Available dimensions '2D' and '3D'.")

        if padding_type in ['constant', 'nearest', 'wrap', 'reflect']:
            self.padding_type = padding_type
        else:
            self.filter_logger.error(f"Wrong padding type '{padding_type}'. "
                                     f"Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")
            raise ValueError(f"Wrong padding type '{padding_type}'. "
                             f"Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")

        if isinstance(sigma_mm, (int, float)):
            self.sigma_mm = sigma_mm
        else:
            self.filter_logger.error(f'Sigma (in mm) should be int or float but {type(sigma_mm)} detected.')
            raise ValueError(f'Sigma (in mm) should be int or float but {type(sigma_mm)} detected.')

        if isinstance(cutoff, (int, float)):
            self.cutoff = cutoff
        else:
            self.filter_logger.error(f'Cutoff should be int or float but {type(cutoff)} detected.')
            raise ValueError(f'Cutoff should be int or float but {type(cutoff)} detected.')

        self.padding_constant = 0.0
        self.res_mm = None

        self.filter_logger.debug(f"Defined {dimensionality} LoG filter with sigma {sigma_mm}, cutoff {cutoff}, "
                                 f"and {padding_type} padding type.")

    def _prepare(self, image):
        try:
            self.res_mm = float(image.spacing[0])
        except (AttributeError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid image spacing data: {e}")

    def _apply_array(self, img):
        sigma = self.sigma_mm / self.res_mm
        if self.dimensionality == "3D":
            filtered_img = ndi.gaussian_laplace(img, sigma=sigma, mode=self.padding_type, cval=self.padding_constant,
                                                truncate=self.cutoff)
        elif self.dimensionality == "2D":
            filtered_img = np.nan * np.ones(img.shape)
            for i in range(img.shape[2]):
                filtered_img[:, :, i] = ndi.gaussian_laplace(img[:, :, i], sigma=sigma, mode=self.padding_type,
                                                             cval=self.padding_constant, truncate=self.cutoff)
        else:
            filtered_img = None
        return filtered_img


class Laws(BaseFilter):
    """Laws-kernel texture filtering in 2D or 3D."""

    def __init__(self, response_map, padding_type, distance, energy_map, dimensionality,
                 rotation_invariance=False, pooling=None):
        super().__init__(
            filtering_method='Laws Kernels',
            response_map=response_map,
            padding_type=padding_type,
            distance=distance,
            energy_map=energy_map,
            dimensionality=dimensionality,
            rotation_invariance=rotation_invariance,
            pooling=pooling
        )

        close_all_loggers()
        self.filter_logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.filter_logger = get_logger(self.filter_logger_date_time+'_Laws_kernels')

        self.type = 'Laws Kernels'

        if dimensionality in ['2D', '3D']:
            self.dimensionality = dimensionality
        else:
            self.filter_logger.error(f"Wrong dimensionality '{dimensionality}'. Available dimensions '2D' and '3D'.")
            raise ValueError(f"Wrong dimensionality '{dimensionality}'. Available dimensions '2D' and '3D'.")

        if padding_type in ['constant', 'nearest', 'wrap', 'reflect']:
            self.padding_type = padding_type
        else:
            self.filter_logger.error(f"Wrong padding type '{padding_type}'. "
                                     "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")
            raise ValueError(f"Wrong padding type '{padding_type}'. "
                             "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")

        if isinstance(distance, int):
            self.distance = distance
        else:
            self.filter_logger.error(f"Distance should be 'int' but '{type(distance)}' detected.")
            raise ValueError(f"Distance should be 'int' but '{type(distance)}' detected.")

        if isinstance(energy_map, bool):
            self.energy_map = energy_map
        else:
            self.filter_logger.error('Energy map can be only True or False.')
            raise ValueError('Energy map can be only True or False.')

        if isinstance(rotation_invariance, bool):
            self.rotation_invariance = rotation_invariance
        else:
            self.filter_logger.error("Rotation Invariance should be "
                                     f"True or False but '{type(rotation_invariance)}' detected.")
            raise ValueError(f"Rotation Invariance should be True or False but '{type(rotation_invariance)}' detected.")

        self.response_map = response_map
        self.pooling = pooling

        self.filter_logger.debug(f"Defined {dimensionality} Laws Kernels filter with energy map is {energy_map}, "
                                 f"response map {response_map}, distance {distance} "
                                 f"pseudo rotation invariance is {rotation_invariance}, pooling {pooling},"
                                 f"and {padding_type} padding type.")

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
        parts = [self.response_map[i:i + 2] for i in range(0, len(self.response_map), 2)]
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
                    final_image = np.maximum(final_image,
                                             self._filter(img[::-1, ::-1, ::-1], response_map)[::-1, ::-1, ::-1])
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
    """Gabor filtering for 3D volumes using 2D kernels on orthogonal planes."""

    _PADDING_MAP = {
        'reflect': cv2.BORDER_REFLECT,
        'mirror': cv2.BORDER_REFLECT_101,
        'constant': cv2.BORDER_CONSTANT,
        'nearest': cv2.BORDER_REPLICATE,
        'wrap': cv2.BORDER_WRAP,
    }

    def __init__(self,
                 padding_type: str,
                 res_mm: float,
                 sigma_mm: float,
                 lambda_mm: float,
                 gamma: float,
                 theta: float,
                 rotation_invariance: bool = False,
                 orthogonal_planes: bool = False,
                 n_stds: float = None):
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
            n_stds=n_stds
        )

        try:
            self._border = self._PADDING_MAP[padding_type]
        except KeyError:
            raise ValueError(f"padding_type must be one of {list(self._PADDING_MAP)}, "
                             f"got {padding_type!r}")
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
            (ksize, ksize), self.sigma_mm / self.res_mm,
            theta, self.lambda_mm / self.res_mm,
            self.gamma, 0, ktype=cv2.CV_32F)
        kern_imag = cv2.getGaborKernel(
            (ksize, ksize), self.sigma_mm / self.res_mm,
            theta, self.lambda_mm / self.res_mm,
            self.gamma, np.pi / 2, ktype=cv2.CV_32F)
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
