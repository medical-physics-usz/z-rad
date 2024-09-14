import sys
from datetime import datetime
from itertools import permutations

import numpy as np
import pywt
from scipy import ndimage as ndi

from .toolbox_logic import get_logger, handle_uncaught_exception, close_all_loggers

sys.excepthook = handle_uncaught_exception


class Mean:
    def __init__(self, padding_type, support, dimensionality):
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

    def apply(self, img):
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


class LoG:
    """LoG"""

    def __init__(self, padding_type, sigma_mm, cutoff, dimensionality):

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

    def apply(self, img):
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


class Wavelets2D:
    """Wavelet filtering in 2D."""

    def __init__(self, wavelet_type, padding_type, response_map, decomposition_level, rotation_invariance=False):

        close_all_loggers()
        self.filter_logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.filter_logger = get_logger(self.filter_logger_date_time+'_Wavelets2D')

        self.type = 'Wavelets'

        self.dimensionality = '2D'

        if padding_type in ['constant', 'nearest', 'wrap', 'reflect']:
            self.padding_type = padding_type
        else:
            self.filter_logger.error(f"Wrong padding type '{padding_type}'. "
                                     "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")
            raise ValueError(f"Wrong padding type '{padding_type}'. "
                             "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")

        if wavelet_type in ['db3', 'db2', 'coif1', 'haar']:
            self.wavelet_type = wavelet_type
        else:
            self.filter_logger.error(f"Wrong wavelet type '{wavelet_type}'. "
                                     "Available wavelet types are: 'db3', 'db2', 'coif1', 'haar'.")
            raise ValueError(f"Wrong wavelet type '{wavelet_type}'. "
                             "Available wavelet types are: 'db3', 'db2', 'coif1', 'haar'.")

        if decomposition_level in [1, 2]:
            self.decomposition_level = decomposition_level
        else:
            self.filter_logger.error(f"Wrong decomposition_level' {decomposition_level}'. "
                                     "Decomposition level should be integer. "
                                     "Available decomposition levels are: 1 and 2.")
            raise ValueError(f"Wrong decomposition_level' {decomposition_level}'. "
                             "Decomposition level should be integer. Available decomposition levels are: 1 and 2.")

        if response_map in ['LL', 'HL', 'LH', 'HH']:
            self.response_map = response_map
        else:
            self.filter_logger.error(f"Wrong response_map' {response_map}'. "
                                     "Available response_maps are: 'LL', 'HL', 'LH', 'HH'.")
            raise ValueError(f"Wrong response_map' {response_map}'. "
                             "Available response_maps are: 'LL', 'HL', 'LH', 'HH'.")

        if isinstance(rotation_invariance, bool):
            self.rotation_invariance = rotation_invariance
        else:
            self.filter_logger.error("Rotation Invariance should be "
                                     f"True or False but '{type(rotation_invariance)}' detected.")
            raise ValueError(f"Rotation Invariance should be True or False but '{type(rotation_invariance)}' detected.")

        self.filter_logger.debug(f"Defined 2D {wavelet_type} Wavelet filter with response map {response_map}, "
                                 f"decomposition level {decomposition_level}, "
                                 f"pseudo rotation invariance is {rotation_invariance}"
                                 f"and {padding_type} padding type.")

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

    def apply(self, img):
        if self.decomposition_level == 1:
            x_filter = self._get_kernel(self.response_map[0])
            y_filter = self._get_kernel(self.response_map[1])
            if self.rotation_invariance:
                final_image = np.zeros(img.shape)
                for i in range(img.shape[2]):
                    for k in range(4):
                        final_image[:, :, i] += np.rot90(self._filter(np.rot90(img[:, :, i], k=k, axes=(0, 1)),
                                                         x_filter, y_filter), k=k, axes=(1, 0))
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
                    img_level0 = np.rot90(img[:, :, i], k=k, axes=(0, 1))  # original
                    img_level1 = self._filter(img_level0, l_filter, l_filter)
                    img_level2 = self._filter(img_level1, x_filter, y_filter)
                    final_image[:, :, i] += np.rot90(img_level2, k=k, axes=(1, 0))
            filtered_img = final_image / 4

        return filtered_img


class Wavelets3D:
    """Wavelet filtering."""

    def __init__(self, wavelet_type, padding_type, response_map, decomposition_level, rotation_invariance=False):

        close_all_loggers()
        self.filter_logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.filter_logger = get_logger(self.filter_logger_date_time+'_Wavelets3D')

        self.type = 'Wavelets'

        self.dimensionality = '3D'

        if padding_type in ['constant', 'nearest', 'wrap', 'reflect']:
            self.padding_type = padding_type
        else:
            self.filter_logger.error(f"Wrong padding type '{padding_type}'. "
                                     "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")
            raise ValueError(f"Wrong padding type '{padding_type}'. "
                             "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'.")

        if wavelet_type in ['db3', 'db2', 'coif1', 'haar']:
            self.wavelet_type = wavelet_type
        else:
            self.filter_logger.error(f"Wrong wavelet type '{wavelet_type}'. "
                                     "Available wavelet types are: 'db3', 'db2', 'coif1', 'haar'.")
            raise ValueError(f"Wrong wavelet type '{wavelet_type}'. "
                             "Available wavelet types are: 'db3', 'db2', 'coif1', 'haar'.")

        if decomposition_level in [1, 2]:
            self.decomposition_level = decomposition_level
        else:
            self.filter_logger.error(f"Wrong decomposition_level' {decomposition_level}'. "
                                     "Decomposition level should be integer. "
                                     "Available decomposition levels are: 1 and 2.")
            raise ValueError(f"Wrong decomposition_level' {decomposition_level}'. "
                             "Decomposition level should be integer. Available decomposition levels are: 1 and 2.")

        if response_map in ['LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', 'HLH', 'HHH']:
            self.response_map = response_map
        else:
            self.filter_logger.error(f"Wrong response_map' {response_map}'. "
                                     "Available response_maps are: "
                                     "'LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', 'HLH', 'HHH'.")
            raise ValueError(f"Wrong response_map' {response_map}'. "
                             "Available response_maps are: 'LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', 'HLH', 'HHH'.")

        if isinstance(rotation_invariance, bool):
            self.rotation_invariance = rotation_invariance
        else:
            self.filter_logger.error("Rotation Invariance should be "
                                     f"True or False but '{type(rotation_invariance)}' detected.")
            raise ValueError(f"Rotation Invariance should be True or False but '{type(rotation_invariance)}' detected.")

        self.pooling = None

        self.filter_logger.debug(f"Defined 3D {wavelet_type} Wavelet filter with response map {response_map}, "
                                 f"decomposition level {decomposition_level}, "
                                 f"pseudo rotation invariance is {rotation_invariance}"
                                 f"and {padding_type} padding type.")

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

    def apply(self, img):
        if self.decomposition_level == 1:
            x_filter = self._get_kernel(self.response_map[0])
            y_filter = self._get_kernel(self.response_map[1])
            z_filter = self._get_kernel(self.response_map[2])
            if self.rotation_invariance:
                final_image = np.zeros(img.shape)
                kernels_permutation = [(x_filter, y_filter, z_filter),
                                       (z_filter, x_filter, y_filter),
                                       (y_filter, z_filter, x_filter)]
                for kernels in kernels_permutation:
                    final_image += self._filter(img, kernels[0], kernels[1], kernels[2])
                    final_image += self._filter(img[::-1, :, :], kernels[0], kernels[1], kernels[2])[::-1, :, :]
                    final_image += self._filter(img[:, ::-1, :], kernels[0], kernels[1], kernels[2])[:, ::-1, :]
                    final_image += self._filter(img[:, :, ::-1], kernels[0], kernels[1], kernels[2])[:, :, ::-1]
                    final_image += self._filter(img[::-1, ::-1, :], kernels[0], kernels[1], kernels[2])[::-1, ::-1, :]
                    final_image += self._filter(img[::-1, :, ::-1], kernels[0], kernels[1], kernels[2])[::-1, :, ::-1]
                    final_image += self._filter(img[:, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[:, ::-1, ::-1]
                    final_image += self._filter(img[::-1, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[::-1, ::-1, ::-1]
                filtered_img = final_image / (8 * len(kernels_permutation))
            else:
                filtered_img = self._filter(img, x_filter, y_filter, z_filter)
        else:
            # First, do low pass LLL
            x_filter = self._get_kernel("L")
            y_filter = self._get_kernel("L")
            z_filter = self._get_kernel("L")
            kernels_permutation = [(x_filter, y_filter, z_filter),
                                   (z_filter, x_filter, y_filter),
                                   (y_filter, z_filter, x_filter)]
            level1_responses = list()
            for kernels in kernels_permutation:
                level1_responses.append(self._filter(img, kernels[0], kernels[1], kernels[2]))
                level1_responses.append(self._filter(img[::-1, :, :], kernels[0], kernels[1], kernels[2])[::-1, :, :])
                level1_responses.append(self._filter(img[:, ::-1, :], kernels[0], kernels[1], kernels[2])[:, ::-1, :])
                level1_responses.append(self._filter(img[:, :, ::-1], kernels[0], kernels[1], kernels[2])[:, :, ::-1])
                level1_responses.append(self._filter(img[::-1, ::-1, :], kernels[0], kernels[1], kernels[2])[::-1, ::-1, :])
                level1_responses.append(self._filter(img[::-1, :, ::-1], kernels[0], kernels[1], kernels[2])[::-1, :, ::-1])
                level1_responses.append(self._filter(img[:, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[:, ::-1, ::-1])
                level1_responses.append(self._filter(img[::-1, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[::-1, ::-1, ::-1])

            # Now do 2nd level filtering
            x_filter = self._get_kernel(self.response_map[0], decomposition_level=2)
            y_filter = self._get_kernel(self.response_map[1], decomposition_level=2)
            z_filter = self._get_kernel(self.response_map[2], decomposition_level=2)
            final_image = np.zeros(img.shape)
            kernels_permutation = [(x_filter, y_filter, z_filter),
                                   (z_filter, x_filter, y_filter),
                                   (y_filter, z_filter, x_filter)]
            for kernels in kernels_permutation:
                final_image += self._filter(level1_responses[0], kernels[0], kernels[1], kernels[2])
                final_image += self._filter(level1_responses[1][::-1, :, :], kernels[0], kernels[1], kernels[2])[::-1, :, :]
                final_image += self._filter(level1_responses[2][:, ::-1, :], kernels[0], kernels[1], kernels[2])[:, ::-1, :]
                final_image += self._filter(level1_responses[3][:, :, ::-1], kernels[0], kernels[1], kernels[2])[:, :, ::-1]
                final_image += self._filter(level1_responses[4][::-1, ::-1, :], kernels[0], kernels[1], kernels[2])[::-1, ::-1, :]
                final_image += self._filter(level1_responses[5][::-1, :, ::-1], kernels[0], kernels[1], kernels[2])[::-1, :, ::-1]
                final_image += self._filter(level1_responses[6][:, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[:, ::-1, ::-1]
                final_image += self._filter(level1_responses[7][::-1, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[::-1, ::-1, ::-1]
            filtered_img = final_image / (8 * len(kernels_permutation))
        return filtered_img


class Laws:
    """Laws2"""

    def __init__(self, response_map, padding_type, distance, energy_map, dimensionality,
                 rotation_invariance=False, pooling=None):

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

    def apply(self, img):
        final_image = None
        if self.rotation_invariance:
            response_maps = self._get_response_maps()

            if self.pooling == "avg":
                # average pooling of Laws filtering was not tested in IBSI 2.
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
                final_image = final_image / 24  # shouldn't it rather be len(response_maps) * 8 = 48?

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