from datetime import datetime

import numpy as np
import pywt
from scipy import ndimage as ndi

from ..toolbox_logic import get_logger, close_all_loggers
from .base import BaseFilter


class Wavelets2D(BaseFilter):
    """2D wavelet response maps evaluated slice-wise on a 3D volume."""

    def __init__(self, wavelet_type, padding_type, response_map, decomposition_level, rotation_invariance=False):
        super().__init__(
            filtering_method='Wavelets',
            wavelet_type=wavelet_type,
            padding_type=padding_type,
            response_map=response_map,
            decomposition_level=decomposition_level,
            rotation_invariance=rotation_invariance,
            dimensionality='2D'
        )

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

    def _apply_array(self, img):
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
                    img_level0 = np.rot90(img[:, :, i], k=k, axes=(0, 1))
                    img_level1 = self._filter(img_level0, l_filter, l_filter)
                    img_level2 = self._filter(img_level1, x_filter, y_filter)
                    final_image[:, :, i] += np.rot90(img_level2, k=k, axes=(1, 0))
            filtered_img = final_image / 4

        return filtered_img


class Wavelets3D(BaseFilter):
    """3D wavelet response maps for volumetric filtering."""

    def __init__(self, wavelet_type, padding_type, response_map, decomposition_level, rotation_invariance=False):
        super().__init__(
            filtering_method='Wavelets',
            wavelet_type=wavelet_type,
            padding_type=padding_type,
            response_map=response_map,
            decomposition_level=decomposition_level,
            rotation_invariance=rotation_invariance,
            dimensionality='3D'
        )

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

    def _apply_array(self, img):
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
