import sys

import numpy as np

from .image import Image
from .toolbox_logic import handle_uncaught_exception
from ..logic.filtering_definitions import Mean, LoG, Wavelets2D, Wavelets3D, Laws

sys.excepthook = handle_uncaught_exception


class Filtering:
    def __init__(self, filtering_method, **kwargs):
        self.filtering_method = filtering_method
        self.filtering_params = kwargs
        self.filter = self._get_filter(filtering_method)

    def _get_filter(self, filtering_method):
        if filtering_method == 'Mean':
            my_filter = Mean(padding_type=self.filtering_params["padding_type"],
                             support=int(self.filtering_params["support"]),
                             dimensionality=self.filtering_params["dimensionality"],
                             )
        elif filtering_method == 'Laplacian of Gaussian':
            my_filter = LoG(padding_type=self.filtering_params["padding_type"],
                            sigma_mm=float(self.filtering_params["sigma_mm"]),
                            cutoff=float(self.filtering_params["cutoff"]),
                            dimensionality=self.filtering_params["dimensionality"],
                            )
        elif filtering_method == 'Laws Kernels':
            my_filter = Laws(response_map=self.filtering_params["response_map"],
                             padding_type=self.filtering_params["padding_type"],
                             dimensionality=self.filtering_params["dimensionality"],
                             rotation_invariance=self.filtering_params["rotation_invariance"],
                             pooling=self.filtering_params["pooling"],
                             energy_map=self.filtering_params["energy_map"],
                             distance=int(self.filtering_params["distance"])
                             )
        elif filtering_method == 'Wavelets':
            if self.filtering_params["dimensionality"] == '2D':
                my_filter = Wavelets2D(wavelet_type=self.filtering_params["wavelet_type"],
                                       padding_type=self.filtering_params["padding_type"],
                                       response_map=self.filtering_params["response_map"],
                                       decomposition_level=self.filtering_params["decomposition_level"],
                                       rotation_invariance=self.filtering_params["rotation_invariance"]
                                       )
            elif self.filtering_params["dimensionality"] == '3D':
                my_filter = Wavelets3D(wavelet_type=self.filtering_params["wavelet_type"],
                                       padding_type=self.filtering_params["padding_type"],
                                       response_map=self.filtering_params["response_map"],
                                       decomposition_level=self.filtering_params["decomposition_level"],
                                       rotation_invariance=self.filtering_params["rotation_invariance"]
                                       )
            else:
                raise ValueError(f"Filter_dimension {self.filtering_params['filter_dimension']} is not supported.")
        else:
            raise ValueError(f"Filter {filtering_method} is not supported.")
        return my_filter

    def apply_filter(self, image):
        # Adjust filter settings based on the filter type
        if self.filtering_method == 'Laplacian of Gaussian':
            try:
                self.filter.res_mm = float(image.spacing[0])
            except (AttributeError, ValueError) as e:
                raise ValueError(f"Invalid image spacing data: {e}")

        # Perform filtering
        try:
            filtered_array = self.filter.apply(image.array.astype(np.float64).transpose(1, 2, 0))
        except Exception as e:
            raise ValueError(f"Error applying filter: {e}")

        # Create the filtered image to save, preserving metadata
        filtered_image = Image(
            array=filtered_array.transpose(2, 0, 1),
            origin=image.origin,
            spacing=image.spacing,
            direction=image.direction,
            shape=image.shape
        )
        return filtered_image
