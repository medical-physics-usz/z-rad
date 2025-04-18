import sys
import numpy as np
from ..image import Image
from ..toolbox_logic import handle_uncaught_exception
from .filtering_definitions import Mean, LoG, Wavelets2D, Wavelets3D, Laws, Gabor

sys.excepthook = handle_uncaught_exception

class Filtering:
    def __init__(self, filtering_method, **kwargs):
        self.filtering_method = filtering_method
        self.filtering_params = kwargs
        self.filter = self._get_filter(filtering_method)

    def _get_filter(self, filtering_method):
        params = self.filtering_params
        if filtering_method == 'Mean':
            return Mean(
                padding_type=params['padding_type'],
                support=int(params['support']),
                dimensionality=params['dimensionality']
            )
        elif filtering_method == 'Laplacian of Gaussian':
            return LoG(
                padding_type=params['padding_type'],
                sigma_mm=float(params['sigma_mm']),
                cutoff=float(params['cutoff']),
                dimensionality=params['dimensionality']
            )
        elif filtering_method == 'Laws Kernels':
            return Laws(
                response_map=params['response_map'],
                padding_type=params['padding_type'],
                dimensionality=params['dimensionality'],
                rotation_invariance=params['rotation_invariance'],
                pooling=params['pooling'],
                energy_map=params['energy_map'],
                distance=int(params['distance'])
            )
        elif filtering_method == 'Gabor':
            return Gabor(
                padding_type=params['padding_type'],
                res_mm=float(params['res_mm']),
                sigma_mm=float(params['sigma_mm']),
                lambda_mm=float(params['lambda_mm']),
                gamma=float(params['gamma']),
                theta=float(params['theta']),
                rotation_invariance=params.get('rotation_invariance', False),
                orthogonal_planes=params.get('orthogonal_planes', False),
                n_stds=params.get('n_stds', None),
            )
        elif filtering_method == 'Wavelets':
            dim = params['dimensionality']
            common = dict(
                wavelet_type=params['wavelet_type'],
                padding_type=params['padding_type'],
                response_map=params['response_map'],
                decomposition_level=int(params['decomposition_level']),
                rotation_invariance=params['rotation_invariance']
            )
            if dim == '2D':
                return Wavelets2D(**common)
            elif dim == '3D':
                return Wavelets3D(**common)
            else:
                raise ValueError(f"Filter_dimension {params['dimensionality']} is not supported.")
        else:
            raise ValueError(f"Filter {filtering_method} is not supported.")

    def apply_filter(self, image):
        # Adjust LoG resolution from image spacing
        if self.filtering_method == 'Laplacian of Gaussian':
            try:
                self.filter.res_mm = float(image.spacing[0])
            except (AttributeError, ValueError, TypeError) as e:
                raise ValueError(f"Invalid image spacing data: {e}")

        # Prepare array: (Z,Y,X) -> (Y,X,Z)
        arr = image.array.astype(np.float64).transpose(1, 2, 0)
        try:
            if hasattr(self.filter, 'apply'):
                filtered = self.filter.apply(arr)
            elif hasattr(self.filter, 'filter'):
                filtered = self.filter.filter(arr)
            else:
                raise ValueError("Filter object lacks apply/filter method")
        except Exception as e:
            raise ValueError(f"Error applying filter: {e}")

        # Restore shape: (Y,X,Z) -> (Z,Y,X)
        out_arr = filtered.transpose(2, 0, 1)
        return Image(
            array=out_arr,
            origin=image.origin,
            spacing=image.spacing,
            direction=image.direction,
            shape=image.shape
        )
