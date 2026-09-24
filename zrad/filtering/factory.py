from .base import BaseFilter
from .spatial import Gabor, Laws, LoG, Mean, RieszLoG
from .wavelet import Simoncelli, Wavelets2D, Wavelets3D


def _parse_riesz_order(value):
    if isinstance(value, str):
        return tuple(int(order.strip()) for order in value.split(','))
    return value


def _parse_optional_riesz_order(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _parse_riesz_order(value)


def create_filter(filtering_method, **kwargs) -> BaseFilter:
    """Create a filter from GUI or configuration-style parameters.

    Direct use of concrete filter classes such as ``Mean`` or ``LoG`` is the
    recommended Python API. This helper is intended for dynamic workflows where
    the filter family is selected by name, for example from GUI controls or
    saved configuration files.

    Parameters
    ----------
    filtering_method : {"Mean", "Laplacian of Gaussian", "Riesz-transformed LoG", "Laws Kernels", "Gabor", "Wavelets", "Simoncelli"}
        Filter family to instantiate.
    **kwargs
        Constructor parameters for the selected filter. For separable
        wavelets, include ``dimensionality`` to choose between ``Wavelets2D``
        and ``Wavelets3D``. ``Riesz-transformed LoG`` accepts ``riesz_order``
        and optional ``structure_tensor_sigma_mm``; ``Simoncelli`` accepts an
        optional ``riesz_order``.

    Returns
    -------
    filter : BaseFilter
        Configured concrete filter instance.
    """
    params = kwargs
    if filtering_method == 'Mean':
        return Mean(
            padding_type=params['padding_type'], support=int(params['support']), dimensionality=params['dimensionality']
        )
    if filtering_method == 'Laplacian of Gaussian':
        return LoG(
            padding_type=params['padding_type'],
            sigma_mm=float(params['sigma_mm']),
            cutoff=float(params['cutoff']),
            dimensionality=params['dimensionality'],
        )
    if filtering_method == 'Riesz-transformed LoG':
        return RieszLoG(
            padding_type=params['padding_type'],
            sigma_mm=float(params['sigma_mm']),
            cutoff=float(params['cutoff']),
            dimensionality=params['dimensionality'],
            riesz_order=_parse_riesz_order(params['riesz_order']),
            structure_tensor_sigma_mm=(
                None if params.get('structure_tensor_sigma_mm') is None else float(params['structure_tensor_sigma_mm'])
            ),
        )
    if filtering_method == 'Laws Kernels':
        return Laws(
            response_map=params['response_map'],
            padding_type=params['padding_type'],
            dimensionality=params['dimensionality'],
            rotation_invariance=params['rotation_invariance'],
            pooling=params['pooling'],
            energy_map=params['energy_map'],
            distance=int(params['distance']),
        )
    if filtering_method == 'Gabor':
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
    if filtering_method == 'Wavelets':
        dim = params['dimensionality']
        common = dict(
            wavelet_type=params['wavelet_type'],
            padding_type=params['padding_type'],
            response_map=params['response_map'],
            decomposition_level=int(params['decomposition_level']),
            rotation_invariance=params['rotation_invariance'],
        )
        if dim == '2D':
            return Wavelets2D(**common)
        if dim == '3D':
            return Wavelets3D(**common)
        raise ValueError(f"Filter_dimension {params['dimensionality']} is not supported.")
    if filtering_method == 'Simoncelli':
        return Simoncelli(
            padding_type=params['padding_type'],
            decomposition_level=int(params['decomposition_level']),
            dimensionality=params.get('dimensionality', '3D'),
            riesz_order=_parse_optional_riesz_order(params.get('riesz_order')),
        )
    raise ValueError(f"Filter {filtering_method} is not supported.")
