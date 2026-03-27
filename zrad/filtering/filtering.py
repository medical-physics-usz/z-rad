from .filtering_definitions import BaseFilter, Mean, LoG, Wavelets2D, Wavelets3D, Laws, Gabor


def create_filter(filtering_method, **kwargs) -> BaseFilter:
    """Create one configured filter instance."""
    params = kwargs
    if filtering_method == 'Mean':
        return Mean(
            padding_type=params['padding_type'],
            support=int(params['support']),
            dimensionality=params['dimensionality']
        )
    if filtering_method == 'Laplacian of Gaussian':
        return LoG(
            padding_type=params['padding_type'],
            sigma_mm=float(params['sigma_mm']),
            cutoff=float(params['cutoff']),
            dimensionality=params['dimensionality']
        )
    if filtering_method == 'Laws Kernels':
        return Laws(
            response_map=params['response_map'],
            padding_type=params['padding_type'],
            dimensionality=params['dimensionality'],
            rotation_invariance=params['rotation_invariance'],
            pooling=params['pooling'],
            energy_map=params['energy_map'],
            distance=int(params['distance'])
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
            rotation_invariance=params['rotation_invariance']
        )
        if dim == '2D':
            return Wavelets2D(**common)
        if dim == '3D':
            return Wavelets3D(**common)
        raise ValueError(f"Filter_dimension {params['dimensionality']} is not supported.")
    raise ValueError(f"Filter {filtering_method} is not supported.")


# Backward-compatible alias for the previous public constructor.
Filtering = create_filter
