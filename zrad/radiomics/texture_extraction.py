"""Shared feature-group execution and optional texture trace retention."""

from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from .texture_matrices import readonly


@dataclass(frozen=True)
class ExtractionResult:
    """Feature results, texture traces, and effective prepared ROI arrays.

    Input slice coordinates refer to these arrays. ``excluded_slices`` lists
    originally nonempty morphological slices removed by extraction validation.
    """

    features: dict
    texture: dict
    discretized_image: object
    morphological_mask: object
    excluded_slices: tuple

    def __post_init__(self):
        object.__setattr__(self, 'features', MappingProxyType(dict(self.features)))
        object.__setattr__(self, 'texture', MappingProxyType(dict(self.texture)))
        for name in ('discretized_image', 'morphological_mask'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, readonly(value))


def build_texture_family(group, context, prepared_data, *, options=None):
    from .glcm import GLCM
    from .gldzm import GLDZM
    from .glrlm import GLRLM
    from .glszm import GLSZM
    from .ngldm import NGLDM
    from .ngtdm import NGTDM

    calculators = dict(glcm=GLCM, glrlm=GLRLM, glszm=GLSZM, gldzm=GLDZM, ngldm=NGLDM, ngtdm=NGTDM)
    family = group.family
    kwargs = dict(aggr_dim=context.aggr_dim, slice_weight=context.slice_weighting, slice_median=context.slice_median)
    if family in ('glcm', 'glrlm'):
        kwargs['aggr_method'] = context.aggr_method
    calculator = calculators[family](**kwargs)
    options = dict(options or {})
    if set(options) - {'directions', 'slice_axis', 'symmetric'}:
        raise ValueError('Texture extraction options support directions, slice_axis and symmetric only.')
    if options.get('slice_axis', 0) != 0:
        raise ValueError('Extraction validates axial slices; use the standalone builder for other slice axes.')
    if family == 'gldzm':
        options['morphological_mask'] = prepared_data.require_analysis_masks().morphological_mask
    matrices = calculator.calculate_matrices(prepared_data.require_discretized_intensity_image(), **options)
    return calculator, matrices


def calculate_texture_family(group, context, prepared_data, *, retain=False, options=None):
    family = group.family
    options = dict(options or {})
    if family == 'glcm':
        symmetric = options.get('symmetric', True)
        if symmetric is not None and not isinstance(symmetric, (bool, np.bool_)):
            raise ValueError('symmetric must be a boolean.')
        if symmetric is not None and not symmetric:
            raise ValueError(
                'Asymmetric GLCM feature extraction is not supported; use calculate_matrices for inspection.'
            )
        if retain:
            options['symmetric'] = False
    calculator, raw = build_texture_family(group, context, prepared_data, options=options)
    matrices = raw.symmetrize() if family == 'glcm' and retain else raw
    trace = calculator.inspect_matrices(matrices)
    features = {
        output: trace.features[base]
        for output, base in zip(group.output_names(context), calculator.get_feature_names())
    }
    return features, replace(trace, raw=raw) if retain else None
