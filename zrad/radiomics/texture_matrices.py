"""Inspectable texture matrices shared by construction and feature extraction.

Public builders use NumPy axis order: (row, column) or (z, y, x). Grey-level
indices are retained, including the unused zero row, so the arrays are also the
exact inputs expected by the feature formulas.
"""

from dataclasses import dataclass, replace
from itertools import product
from types import MappingProxyType

import numpy as np

from ..exceptions import DataStructureError
from ..image import Image
from .texture_base import crop_to_valid_bbox

TEXTURE_FAMILIES = ('glcm', 'glrlm', 'glszm', 'gldzm', 'ngtdm', 'ngldm')


def readonly(array):
    """Make a detached array whose backing storage cannot be made writable."""
    array = np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class TextureMatrix:
    """One matrix, its labelled axes, and its construction provenance.

    ``voxel_count`` is the feature denominator (including direction multiplicity
    for merged GLRLMs). ``roi_voxel_count`` counts each contributing slice once.
    ``source_ids`` identify the unmerged matrices within a collection.
    """

    family: str
    values: np.ndarray
    axes: tuple
    axis_names: tuple
    direction: tuple | None = None
    slice_index: int | None = None
    slice_axis: int | None = None
    symmetric: bool | None = None
    normalized: bool = False
    voxel_count: int = 0
    roi_counts: tuple = ()
    source_ids: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'values', readonly(self.values))
        object.__setattr__(self, 'axes', tuple(tuple(axis) for axis in self.axes))
        if self.values.ndim != 2 or self.values.shape != tuple(map(len, self.axes)):
            raise ValueError('Matrix shape must match its two labelled axes.')

    @property
    def roi_voxel_count(self):
        return sum(count for _, count in self.roi_counts)

    @property
    def contributing_voxel_count(self):
        """NGTDM excludes voxels without any valid neighbours."""
        return int(self.n_i.sum()) if self.family == 'ngtdm' else self.roi_voxel_count

    @property
    def status(self):
        return 'empty' if not np.any(self.values) else 'ok'

    def symmetrize(self):
        """Return GLCM counts C + C.T, including doubled diagonal entries."""
        if self.family != 'glcm' or self.normalized:
            raise ValueError('Symmetrization requires GLCM counts.')
        if self.symmetric:
            return self
        return replace(self, values=self.values + self.values.T, symmetric=True)

    def probabilities(self):
        """Return a probability array for a count matrix (not NGTDM)."""
        if self.family == 'ngtdm':
            raise ValueError('NGTDM contains n_i and s_i; use p_i instead.')
        total = self.values.sum()
        if total == 0:
            raise DataStructureError('An empty matrix has no probability distribution.')
        return readonly(self.values / total)

    @property
    def n_i(self):
        if self.family != 'ngtdm':
            raise ValueError('n_i is defined only for NGTDM.')
        return self.values[:, 0]

    @property
    def s_i(self):
        if self.family != 'ngtdm':
            raise ValueError('s_i is defined only for NGTDM.')
        return self.values[:, 1]

    @property
    def p_i(self):
        total = self.n_i.sum()
        if total == 0:
            raise DataStructureError('NGTDM has no voxels with valid neighbours.')
        return readonly(self.n_i / total)


@dataclass(frozen=True)
class TextureMatrixCollection:
    """Ordered matrices with explicit selection and count merging.

    ``configuration`` records effective construction settings. Slice indices are
    relative to the supplied array, never renumbered after skipping empty slices.
    """

    matrices: tuple
    configuration: dict
    skipped_slices: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'matrices', tuple(self.matrices))
        object.__setattr__(self, 'configuration', MappingProxyType(dict(self.configuration)))

    def __iter__(self):
        return iter(self.matrices)

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, index):
        return self.matrices[index]

    def only(self):
        if len(self) != 1:
            raise ValueError(f'Expected exactly one matrix, found {len(self)}.')
        return self.matrices[0]

    def select(self, **criteria):
        """Select by direction and/or original slice_index."""
        if set(criteria) - {'direction', 'slice_index'}:
            raise ValueError('Select accepts direction and slice_index only.')
        return replace(
            self,
            matrices=tuple(
                matrix for matrix in self if all(getattr(matrix, key) == value for key, value in criteria.items())
            ),
        )

    def symmetrize(self):
        directions = {m.direction for m in self if m.direction is not None}
        if any(tuple(-v for v in d) in directions for d in directions):
            raise ValueError('Cannot symmetrize a collection with redundant opposite directions.')
        return replace(
            self,
            matrices=tuple(matrix.symmetrize() for matrix in self),
            configuration={**self.configuration, 'symmetric': True},
        )

    def merge(self, over):
        """Sum counts over directions, slices, or both; never average probabilities."""
        over = (over,) if isinstance(over, str) else tuple(over)
        if not over or set(over) - {'directions', 'slices'}:
            raise ValueError("over must contain 'directions' and/or 'slices'.")
        if 'directions' in over and self.configuration['family'] not in ('glcm', 'glrlm'):
            raise ValueError('This family has no directional matrices.')
        groups = {}
        for matrix in self:
            if matrix.normalized:
                raise ValueError('Merge counts before normalization.')
            key = (None if 'slices' in over else matrix.slice_index, None if 'directions' in over else matrix.direction)
            groups.setdefault(key, []).append(matrix)
        merged = []
        for (slice_index, direction), members in groups.items():
            first = members[0]
            if any(m.family != first.family or m.symmetric != first.symmetric or m.axes != first.axes for m in members):
                raise ValueError('Merged matrices must have matching families, symmetry and labelled axes.')
            roi_counts = dict(item for m in members for item in m.roi_counts)
            merged.append(
                replace(
                    first,
                    values=np.sum([m.values for m in members], axis=0),
                    slice_index=slice_index,
                    direction=direction,
                    voxel_count=sum(m.voxel_count for m in members),
                    roi_counts=tuple(roi_counts.items()),
                    source_ids=tuple(source for m in members for source in m.source_ids),
                )
            )
        return replace(self, matrices=tuple(merged))


@dataclass(frozen=True)
class TextureTrace:
    """Matrices and reductions sufficient to reproduce the reported features."""

    raw: TextureMatrixCollection
    feature_inputs: TextureMatrixCollection
    per_matrix_features: tuple
    aggregation: dict
    features: dict

    def __post_init__(self):
        object.__setattr__(
            self, 'per_matrix_features', tuple(MappingProxyType(dict(v)) for v in self.per_matrix_features)
        )
        object.__setattr__(self, 'aggregation', MappingProxyType(dict(self.aggregation)))
        object.__setattr__(self, 'features', MappingProxyType(dict(self.features)))


def _array(value):
    return np.asarray(value.array if isinstance(value, Image) else value)


def _aligned_array(reference, value):
    if isinstance(reference, Image) and isinstance(value, Image):
        for name in ('origin', 'spacing', 'direction'):
            a, b = getattr(reference, name), getattr(value, name)
            if a is not None and b is not None and not np.array_equal(a, b):
                raise ValueError(f'Image and mask {name} must match; resample explicitly first.')
    return _array(value)


def _directions(ndim, slice_axis, directions, family, symmetric):
    if directions is None:
        active = [axis for axis in range(ndim) if axis != slice_axis]
        directions = []
        for offset in product((-1, 0, 1), repeat=len(active)):
            nonzero = [v for v in offset if v]
            if not nonzero or nonzero[0] < 0:
                continue
            direction = [0] * ndim
            for axis, value in zip(active, offset):
                direction[axis] = value
            directions.append(tuple(direction))
    result = []
    seen = set()
    for direction in directions:
        direction = tuple(direction)
        if (
            len(direction) != ndim
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in direction)
            or not any(direction)
        ):
            raise ValueError(f'Directions must be nonzero integer offsets of length {ndim}.')
        if slice_axis is not None and direction[slice_axis] != 0:
            raise ValueError('Slice-wise directions must lie in the selected slice plane.')
        if family == 'glrlm' and any(abs(v) > 1 for v in direction):
            raise ValueError('GLRLM supports adjacent directions only (components -1, 0, 1).')
        opposite = tuple(-v for v in direction)
        if direction in seen or ((family == 'glrlm' or symmetric) and opposite in seen):
            raise ValueError('Duplicate or redundant opposite direction.')
        seen.add(direction)
        result.append(direction)
    if not result:
        raise ValueError('At least one direction is required.')
    return tuple(result)


def _glcm_counts(array, direction, levels):
    source, target = [], []
    for size, delta in zip(array.shape, direction):
        if abs(delta) >= size:
            return np.zeros((levels, levels), dtype=np.int64)
        source.append(slice(max(0, -delta), min(size, size - delta)))
        target.append(slice(max(0, delta), min(size, size + delta)))
    a, b = array[tuple(source)], array[tuple(target)]
    valid = np.isfinite(a) & np.isfinite(b)
    pairs = a[valid].astype(int) * levels + b[valid].astype(int)
    return np.bincount(pairs, minlength=levels * levels).reshape(levels, levels)


class TextureMatrixMixin:
    """Public matrix API shared by the six texture calculators."""

    def calculate_matrices(
        self, image, *, mask=None, morphological_mask=None, directions=None, symmetric=None, slice_axis=0
    ):
        """Build unmerged matrices from discretized NumPy data or an Image.

        Input is (row, column) or (z, y, x), with positive integer grey levels
        and NaN outside the ROI. No discretization or minimum-ROI filtering is
        performed. In 2D/2.5D mode a volume is sliced along ``slice_axis``.
        GLCM alone accepts ``symmetric`` (default True); GLCM and GLRLM accept
        explicit offsets in input array coordinates. GLDZM requires a separate
        morphological_mask, containing every valid intensity voxel.
        """
        family = self.matrix_family
        array = np.asarray(_array(image), dtype=float).copy()
        if array.ndim not in (2, 3):
            raise ValueError('Texture input must be a 2D or 3D array.')
        if self.aggr_dim not in ('2D', '2.5D', '3D'):
            raise ValueError('aggr_dim must be 2D, 2.5D or 3D.')
        if array.ndim == 2 and self.aggr_dim == '3D':
            raise ValueError('Use aggr_dim="2D" for a 2D input array.')
        if mask is not None:
            mask_array = _aligned_array(image, mask)
            if mask_array.shape != array.shape or not np.isfinite(mask_array).all():
                raise ValueError('mask must be finite and match the input shape.')
            array[mask_array <= 0] = np.nan
        valid = ~np.isnan(array)
        values = array[valid]
        if not np.all(np.isfinite(values) & (values >= 1) & (values == np.floor(values))):
            raise ValueError('ROI values must be finite positive integer grey levels; use NaN outside the ROI.')
        levels = int(values.max()) + 1 if values.size else 1
        if array.ndim == 3 and self.aggr_dim != '3D':
            if (
                isinstance(slice_axis, bool)
                or not isinstance(slice_axis, (int, np.integer))
                or slice_axis not in (0, 1, 2)
            ):
                raise ValueError('slice_axis must be 0, 1 or 2.')
            effective_axis = int(slice_axis)
        else:
            effective_axis = None
        morphology = None
        if family == 'gldzm':
            if morphological_mask is None:
                raise ValueError('GLDZM requires morphological_mask separately from the intensity ROI.')
            morphology = _aligned_array(image, morphological_mask)
            if morphology.shape != array.shape or not np.isfinite(morphology).all():
                raise ValueError('morphological_mask must be finite and match the input shape.')
            morphology = (morphology > 0).astype(np.int8)
            if np.any(valid & (morphology == 0)):
                raise ValueError('The morphological mask must contain the intensity ROI.')
        elif morphological_mask is not None:
            raise ValueError('morphological_mask is only supported for GLDZM.')
        if family != 'glcm' and symmetric is not None:
            raise ValueError('symmetric is only supported for GLCM.')
        if symmetric is not None and not isinstance(symmetric, (bool, np.bool_)):
            raise ValueError('symmetric must be a boolean.')
        symmetric = (True if symmetric is None else bool(symmetric)) if family == 'glcm' else None
        if family in ('glcm', 'glrlm'):
            offsets = _directions(array.ndim, effective_axis, directions, family, symmetric)
        else:
            if directions is not None:
                raise ValueError('directions are only supported for GLCM and GLRLM.')
            offsets = (None,)
        if effective_axis is None:
            planes = [(None, array, morphology)]
        else:
            planes = [
                (
                    index,
                    np.take(array, index, axis=effective_axis),
                    None if morphology is None else np.take(morphology, index, axis=effective_axis),
                )
                for index in range(array.shape[effective_axis])
            ]
        matrices, skipped = [], []
        for index, plane, morph_plane in planes:
            count = int(np.isfinite(plane).sum())
            if not count and index is not None:
                skipped.append(index)
                continue
            if family == 'glcm':
                plane = crop_to_valid_bbox(plane)
            for direction in offsets:
                local_direction = (
                    tuple(v for axis, v in enumerate(direction) if axis != effective_axis)
                    if direction is not None
                    else None
                )
                matrix = self._build_matrix(plane, morph_plane, local_direction, levels)
                if family == 'glcm' and symmetric:
                    matrix = matrix + matrix.T
                second_name = {
                    'glcm': 'neighbour_grey_level',
                    'glrlm': 'run_length',
                    'glszm': 'zone_size',
                    'gldzm': 'distance',
                    'ngldm': 'dependence_size',
                    'ngtdm': 'statistic',
                }[family]
                second_axis = (
                    ('n_i', 's_i')
                    if family == 'ngtdm'
                    else tuple(range(0 if family == 'glcm' else 1, matrix.shape[1] + (0 if family == 'glcm' else 1)))
                )
                matrices.append(
                    TextureMatrix(
                        family,
                        matrix,
                        (tuple(range(levels)), second_axis),
                        ('grey_level', second_name),
                        direction=direction,
                        slice_index=index,
                        slice_axis=effective_axis,
                        symmetric=symmetric,
                        voxel_count=count,
                        roi_counts=((index, count),),
                        source_ids=(len(matrices),),
                    )
                )
        # A common padded axis makes arbitrary slice/direction merges exact.
        if matrices:
            width = max(m.values.shape[1] for m in matrices)
            matrices = [
                replace(
                    m,
                    values=np.pad(m.values, ((0, 0), (0, width - m.values.shape[1]))),
                    axes=(m.axes[0], tuple(range(1, width + 1))),
                )
                if m.values.shape[1] < width
                else m
                for m in matrices
            ]
        configuration = dict(
            family=family,
            spatial_mode='3D' if self.aggr_dim == '3D' else '2D',
            input_shape=tuple(array.shape),
            array_order='row,column' if array.ndim == 2 else 'z,y,x',
            slice_axis=effective_axis,
            directions=offsets if offsets != (None,) else (),
            symmetric=symmetric,
        )
        if isinstance(image, Image):
            for name in ('origin', 'spacing', 'direction'):
                value = getattr(image, name)
                configuration['image_' + name] = None if value is None else tuple(np.asarray(value).ravel())
        if family in ('glszm', 'gldzm'):
            configuration['connectivity'] = 26 if self.aggr_dim == '3D' else 8
        if family == 'gldzm':
            configuration['distance_metric'] = 'taxicab'
        if family in ('ngtdm', 'ngldm'):
            configuration.update(neighbourhood_radius=1, neighbourhood_metric='chebyshev')
        if family == 'ngldm':
            configuration['dependence_tolerance'] = 0
        return TextureMatrixCollection(tuple(matrices), configuration, tuple(skipped))

    def _build_matrix(self, array, morphology, direction, levels):
        family = self.matrix_family
        if family == 'glcm':
            return _glcm_counts(array, direction, levels)
        if family == 'glrlm':
            return self._rlm_for_direction(array, np.isfinite(array), direction, levels, max(array.shape))
        if self.aggr_dim == '3D':
            if family == 'glszm':
                return self._calc_glsz_3d_matrix(array, levels)[0]
            if family == 'gldzm':
                return self._calc_gldz_3d_matrix(array, morphology, levels)[0]
            return self._calc_3d_matrix(array, levels)
        expanded = array[..., np.newaxis]
        if not np.isfinite(array).any():
            width = 2 if family == 'ngtdm' else 9 if family == 'ngldm' else 0
            return np.zeros((levels, width))
        if family == 'glszm':
            return self._calc_glsz_2d_matrices(expanded, levels)[0][0]
        if family == 'gldzm':
            return self._calc_gldz_2d_matrices(expanded, morphology[..., np.newaxis], levels)[0][0]
        return self._calc_2d_matrices(expanded, levels)[0][0]

    def calculate_features_from_matrices(self, matrices):
        """Evaluate existing matrices without rebuilding them from an image."""
        return dict(self.inspect_matrices(matrices).features)

    def inspect_matrices(self, matrices):
        """Evaluate matrices and retain exact formula inputs and reduction weights."""
        family = self.matrix_family
        if not isinstance(matrices, TextureMatrixCollection) or matrices.configuration['family'] != family:
            raise ValueError('Expected a matrix collection for this calculator family.')
        if matrices.configuration['spatial_mode'] != ('3D' if self.aggr_dim == '3D' else '2D'):
            raise ValueError('Matrix spatial mode does not match this calculator.')
        if not len(matrices):
            raise DataStructureError('No matrices are available for feature evaluation.')
        if self.slice_weight and self.slice_median:
            raise ValueError('Weighted median is not supported.')
        if family == 'glcm' and any(not m.symmetric for m in matrices):
            raise ValueError('Asymmetric GLCM feature formulas are not supported; symmetrize explicitly.')
        if any(m.normalized for m in matrices):
            raise ValueError('Feature evaluation requires count matrices, not normalized inputs.')
        method = getattr(self, 'aggr_method', None)
        inputs = matrices
        slice_average = False
        if family in ('glcm', 'glrlm'):
            if self.aggr_dim == '3D':
                if method not in ('AVER', 'MERG'):
                    raise ValueError('3D directional aggregation requires AVER or MERG.')
                merge_over = ('directions',) if method == 'MERG' else ()
            else:
                choices = {
                    'AVER': (),
                    'SLICE_MERG': ('directions',),
                    'DIR_MERG': ('slices',),
                    'MERG': ('slices', 'directions'),
                }
                if method not in choices:
                    raise ValueError('Unsupported directional aggregation method.')
                merge_over = choices[method]
                slice_average = method in ('AVER', 'SLICE_MERG')
        else:
            merge_over = ('slices',) if self.aggr_dim == '2.5D' else ()
            slice_average = self.aggr_dim == '2D'
        if merge_over:
            inputs = inputs.merge(merge_over)
        if family == 'glcm':
            inputs = replace(
                inputs, matrices=tuple(replace(m, values=m.probabilities(), normalized=True) for m in inputs)
            )
        feature_dicts = []
        for matrix in inputs:
            if matrix.status == 'empty':
                raise DataStructureError(f'Empty {family} matrix from sources {matrix.source_ids} cannot be evaluated.')
            if family == 'glcm':
                values = self._feature_values(matrix.values)
            elif family == 'ngtdm':
                values = self._matrix_feature_values(matrix.values)
            else:
                values = self._map_feature_names(
                    self._matrix_feature_values(
                        matrix.values, matrix.voxel_count, **({'include_energy': True} if family == 'ngldm' else {})
                    )
                )
            feature_dicts.append(values)
        # Preserve existing GLCM median support across directions; other families
        # use median only for slice-wise feature aggregation.
        median = self.slice_median and (slice_average or family == 'glcm') and len(inputs) > 1
        weights = np.array(
            [m.roi_voxel_count if slice_average and self.slice_weight else 1 for m in inputs], dtype=float
        )
        weights /= weights.sum()
        features = {
            name: float(
                np.median([v[name] for v in feature_dicts])
                if median
                else np.average([v[name] for v in feature_dicts], weights=weights)
            )
            for name in feature_dicts[0]
        }
        return TextureTrace(
            matrices,
            inputs,
            tuple(feature_dicts),
            dict(
                aggr_dim=self.aggr_dim,
                aggr_method=method,
                slice_weighting=self.slice_weight,
                slice_median=self.slice_median,
                merge_over=merge_over,
                reduction='median' if median else 'mean',
                weights=None if median else tuple(float(w) for w in weights),
                voxel_denominators=tuple(m.voxel_count for m in inputs),
            ),
            features,
        )

    def _legacy_features(self, array, morphology=None):
        # Existing family calculate_features methods accept (x, y, z). Keep that
        # contract while new public builders and extraction use Image.array order.
        array = np.asarray(array)
        matrices = self.calculate_matrices(
            array.T, **({'morphological_mask': np.asarray(morphology).T} if morphology is not None else {})
        )
        return self.calculate_features_from_matrices(matrices)
