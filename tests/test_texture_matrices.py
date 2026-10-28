"""Matrix-level contracts, hand counts, and reproducible extraction traces."""

import numpy as np
import pytest

from zrad.exceptions import DataStructureError
from zrad.image import Image
from zrad.preprocessing import IntensityMaskBuilder, RoiData, TextureDiscretizer
from zrad.radiomics import Radiomics
from zrad.radiomics.glcm import GLCM
from zrad.radiomics.gldzm import GLDZM
from zrad.radiomics.glrlm import GLRLM
from zrad.radiomics.glszm import GLSZM
from zrad.radiomics.ngldm import NGLDM
from zrad.radiomics.ngtdm import NGTDM

pytestmark = pytest.mark.unit

CALCULATORS = (GLCM, GLRLM, GLSZM, GLDZM, NGTDM, NGLDM)


def calculator(cls, dimension='2D', method='AVER', **kwargs):
    if cls in (GLCM, GLRLM):
        kwargs['aggr_method'] = method
    return cls(aggr_dim=dimension, **kwargs)


def image(array):
    array = np.asarray(array, dtype=float)
    return Image(array=array, shape=array.shape[::-1])


def roi(array, mask=None):
    data = IntensityMaskBuilder().apply(
        RoiData(
            image=image(array),
            morphological_mask=image(np.ones_like(array) if mask is None else mask),
        )
    )
    return TextureDiscretizer(number_of_bins=3).apply(data)


def test_hand_counted_directed_glcm_and_symmetrization():
    array = np.array([[1, 1, 2], [2, 3, 3], [1, 2, 3]])
    calc = calculator(GLCM)
    raw = calc.calculate_matrices(array, directions=[(0, 1)], symmetric=False)
    expected = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 0, 0, 2], [0, 0, 0, 1]])
    np.testing.assert_array_equal(raw.only().values, expected)
    reverse = calc.calculate_matrices(array, directions=[(0, -1)], symmetric=False)
    np.testing.assert_array_equal(reverse.only().values, expected.T)
    symmetric = raw.symmetrize()
    np.testing.assert_array_equal(symmetric.only().values, expected + expected.T)
    assert symmetric.only().values.sum() == 12
    np.testing.assert_allclose(symmetric.only().probabilities().sum(), 1)
    with pytest.raises(ValueError, match='Asymmetric'):
        calc.calculate_features_from_matrices(raw)
    assert calc.calculate_features_from_matrices(symmetric)['cm_contrast'] == pytest.approx(4 / 6)
    with pytest.raises(ValueError):
        raw.only().values[1, 1] = 99
    with pytest.raises(ValueError):
        raw.only().values.setflags(write=True)
    assert raw.only().axes == ((0, 1, 2, 3), (0, 1, 2, 3))


@pytest.mark.parametrize('cls', CALCULATORS)
def test_numpy_and_image_equivalence(cls):
    array = np.array([[1, 1, 2], [2, 3, 3], [1, 2, 3]])
    calc = calculator(cls)
    kwargs2d = {'morphological_mask': np.ones_like(array)} if cls is GLDZM else {}
    kwargs3d = {'morphological_mask': image(np.ones_like(array)[None])} if cls is GLDZM else {}
    a = calc.calculate_matrices(array, **kwargs2d)
    b = calc.calculate_matrices(image(array[None]), **kwargs3d)
    for first, second in zip(a, b):
        np.testing.assert_array_equal(first.values, second.values)
        if first.direction is not None:
            assert (0, *first.direction) == second.direction
    assert calc.calculate_features_from_matrices(a) == pytest.approx(calc.calculate_features_from_matrices(b))


def test_hand_counted_run_zone_dependence_and_difference_matrices():
    array = np.ones((2, 3), dtype=int)
    runs = calculator(GLRLM).calculate_matrices(array, directions=[(0, 1)]).only()
    assert runs.values[1, 2] == 2
    assert runs.values.sum() == 2
    assert runs.axes[1] == (1, 2, 3)
    zones = calculator(GLSZM).calculate_matrices(array).only()
    assert zones.values[1, 5] == 1
    assert zones.values.sum() == 1
    distance = calculator(GLDZM).calculate_matrices(array, morphological_mask=np.ones_like(array)).only()
    assert distance.values[1, 0] == 1
    dependence = calculator(NGLDM).calculate_matrices(array).only()
    assert dependence.values[1, 3] == 4  # corners: centre + 3 neighbours
    assert dependence.values[1, 5] == 2
    ngtdm = calculator(NGTDM).calculate_matrices(array).only()
    assert ngtdm.n_i[1] == 6
    assert ngtdm.s_i[1] == 0
    assert ngtdm.p_i[1] == 1
    with pytest.raises(ValueError, match='p_i'):
        ngtdm.probabilities()


def test_gldzm_uses_separate_morphology():
    array = np.full((5, 5), np.nan)
    array[2, 2] = 1
    calc = calculator(GLDZM)
    matrix = calc.calculate_matrices(array, morphological_mask=np.ones((5, 5))).only()
    assert matrix.values[1, 2] == 1
    with pytest.raises(ValueError, match='contain'):
        calc.calculate_matrices(array, morphological_mask=np.zeros((5, 5)))
    with pytest.raises(ValueError, match='requires'):
        calc.calculate_matrices(array)


def test_slice_identity_selection_merge_and_glrlm_denominator():
    array = np.ones((3, 2, 3))
    array[1] = np.nan
    calc = calculator(GLRLM, method='MERG', dimension='2.5D')
    matrices = calc.calculate_matrices(array, directions=[(0, 0, 1), (0, 1, 0)])
    assert matrices.skipped_slices == (1,)
    assert len(matrices.select(slice_index=2)) == 2
    assert len(matrices.select(direction=(0, 0, 1))) == 2
    merged = matrices.merge(('slices', 'directions')).only()
    assert merged.voxel_count == 24
    assert merged.roi_voxel_count == 12
    assert merged.source_ids == (0, 1, 2, 3)
    assert calc.calculate_features_from_matrices(matrices)['rlm_r_perc'] == pytest.approx(10 / 24)
    sequential = matrices.merge('directions').merge('slices').only()
    np.testing.assert_array_equal(sequential.values, merged.values)
    assert sequential.voxel_count == merged.voxel_count


def test_slice_axis_and_arbitrary_glcm_offsets():
    array = np.arange(1, 25).reshape(2, 3, 4)
    calc = calculator(GLCM)
    a = calc.calculate_matrices(array, directions=[(0, 0, 2)], slice_axis=1, symmetric=False)
    assert [m.slice_index for m in a] == [0, 1, 2]
    assert all(m.values.sum() == 4 for m in a)
    matrix = a[0].values
    assert matrix[1, 3] == 1
    assert matrix[13, 15] == 1
    assert a[0].slice_axis == 1


@pytest.mark.parametrize('cls', CALCULATORS)
@pytest.mark.parametrize(
    'dimension,method',
    [('2D', 'AVER'), ('2D', 'SLICE_MERG'), ('2.5D', 'DIR_MERG'), ('2.5D', 'MERG'), ('3D', 'AVER'), ('3D', 'MERG')],
)
def test_trace_reproduces_extraction_and_exact_formula_inputs(cls, dimension, method):
    data = roi(np.random.default_rng(14).integers(1, 4, size=(3, 3, 4)))
    family = cls.matrix_family
    extractor = Radiomics(aggr_dim=dimension, aggr_method=method)
    result = extractor.extract_with_details(data, families=[family])
    ordinary = extractor.extract_features(data, families=[family])
    assert dict(result.features) == pytest.approx(ordinary)
    trace = result.texture[family]
    calc = calculator(cls, dimension, method)
    raw = trace.raw.symmetrize() if cls is GLCM else trace.raw
    assert calc.calculate_features_from_matrices(raw) == pytest.approx(dict(trace.features))
    for matrix, values in zip(trace.feature_inputs, trace.per_matrix_features):
        if cls is GLCM:
            reproduced = calc._feature_values(matrix.values)
            assert matrix.normalized
        elif cls is NGTDM:
            reproduced = calc._matrix_feature_values(matrix.values)
        else:
            kwargs = {'include_energy': True} if cls is NGLDM else {}
            reproduced = calc._map_feature_names(
                calc._matrix_feature_values(matrix.values, matrix.voxel_count, **kwargs)
            )
        assert reproduced == pytest.approx(dict(values))
    for name, value in trace.features.items():
        assert value == pytest.approx(
            np.average([v[name] for v in trace.per_matrix_features], weights=trace.aggregation['weights'])
        )


@pytest.mark.parametrize('cls', CALCULATORS)
def test_weighted_and_median_reductions(cls):
    array = np.random.default_rng(7).integers(1, 4, size=(3, 4, 4)).astype(float)
    array[0, 0] = np.nan
    kwargs = {'morphological_mask': np.ones_like(array)} if cls is GLDZM else {}
    for median in (False, True):
        calc = calculator(cls, slice_median=median, slice_weight=not median)
        trace = calc.inspect_matrices(calc.calculate_matrices(array, **kwargs))
        for name, value in trace.features.items():
            values = [v[name] for v in trace.per_matrix_features]
            expected = (
                np.median(values)
                if median
                else np.average(values, weights=[m.roi_voxel_count for m in trace.feature_inputs])
            )
            assert value == pytest.approx(expected)


def test_extraction_records_excluded_slices_and_selected_features():
    mask = np.ones((3, 3, 3))
    mask[1] = 0
    mask[1, 1, 1] = 1
    data = roi(np.tile(np.arange(9).reshape(1, 3, 3), (3, 1, 1)), mask)
    extractor = Radiomics(aggr_dim='2D', aggr_method='AVER')
    result = extractor.extract_with_details(
        data, features=['cm_contrast'], texture_options={'glcm': {'directions': [(0, 0, 1)]}}
    )
    assert result.excluded_slices == (1,)
    assert list(result.features) == ['cm_contrast_2D_avg']
    assert [m.slice_index for m in result.texture['glcm'].raw] == [0, 2]
    assert np.isnan(result.discretized_image[1]).all()
    assert np.all(result.morphological_mask[1] == 0)
    assert np.any(data.morphological_mask.array[1])


@pytest.mark.parametrize('directions', [[(0, 0)], [(0, 1.5)], [(0, 1), (0, -1)], []])
def test_invalid_directions(directions):
    with pytest.raises(ValueError):
        calculator(GLCM).calculate_matrices(np.ones((2, 2)), directions=directions)


@pytest.mark.parametrize('array', [np.array([[0]]), np.array([[-1]]), np.array([[1.5]]), np.array([[np.inf]])])
def test_invalid_grey_levels(array):
    with pytest.raises(ValueError, match='positive integer'):
        calculator(GLCM).calculate_matrices(array)


def test_empty_matrices_and_isolated_ngtdm_voxel_are_inspectable():
    calc = calculator(GLCM)
    matrix = calc.calculate_matrices(np.ones((1, 1)), directions=[(0, 1)]).only()
    assert matrix.status == 'empty'
    with pytest.raises(DataStructureError):
        matrix.probabilities()
    ngtdm = calculator(NGTDM).calculate_matrices(np.ones((1, 1))).only()
    assert ngtdm.roi_voxel_count == 1
    assert ngtdm.contributing_voxel_count == 0
    with pytest.raises(DataStructureError):
        ngtdm.p_i
    assert len(calc.calculate_matrices(np.full((2, 2, 2), np.nan))) == 0


def test_unsupported_family_controls_and_redundant_directions():
    array = np.ones((2, 2))
    with pytest.raises(ValueError, match='symmetric'):
        calculator(GLRLM).calculate_matrices(array, symmetric=False)
    with pytest.raises(ValueError, match='directions'):
        calculator(GLSZM).calculate_matrices(array, directions=[(0, 1)])
    with pytest.raises(ValueError, match='adjacent'):
        calculator(GLRLM).calculate_matrices(array, directions=[(0, 2)])
    with pytest.raises(ValueError, match='redundant'):
        calculator(GLRLM).calculate_matrices(array, directions=[(0, 1), (0, -1)])
    with pytest.raises(ValueError, match='no directional'):
        calculator(GLSZM).calculate_matrices(array).merge('directions')


def test_feature_average_is_not_merged_matrix_feature():
    array = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    matrices = calculator(GLCM).calculate_matrices(array, directions=[(0, 1), (1, 0)])
    avg = calculator(GLCM).calculate_features_from_matrices(matrices)
    merged = calculator(GLCM, method='SLICE_MERG').calculate_features_from_matrices(matrices)
    assert avg['cm_energy'] != pytest.approx(merged['cm_energy'])


def test_all_matrices_from_roi_without_feature_evaluation():
    # Constant grey levels make some GLCM formulas undefined, but construction
    # must still succeed for every family.
    data = roi(np.ones((3, 3, 3)))
    extractor = Radiomics()
    matrices = extractor.calculate_texture_matrices(data, texture_options={'glcm': {'symmetric': False}})
    assert set(matrices) == {cls.matrix_family for cls in CALCULATORS}
    assert len(matrices['glcm']) == 13
    assert not matrices['glcm'][0].symmetric
    for cls in CALCULATORS:
        assert len(matrices[cls.matrix_family]) == (13 if cls in (GLCM, GLRLM) else 1)


def test_geometry_mismatch_and_invalid_retained_symmetry():
    reference = image(np.ones((3, 3, 3)))
    reference.spacing = (1, 1, 1)
    mask = image(np.ones((3, 3, 3)))
    mask.spacing = (2, 2, 2)
    with pytest.raises(ValueError, match='spacing'):
        calculator(GLCM, '3D').calculate_matrices(reference, mask=mask)
    data = roi(np.arange(27).reshape(3, 3, 3))
    for symmetry in (False, np.bool_(False), 'yes'):
        with pytest.raises(ValueError):
            Radiomics().extract_with_details(data, families='glcm', texture_options={'glcm': {'symmetric': symmetry}})


def test_3d_offsets_follow_numpy_axis_order():
    array = np.arange(1, 25).reshape(2, 3, 4)
    calc = calculator(GLCM, dimension='3D')
    matrix = calc.calculate_matrices(array, directions=[(1, 0, 0)], symmetric=False).only()
    assert matrix.values[1, 13] == 1
    assert matrix.values[1, 2] == 0
    assert matrix.values.sum() == 12
    runs = calculator(GLRLM, dimension='3D').calculate_matrices(np.ones((2, 3, 4)), directions=[(1, 0, 0)]).only()
    assert runs.values[1, 1] == 12
    assert runs.values.sum() == 12


@pytest.mark.parametrize('cls', CALCULATORS)
def test_reuse_slice_matrices_for_2_5d_aggregation(cls):
    array = np.random.default_rng(84).integers(1, 4, size=(3, 4, 5))
    kwargs = {'morphological_mask': np.ones_like(array)} if cls is GLDZM else {}
    matrices = calculator(cls).calculate_matrices(array, **kwargs)
    merger = calculator(cls, dimension='2.5D', method='MERG')
    assert merger.calculate_features_from_matrices(matrices) == pytest.approx(
        merger.calculate_features_from_matrices(merger.calculate_matrices(array, **kwargs))
    )
    with pytest.raises(ValueError, match='spatial mode'):
        calculator(cls, dimension='3D').calculate_features_from_matrices(matrices)
