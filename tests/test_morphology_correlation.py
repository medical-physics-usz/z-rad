"""Independent direct oracles and dispatch tests for full inverse-distance features."""

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from zrad.radiomics import morphology
from zrad.radiomics.morphology import MorphologyCorrelationFeatures

TAGS = ('morph_moran_i', 'morph_geary_c')


def direct(mask, image, spacing):
    # The previous dense definition, using float64 to avoid integer overflow.
    indices = np.argwhere(mask)
    values = image[tuple(indices.T)].astype(np.float64)
    valid = ~np.isnan(values)
    values, indices = values[valid], indices[valid]
    weights = squareform(pdist(indices * spacing))
    np.fill_diagonal(weights, np.inf)
    weights = 1 / weights
    z = values - values.mean()
    q = np.sum(z**2)
    n = len(values)
    return {
        TAGS[0]: n / weights.sum() * np.sum(weights * np.outer(z, z)) / q,
        TAGS[1]: (n - 1) / (2 * weights.sum()) * np.sum(weights * (values[:, None] - values) ** 2) / q,
    }


@pytest.fixture(params=['blocked', 'fft'])
def method(request, monkeypatch):
    monkeypatch.setattr(morphology, '_correlation_method', lambda *args: request.param)
    return request.param


@pytest.mark.unit
@pytest.mark.parametrize('shape', [(1, 1, 2), (1, 5, 9), (7, 8, 9), (9, 10, 11)])
@pytest.mark.parametrize('spacing', [(1, 1, 1), (3, 0.977, 0.81), (0.001, 100, 0.3)])
@pytest.mark.parametrize('occupancy', [0.15, 0.7, 1])
def test_matches_dense_definition(method, shape, spacing, occupancy):
    rng = np.random.default_rng(913)
    mask = rng.random(shape) < occupancy
    mask.flat[0] = mask.flat[-1] = True
    image = rng.normal(-200, 100, shape)
    image[(rng.random(shape) < 0.1) & ~mask] = np.nan
    if mask.sum() > 2:
        image.flat[np.flatnonzero(mask)[1]] = np.nan
    before = image.copy()
    expected = direct(mask, image, spacing)
    actual = MorphologyCorrelationFeatures(spacing).calculate_features(mask, image)
    assert actual == pytest.approx(expected, abs=2e-12, rel=2e-12)
    np.testing.assert_array_equal(image, before)


@pytest.mark.unit
@pytest.mark.parametrize('dtype', [np.int16, np.int32, np.float32, np.float64])
def test_dtype_and_geometry_invariance(method, dtype):
    rng = np.random.default_rng(41)
    image = rng.integers(-1000, 401, (5, 6, 7)).astype(dtype)
    mask = rng.random(image.shape) > 0.3
    spacing = np.array([2.5, 0.8, 1.1])
    expected = direct(mask, image, spacing)
    calc = MorphologyCorrelationFeatures(spacing)
    assert calc.calculate_features(mask, image) == pytest.approx(expected, abs=2e-12)
    assert calc.calculate_features(np.pad(mask, 5), np.pad(image, 5)) == pytest.approx(expected, abs=2e-12)
    assert calc.calculate_features(mask, image.astype(float) * -3 + 1e9) == pytest.approx(expected, abs=2e-12)
    permuted = MorphologyCorrelationFeatures(spacing[[2, 0, 1]])
    assert permuted.calculate_features(mask.transpose(2, 0, 1), image.transpose(2, 0, 1)) == pytest.approx(
        expected, abs=2e-12
    )


@pytest.mark.unit
@pytest.mark.parametrize('count', [0, 1])
def test_empty_or_singleton_is_nan(count):
    mask = np.zeros((3, 3, 3))
    mask.flat[:count] = 1
    result = MorphologyCorrelationFeatures((1, 1, 1)).calculate_features(mask, mask)
    assert all(np.isnan(v) for v in result.values())


@pytest.mark.unit
@pytest.mark.parametrize('value', [np.inf, -np.inf])
def test_infinity_propagates_without_changing_population(value):
    image = np.arange(27, dtype=float).reshape(3, 3, 3)
    image.flat[0] = value
    result = MorphologyCorrelationFeatures((1, 1, 1)).calculate_features(np.ones_like(image), image)
    assert all(np.isnan(v) for v in result.values())


@pytest.mark.unit
@pytest.mark.parametrize('shape', [(4, 5, 6), (10, 10, 10)])
@pytest.mark.parametrize('value', [0.1, 1.1, 50.0])
def test_constant_detected_before_spatial_work(monkeypatch, shape, value):
    def unexpected(*args):
        pytest.fail('Constant intensities should not reach the selector')

    monkeypatch.setattr(morphology, '_correlation_method', unexpected)
    result = MorphologyCorrelationFeatures((1, 1, 1)).calculate_features(np.ones(shape), np.full(shape, value))
    assert set(result) == set(TAGS)
    assert all(np.isnan(value) for value in result.values())


@pytest.mark.unit
@pytest.mark.parametrize('spacing', [(0, 1, 1), (-1, 1, 1), (np.nan, 1, 1), (np.inf, 1, 1), (1, 1)])
def test_invalid_spacing(spacing):
    with pytest.raises(ValueError, match='positive finite'):
        MorphologyCorrelationFeatures(spacing).calculate_features(np.ones((3, 3, 3)), np.arange(27).reshape(3, 3, 3))


@pytest.mark.unit
def test_selector_uses_population_and_geometry():
    select = morphology._correlation_method
    assert select(74, (7, 7, 9)) == 'blocked'
    assert select(256, (1, 1, 511)) == 'blocked'
    assert select(257, (1, 1, 513)) == 'fft'
    assert select(1000, (32, 25, 21)) == 'fft'
    assert select(1000, (256, 256, 256)) == 'blocked'
    assert select(2_000_000, (512, 512, 512)) == 'fft'


@pytest.mark.unit
def test_selector_receives_valid_intensity_population_and_cropped_geometry(monkeypatch):
    image = np.full((20, 21, 22), np.nan)
    image[5, 6, 7], image[7, 9, 11] = 1, 3
    original = morphology._correlation_method
    calls = []

    def select(n, fft_shape):
        calls.append((n, fft_shape))
        return original(n, fft_shape)

    monkeypatch.setattr(morphology, '_correlation_method', select)
    result = MorphologyCorrelationFeatures((2, 1, 1)).calculate_features(np.ones_like(image), image)
    assert calls == [(2, (5, 7, 9))]
    assert result == pytest.approx({'morph_moran_i': -1, 'morph_geary_c': 1})


@pytest.mark.unit
def test_selector_cost_boundary(monkeypatch):
    # For P=1024 and N=1000, equality must choose blocked.
    pairs = 1000 * 999 // 2
    monkeypatch.setattr(morphology, '_CORRELATION_FFT_COST_FACTOR', pairs / (1024 * 10))
    assert morphology._correlation_method(1000, (1, 1, 1024)) == 'blocked'
    monkeypatch.setattr(morphology, '_CORRELATION_FFT_COST_FACTOR', pairs / (1024 * 10) - 1e-6)
    assert morphology._correlation_method(1000, (1, 1, 1024)) == 'fft'


@pytest.mark.unit
def test_digital_phantom_reference(method):
    # Official IBSI digital phantom, z/y/x order, 2 mm isotropic voxels.
    image = np.array(
        [
            [[1, 4, 4, 1, 1], [1, 4, 6, 1, 1], [4, 1, 6, 4, 1], [4, 4, 6, 4, 1]],
            [[1, 4, 4, 1, 1], [1, 1, 6, 1, 1], [1, 1, 3, 1, 1], [4, 4, 6, 1, 1]],
            [[1, 4, 4, 1, 1], [1, 1, 1, 1, 1], [1, 1, 9, 1, 1], [1, 1, 6, 1, 1]],
            [[1, 4, 4, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 6, 1, 1]],
        ]
    )
    mask = np.ones_like(image)
    mask[1, 2, 0] = mask[2, 2, 2] = 0
    mask[2:, 0, 3:] = 0
    actual = MorphologyCorrelationFeatures((2, 2, 2)).calculate_features(mask, image)
    assert set(actual) == set(TAGS)
    with (Path(__file__).parent / 'data/ibsi_1_reference_values_digital_phantom.csv').open() as f:
        refs = {r['tag']: r for r in csv.DictReader(f)}
    for tag in TAGS:
        assert actual[tag] == pytest.approx(float(refs[tag]['reference value']), abs=float(refs[tag]['tolerance']))
    assert actual == pytest.approx(direct(mask, image, (2, 2, 2)), abs=2e-12)


@pytest.mark.unit
def test_api_uses_resegmented_intensities_not_texture_bins():
    from zrad.image import Image
    from zrad.preprocessing import IntensityMaskBuilder, Resegmenter, RoiData, TextureDiscretizer
    from zrad.radiomics import Radiomics

    array = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6) - 60

    def image(values):
        return Image(
            array=values,
            origin=(0, 0, 0),
            spacing=np.array([0.8, 1.1, 2.5]),
            direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
            shape=(6, 5, 4),
        )

    mask = np.ones_like(array)
    mask[1, 2, 3] = 0
    roi = IntensityMaskBuilder().apply(RoiData(image=image(array), morphological_mask=image(mask)))
    roi = Resegmenter(intensity_range=[-30, 50]).apply(roi)
    roi = TextureDiscretizer(number_of_bins=4).apply(roi)
    expected = direct(mask, np.where((array >= -30) & (array <= 50), array, np.nan), (2.5, 1.1, 0.8))
    extractor = Radiomics()
    actual = extractor.extract_features(roi_data=roi, features=list(TAGS))
    assert actual == pytest.approx(expected, abs=2e-12)
    for tag in TAGS:
        assert extractor.extract_features(roi_data=roi, features=[tag]) == pytest.approx(
            {tag: expected[tag]}, abs=2e-12
        )


@pytest.mark.unit
@pytest.mark.parametrize('shape', [(3, 3), (3, 3, 4)])
def test_misaligned_input_rejected(shape):
    with pytest.raises(ValueError, match='aligned 3D'):
        MorphologyCorrelationFeatures((1, 1, 1)).calculate_features(np.ones((3, 3, 3)), np.ones(shape))


@pytest.mark.unit
def test_low_contrast_is_not_treated_as_constant(method):
    image = 0.1 + np.arange(120).reshape(4, 5, 6) * 1e-12
    mask = np.ones_like(image)
    result = MorphologyCorrelationFeatures((1, 1, 1)).calculate_features(mask, image)
    assert all(np.isfinite(value) for value in result.values())
    assert result == pytest.approx(direct(mask, image, (1, 1, 1)), abs=2e-12, rel=2e-12)
