import numpy as np
import pytest

from zrad.image import Image
from zrad.preprocessing import (
    IntensityMaskBuilder,
    IVHIntensityDiscretizer,
    Resegmenter,
    RoiCropper,
    RoiData,
    TextureDiscretizer,
)
from zrad.radiomics import Radiomics
from zrad.radiomics.gldzm import GLDZM
from zrad.radiomics.morphology import MorphologicalFeatures


def _make_image(array):
    return Image(
        array=np.asarray(array, dtype=np.float64),
        origin=(0.0, 0.0, 0.0),
        spacing=np.array([1.0, 1.0, 1.0]),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        shape=(array.shape[2], array.shape[1], array.shape[0]),
    )


def _roi_data(image, mask, filtered_image=None):
    return IntensityMaskBuilder().apply(
        RoiData(
            image=image,
            filtered_image=filtered_image,
            morphological_mask=mask,
        )
    )


@pytest.mark.unit
def test_radiomics_extract_features_returns_dict_from_roi_data():
    image = _make_image(np.arange(27, dtype=np.float64).reshape(3, 3, 3))
    mask = _make_image(np.ones((3, 3, 3), dtype=np.float64))

    features = Radiomics().extract_features(
        roi_data=_roi_data(image, mask),
        families=['local_intensity'],
    )

    assert isinstance(features, dict)
    assert set(features) == {'loc_peak_loc', 'loc_peak_glob'}


@pytest.mark.unit
def test_morphology_mesh_handles_roi_touching_image_border():
    mask_array = np.ones((3, 3, 3), dtype=np.float64)

    mesh_verts, mesh_faces = MorphologicalFeatures(spacing=(1.0, 1.0, 1.0))._calc_mesh(mask_array)

    assert mesh_verts.shape[1] == 3
    assert mesh_faces.shape[1] == 3
    assert mesh_faces.size > 0
    assert np.min(mesh_verts) == pytest.approx(-0.5)
    assert np.max(mesh_verts) == pytest.approx(2.5)


@pytest.mark.unit
def test_radiomics_rejects_direct_image_mask_inputs():
    image = _make_image(np.arange(27, dtype=np.float64).reshape(3, 3, 3))
    mask = _make_image(np.ones((3, 3, 3), dtype=np.float64))

    with pytest.raises(TypeError, match="unexpected keyword"):
        Radiomics().extract_features(image=image, mask=mask)


@pytest.mark.unit
def test_radiomics_requires_roi_data():
    with pytest.raises(TypeError, match="roi_data"):
        Radiomics().extract_features()


@pytest.mark.unit
def test_radiomics_feature_subset_returns_only_requested_keys():
    image = _make_image(np.arange(1, 28, dtype=np.float64).reshape(3, 3, 3))
    mask = _make_image(np.ones((3, 3, 3), dtype=np.float64))

    features = Radiomics().extract_features(
        roi_data=_roi_data(image, mask),
        features=['stat_mean', 'stat_max'],
    )

    assert list(features) == ['stat_mean', 'stat_max']
    assert features['stat_mean'] == pytest.approx(np.mean(image.array))
    assert features['stat_max'] == pytest.approx(np.max(image.array))


@pytest.mark.unit
def test_default_extraction_uses_available_prepared_families():
    image = _make_image(np.arange(1, 217, dtype=np.float64).reshape(6, 6, 6))
    mask_array = np.zeros((6, 6, 6), dtype=np.float64)
    mask_array[1:5, 1:4, 1:4] = 1
    mask_array[4, 3, 3] = 1
    mask = _make_image(mask_array)
    roi_data = TextureDiscretizer(number_of_bins=4).apply(_roi_data(image, mask))

    features = Radiomics().extract_features(roi_data=roi_data)

    assert 'stat_mean' in features
    assert 'ih_mean' in features
    assert 'cm_joint_max_3D_avg' in features
    assert 'ivh_v10' not in features
    assert 'morph_moran_i' in features
    assert 'morph_geary_c' in features


@pytest.mark.unit
def test_explicit_missing_texture_family_raises_clear_error():
    image = _make_image(np.arange(1, 28, dtype=np.float64).reshape(3, 3, 3))
    mask = _make_image(np.ones((3, 3, 3), dtype=np.float64))

    with pytest.raises(Exception, match="not supported"):
        Radiomics().extract_features(
            roi_data=_roi_data(image, mask),
            families=['glcm'],
        )


@pytest.mark.unit
def test_explicit_missing_ivh_family_raises_clear_error():
    image = _make_image(np.arange(1, 28, dtype=np.float64).reshape(3, 3, 3))
    mask = _make_image(np.ones((3, 3, 3), dtype=np.float64))

    with pytest.raises(Exception, match="not supported"):
        Radiomics().extract_features(
            roi_data=_roi_data(image, mask),
            families=['ivh'],
        )


@pytest.mark.unit
def test_radiomics_metadata_is_opt_in():
    image = _make_image(np.arange(1, 28, dtype=np.float64).reshape(3, 3, 3))
    mask = _make_image(np.ones((3, 3, 3), dtype=np.float64))
    roi_data = TextureDiscretizer(number_of_bins=8).apply(_roi_data(image, mask))

    without_metadata = Radiomics().extract_features(
        roi_data=roi_data,
        families=['intensity_histogram'],
    )
    with_metadata = Radiomics().extract_features(
        roi_data=roi_data,
        families=['intensity_histogram'],
        include_metadata=True,
    )

    assert 'bounding_box_min' not in without_metadata
    assert 'no_voxels' not in without_metadata
    assert 'no_bins' not in without_metadata
    assert with_metadata['bounding_box_min'] == 3
    assert with_metadata['no_voxels'] == 27
    assert with_metadata['no_bins'] == 8


@pytest.mark.unit
def test_ivh_features_use_prepared_image_and_metadata():
    image = _make_image(
        np.array(
            [
                [[2.0, 4.0, 8.0], [2.0, 4.0, 8.0], [2.0, 4.0, 8.0]],
                [[2.0, 4.0, 8.0], [2.0, 4.0, 8.0], [2.0, 4.0, 8.0]],
                [[2.0, 4.0, 8.0], [2.0, 4.0, 8.0], [2.0, 4.0, 8.0]],
            ]
        )
    )
    mask = _make_image(np.ones((3, 3, 3), dtype=np.float64))
    roi_data = Resegmenter(intensity_range=(0.0, 10.0)).apply(_roi_data(image, mask))
    roi_data = IVHIntensityDiscretizer(method='direct').apply(roi_data)

    features = Radiomics().extract_features(roi_data=roi_data, families=['ivh'])

    assert roi_data.intensity_range == (0.0, 10.0)
    assert roi_data.ivh_discretization_method == 'direct'
    assert roi_data.ivh_discretization_step == 1
    assert set(features) == {'ivh_v10', 'ivh_v90', 'ivh_i10', 'ivh_i90', 'ivh_diff_v10_v90', 'ivh_diff_i10_i90'}


@pytest.mark.unit
def test_radiomics_filtered_image_uses_original_image_for_masking():
    original = np.zeros((3, 3, 3), dtype=np.float64)
    original[1, 1, 1] = 10.0
    filtered = np.full((3, 3, 3), 50.0, dtype=np.float64)
    filtered[1, 1, 1] = 100.0

    image = _make_image(original)
    filtered_image = _make_image(filtered)
    mask = _make_image(np.ones((3, 3, 3), dtype=np.float64))

    roi_data = _roi_data(image, mask, filtered_image=filtered_image)
    roi_data = Resegmenter(intensity_range=[10.0, 10.0]).apply(roi_data)
    features = Radiomics().extract_features(
        roi_data=roi_data,
        families=['intensity_statistics'],
    )

    assert features['stat_mean'] == pytest.approx(100.0)
    assert features['stat_max'] == pytest.approx(100.0)


@pytest.mark.unit
def test_explicit_roi_cropping_preserves_feature_values():
    image_array = np.zeros((6, 6, 6), dtype=np.float64)
    image_array[2:5, 1:4, 2:5] = np.arange(27, dtype=np.float64).reshape(3, 3, 3) + 1
    mask_array = np.zeros_like(image_array)
    mask_array[2:5, 1:4, 2:5] = 1

    image = _make_image(image_array)
    mask = _make_image(mask_array)

    families = ['intensity_statistics', 'intensity_histogram', 'glcm', 'gldzm']
    roi_data = TextureDiscretizer(number_of_bins=4).apply(_roi_data(image, mask))
    uncropped = Radiomics().extract_features(roi_data=roi_data, families=families)
    cropped = Radiomics().extract_features(
        roi_data=RoiCropper().apply(roi_data),
        families=families,
    )

    assert set(cropped) == set(uncropped)
    for name, value in uncropped.items():
        assert cropped[name] == pytest.approx(value)


@pytest.mark.unit
def test_gldzm_distances_use_morphological_mask_after_resegmentation_edges_are_excluded():
    discretized_image = np.full((3, 3, 3), np.nan, dtype=np.float64)
    discretized_image[1, 1, 1] = 1.0
    morphological_mask = np.ones((3, 3, 3), dtype=np.float64)

    features = GLDZM(aggr_dim='3D').calculate_features(discretized_image, morphological_mask)

    assert features['dzm_sde'] == pytest.approx(0.25)
    assert features['dzm_lde'] == pytest.approx(4.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    'families',
    [
        None,
        ['morphology'],
        ['morphology', 'morphology'],
        'all',
    ],
)
def test_morphology_includes_correlation_once(families, monkeypatch):
    from zrad.radiomics.morphology import MorphologyCorrelationFeatures

    image = _make_image(np.arange(1, 217, dtype=float).reshape(6, 6, 6))
    mask_array = np.zeros_like(image.array)
    mask_array[1:5, 1:4, 1:4] = 1
    mask_array[4, 3, 3] = 0
    mask = _make_image(mask_array)
    roi = TextureDiscretizer(number_of_bins=4).apply(_roi_data(image, mask))
    roi = IVHIntensityDiscretizer(method='direct').apply(roi)
    expected = Radiomics().extract_features(roi_data=roi, families=['morphology'])
    original = MorphologyCorrelationFeatures.calculate_features
    calls = []

    def calculate(self, *args):
        calls.append(1)
        return original(self, *args)

    monkeypatch.setattr(MorphologyCorrelationFeatures, 'calculate_features', calculate)
    result = Radiomics().extract_features(roi_data=roi, families=families)
    assert 'morph_volume' in result
    assert {tag: result[tag] for tag in expected} == pytest.approx(expected)
    assert len(calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    'selection',
    [
        {'features': ['morph_moran_i']},
        {'features': ['morph_geary_c']},
        {'features': ['morph_moran_i', 'morph_geary_c', 'morph_moran_i']},
        {'features': ['morph_volume', 'morph_geary_c', 'morph_moran_i']},
    ],
)
def test_morphology_individual_selection(selection, monkeypatch):
    image = _make_image(np.arange(1, 217, dtype=float).reshape(6, 6, 6))
    mask_array = np.zeros_like(image.array)
    mask_array[1:5, 1:4, 1:4] = 1
    mask_array[4, 3, 3] = 0
    roi = _roi_data(image, _make_image(mask_array))
    expected = Radiomics().extract_features(roi_data=roi, families=['morphology'])
    assert set(expected) == set(MorphologicalFeatures((1, 1, 1)).get_feature_names())
    from zrad.radiomics.morphology import MorphologyCorrelationFeatures

    original = MorphologyCorrelationFeatures.calculate_features
    calls = []

    def calculate(self, *args):
        calls.append(1)
        return original(self, *args)

    monkeypatch.setattr(MorphologyCorrelationFeatures, 'calculate_features', calculate)
    result = Radiomics().extract_features(roi_data=roi, **selection)
    tags = set(selection.get('features', expected))
    assert result == pytest.approx({tag: expected[tag] for tag in tags})
    assert len(calls) == 1


@pytest.mark.unit
def test_slice_image_still_omits_3d_morphology():
    image = _make_image(np.arange(36, dtype=float).reshape(1, 6, 6))
    roi = _roi_data(image, _make_image(np.ones_like(image.array)))
    result = Radiomics().extract_features(roi_data=roi)
    assert not any(tag.startswith('morph_') for tag in result)


@pytest.mark.unit
def test_removed_correlation_family_is_rejected():
    image = _make_image(np.arange(27, dtype=float).reshape(3, 3, 3))
    roi = _roi_data(image, _make_image(np.ones_like(image.array)))
    with pytest.raises(ValueError, match="Feature family 'morphology_correlation' is not supported"):
        Radiomics().extract_features(roi_data=roi, families=['morphology_correlation'])


@pytest.mark.unit
@pytest.mark.parametrize('selection', [{}, {'families': ['morphology']}, {'features': ['morph_volume']}])
@pytest.mark.parametrize('value', [0.1, 1.1, 50.0])
def test_constant_intensity_preserves_morphology_results(selection, value):
    image = _make_image(np.full((4, 5, 6), value))
    roi = _roi_data(image, _make_image(np.ones_like(image.array)))
    result = Radiomics().extract_features(roi_data=roi, **selection)
    assert result['morph_volume'] == pytest.approx(113.16666666666667)
    if 'features' not in selection:
        assert np.isnan(result['morph_moran_i'])
        assert np.isnan(result['morph_geary_c'])
        assert all(np.isfinite(value) for tag, value in result.items() if tag not in {'morph_moran_i', 'morph_geary_c'})
    else:
        assert set(result) == {'morph_volume'}


@pytest.mark.unit
@pytest.mark.parametrize('zero_sum', [False, True])
@pytest.mark.parametrize(
    'features',
    [
        ['morph_moran_i'],
        ['morph_geary_c'],
        ['morph_geary_c', 'stat_mean', 'morph_moran_i', 'morph_geary_c'],
    ],
)
def test_correlation_selection_avoids_unrelated_shape_failures(zero_sum, features, monkeypatch):
    from zrad.radiomics.morphology import MorphologyCorrelationFeatures

    values = np.arange(1, 126, dtype=float).reshape(5, 5, 5)
    if zero_sum:
        values -= values.mean()
    image = _make_image(values)
    mask = np.ones_like(values)
    roi = _roi_data(image, _make_image(mask))
    # Direct pairwise definition for this cube, independently validated on master.
    expected = {'morph_moran_i': 0.19822002516539844, 'morph_geary_c': 0.7404778956266241, 'stat_mean': values.mean()}
    original = MorphologyCorrelationFeatures.calculate_features
    calls = []

    def calculate(self, *args):
        calls.append(1)
        return original(self, *args)

    def unexpected(*args, **kwargs):
        pytest.fail('Correlation-only morphology selection must not evaluate shape features')

    monkeypatch.setattr(MorphologicalFeatures, 'calculate_features', unexpected)
    monkeypatch.setattr(MorphologyCorrelationFeatures, 'calculate_features', calculate)
    result = Radiomics().extract_features(roi_data=roi, features=features)
    assert result == pytest.approx({name: expected[name] for name in features}, abs=2e-12)
    assert list(result) == list(dict.fromkeys(features))
    assert len(calls) == 1


@pytest.mark.unit
def test_shape_selection_skips_spatial_calculation(monkeypatch):
    from zrad.radiomics.morphology import MorphologyCorrelationFeatures

    image = _make_image(np.full((4, 5, 6), 50.0))
    roi = _roi_data(image, _make_image(np.ones_like(image.array)))

    def unexpected(*args):
        pytest.fail('Shape-only selection must not evaluate Moran or Geary')

    monkeypatch.setattr(MorphologyCorrelationFeatures, 'calculate_features', unexpected)
    result = Radiomics().extract_features(roi_data=roi, features=['morph_volume'])
    assert result == pytest.approx({'morph_volume': 113.16666666666667})


@pytest.mark.unit
def test_global_peak_preserves_constant_intensity_at_image_boundary():
    from zrad.radiomics.intensity import LocalIntensityFeatures

    image = np.full((3, 3, 3), 7.0)
    masked = np.full_like(image, np.nan)
    masked[0, 0, 0] = 7.0
    assert LocalIntensityFeatures((2, 2, 2))._calc_global_intensity_peak(image, masked) == pytest.approx(7.0)


@pytest.mark.unit
def test_histogram_gradient_keeps_empty_bins():
    from zrad.radiomics.intensity import IntensityHistogramFeatures

    values, gradient = IntensityHistogramFeatures._histogram_gradient(np.array([1.0, 1.0, 3.0, 3.0, 3.0]))
    np.testing.assert_array_equal(values, [1, 2, 3])
    np.testing.assert_array_equal(gradient, [-2, 0.5, 3])


@pytest.mark.unit
@pytest.mark.parametrize('array', [np.array([]), np.array([np.nan])])
def test_histogram_gradient_rejects_empty_roi(array):
    from zrad.exceptions import DataStructureError
    from zrad.radiomics.intensity import IntensityHistogramFeatures

    with pytest.raises(DataStructureError, match='Not enough bins'):
        IntensityHistogramFeatures._histogram_gradient(array)
