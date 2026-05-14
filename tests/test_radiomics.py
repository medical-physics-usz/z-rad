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
    assert 'morph_moran_i' not in features


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

    families = ['intensity_statistics', 'intensity_histogram', 'glcm']
    roi_data = TextureDiscretizer(number_of_bins=4).apply(_roi_data(image, mask))
    uncropped = Radiomics().extract_features(roi_data=roi_data, families=families)
    cropped = Radiomics().extract_features(
        roi_data=RoiCropper().apply(roi_data),
        families=families,
    )

    assert set(cropped) == set(uncropped)
    for name, value in uncropped.items():
        assert cropped[name] == pytest.approx(value)
