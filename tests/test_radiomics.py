import numpy as np
import pytest

from zrad.image import Image
from zrad.preprocessing import Resegmenter, RoiMaskBuilder
from zrad.radiomics import Radiomics


def _make_image(array):
    return Image(
        array=np.asarray(array, dtype=np.float64),
        origin=(0.0, 0.0, 0.0),
        spacing=np.array([1.0, 1.0, 1.0]),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        shape=(array.shape[2], array.shape[1], array.shape[0]),
    )


@pytest.mark.unit
def test_radiomics_extract_features_returns_dict():
    image = _make_image(np.arange(27, dtype=np.float64).reshape(3, 3, 3))
    mask = _make_image(np.ones((3, 3, 3), dtype=np.float64))

    radiomics = Radiomics()
    features = radiomics.extract_features(image=image, mask=mask, families=['local_intensity'])

    assert isinstance(features, dict)
    assert set(features) == {'loc_peak_loc', 'loc_peak_glob'}


@pytest.mark.unit
def test_radiomics_feature_subset_returns_only_requested_keys():
    image = _make_image(np.arange(1, 28, dtype=np.float64).reshape(3, 3, 3))
    mask = _make_image(np.ones((3, 3, 3), dtype=np.float64))

    radiomics = Radiomics()
    features = radiomics.extract_features(
        image=image,
        mask=mask,
        features=['stat_mean', 'stat_max'],
    )

    assert list(features) == ['stat_mean', 'stat_max']
    assert features['stat_mean'] == pytest.approx(np.mean(image.array))
    assert features['stat_max'] == pytest.approx(np.max(image.array))


@pytest.mark.unit
def test_radiomics_metadata_is_opt_in():
    image = _make_image(np.arange(1, 28, dtype=np.float64).reshape(3, 3, 3))
    mask = _make_image(np.ones((3, 3, 3), dtype=np.float64))

    radiomics = Radiomics(number_of_bins=8)

    without_metadata = radiomics.extract_features(
        image=image,
        mask=mask,
        families=['intensity_histogram'],
    )
    with_metadata = radiomics.extract_features(
        image=image,
        mask=mask,
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
def test_radiomics_filtered_image_uses_original_image_for_masking():
    original = np.zeros((3, 3, 3), dtype=np.float64)
    original[1, 1, 1] = 10.0

    filtered = np.full((3, 3, 3), 50.0, dtype=np.float64)
    filtered[1, 1, 1] = 100.0

    image = _make_image(original)
    filtered_image = _make_image(filtered)
    mask = _make_image(np.ones((3, 3, 3), dtype=np.float64))

    radiomics = Radiomics(intensity_range=[10.0, 10.0])
    features = radiomics.extract_features(
        image=image,
        mask=mask,
        filtered_image=filtered_image,
        families=['intensity_statistics'],
    )

    assert list(features) == ['stat_mean', 'stat_var', 'stat_skew', 'stat_kurt', 'stat_median', 'stat_min', 'stat_p10',
                              'stat_p90', 'stat_max', 'stat_iqr', 'stat_range', 'stat_mad', 'stat_rmad',
                              'stat_medad', 'stat_cov', 'stat_qcod', 'stat_energy', 'stat_rms']
    assert features['stat_mean'] == pytest.approx(100.0)
    assert features['stat_max'] == pytest.approx(100.0)


@pytest.mark.unit
def test_radiomics_accepts_prepared_roi_data():
    original = np.zeros((3, 3, 3), dtype=np.float64)
    original[1, 1, 1] = 10.0

    filtered = np.full((3, 3, 3), 50.0, dtype=np.float64)
    filtered[1, 1, 1] = 100.0

    image = _make_image(original)
    filtered_image = _make_image(filtered)
    mask = _make_image(np.ones((3, 3, 3), dtype=np.float64))

    roi_data = RoiMaskBuilder().apply(
        image=image,
        mask=mask,
        filtered_image=filtered_image,
    )
    roi_data = Resegmenter(intensity_range=[10.0, 10.0]).apply(
        roi_data,
        reference_image=image,
    )

    features = Radiomics().extract_features(
        roi_data=roi_data,
        families=['intensity_statistics'],
    )

    assert features['stat_mean'] == pytest.approx(100.0)
    assert features['stat_max'] == pytest.approx(100.0)


@pytest.mark.unit
def test_radiomics_crop_to_roi_preserves_feature_values():
    image_array = np.zeros((6, 6, 6), dtype=np.float64)
    image_array[2:5, 1:4, 2:5] = np.arange(27, dtype=np.float64).reshape(3, 3, 3) + 1
    mask_array = np.zeros_like(image_array)
    mask_array[2:5, 1:4, 2:5] = 1

    image = _make_image(image_array)
    mask = _make_image(mask_array)

    families = ['intensity_statistics', 'intensity_histogram', 'glcm']
    uncropped = Radiomics(number_of_bins=4, crop_to_roi=False).extract_features(
        image=image,
        mask=mask,
        families=families,
    )
    cropped = Radiomics(number_of_bins=4, crop_to_roi=True).extract_features(
        image=image,
        mask=mask,
        families=families,
    )

    assert set(cropped) == set(uncropped)
    for name, value in uncropped.items():
        assert cropped[name] == pytest.approx(value)
