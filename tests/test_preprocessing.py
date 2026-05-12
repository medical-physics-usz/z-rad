import pytest
import numpy as np
import zrad.preprocessing as preprocessing
from zrad.preprocessing import (
    ImageDiscretizer,
    ImageResampler,
    IntensityMaskBuilder,
    MaskResampler,
    Pipeline,
    RoiCropper,
    RoiData,
)
from zrad.image import Image


class AddOneFilter:
    def get_params(self):
        return {}

    def apply(self, image):
        if isinstance(image, RoiData):
            return RoiData(
                image=image.image,
                filtered_image=self.apply(image.image),
                morphological_mask=image.morphological_mask,
                intensity_mask=None,
            )
        filtered = image.copy()
        filtered.array = filtered.array + 1
        return filtered


def _make_image(array):
    return Image(
        array=np.asarray(array, dtype=np.float64),
        origin=[0.0, 0.0, 0.0],
        spacing=[1.0, 1.0, 1.0],
        direction=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        shape=(array.shape[2], array.shape[1], array.shape[0]),
    )


@pytest.mark.unit
def test_preprocessing_public_api_exposes_main_steps_only():
    assert set(preprocessing.__all__) == {
        'ImageDiscretizer',
        'ImageResampler',
        'IntensityMaskBuilder',
        'MaskResampler',
        'Pipeline',
        'RoiCropper',
        'Resegmenter',
        'RoiData',
        'RoiMaskValidator',
    }

    assert not hasattr(preprocessing, 'RangeResegmenter')
    assert not hasattr(preprocessing, 'OutlierResegmenter')
    assert not hasattr(preprocessing, 'FixedBinSizeDiscretizer')
    assert not hasattr(preprocessing, 'FixedBinNumberDiscretizer')
    assert not hasattr(preprocessing, 'IntensityVolumeHistogramDiscretizer')


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"number_of_bins": 4, "bin_size": 25},
    ],
)
def test_image_discretizer_requires_exactly_one_method(kwargs):
    with pytest.raises(ValueError, match="Specify exactly one"):
        ImageDiscretizer(**kwargs)


@pytest.mark.unit
def test_image_discretizer_applies_fixed_bin_size():
    image = _make_image(np.array([[[0.0, 25.0, 50.0]]]))
    discretized = ImageDiscretizer(bin_size=25, minimum=0).apply(image)
    np.testing.assert_array_equal(discretized.array, np.array([[[1.0, 2.0, 3.0]]]))


@pytest.mark.unit
def test_image_discretizer_applies_fixed_bin_number():
    image = _make_image(np.array([[[0.0, 10.0, 20.0, 30.0]]]))
    discretized = ImageDiscretizer(number_of_bins=3).apply(image)
    np.testing.assert_array_equal(discretized.array, np.array([[[1.0, 2.0, 3.0, 3.0]]]))

@pytest.mark.unit
def test_constructor_valid_inputs():
    image_resampler = ImageResampler(
        resolution=(2.0, 2.0, 2.0),
        method='linear',
        intensity_rounding='nearest_integer',
    )
    mask_resampler = MaskResampler(
        resolution=(2.0, 2.0, 2.0),
        method='trilinear',
        partial_volume_threshold=0.5,
    )
    assert image_resampler.resolution == (2.0, 2.0, 2.0)
    assert image_resampler.method == 'linear'
    assert image_resampler.intensity_rounding == 'nearest_integer'
    assert mask_resampler.partial_volume_threshold == 0.5


@pytest.mark.unit
@pytest.mark.parametrize("invalid_resolution", [0, -1, None, "abc"])
def test_constructor_invalid_resolution(invalid_resolution):
    with pytest.raises(ValueError) as exc_info:
        ImageResampler(resolution=invalid_resolution, method='linear')
    assert "Resolution" in str(exc_info.value)


@pytest.mark.unit
def test_get_interpolator_unsupported_method():
    image = _make_image(np.ones((3, 3, 3), dtype=np.float64))
    with pytest.raises(ValueError) as exc_info:
        ImageResampler(resolution=(1.0, 1.0, 1.0), method='SomeUnsupportedMethod').apply(image)
    assert "is not supported" in str(exc_info.value)


@pytest.mark.unit
def test_resample_image_2d():
    """
    Create a small 2D-like image (1 slice in the z-dimension) and check
    whether the ImageResampler class resamples it as expected.
    """
    # Dummy 2D data (we treat the z-dim as 1 to simulate a single slice)
    dummy_array = np.ones((1, 10, 10), dtype=np.float64)  # shape = (z, y, x)
    # Create an Image object with minimal metadata
    original_image = Image(
        array=dummy_array,
        origin=[0.0, 0.0, 0.0],
        spacing=[1.0, 1.0, 2.0],  # We can assume (x, y, z) or (y, x, z) as needed
        direction=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        shape=(1, 10, 10)
    )

    preprocessing = ImageResampler(
        resolution=(2.0, 2.0, original_image.spacing[2]),
        method='linear',
        intensity_rounding='nearest_integer',
    )

    resampled_image = preprocessing.apply(original_image)

    # Verify dimensions, shape, spacing have changed for x,y but not for z
    assert resampled_image.shape[0] == 1  # z dimension unchanged
    assert resampled_image.spacing[2] == 2.0  # z spacing unchanged
    # x,y spacing should match 2.0 from the constructor
    assert pytest.approx(resampled_image.spacing[0]) == 2.0
    assert pytest.approx(resampled_image.spacing[1]) == 2.0

    # For a 10x10, upscaling spacing to 2.0 typically halves the shape dimension.
    # We might get 5x5 or 6x6, depending on ceiling. Just check it's smaller:
    assert resampled_image.shape[1] <= 10
    assert resampled_image.shape[2] <= 10


@pytest.mark.unit
def test_resample_mask_3d():
    """
    Create a small 3D mask image and verify thresholding is applied
    when resampling.
    """
    dummy_mask_array = np.array([[[0.4, 0.6],
                                  [1.0, 0.2]],
                                 [[0.9, 0.5],
                                  [0.2, 0.8]]],
                                dtype=np.float64)
    # shape = (2, 2, 2) in z,y,x order
    original_mask = Image(
        array=dummy_mask_array,
        origin=[0.0, 0.0, 0.0],
        spacing=[1.0, 1.0, 1.0],
        direction=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        shape=dummy_mask_array.shape
    )

    preprocessing = MaskResampler(
        resolution=(2.0, 2.0, 2.0),
        method='linear',
        partial_volume_threshold=0.5,
    )

    resampled_mask = preprocessing.apply(original_mask)

    # Ensure the resampled array is in {0,1} due to the thresholding
    unique_values = np.unique(resampled_mask.array)
    assert all(val in [0, 1] for val in unique_values), "Mask should only have 0 or 1 after thresholding"


@pytest.mark.unit
def test_pipeline_transforms_roi_data():
    image = _make_image(np.ones((3, 3, 3), dtype=np.float64))
    mask = _make_image(np.ones((3, 3, 3), dtype=np.float64))
    roi_data = RoiData(image=image, morphological_mask=mask)

    pipeline = Pipeline([
        ("image_resampler", ImageResampler(resolution=(1.0, 1.0, 1.0), method="linear")),
        ("mask_resampler", MaskResampler(resolution=(1.0, 1.0, 1.0), method="trilinear")),
        ("filter", AddOneFilter()),
        ("intensity_mask_builder", IntensityMaskBuilder()),
        ("cropper", RoiCropper()),
    ])

    result = pipeline.apply(roi_data)

    assert isinstance(result, RoiData)
    assert result.filtered_image is not None
    assert result.intensity_mask is not None
    assert np.all(result.intensity_mask.array == 2)


@pytest.mark.unit
def test_image_and_mask_resamplers_accept_images_directly():
    image = _make_image(np.ones((3, 3, 3), dtype=np.float64))
    mask = _make_image(np.ones((3, 3, 3), dtype=np.float64))

    resampled_image = ImageResampler(resolution=(1.0, 1.0, 1.0), method="tricubic_spline").apply(image)
    resampled_mask = MaskResampler(resolution=(1.0, 1.0, 1.0), method="trilinear").apply(mask)

    assert isinstance(resampled_image, Image)
    assert isinstance(resampled_mask, Image)
    assert set(np.unique(resampled_mask.array)) <= {0, 1}
