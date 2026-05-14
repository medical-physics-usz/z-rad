import pytest
import numpy as np
import zrad.preprocessing as preprocessing
from zrad.preprocessing.discretization import IntensityVolumeHistogramDiscretizer
from zrad.preprocessing.resegmentation import OutlierResegmenter, RangeResegmenter, Resegmenter
from zrad.preprocessing import (
    ImageResampler,
    IntensityMaskBuilder,
    IVHIntensityDiscretizer,
    MaskResampler,
    Pipeline,
    RoiCropper,
    RoiData,
    TextureDiscretizer,
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
        'IVHIntensityDiscretizer',
        'ImageResampler',
        'IntensityMaskBuilder',
        'MaskResampler',
        'Pipeline',
        'RoiCropper',
        'Resegmenter',
        'RoiData',
        'RoiMaskValidator',
        'TextureDiscretizer',
    }

    assert not hasattr(preprocessing, 'RangeResegmenter')
    assert not hasattr(preprocessing, 'OutlierResegmenter')
    assert not hasattr(preprocessing, 'FixedBinSizeDiscretizer')
    assert not hasattr(preprocessing, 'FixedBinNumberDiscretizer')
    assert not hasattr(preprocessing, 'ImageDiscretizer')
    assert not hasattr(preprocessing, 'IntensityVolumeHistogramDiscretizer')


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"number_of_bins": 4, "bin_size": 25},
    ],
)
def test_texture_discretizer_requires_exactly_one_method(kwargs):
    with pytest.raises(ValueError, match="Specify exactly one"):
        TextureDiscretizer(**kwargs)


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"number_of_bins": 0}, "positive integer"),
        ({"number_of_bins": 1.5}, "positive integer"),
        ({"bin_size": 0}, "positive number"),
        ({"bin_size": -1}, "positive number"),
    ],
)
def test_texture_discretizer_validates_method_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TextureDiscretizer(**kwargs)


@pytest.mark.unit
def test_texture_discretizer_applies_fixed_bin_size_to_roi_data():
    image = _make_image(np.array([[[0.0, 25.0, 50.0]]]))
    mask = _make_image(np.ones((1, 1, 3), dtype=np.float64))
    roi_data = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))
    roi_data = Resegmenter(intensity_range=(0.0, 50.0)).apply(roi_data)
    discretized = TextureDiscretizer(bin_size=25).apply(roi_data)
    np.testing.assert_array_equal(discretized.texture_discretized_image.array, np.array([[[1.0, 2.0, 3.0]]]))
    assert discretized.intensity_range == (0.0, 50.0)


@pytest.mark.unit
def test_texture_discretizer_fixed_bin_size_requires_resegmentation_range():
    image = _make_image(np.array([[[0.0, 25.0, 50.0]]]))
    mask = _make_image(np.ones((1, 1, 3), dtype=np.float64))
    roi_data = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))

    with pytest.raises(ValueError, match="intensity_range"):
        TextureDiscretizer(bin_size=25).apply(roi_data)


@pytest.mark.unit
def test_texture_discretizer_applies_fixed_bin_number_to_roi_data():
    image = _make_image(np.array([[[0.0, 10.0, 20.0, 30.0]]]))
    mask = _make_image(np.ones((1, 1, 4), dtype=np.float64))
    roi_data = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))
    discretized = TextureDiscretizer(number_of_bins=3).apply(roi_data)
    np.testing.assert_array_equal(discretized.texture_discretized_image.array, np.array([[[1.0, 2.0, 3.0, 3.0]]]))


@pytest.mark.unit
def test_fixed_bin_number_constant_image_maps_to_first_bin():
    image = _make_image(np.array([[[5.0, 5.0, np.nan]]]))
    mask = _make_image(np.array([[[1.0, 1.0, 0.0]]]))
    roi_data = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))
    discretized = TextureDiscretizer(number_of_bins=4).apply(roi_data)
    np.testing.assert_array_equal(discretized.texture_discretized_image.array, np.array([[[1.0, 1.0, np.nan]]]))


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"number_of_bins": 4, "bin_size": 25},
    ],
)
def test_ivh_discretizer_requires_exactly_one_method(kwargs):
    with pytest.raises(ValueError, match="Specify exactly one"):
        IntensityVolumeHistogramDiscretizer(**kwargs)


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"method": "direct", "number_of_bins": 4}, "direct IVH"),
        ({"method": "fixed_bin_number"}, "requires number_of_bins"),
        ({"method": "fixed_bin_number", "number_of_bins": 4, "bin_size": 1}, "bin_size"),
        ({"method": "fixed_bin_size", "number_of_bins": 4, "bin_size": 1}, "number_of_bins"),
        ({"method": "unsupported"}, "method"),
    ],
)
def test_ivh_intensity_discretizer_validates_methods(kwargs, message):
    with pytest.raises(ValueError, match=message):
        IVHIntensityDiscretizer(**kwargs)


@pytest.mark.unit
def test_ivh_intensity_discretizer_direct_writes_image_and_metadata():
    image = _make_image(np.array([[[2.0, 4.0, 8.0]]]))
    mask = _make_image(np.ones((1, 1, 3), dtype=np.float64))
    roi_data = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))

    prepared = IVHIntensityDiscretizer(method='direct').apply(roi_data)

    np.testing.assert_array_equal(prepared.ivh_intensity_image.array, image.array)
    assert prepared.ivh_discretization_method == 'direct'
    assert prepared.ivh_discretization_step == 1


@pytest.mark.unit
def test_ivh_intensity_discretizer_fixed_bin_size_uses_bin_centres():
    image = _make_image(np.array([[[0.0, 1.0, 2.0]]]))
    mask = _make_image(np.ones((1, 1, 3), dtype=np.float64))
    roi_data = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))
    roi_data = Resegmenter(intensity_range=(0.0, 3.0)).apply(roi_data)

    prepared = IVHIntensityDiscretizer(
        method='fixed_bin_size',
        bin_size=1.0,
    ).apply(roi_data)

    np.testing.assert_array_equal(prepared.ivh_intensity_image.array, np.array([[[0.5, 1.5, 2.5]]]))
    assert prepared.ivh_discretization_method == 'fixed_bin_size'
    assert prepared.ivh_discretization_step == 1.0


@pytest.mark.unit
def test_ivh_intensity_discretizer_fixed_bin_size_requires_resegmentation_range():
    image = _make_image(np.array([[[0.0, 1.0, 2.0]]]))
    mask = _make_image(np.ones((1, 1, 3), dtype=np.float64))
    roi_data = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))

    with pytest.raises(ValueError, match="intensity_range"):
        IVHIntensityDiscretizer(method='fixed_bin_size', bin_size=1.0).apply(roi_data)


@pytest.mark.unit
def test_ivh_intensity_discretizer_fixed_bin_number_writes_metadata():
    image = _make_image(np.array([[[0.0, 10.0, 20.0, 30.0]]]))
    mask = _make_image(np.ones((1, 1, 4), dtype=np.float64))
    roi_data = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))

    prepared = IVHIntensityDiscretizer(method='fixed_bin_number', number_of_bins=3).apply(roi_data)

    np.testing.assert_array_equal(prepared.ivh_intensity_image.array, np.array([[[1.0, 2.0, 3.0, 3.0]]]))
    assert prepared.ivh_discretization_method == 'fixed_bin_number'
    assert prepared.ivh_discretization_step == 1


@pytest.mark.unit
@pytest.mark.parametrize("intensity_range", [[1], [2, 1], [np.nan, 1], ["a", 1]])
def test_range_resegmenter_validates_intensity_range(intensity_range):
    with pytest.raises(ValueError, match="intensity_range"):
        RangeResegmenter(intensity_range)


@pytest.mark.unit
@pytest.mark.parametrize("outlier_range", [0, -1, np.inf, "not-a-number"])
def test_outlier_resegmenter_validates_outlier_range(outlier_range):
    with pytest.raises(ValueError, match="outlier_range"):
        OutlierResegmenter(outlier_range)


@pytest.mark.unit
def test_outlier_resegmenter_accepts_numeric_string():
    assert OutlierResegmenter("3").outlier_range == 3.0


@pytest.mark.unit
def test_outlier_resegmenter_uses_current_intensity_mask_after_range_resegmentation():
    image = _make_image(np.array([[[0.0, 100.0, 100.0, 100.0, 1000.0]]]))
    mask = _make_image(np.ones((1, 1, 5), dtype=np.float64))
    roi_data = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))

    resegmented = Resegmenter(intensity_range=(0.0, 100.0), outlier_range=1.0).apply(roi_data)

    np.testing.assert_array_equal(
        resegmented.intensity_mask.array,
        np.array([[[np.nan, 100.0, 100.0, 100.0, np.nan]]]),
    )


@pytest.mark.unit
def test_resegmenter_clears_prepared_texture_and_ivh_fields():
    image = _make_image(np.array([[[0.0, 1.0, 2.0]]]))
    mask = _make_image(np.ones((1, 1, 3), dtype=np.float64))
    roi_data = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))
    roi_data = TextureDiscretizer(number_of_bins=2).apply(roi_data)
    roi_data = IVHIntensityDiscretizer(method='direct').apply(roi_data)

    resegmented = Resegmenter(intensity_range=(0.0, 2.0)).apply(roi_data)

    assert resegmented.texture_discretized_image is None
    assert resegmented.ivh_intensity_image is None
    assert resegmented.ivh_discretization_method is None
    assert resegmented.ivh_discretization_step is None
    assert resegmented.intensity_range == (0.0, 2.0)


@pytest.mark.unit
def test_intensity_mask_builder_clears_range_and_prepared_fields():
    image = _make_image(np.array([[[0.0, 1.0, 2.0]]]))
    mask = _make_image(np.ones((1, 1, 3), dtype=np.float64))
    roi_data = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))
    roi_data = Resegmenter(intensity_range=(0.0, 2.0)).apply(roi_data)
    roi_data = TextureDiscretizer(number_of_bins=2).apply(roi_data)
    roi_data = IVHIntensityDiscretizer(method='direct').apply(roi_data)

    rebuilt = IntensityMaskBuilder().apply(roi_data)

    assert rebuilt.intensity_range is None
    assert rebuilt.texture_discretized_image is None
    assert rebuilt.ivh_intensity_image is None
    assert rebuilt.ivh_discretization_method is None
    assert rebuilt.ivh_discretization_step is None


@pytest.mark.unit
def test_roi_cropper_crops_prepared_texture_and_ivh_images():
    image = _make_image(np.arange(27, dtype=np.float64).reshape(3, 3, 3))
    mask_array = np.zeros((3, 3, 3), dtype=np.float64)
    mask_array[1:, 1:, 1:] = 1
    mask = _make_image(mask_array)
    roi_data = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))
    roi_data = Resegmenter(intensity_range=(0.0, 26.0)).apply(roi_data)
    roi_data = TextureDiscretizer(number_of_bins=4).apply(roi_data)
    roi_data = IVHIntensityDiscretizer(method='direct').apply(roi_data)

    cropped = RoiCropper().apply(roi_data)

    assert cropped.intensity_mask.array.shape == (2, 2, 2)
    assert cropped.texture_discretized_image.array.shape == (2, 2, 2)
    assert cropped.ivh_intensity_image.array.shape == (2, 2, 2)
    assert cropped.intensity_range == roi_data.intensity_range
    assert cropped.ivh_discretization_method == roi_data.ivh_discretization_method
    assert cropped.ivh_discretization_step == roi_data.ivh_discretization_step

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
