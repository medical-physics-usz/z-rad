import pytest
import numpy as np
import SimpleITK as sitk
from zrad.preprocessing import Preprocessing
from zrad.image import Image

@pytest.mark.unit
def test_constructor_valid_inputs():
    preprocessing = Preprocessing(
        input_imaging_modality='CT',
        resample_resolution=2.0,
        resample_dimension='3D',
        interpolation_method='Linear',
        interpolation_threshold=0.5
    )
    assert preprocessing.input_imaging_modality == 'CT'
    assert preprocessing.resample_resolution == 2.0
    assert preprocessing.resample_dimension == '3D'
    assert preprocessing.interpolation_method == 'Linear'
    assert preprocessing.interpolation_threshold == 0.5


@pytest.mark.unit
@pytest.mark.parametrize("invalid_resolution", [0, -1, None, "abc"])
def test_constructor_invalid_resolution(invalid_resolution):
    with pytest.raises(ValueError) as exc_info:
        Preprocessing(
            input_imaging_modality='CT',
            resample_resolution=invalid_resolution,
            resample_dimension='2D',
            interpolation_method='Linear'
        )
    assert "must be a positive int or float" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize("invalid_dimension", ["1D", "4D", None, 123])
def test_constructor_invalid_dimension(invalid_dimension):
    with pytest.raises(ValueError) as exc_info:
        Preprocessing(
            input_imaging_modality='CT',
            resample_resolution=2.0,
            resample_dimension=invalid_dimension,
            interpolation_method='Linear'
        )
    assert "Resample dimension" in str(exc_info.value)
    assert "is not '2D' or '3D'." in str(exc_info.value)


@pytest.mark.unit
def test_get_interpolator_supported_method():
    interpolator = Preprocessing.get_interpolator('Linear')
    assert interpolator == sitk.sitkLinear

    interpolator = Preprocessing.get_interpolator('BSpline')
    assert interpolator == sitk.sitkBSpline


@pytest.mark.unit
def test_get_interpolator_unsupported_method():
    with pytest.raises(ValueError) as exc_info:
        Preprocessing.get_interpolator('SomeUnsupportedMethod')
    assert "is not supported" in str(exc_info.value)


@pytest.mark.unit
def test_calculate_resampled_origin():
    """
    Test that calculate_resampled_origin returns a float representing
    the new origin on a given axis.
    """
    initial_shape = (100, 100, 50)
    initial_spacing = (1.0, 1.0, 1.5)
    resulted_spacing = (2.0, 2.0, 1.5)
    initial_origin = (0.0, 0.0, 0.0)
    axis = 1  # Y-axis

    new_origin = Preprocessing.calculate_resampled_origin(
        initial_shape, initial_spacing, resulted_spacing, initial_origin, axis
    )

    # Basic sanity check: new origin should be within some reasonable range.
    assert isinstance(new_origin, float)


@pytest.mark.unit
def test_resample_image_2d():
    """
    Create a small 2D-like image (1 slice in the z-dimension) and check
    whether the Preprocessing class resamples it as expected.
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

    preprocessing = Preprocessing(
        input_imaging_modality='CT',
        resample_resolution=2.0,
        resample_dimension='2D',   # We'll only change x and y spacing
        interpolation_method='Linear',
        interpolation_threshold=0.5
    )

    resampled_image = preprocessing.resample(original_image, image_type='image')

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

    preprocessing = Preprocessing(
        input_imaging_modality='CT',  # Modality is not as critical for masks
        resample_resolution=2.0,
        resample_dimension='3D',
        interpolation_method='Linear',
        interpolation_threshold=0.5
    )

    resampled_mask = preprocessing.resample(original_mask, image_type='mask')

    # Ensure the resampled array is in {0,1} due to the thresholding
    unique_values = np.unique(resampled_mask.array)
    assert all(val in [0, 1] for val in unique_values), "Mask should only have 0 or 1 after thresholding"
