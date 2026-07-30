import numpy as np
import pytest
import SimpleITK as sitk

from zrad.image import Image


def _make_image(array, *, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)):
    return Image(
        array=np.asarray(array, dtype=np.float64),
        origin=origin,
        spacing=spacing,
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        shape=(array.shape[2], array.shape[1], array.shape[0]),
    )


@pytest.mark.unit
def test_resample_to_target_uses_target_grid_and_minimum_background():
    moving = _make_image(np.array([[[2.0, 4.0], [6.0, 8.0]]]))
    target = _make_image(
        np.zeros((1, 3, 4)),
        origin=(-1.0, 0.0, 0.0),
        spacing=(0.5, 1.0, 1.0),
    )

    resampled = moving.resample_to_target(target, interpolator=sitk.sitkNearestNeighbor)

    assert resampled is not moving
    assert resampled.shape == target.shape
    assert resampled.origin == target.origin
    np.testing.assert_array_equal(resampled.spacing, target.spacing)
    assert resampled.direction == target.direction
    np.testing.assert_array_equal(resampled.array[0, 0], [2.0, 2.0, 2.0, 4.0])


@pytest.mark.unit
def test_resample_to_target_rejects_non_image_target():
    moving = _make_image(np.ones((1, 1, 1)))

    with pytest.raises(TypeError, match="Expected target to be Image"):
        moving.resample_to_target(object())
