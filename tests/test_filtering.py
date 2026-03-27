import numpy as np
import pytest
from zrad.filtering.filtering import create_filter
from zrad.image import Image


@pytest.mark.unit
def test_filtering_constructor_valid_mean():
    # A basic valid constructor for a Mean filter
    flt = create_filter(
        filtering_method='Mean',
        padding_type='constant',
        support=3,
        dimensionality='2D'
    )
    assert flt.filtering_method == 'Mean'
    assert flt.filtering_params['support'] == 3
    assert flt.filter is not None, "Filter instance should be created."


@pytest.mark.unit
def test_filtering_constructor_valid_wavelets_2d():
    # A valid constructor for Wavelets in 2D
    flt = create_filter(
        filtering_method='Wavelets',
        wavelet_type='haar',
        padding_type='constant',
        response_map='LL',
        decomposition_level=2,
        rotation_invariance=False,
        dimensionality='2D'
    )
    assert flt.filtering_method == 'Wavelets'
    assert flt.filtering_params['wavelet_type'] == 'haar'
    assert flt.filter is not None


@pytest.mark.unit
def test_filtering_constructor_valid_wavelets_3d():
    # A valid constructor for Wavelets in 3D
    flt = create_filter(
        filtering_method='Wavelets',
        wavelet_type='haar',
        padding_type='constant',
        response_map='LLL',
        decomposition_level=1,
        rotation_invariance=True,
        dimensionality='3D'
    )
    assert flt.filtering_method == 'Wavelets'
    assert flt.filtering_params['dimensionality'] == '3D'
    assert flt.filter is not None


@pytest.mark.unit
def test_filtering_constructor_unsupported_method():
    # Trying a non-existing filtering method
    with pytest.raises(ValueError) as exc_info:
        create_filter(filtering_method='UnknownFilter')
    assert "Filter UnknownFilter is not supported." in str(exc_info.value)


@pytest.mark.unit
def test_filtering_constructor_laws_kernels():
    # Constructor for Laws Kernels
    flt = create_filter(
        filtering_method='Laws Kernels',
        response_map='custom',
        padding_type='constant',
        dimensionality='2D',
        rotation_invariance=False,
        pooling=None,
        energy_map=True,
        distance=1
    )
    assert flt.filtering_method == 'Laws Kernels'
    assert flt.filter is not None


@pytest.mark.unit
def test_filter_apply_returns_image():
    flt = create_filter(
        filtering_method='Mean',
        padding_type='reflect',
        support=3,
        dimensionality='3D'
    )
    image = Image(
        array=np.ones((2, 3, 4), dtype=np.float64),
        origin=(0.0, 0.0, 0.0),
        spacing=np.array([1.0, 1.0, 1.0]),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        shape=(4, 3, 2)
    )

    filtered = flt.apply(image)

    assert isinstance(filtered, Image)
    assert filtered.array.shape == image.array.shape
    assert np.array_equal(filtered.spacing, image.spacing)
