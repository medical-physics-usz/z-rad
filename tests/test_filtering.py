import pytest
from zrad.filtering.filtering import Filtering


@pytest.mark.unit
def test_filtering_constructor_valid_mean():
    # A basic valid constructor for a Mean filter
    flt = Filtering(
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
    flt = Filtering(
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
    flt = Filtering(
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
        Filtering(filtering_method='UnknownFilter')
    assert "Filter UnknownFilter is not supported." in str(exc_info.value)


@pytest.mark.unit
def test_filtering_constructor_laws_kernels():
    # Constructor for Laws Kernels
    flt = Filtering(
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
