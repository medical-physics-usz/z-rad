import numpy as np
import pytest

from zrad.filtering import Mean, RieszLoG, Simoncelli, create_filter
from zrad.image import Image


@pytest.mark.unit
def test_concrete_filter_constructor_valid_mean():
    flt = Mean(padding_type='constant', support=3, dimensionality='2D')
    assert flt.filtering_method == 'Mean'
    assert flt.get_params()['support'] == 3


@pytest.mark.unit
def test_factory_creates_mean_from_config_parameters():
    flt = create_filter(filtering_method='Mean', padding_type='constant', support=3, dimensionality='2D')
    assert isinstance(flt, Mean)


@pytest.mark.unit
def test_factory_composes_riesz_transform_with_log():
    flt = create_filter(
        filtering_method='Riesz-transformed LoG',
        padding_type='constant',
        sigma_mm=3.0,
        cutoff=4,
        dimensionality='3D',
        riesz_order=(1, 0, 0),
    )

    assert isinstance(flt, RieszLoG)
    assert flt.get_params()['riesz_order'] == (1, 0, 0)


@pytest.mark.unit
def test_riesz_log_supports_structure_tensor_alignment():
    flt = create_filter(
        filtering_method='Riesz-transformed LoG',
        padding_type='constant',
        sigma_mm=3.0,
        cutoff=4,
        dimensionality='3D',
        riesz_order=(0, 2, 0),
        structure_tensor_sigma_mm=1.0,
    )
    image = Image(
        array=np.pad(np.ones((3, 3, 3)), 2),
        origin=(0.0, 0.0, 0.0),
        spacing=np.ones(3),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        shape=(7, 7, 7),
    )

    filtered = flt.apply(image)

    assert filtered.array.shape == image.array.shape
    assert np.all(np.isfinite(filtered.array))


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
        dimensionality='2D',
    )
    assert flt.filtering_method == 'Wavelets'
    assert flt.get_params()['wavelet_type'] == 'haar'


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
        dimensionality='3D',
    )
    assert flt.filtering_method == 'Wavelets'
    assert flt.get_params()['dimensionality'] == '3D'


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
        distance=1,
    )
    assert flt.filtering_method == 'Laws Kernels'
    assert flt.get_params()['energy_map'] is True


@pytest.mark.unit
def test_filter_apply_returns_image():
    flt = create_filter(filtering_method='Mean', padding_type='reflect', support=3, dimensionality='3D')
    image = Image(
        array=np.ones((2, 3, 4), dtype=np.float64),
        origin=(0.0, 0.0, 0.0),
        spacing=np.array([1.0, 1.0, 1.0]),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        shape=(4, 3, 2),
    )

    filtered = flt.apply(image)

    assert isinstance(filtered, Image)
    assert filtered.array.shape == image.array.shape
    assert np.array_equal(filtered.spacing, image.spacing)


@pytest.mark.unit
@pytest.mark.parametrize('level', [1, 2, 3])
def test_simoncelli_b_maps_match_radial_definition(level):
    flt = Simoncelli(padding_type='periodic', decomposition_level=level, dimensionality='3D')
    shape = (8, 8, 8)
    response = flt._frequency_response(shape)
    frequencies = np.fft.ifftshift(np.linspace(-np.pi, np.pi, shape[0]))
    frequencies[0] = 0.0
    radius = np.sqrt(frequencies[1] ** 2 + frequencies[0] ** 2 + frequencies[0] ** 2)
    nyquist = np.pi / 2 ** (level - 1)
    expected = 0.0
    if nyquist / 4 <= radius <= nyquist:
        expected = np.cos(np.pi / 2 * np.log2(2 * radius / nyquist))
    assert response[1, 0, 0] == pytest.approx(expected)
    assert np.all(
        response[np.sqrt(sum(grid**2 for grid in np.meshgrid(frequencies, frequencies, frequencies))) > np.pi] == 0
    )


@pytest.mark.unit
def test_simoncelli_is_isotropic_and_validates_padding():
    flt = create_filter(filtering_method='Simoncelli', padding_type='wrap', decomposition_level=1, dimensionality='3D')
    response = flt._frequency_response((8, 8, 8))
    assert response[1, 0, 0] == pytest.approx(response[0, 1, 0])
    assert response[1, 1, 0] == pytest.approx(response[1, 0, 1])
    assert Simoncelli(padding_type='nearest', decomposition_level=1).padding_type == 'nearest'
    with pytest.raises(ValueError, match='Simoncelli padding'):
        Simoncelli(padding_type='reflect', decomposition_level=1)


@pytest.mark.unit
def test_simoncelli_second_order_riesz_multiplier():
    base = Simoncelli('wrap', 1, '3D')._frequency_response((8, 8, 8))
    riesz = Simoncelli('wrap', 1, '3D', riesz_order=(0, 2, 0))._frequency_response((8, 8, 8))
    frequencies = np.fft.ifftshift(np.linspace(-np.pi, np.pi, 8))
    frequencies[0] = 0.0
    expected_multiplier = -(frequencies[1] ** 2) / (frequencies[0] ** 2 + frequencies[1] ** 2 + frequencies[0] ** 2)
    assert riesz[1, 0, 0] == pytest.approx(base[1, 0, 0] * expected_multiplier)


@pytest.mark.unit
def test_simoncelli_first_order_riesz_response_is_real_and_nonzero():
    image = np.zeros((8, 8, 8))
    image[3, 3, 3] = 1.0
    response = Simoncelli('wrap', 1, '3D', riesz_order=(1, 0, 0))._filter_periodic(image)

    assert np.isrealobj(response)
    assert np.max(np.abs(response)) > 0.0


@pytest.mark.unit
def test_simoncelli_2d_filters_slices_independently():
    image = np.zeros((4, 5, 6))
    image[:, :, 2] = np.arange(20).reshape(4, 5)
    response = Simoncelli('wrap', 1, '2D')._apply_array(image)

    assert np.all(response[:, :, :2] == 0.0)
    assert np.all(response[:, :, 3:] == 0.0)
    assert np.any(response[:, :, 2] != 0.0)


@pytest.mark.unit
def test_simoncelli_nearest_padding_avoids_circular_boundary_response():
    image = np.zeros((16, 16))
    image[:, 0] = 1.0

    nearest = Simoncelli('nearest', 1, '2D')._apply_array(image)
    wrapped = Simoncelli('wrap', 1, '2D')._apply_array(image)

    assert not np.allclose(nearest, wrapped)
    assert np.max(np.abs(nearest[:, -1])) < np.max(np.abs(wrapped[:, -1]))
