"""Asset and mathematical checks; these are not IBSI consensus benchmarks."""

import itertools

import numpy as np
import pytest
import SimpleITK as sitk

from zrad.filtering.spatial import Laws, LoG, Mean
from zrad.image import Image

pytestmark = pytest.mark.integration
PHANTOMS = ['checkerboard', 'empty', 'impulse', 'noise', 'orientation', 'pattern_1', 'pattern_2', 'pattern_3', 'sphere']


def load_phantom(root, name):
    return Image.from_nifti(root / 'nifti' / name / 'image' / f'{name}.nii.gz')


def assert_geometry(actual, expected):
    for field in ('origin', 'spacing', 'direction', 'shape'):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert actual.array.shape == expected.array.shape


@pytest.mark.parametrize('name', PHANTOMS)
def test_ibsi_supplemental_asset_geometry(ibsi_ii_digital_data_dir, name):
    root = ibsi_ii_digital_data_dir
    image = load_phantom(root, name)
    shape = (64, 48, 32) if name == 'orientation' else (64, 64, 64)
    assert image.array.shape == shape
    np.testing.assert_array_equal(image.shape, shape[::-1])
    np.testing.assert_array_equal(image.origin, (0, 0, 0))
    np.testing.assert_array_equal(image.spacing, (2, 2, 2))
    np.testing.assert_array_equal(image.direction, np.eye(3).ravel())
    assert np.isfinite(image.array).all()

    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(root / 'dicom' / name))
    assert len(files) == 64
    reader.SetFileNames(files)
    dicom = reader.Execute()
    assert dicom.GetSize() == shape[::-1]
    assert dicom.GetSpacing() == (2, 2, 2)
    assert dicom.GetOrigin() == (0, 0, -126)
    np.testing.assert_array_equal(dicom.GetDirection(), np.eye(3).ravel())
    # A documented source-format relationship, not physical-grid equivalence.
    np.testing.assert_array_equal(sitk.GetArrayFromImage(dicom)[::-1], image.array)

    if name != 'orientation':
        mask = Image.from_nifti_mask(root / 'nifti' / name / 'mask' / 'mask.nii.gz', image)
        assert_geometry(mask, image)
        np.testing.assert_array_equal(mask.array, np.ones(shape))


@pytest.mark.parametrize('name', PHANTOMS)
def test_ibsi_supplemental_dicom_loading(ibsi_ii_digital_data_dir, name):
    image = Image.from_dicom(ibsi_ii_digital_data_dir / 'dicom' / name, modality='CT')
    nifti = load_phantom(ibsi_ii_digital_data_dir, name)
    np.testing.assert_array_equal(image.array[::-1], nifti.array)
    np.testing.assert_array_equal(image.shape, nifti.shape)
    np.testing.assert_array_equal(image.spacing, (2, 2, 2))
    np.testing.assert_array_equal(image.direction, np.eye(3).ravel())
    np.testing.assert_array_equal(image.origin, (0, 0, -126))
    assert image.array.shape == nifti.array.shape


@pytest.mark.parametrize('dimension', ['2D', '3D'])
@pytest.mark.parametrize('padding', ['constant', 'nearest', 'wrap', 'reflect'])
@pytest.mark.parametrize('kind', ['mean', 'log', 'laws'])
def test_ibsi_supplemental_zero_input(ibsi_ii_digital_data_dir, dimension, padding, kind):
    image = load_phantom(ibsi_ii_digital_data_dir, 'empty')
    filters = {
        'mean': lambda: Mean(padding, 3, dimension),
        'log': lambda: LoG(padding, 3.0, 4.0, dimension),
        'laws': lambda: Laws('E3L3' if dimension == '2D' else 'E3L3L3', padding, 0, False, dimension),
    }
    result = filters[kind]().apply(image)
    assert_geometry(result, image)
    np.testing.assert_array_equal(result.array, np.zeros_like(image.array))


@pytest.mark.parametrize('name', ['orientation', 'noise'])
def test_ibsi_supplemental_mean_orientation(ibsi_ii_digital_data_dir, name):
    image = load_phantom(ibsi_ii_digital_data_dir, name)
    original = image.array.copy()
    # Explicit periodic 3x3x3 neighbourhood average, independent of filter internals.
    expected = sum(np.roll(original, shift, axis=(0, 1, 2)) for shift in itertools.product((-1, 0, 1), repeat=3)) / 27
    result = Mean('wrap', 3, '3D').apply(image)
    np.testing.assert_allclose(result.array, expected, rtol=0, atol=1e-12)
    assert_geometry(result, image)
    np.testing.assert_array_equal(image.array, original)


@pytest.mark.parametrize('name', ['orientation', 'pattern_2', 'pattern_3'])
@pytest.mark.parametrize('axis,response', [(2, 'E3L3L3'), (1, 'L3E3L3'), (0, 'L3L3E3')])
def test_ibsi_supplemental_directional_laws(ibsi_ii_digital_data_dir, name, axis, response):
    image = load_phantom(ibsi_ii_digital_data_dir, name)
    # Published three-tap E=(-1,0,1)/sqrt(2), L=(1,2,1)/sqrt(6).
    # Enumerate the convolution stencil in public NumPy (z,y,x) coordinates.
    kernels = [(1, 2, 1)] * 3
    kernels[axis] = (-1, 0, 1)
    expected = np.zeros_like(image.array, dtype=float)
    for shifts in itertools.product((-1, 0, 1), repeat=3):
        weight = np.prod([kernels[a][offset + 1] for a, offset in enumerate(shifts)])
        expected += weight * np.roll(image.array, shifts, axis=(0, 1, 2)) / (6 * np.sqrt(2))
    result = Laws(response, 'wrap', 0, False, '3D').apply(image)
    np.testing.assert_allclose(result.array, expected, rtol=0, atol=1e-11)
    assert_geometry(result, image)
    if name == 'orientation':
        assert np.max(np.abs(expected)) > 0
