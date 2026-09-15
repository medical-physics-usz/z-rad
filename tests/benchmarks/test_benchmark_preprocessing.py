import pytest

from .workloads import preprocessing, resampling, texture_fixed_bin_size

pytestmark = pytest.mark.benchmark(group='preprocessing')


@pytest.mark.parametrize('method,size', [
    pytest.param('Linear', 'small', id='linear-small'),
    pytest.param('Linear', 'medium', id='linear-medium'),
    pytest.param('Linear', 'large', id='linear-large', marks=pytest.mark.benchmark_slow),
    pytest.param('BSpline', 'small', id='bspline-small'),
    pytest.param('BSpline', 'medium', id='bspline-medium'),
    pytest.param('BSpline', 'large', id='bspline-large', marks=pytest.mark.benchmark_slow),
    pytest.param('NN', 'medium', id='nearest_neighbor-medium'),
    pytest.param('Gaussian', 'medium', id='gaussian-medium'),
])
def test_image_resampling_isotropic(benchmark, measure, method, size):
    measure(resampling(size, method))


@pytest.mark.parametrize('method', [
    pytest.param('NN', id='nearest_neighbor-medium'),
    pytest.param('Linear', id='linear-medium'),
    pytest.param('BSpline', id='bspline-medium'),
    pytest.param('Gaussian', id='gaussian-medium'),
])
def test_mask_resampling_isotropic(benchmark, measure, method):
    measure(resampling('medium', method, mask=True))


@pytest.mark.parametrize('method,size', [pytest.param('Linear', 'medium', id='linear-medium')])
def test_image_resampling_in_plane(benchmark, measure, method, size):
    measure(resampling(size, method, dimension='2D'))


@pytest.mark.parametrize('method', [
    pytest.param('NN', id='nearest_neighbor-medium'),
    pytest.param('Linear', id='linear-medium'),
])
def test_mask_resampling_in_plane(benchmark, measure, method):
    measure(resampling('medium', method, mask=True, dimension='2D'))


def test_intensity_mask_building(benchmark, measure):
    measure(preprocessing('roi'))


def test_range_and_outlier_resegmentation(benchmark, measure):
    measure(preprocessing('resegment'))


def test_texture_discretization_fixed_bin_number(benchmark, measure):
    measure(preprocessing('discretize'))


def test_texture_discretization_fixed_bin_size(benchmark, measure):
    measure(texture_fixed_bin_size())
