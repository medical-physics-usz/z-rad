import pytest

from .workloads import preprocessing, resampling

pytestmark = pytest.mark.benchmark(group='preprocessing')


@pytest.mark.parametrize('method,size', [
    ('Linear', 'small'),
    ('Linear', 'medium'),
    pytest.param('Linear', 'large', marks=pytest.mark.benchmark_slow),
    ('BSpline', 'small'),
    ('BSpline', 'medium'),
    pytest.param('BSpline', 'large', marks=pytest.mark.benchmark_slow),
    ('NN', 'medium'),
    ('Gaussian', 'medium'),
])
def test_image_resampling_isotropic(benchmark, measure, method, size):
    measure(resampling(size, method))


@pytest.mark.parametrize('method', ['NN', 'Linear', 'BSpline', 'Gaussian'])
def test_mask_resampling_isotropic(benchmark, measure, method):
    measure(resampling('medium', method, mask=True))


def test_image_resampling_in_plane(benchmark, measure):
    measure(resampling('medium', 'Linear', dimension='2D'))


@pytest.mark.parametrize('method', ['NN', 'Linear'])
def test_mask_resampling_in_plane(benchmark, measure, method):
    measure(resampling('medium', method, mask=True, dimension='2D'))


@pytest.mark.parametrize('operation', ['roi', 'resegment', 'discretize'])
def test_roi_preparation(benchmark, measure, operation):
    measure(preprocessing(operation))
