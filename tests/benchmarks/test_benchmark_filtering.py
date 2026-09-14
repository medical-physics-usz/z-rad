import pytest

from .workloads import filtering

pytestmark = pytest.mark.benchmark(group='filtering')


@pytest.mark.parametrize('size', ['small', 'medium', pytest.param('large', marks=pytest.mark.benchmark_slow)])
def test_log(benchmark, measure, size):
    measure(filtering('log', size))


@pytest.mark.parametrize(
    ('kind', 'size'),
    [
        ('mean', 'medium'),
        ('wavelet', 'medium'),
        ('laws', 'small'),
        ('gabor', 'small'),
        ('simoncelli', 'medium'),
        pytest.param('wavelet_2d', 'medium', marks=pytest.mark.benchmark_slow),
        pytest.param('riesz', 'medium', marks=pytest.mark.benchmark_slow),
        pytest.param('riesz_simoncelli', 'medium', marks=pytest.mark.benchmark_slow),
    ],
)
def test_filter(benchmark, measure, kind, size):
    measure(filtering(kind, size))


@pytest.mark.benchmark_slow
@pytest.mark.parametrize('kind', ['laws', 'gabor'])
def test_expensive_filter_medium(benchmark, measure, kind):
    measure(filtering(kind, 'medium'))
