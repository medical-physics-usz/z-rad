import pytest

from .workloads import filtering

pytestmark = pytest.mark.benchmark(group='filtering')


@pytest.mark.parametrize('size', ['small', 'medium', pytest.param('large', marks=pytest.mark.benchmark_slow)])
def test_log(benchmark, measure, size):
    measure(filtering('log', size))


@pytest.mark.parametrize('kind', ['mean', 'wavelet', pytest.param('riesz', marks=pytest.mark.benchmark_slow)])
def test_filter(benchmark, measure, kind):
    measure(filtering(kind, 'medium'))
