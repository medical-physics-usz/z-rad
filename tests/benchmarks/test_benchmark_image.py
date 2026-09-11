import pytest

from .workloads import resampling

pytestmark = pytest.mark.benchmark(group='image')


@pytest.mark.parametrize('size', ['small', 'medium', pytest.param('large', marks=pytest.mark.benchmark_slow)])
def test_target_resampling(benchmark, measure, size):
    measure(resampling(size, target=True))
