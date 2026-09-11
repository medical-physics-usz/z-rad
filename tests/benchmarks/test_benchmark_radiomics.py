import pytest

from .workloads import radiomics

pytestmark = pytest.mark.benchmark(group='radiomics')


@pytest.mark.parametrize('size', ['small', 'medium', pytest.param('large', marks=pytest.mark.benchmark_slow)])
def test_complete_fresh_extraction(benchmark, measure, size):
    measure(radiomics(size))


@pytest.mark.parametrize('size', ['small', 'medium', pytest.param('large', marks=pytest.mark.benchmark_slow)])
def test_spatial_statistics(benchmark, measure, size):
    measure(radiomics(size, 'spatial'))


def test_texture_families(benchmark, measure):
    measure(radiomics('medium', 'texture'))
