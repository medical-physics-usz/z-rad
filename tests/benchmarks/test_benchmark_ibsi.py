import pytest

from .workloads import ibsi

pytestmark = [pytest.mark.benchmark(group='ibsi'), pytest.mark.benchmark_slow]


@pytest.mark.parametrize('phase', ['i', 'ii'])
def test_ibsi_workflow(benchmark, measure, ct_pair, phase):
    measure(ibsi(phase, ct_pair))
