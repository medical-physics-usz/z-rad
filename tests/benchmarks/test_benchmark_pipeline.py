import pytest

from .workloads import pipeline

pytestmark = [pytest.mark.benchmark(group='pipeline'), pytest.mark.benchmark_slow]


def test_log_radiomics(benchmark, measure):
    measure(pipeline())
