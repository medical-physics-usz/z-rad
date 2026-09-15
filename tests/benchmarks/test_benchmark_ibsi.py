import pytest
from ibsi_cases import IBSI_I_FEATURE_CASES, IBSI_II_FEATURE_CASES, IBSI_II_FILTER_CASES

from .ibsi_workloads import ibsi_feature, ibsi_filter

pytestmark = [
    pytest.mark.benchmark_slow,
    pytest.mark.benchmark_ibsi,
]


@pytest.mark.benchmark(group='ibsi1')
@pytest.mark.parametrize('case', IBSI_I_FEATURE_CASES, ids=lambda case: case.identifier)
def test_ibsi_i_feature_workflow(benchmark, measure, ibsi_feature_sources, case):
    measure(ibsi_feature(case, ibsi_feature_sources))


@pytest.mark.benchmark_exhaustive
@pytest.mark.benchmark(group='ibsi2_phase1')
@pytest.mark.parametrize('case', IBSI_II_FILTER_CASES, ids=lambda case: case.identifier)
def test_ibsi_ii_filter_response(benchmark, measure, ibsi_ii_filter_sources, ibsi_ii_response_maps_dir, case):
    measure(ibsi_filter(case, ibsi_ii_filter_sources, ibsi_ii_response_maps_dir))


@pytest.mark.benchmark_exhaustive
@pytest.mark.benchmark(group='ibsi2_phase2')
@pytest.mark.parametrize('case', IBSI_II_FEATURE_CASES, ids=lambda case: case.identifier)
def test_ibsi_ii_feature_workflow(benchmark, measure, ibsi_ct_sources, case):
    measure(ibsi_feature(case, ibsi_ct_sources))
