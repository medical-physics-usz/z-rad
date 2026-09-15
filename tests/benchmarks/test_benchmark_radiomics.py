import pytest

from .workloads import radiomics

pytestmark = pytest.mark.benchmark(group='radiomics')


@pytest.mark.parametrize('size', ['medium'])
def test_complete_fresh_extraction(benchmark, measure, size):
    measure(radiomics(size))


@pytest.mark.parametrize('size', ['medium'])
def test_spatial_statistics(benchmark, measure, size):
    measure(radiomics(size, 'spatial'))


@pytest.mark.parametrize(
    'family',
    [
        'morphology',
        'local_intensity',
        'intensity_statistics',
        'intensity_histogram',
        'glcm',
        'glrlm',
        'glszm',
        'gldzm',
        'ngtdm',
        'ngldm',
        'ivh',
    ],
)
def test_feature_family(benchmark, measure, family):
    measure(radiomics('medium', family))


_DIRECTIONAL_AGGREGATIONS = (
    ('2D', 'AVER'),
    ('2D', 'SLICE_MERG'),
    ('2.5D', 'DIR_MERG'),
    ('2.5D', 'MERG'),
    ('3D', 'AVER'),
)
_DIMENSIONAL_AGGREGATIONS = (('2D', 'AVER'), ('2.5D', 'AVER'), ('3D', 'AVER'))
_AGGREGATION_CASES = tuple(
    (family, aggregation)
    for family in ('glcm', 'glrlm')
    for aggregation in _DIRECTIONAL_AGGREGATIONS
) + tuple(
    (family, aggregation)
    for family in ('glszm', 'gldzm', 'ngtdm', 'ngldm')
    for aggregation in _DIMENSIONAL_AGGREGATIONS
)


@pytest.mark.benchmark_slow
@pytest.mark.parametrize(
    ('family', 'aggregation'),
    _AGGREGATION_CASES,
    ids=lambda value: '/'.join(value) if isinstance(value, tuple) else value,
)
def test_texture_aggregation(benchmark, measure, family, aggregation):
    measure(radiomics('medium', family, aggregation))
