import pytest

from .workloads import preprocessing, resampling

pytestmark = pytest.mark.benchmark(group='preprocessing')


@pytest.mark.parametrize('size', ['small', 'medium', pytest.param('large', marks=pytest.mark.benchmark_slow)])
@pytest.mark.parametrize('method', ['Linear', 'BSpline'])
def test_image_resampling(benchmark, measure, size, method):
    measure(resampling(size, method))


def test_mask_resampling(benchmark, measure):
    measure(resampling('medium', mask=True))


@pytest.mark.parametrize('operation', ['roi', 'resegment', 'discretize'])
def test_roi_preparation(benchmark, measure, operation):
    measure(preprocessing(operation))
