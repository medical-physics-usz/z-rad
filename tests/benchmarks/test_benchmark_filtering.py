"""Matched operation-only filtering paths."""

import pytest

from .workloads import filtering

pytestmark = pytest.mark.benchmark(group='filtering')


@pytest.mark.parametrize(
    ('kind', 'size'),
    [
        ('mean_3d', 'medium'),
        ('mean_2d', 'medium'),
        ('log_3d', 'medium'),
        ('log_2d', 'medium'),
        ('wavelet_3d', 'medium'),
        pytest.param('wavelet_3d_rot', 'medium', marks=pytest.mark.benchmark_slow),
        pytest.param('wavelet_2d_l1', 'small', marks=pytest.mark.benchmark_slow),
        pytest.param('wavelet_2d_l2', 'small', marks=pytest.mark.benchmark_slow),
        ('laws_plain', 'medium'),
        pytest.param('laws_rot_energy', 'medium', marks=pytest.mark.benchmark_slow),
        ('gabor_fixed', 'small'),
        pytest.param('gabor_rot_plane', 'small', marks=pytest.mark.benchmark_slow),
        pytest.param('gabor_rot_orthogonal', 'small', marks=pytest.mark.benchmark_slow),
        ('simoncelli_wrap_3d', 'medium'),
        pytest.param('simoncelli_nearest_3d', 'medium', marks=pytest.mark.benchmark_slow),
        pytest.param('simoncelli_wrap_2d', 'medium', marks=pytest.mark.benchmark_slow),
        pytest.param('simoncelli_riesz', 'medium', marks=pytest.mark.benchmark_slow),
        pytest.param('riesz_first', 'medium', marks=pytest.mark.benchmark_slow),
        pytest.param('riesz_second', 'small', marks=pytest.mark.benchmark_slow),
        pytest.param('riesz_aligned', 'small', marks=pytest.mark.benchmark_slow),
    ],
)
def test_apply(benchmark, measure, kind, size):
    measure(filtering(kind, size))
