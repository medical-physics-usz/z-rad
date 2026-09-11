import pytest

from .workloads import pet_suv

pytestmark = pytest.mark.benchmark(group='pet_suv')


def test_enhanced_conversion(benchmark, measure, ibsi_suv_data_dir):
    measure(pet_suv(ibsi_suv_data_dir))


@pytest.mark.benchmark_slow
@pytest.mark.benchmark_io
def test_dicom_load_and_convert(benchmark, measure, ibsi_suv_data_dir):
    measure(pet_suv(ibsi_suv_data_dir, io=True))
