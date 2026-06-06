import numpy as np
import pytest

import zrad.batch as batch
import zrad.batch.filtering as batch_filtering
from zrad.batch import BatchFilter, BatchResult, FilteringCaseResult
from zrad.exceptions import InvalidInputParametersError
from zrad.gui.filt_tab import create_batch_filter_from_input_params
from zrad.image import Image


def _make_image(array=None):
    if array is None:
        array = np.ones((3, 3, 3), dtype=np.float64)
    return Image(
        array=np.asarray(array, dtype=np.float64),
        origin=[0.0, 0.0, 0.0],
        spacing=[1.0, 1.0, 1.0],
        direction=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        shape=(array.shape[2], array.shape[1], array.shape[0]),
    )


def _write_case(input_dir, case_name, image_name='image'):
    case_dir = input_dir / case_name
    case_dir.mkdir(parents=True)
    _make_image().save_as_nifti(case_dir / f'{image_name}.nii.gz')
    return case_dir


def _mean_filter(input_dir, output_dir, **kwargs):
    params = {
        'input_directory': input_dir,
        'output_directory': output_dir,
        'input_data_type': 'nifti',
        'modality': 'CT',
        'nifti_image_name': 'image',
        'number_of_threads': 1,
        'filter_type': 'Mean',
        'filter_dimension': '3D',
        'padding_type': 'reflect',
        'mean_support': 3,
    }
    params.update(kwargs)
    return BatchFilter(**params)


@pytest.mark.unit
def test_batch_public_api_exposes_filtering_classes():
    assert BatchFilter is batch.BatchFilter
    assert FilteringCaseResult is batch.FilteringCaseResult
    assert BatchResult is batch.BatchResult


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({'input_data_type': 'unsupported'}, "input_data_type"),
        ({'number_of_threads': 0}, "number_of_threads"),
        ({'nifti_image_name': None}, "nifti_image_name"),
        ({'mean_support': None}, "mean_support"),
    ],
)
def test_batch_filter_validates_inputs(tmp_path, kwargs, message):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()

    batch_filter = _mean_filter(input_dir, output_dir, **kwargs)

    with pytest.raises(InvalidInputParametersError, match=message):
        batch_filter.validate()


@pytest.mark.unit
def test_batch_filter_selects_all_patient_folders(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()
    (input_dir / '.hidden').mkdir()
    (input_dir / 'case_a').mkdir()
    (input_dir / 'case_b').mkdir()

    assert _mean_filter(input_dir, output_dir).plan() == ['case_a', 'case_b']


@pytest.mark.unit
def test_batch_filter_selects_explicit_patient_folders(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()

    batch_filter = _mean_filter(input_dir, output_dir, patient_folders=['case_b', 'case_a'])

    assert batch_filter.plan() == ['case_b', 'case_a']


@pytest.mark.unit
def test_batch_filter_selects_numeric_folder_range(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()
    for folder in ['1', '2', '3', 'case_a']:
        (input_dir / folder).mkdir()

    batch_filter = _mean_filter(input_dir, output_dir, start_folder=2, stop_folder=3)

    assert batch_filter.plan() == ['2', '3']


@pytest.mark.unit
def test_nifti_mean_filter_writes_expected_output(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a')

    result = _mean_filter(input_dir, output_dir).run()

    expected = output_dir / 'case_a' / 'Mean_3D_3support_reflect.nii.gz'
    case_result = result.case_results[0]
    assert case_result.status == 'processed'
    assert case_result.output_path == expected
    assert expected.exists()


@pytest.mark.unit
def test_missing_nifti_image_is_recorded_as_skipped_case(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    (input_dir / 'case_a').mkdir(parents=True)

    result = _mean_filter(input_dir, output_dir).run()

    case_result = result.case_results[0]
    assert case_result.status == 'skipped'
    assert case_result.error
    assert result.skipped_count == 1


@pytest.mark.unit
def test_unexpected_filter_failure_is_recorded_as_failed_case(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a')

    def fail_create_filter(_self):
        raise RuntimeError("filter failed")

    monkeypatch.setattr(BatchFilter, '_create_filter', fail_create_filter)

    result = _mean_filter(input_dir, output_dir).run()

    case_result = result.case_results[0]
    assert case_result.status == 'failed'
    assert case_result.error == "filter failed"
    assert result.failed_count == 1


@pytest.mark.unit
def test_dicom_filter_uses_dicom_reader(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    (input_dir / 'case_a').mkdir(parents=True)

    monkeypatch.setattr(batch_filtering.Image, 'from_dicom', staticmethod(lambda *args, **kwargs: _make_image()))

    result = BatchFilter(
        input_directory=input_dir,
        output_directory=output_dir,
        input_data_type='dicom',
        modality='CT',
        filter_type='Mean',
        filter_dimension='3D',
        padding_type='reflect',
        mean_support=3,
    ).run()

    expected = output_dir / 'case_a' / 'Mean_3D_3support_reflect.nii.gz'
    case_result = result.case_results[0]
    assert case_result.status == 'processed'
    assert case_result.output_path == expected
    assert expected.exists()


@pytest.mark.unit
def test_sequential_progress_callback_reports_total_case_count(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a')
    _write_case(input_dir, 'case_b')
    progress_steps = []

    _mean_filter(input_dir, output_dir).run(progress_callback=progress_steps.append)

    assert sum(progress_steps) == 2


@pytest.mark.unit
def test_parallel_progress_callback_reports_total_case_count(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a')
    _write_case(input_dir, 'case_b')
    progress_steps = []

    _mean_filter(
        input_dir,
        output_dir,
        number_of_threads=2,
        parallel_backend='threads',
    ).run(progress_callback=progress_steps.append)

    assert sum(progress_steps) == 2


def _gui_input_params(**kwargs):
    params = {
        'input_directory': '/input',
        'output_directory': '/output',
        'input_data_type': 'nifti',
        'input_image_modality': 'CT',
        'number_of_threads': 4,
        'list_of_patient_folders': ['case_a'],
        'start_folder': None,
        'stop_folder': None,
        'nifti_image_name': 'image',
        'filter_type': 'Mean',
        'filter_dimension': '3D',
        'filter_padding_type': 'reflect',
        'filter_mean_support': '3',
        'filter_log_sigma': '2',
        'filter_log_cutoff': '4',
        'filter_laws_response_map': 'L5E5',
        'filter_laws_rot_inv': 'Enable',
        'filter_laws_distance': '5',
        'filter_laws_pooling': 'average',
        'filter_laws_energy_map': 'Disable',
        'filter_wavelet_resp_map_2D': 'LL',
        'filter_wavelet_resp_map_3D': 'LLL',
        'filter_wavelet_type': 'haar',
        'filter_wavelet_decomp_lvl': '1',
        'filter_wavelet_rot_inv': 'Enable',
        'filter_gabor_res_mm': '1',
        'filter_gabor_sigma_mm': '2',
        'filter_gabor_lambda_mm': '3',
        'filter_gabor_gamma': '0.5',
        'filter_gabor_theta': '0',
        'filter_gabor_rotinv': 'Enable',
        'filter_gabor_ortho': 'Disable',
    }
    params.update(kwargs)
    return params


@pytest.mark.unit
@pytest.mark.parametrize(
    "filter_type, expected",
    [
        ('Mean', {'mean_support': '3'}),
        ('Laplacian of Gaussian', {'log_sigma': '2', 'log_cutoff': '4'}),
        ('Laws Kernels', {'laws_response_map': 'L5E5', 'laws_rotation_invariance': 'Enable'}),
        ('Gabor', {'gabor_res_mm': '1', 'gabor_rotation_invariance': 'Enable'}),
        ('Wavelets', {'wavelet_type': 'haar', 'wavelet_response_map': 'LLL'}),
    ],
)
def test_gui_mapping_creates_batch_filter_for_filter_families(filter_type, expected):
    batch_filter = create_batch_filter_from_input_params(
        _gui_input_params(filter_type=filter_type),
        parallel_backend='threads',
    )

    assert batch_filter.filter_type == filter_type
    assert batch_filter.parallel_backend == 'threads'
    for key, value in expected.items():
        assert getattr(batch_filter, key) == value
