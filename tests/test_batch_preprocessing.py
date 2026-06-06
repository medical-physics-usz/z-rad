import numpy as np
import pytest

import zrad.batch as batch
import zrad.batch.preprocessing as batch_preprocessing
from zrad.batch import BatchPreprocessor, BatchResult, PreprocessingCaseResult
from zrad.exceptions import InvalidInputParametersError
from zrad.image import Image


def _make_image(array=None, spacing=None):
    if array is None:
        array = np.ones((2, 2, 2), dtype=np.float64)
    if spacing is None:
        spacing = [1.0, 1.0, 1.0]
    return Image(
        array=np.asarray(array, dtype=np.float64),
        origin=[0.0, 0.0, 0.0],
        spacing=spacing,
        direction=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        shape=(array.shape[2], array.shape[1], array.shape[0]),
    )


def _write_case(input_dir, case_name, image_name='image', masks=None):
    case_dir = input_dir / case_name
    case_dir.mkdir(parents=True)
    _make_image().save_as_nifti(case_dir / f'{image_name}.nii.gz')
    for mask_name in masks or []:
        _make_image().save_as_nifti(case_dir / f'{mask_name}.nii.gz')
    return case_dir


def _nifti_preprocessor(input_dir, output_dir, **kwargs):
    params = {
        'input_directory': input_dir,
        'output_directory': output_dir,
        'input_data_type': 'nifti',
        'modality': 'CT',
        'nifti_image_name': 'image',
        'number_of_threads': 1,
        'structures': ['mask'],
        'just_save_as_nifti': True,
    }
    params.update(kwargs)
    return BatchPreprocessor(**params)


@pytest.mark.unit
def test_batch_public_api_exposes_preprocessing_classes():
    assert set(batch.__all__) == {
        'BatchPreprocessor',
        'BatchResult',
        'PreprocessingCaseResult',
    }
    assert BatchPreprocessor is batch.BatchPreprocessor
    assert BatchResult is batch.BatchResult
    assert PreprocessingCaseResult is batch.PreprocessingCaseResult


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({'input_data_type': 'unsupported'}, "input_data_type"),
        ({'number_of_threads': 0}, "number_of_threads"),
        ({'nifti_image_name': None}, "nifti_image_name"),
        ({'use_all_structures': True}, "use_all_structures"),
        ({'just_save_as_nifti': False, 'resample_resolution': None}, "resample_resolution"),
    ],
)
def test_batch_preprocessor_validates_inputs(tmp_path, kwargs, message):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()

    preprocessor = _nifti_preprocessor(input_dir, output_dir, **kwargs)

    with pytest.raises(InvalidInputParametersError, match=message):
        preprocessor.validate()


@pytest.mark.unit
def test_batch_preprocessor_selects_all_patient_folders(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()
    (input_dir / '.hidden').mkdir()
    (input_dir / 'case_a').mkdir()
    (input_dir / 'case_b').mkdir()

    assert _nifti_preprocessor(input_dir, output_dir).plan() == ['case_a', 'case_b']


@pytest.mark.unit
def test_batch_preprocessor_selects_explicit_patient_folders(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()

    preprocessor = _nifti_preprocessor(input_dir, output_dir, patient_folders=['case_b', 'case_a'])

    assert preprocessor.plan() == ['case_b', 'case_a']


@pytest.mark.unit
def test_batch_preprocessor_selects_numeric_folder_range(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()
    for folder in ['1', '2', '3', 'case_a']:
        (input_dir / folder).mkdir()

    preprocessor = _nifti_preprocessor(input_dir, output_dir, start_folder=2, stop_folder=3)

    assert preprocessor.plan() == ['2', '3']


@pytest.mark.unit
def test_nifti_convert_only_writes_image_and_masks(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a', masks=['mask'])

    result = _nifti_preprocessor(input_dir, output_dir).run()

    case_result = result.case_results[0]
    assert result.processed_count == 1
    assert case_result.status == 'processed'
    assert case_result.image_output_path == output_dir / 'case_a' / 'image.nii.gz'
    assert case_result.mask_output_paths == {'mask': output_dir / 'case_a' / 'mask.nii.gz'}
    assert (output_dir / 'case_a' / 'image.nii.gz').exists()
    assert (output_dir / 'case_a' / 'mask.nii.gz').exists()


@pytest.mark.unit
def test_nifti_resampling_writes_outputs(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a', masks=['mask'])

    result = _nifti_preprocessor(
        input_dir,
        output_dir,
        just_save_as_nifti=False,
        resample_resolution=1.0,
        resample_dimension='3D',
        image_interpolation_method='linear',
        mask_interpolation_method='NN',
    ).run()

    case_result = result.case_results[0]
    assert case_result.status == 'processed'
    assert (output_dir / 'case_a' / 'image.nii.gz').exists()
    assert (output_dir / 'case_a' / 'mask.nii.gz').exists()


@pytest.mark.unit
def test_missing_nifti_mask_is_recorded_as_skipped_structure(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a')

    result = _nifti_preprocessor(input_dir, output_dir, structures=['missing']).run()

    case_result = result.case_results[0]
    assert case_result.status == 'processed'
    assert case_result.processed_structures == []
    assert case_result.skipped_structures == ['missing']


@pytest.mark.unit
def test_nifti_mask_union_is_written_when_enabled(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a', masks=['mask_a', 'mask_b'])

    result = _nifti_preprocessor(
        input_dir,
        output_dir,
        structures=['mask_a', 'mask_b'],
        mask_union=True,
    ).run()

    case_result = result.case_results[0]
    assert case_result.mask_union_output_path == output_dir / 'case_a' / 'mask_union.nii.gz'
    assert (output_dir / 'case_a' / 'mask_union.nii.gz').exists()


@pytest.mark.unit
def test_dicom_preprocessor_uses_explicit_structures(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    (input_dir / 'case_a').mkdir(parents=True)

    monkeypatch.setattr(batch_preprocessing.Image, 'from_dicom', staticmethod(lambda *args, **kwargs: _make_image()))
    monkeypatch.setattr(
        batch_preprocessing.Image,
        'from_dicom_mask',
        staticmethod(lambda *args, **kwargs: _make_image()),
    )
    monkeypatch.setattr(
        batch_preprocessing,
        'get_dicom_files',
        lambda *args, **kwargs: [{'file_path': str(input_dir / 'case_a' / 'rtstruct.dcm')}],
    )

    result = BatchPreprocessor(
        input_directory=input_dir,
        output_directory=output_dir,
        input_data_type='dicom',
        modality='CT',
        structures=['GTV'],
        just_save_as_nifti=True,
    ).run()

    case_result = result.case_results[0]
    assert case_result.status == 'processed'
    assert case_result.processed_structures == ['GTV']
    assert (output_dir / 'case_a' / 'GTV.nii.gz').exists()


@pytest.mark.unit
def test_dicom_preprocessor_resolves_all_structures(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    (input_dir / 'case_a').mkdir(parents=True)

    monkeypatch.setattr(batch_preprocessing.Image, 'from_dicom', staticmethod(lambda *args, **kwargs: _make_image()))
    monkeypatch.setattr(
        batch_preprocessing.Image,
        'from_dicom_mask',
        staticmethod(lambda *args, **kwargs: _make_image()),
    )
    monkeypatch.setattr(
        batch_preprocessing,
        'get_dicom_files',
        lambda *args, **kwargs: [{'file_path': str(input_dir / 'case_a' / 'rtstruct.dcm')}],
    )
    monkeypatch.setattr(batch_preprocessing, 'get_all_structure_names', lambda _path: ['GTV', 'CTV'])

    result = BatchPreprocessor(
        input_directory=input_dir,
        output_directory=output_dir,
        input_data_type='dicom',
        modality='CT',
        use_all_structures=True,
        just_save_as_nifti=True,
    ).run()

    case_result = result.case_results[0]
    assert case_result.processed_structures == ['GTV', 'CTV']


@pytest.mark.unit
def test_batch_continues_after_case_level_failure(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_good', masks=['mask'])
    (input_dir / 'case_bad').mkdir()

    result = _nifti_preprocessor(
        input_dir,
        output_dir,
        patient_folders=['case_good', 'case_bad'],
    ).run()

    assert result.total_count == 2
    assert result.processed_count == 1
    assert result.skipped_count == 1
    assert [case.case_name for case in result.errors] == ['case_bad']
