import numpy as np
import pytest

import zrad.batch as batch
import zrad.batch.preprocessing as batch_preprocessing
from zrad.batch import BatchPreprocessor, BatchResult, PreprocessingCaseResult
from zrad.batch._utils import find_nifti_file
from zrad.exceptions import InvalidInputParametersError
from zrad.gui.prep_tab import create_batch_preprocessor_from_input_params
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


def _write_case(input_dir, case_name, image_name='image', masks=None, mask_arrays=None):
    case_dir = input_dir / case_name
    case_dir.mkdir(parents=True)
    _make_image().save_as_nifti(case_dir / f'{image_name}.nii.gz')
    for mask_name in masks or []:
        array = (mask_arrays or {}).get(mask_name)
        _make_image(array=array).save_as_nifti(case_dir / f'{mask_name}.nii.gz')
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
def test_find_nifti_file_prefers_nii_gz_for_extensionless_stem(tmp_path):
    case_dir = tmp_path / 'case_a'
    case_dir.mkdir()
    nii_path = case_dir / 'image.nii'
    nii_gz_path = case_dir / 'image.nii.gz'
    nii_path.touch()
    nii_gz_path.touch()

    assert find_nifti_file(case_dir, 'image') == nii_gz_path


@pytest.mark.unit
def test_find_nifti_file_returns_nii_for_extensionless_stem_when_nii_gz_is_missing(tmp_path):
    case_dir = tmp_path / 'case_a'
    case_dir.mkdir()
    nii_path = case_dir / 'image.nii'
    nii_path.touch()

    assert find_nifti_file(case_dir, 'image') == nii_path


@pytest.mark.unit
def test_find_nifti_file_returns_explicit_nii_gz_path(tmp_path):
    case_dir = tmp_path / 'case_a'
    case_dir.mkdir()
    nii_gz_path = case_dir / 'image.nii.gz'
    nii_gz_path.touch()

    assert find_nifti_file(case_dir, 'image.nii.gz') == nii_gz_path


@pytest.mark.unit
def test_find_nifti_file_returns_explicit_nii_path(tmp_path):
    case_dir = tmp_path / 'case_a'
    case_dir.mkdir()
    nii_path = case_dir / 'image.nii'
    nii_path.touch()

    assert find_nifti_file(case_dir, 'image.nii') == nii_path


@pytest.mark.unit
def test_find_nifti_file_returns_none_for_missing_file(tmp_path):
    case_dir = tmp_path / 'case_a'
    case_dir.mkdir()

    assert find_nifti_file(case_dir, 'image') is None


@pytest.mark.unit
def test_batch_public_api_exposes_preprocessing_classes():
    assert set(batch.__all__) == {
        'BatchFilter',
        'BatchPreprocessor',
        'BatchRadiomicsExtractor',
        'BatchResult',
        'FilteringCaseResult',
        'PreprocessingCaseResult',
        'RadiomicsCaseResult',
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
def test_batch_preprocessor_validate_normalizes_public_attributes(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()

    preprocessor = BatchPreprocessor(
        input_directory=str(input_dir),
        output_directory=str(output_dir),
        input_data_type=' NIfTI ',
        modality='ct',
        number_of_threads='2',
        patient_folders='case_a, case_b',
        structures='mask_a, mask_b',
        nifti_image_name=' image ',
        just_save_as_nifti=True,
        parallel_backend=' Threads ',
    )

    preprocessor.validate()

    assert preprocessor.input_directory == input_dir
    assert preprocessor.output_directory == output_dir
    assert preprocessor.input_data_type == 'nifti'
    assert preprocessor.modality == 'CT'
    assert preprocessor.number_of_threads == 2
    assert preprocessor.patient_folders == ['case_a', 'case_b']
    assert preprocessor.structures == ['mask_a', 'mask_b']
    assert preprocessor.nifti_image_name == 'image'
    assert preprocessor.parallel_backend == 'threads'


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
def test_batch_preprocessor_plan_then_run_is_safe_after_normalization(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a', masks=['mask'])

    preprocessor = _nifti_preprocessor(
        str(input_dir),
        str(output_dir),
        patient_folders='case_a',
        structures='mask',
    )

    assert preprocessor.plan() == ['case_a']
    result = preprocessor.run()

    assert result.processed_count == 1
    assert (output_dir / 'case_a' / 'image.nii.gz').exists()


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
def test_nifti_convert_only_prefers_nii_gz_when_stem_matches_multiple_files(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    case_dir = input_dir / 'case_a'
    case_dir.mkdir(parents=True)
    (case_dir / 'image.nii').touch()
    (case_dir / 'image.nii.gz').touch()

    loaded_paths = []

    def load_nifti(path):
        loaded_paths.append(path)
        return _make_image()

    monkeypatch.setattr(batch_preprocessing.Image, 'from_nifti', staticmethod(load_nifti))

    result = _nifti_preprocessor(input_dir, output_dir, structures=None).run()

    assert result.processed_count == 1
    assert loaded_paths == [case_dir / 'image.nii.gz']
    assert (output_dir / 'case_a' / 'image.nii.gz').exists()


@pytest.mark.unit
def test_nifti_image_only_case_is_processed(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a')

    result = _nifti_preprocessor(input_dir, output_dir, structures=None).run()

    case_result = result.case_results[0]
    assert case_result.status == 'processed'
    assert case_result.processed_structures == []
    assert case_result.skipped_structures == []
    assert (output_dir / 'case_a' / 'image.nii.gz').exists()


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
def test_missing_nifti_image_is_recorded_as_skipped_case(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    (input_dir / 'case_a').mkdir(parents=True)

    result = _nifti_preprocessor(input_dir, output_dir).run()

    case_result = result.case_results[0]
    assert case_result.status == 'skipped'
    assert case_result.error
    assert result.skipped_count == 1


@pytest.mark.unit
def test_unexpected_processing_exception_is_recorded_as_failed_case(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a')

    def fail_resampling(_self, _image):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(BatchPreprocessor, '_resample_image', fail_resampling)

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
    assert case_result.status == 'failed'
    assert case_result.error == "unexpected failure"
    assert result.failed_count == 1


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
def test_empty_mask_is_recorded_as_skipped_structure(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a')

    monkeypatch.setattr(BatchPreprocessor, '_load_mask', lambda *args, **kwargs: Image())

    result = _nifti_preprocessor(input_dir, output_dir, structures=['empty']).run()

    case_result = result.case_results[0]
    assert case_result.status == 'processed'
    assert case_result.processed_structures == []
    assert case_result.skipped_structures == ['empty']


@pytest.mark.unit
@pytest.mark.parametrize(
    "mask_arrays",
    [
        None,
        {
            'mask_a': np.array([[[0.0, 0.8], [1.2, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
            'mask_b': np.array([[[0.0, 0.0], [0.5, 1.0]], [[0.0, 1.0], [0.0, 0.0]]]),
        },
    ],
)
def test_nifti_mask_union_is_written_when_enabled(tmp_path, mask_arrays):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a', masks=['mask_a', 'mask_b'], mask_arrays=mask_arrays)

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
def test_sequential_progress_callback_reports_total_case_count(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a')
    _write_case(input_dir, 'case_b')
    progress_steps = []

    _nifti_preprocessor(input_dir, output_dir, structures=None).run(progress_callback=progress_steps.append)

    assert sum(progress_steps) == 2


@pytest.mark.unit
def test_parallel_progress_callback_reports_total_case_count(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a')
    _write_case(input_dir, 'case_b')
    progress_steps = []

    _nifti_preprocessor(
        input_dir,
        output_dir,
        structures=None,
        number_of_threads=2,
        parallel_backend='threads',
    ).run(progress_callback=progress_steps.append)

    assert sum(progress_steps) == 2


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


def _gui_input_params(**kwargs):
    params = {
        'input_directory': '/input',
        'output_directory': '/output',
        'input_data_type': 'dicom',
        'input_imaging_modality': 'CT',
        'number_of_threads': 4,
        'list_of_patient_folders': ['case_a'],
        'start_folder': None,
        'stop_folder': None,
        'dicom_structures': ['GTV'],
        'nifti_structures': ['mask'],
        'nifti_image_name': 'image',
        'just_save_as_nifti': False,
        'resample_resolution': 1.0,
        'resample_dimension': '3D',
        'image_interpolation_method': 'linear',
        'mask_interpolation_method': 'NN',
        'mask_interpolation_threshold': 0.5,
        'use_all_structures': False,
        'mask_union': False,
    }
    params.update(kwargs)
    return params


@pytest.mark.unit
def test_gui_mapping_uses_explicit_dicom_structures():
    preprocessor = create_batch_preprocessor_from_input_params(_gui_input_params(), parallel_backend='processes')

    assert preprocessor.input_data_type == 'dicom'
    assert preprocessor.structures == ['GTV']
    assert preprocessor.use_all_structures is False
    assert preprocessor.parallel_backend == 'processes'


@pytest.mark.unit
def test_gui_mapping_uses_no_explicit_structures_when_all_dicom_structures_selected():
    preprocessor = create_batch_preprocessor_from_input_params(
        _gui_input_params(use_all_structures=True),
        parallel_backend='threads',
    )

    assert preprocessor.structures is None
    assert preprocessor.use_all_structures is True
    assert preprocessor.parallel_backend == 'threads'


@pytest.mark.unit
def test_gui_mapping_uses_nifti_mask_names():
    preprocessor = create_batch_preprocessor_from_input_params(
        _gui_input_params(input_data_type='nifti', nifti_structures=['mask_a', 'mask_b']),
        parallel_backend='processes',
    )

    assert preprocessor.input_data_type == 'nifti'
    assert preprocessor.structures == ['mask_a', 'mask_b']
    assert preprocessor.nifti_image_name == 'image'


@pytest.mark.unit
def test_gui_mapping_ignores_stale_all_structures_for_nifti(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()

    preprocessor = create_batch_preprocessor_from_input_params(
        _gui_input_params(
            input_directory=str(input_dir),
            output_directory=str(output_dir),
            input_data_type='nifti',
            nifti_structures=['mask_a', 'mask_b'],
            use_all_structures=True,
            just_save_as_nifti=True,
        ),
        parallel_backend='processes',
    )

    assert preprocessor.use_all_structures is False
    assert preprocessor.structures == ['mask_a', 'mask_b']
    preprocessor.validate()
