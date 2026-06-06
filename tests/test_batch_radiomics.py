import csv

import numpy as np
import pytest

import zrad.batch as batch
import zrad.batch.radiomics as batch_radiomics
from zrad.batch import BatchRadiomicsExtractor, BatchResult, RadiomicsCaseResult
from zrad.exceptions import DataStructureError, InvalidInputParametersError
from zrad.gui.rad_tab import create_batch_radiomics_extractor_from_input_params
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


def _write_case(input_dir, case_name, image_name='image', masks=None, filtered_name=None):
    case_dir = input_dir / case_name
    case_dir.mkdir(parents=True)
    _make_image(np.arange(1, 28, dtype=np.float64).reshape(3, 3, 3)).save_as_nifti(
        case_dir / f'{image_name}.nii.gz'
    )
    if filtered_name:
        _make_image(np.full((3, 3, 3), 2.0)).save_as_nifti(case_dir / f'{filtered_name}.nii.gz')
    for mask_name in masks or []:
        _make_image(np.ones((3, 3, 3), dtype=np.float64)).save_as_nifti(case_dir / f'{mask_name}.nii.gz')
    return case_dir


def _extractor(input_dir, output_dir, **kwargs):
    params = {
        'input_directory': input_dir,
        'output_directory': output_dir,
        'input_data_type': 'nifti',
        'modality': 'CT',
        'nifti_image_name': 'image',
        'number_of_threads': 1,
        'structures': ['mask'],
        'aggregation_dimension': '3D',
        'aggregation_method': 'MERG',
        'discretization_method': 'Number of Bins',
        'number_of_bins': 4,
    }
    params.update(kwargs)
    return BatchRadiomicsExtractor(**params)


@pytest.mark.unit
def test_batch_public_api_exposes_radiomics_classes():
    assert BatchRadiomicsExtractor is batch.BatchRadiomicsExtractor
    assert RadiomicsCaseResult is batch.RadiomicsCaseResult
    assert BatchResult is batch.BatchResult


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({'input_data_type': 'unsupported'}, "input_data_type"),
        ({'number_of_threads': 0}, "number_of_threads"),
        ({'nifti_image_name': None}, "nifti_image_name"),
        ({'structures': None}, "structures"),
        ({'use_all_structures': True}, "use_all_structures"),
        ({'discretization_method': 'Number of Bins', 'number_of_bins': None}, "number_of_bins"),
        ({'discretization_method': 'Bin Size', 'bin_size': None}, "bin_size"),
        ({'discretization_method': 'Bin Size', 'bin_size': 25.0, 'intensity_range': None}, "intensity_range"),
        (
            {'discretization_method': 'Bin Size', 'bin_size': 25.0, 'intensity_range': [float('-inf'), 100.0]},
            "intensity_range",
        ),
        ({'aggregation_method': 'BAD'}, "aggregation_method"),
        ({'slice_weighting': True, 'slice_median': True}, "slice_weighting"),
    ],
)
def test_batch_radiomics_validates_inputs(tmp_path, kwargs, message):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()

    extractor = _extractor(input_dir, output_dir, **kwargs)

    with pytest.raises(InvalidInputParametersError, match=message):
        extractor.validate()


@pytest.mark.unit
def test_batch_radiomics_validate_normalizes_public_attributes(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()

    extractor = BatchRadiomicsExtractor(
        input_directory=str(input_dir),
        output_directory=str(output_dir),
        input_data_type=' NIfTI ',
        modality='ct',
        number_of_threads='2',
        patient_folders='case_a, case_b',
        structures='mask_a, mask_b',
        nifti_image_name=' image ',
        aggregation_dimension='3d',
        aggregation_method='merg',
        discretization_method='Number of Bins',
        number_of_bins='8',
        intensity_range='0, 100',
        parallel_backend=' Threads ',
    )

    extractor.validate()

    assert extractor.input_directory == input_dir
    assert extractor.output_directory == output_dir
    assert extractor.input_data_type == 'nifti'
    assert extractor.modality == 'CT'
    assert extractor.number_of_threads == 2
    assert extractor.patient_folders == ['case_a', 'case_b']
    assert extractor.structures == ['mask_a', 'mask_b']
    assert extractor.nifti_image_name == 'image'
    assert extractor.aggregation_dimension == '3D'
    assert extractor.aggregation_method == 'MERG'
    assert extractor.number_of_bins == 8
    assert extractor.intensity_range == (0.0, 100.0)
    assert extractor.parallel_backend == 'threads'


@pytest.mark.unit
def test_batch_radiomics_plan_then_run_is_safe_after_normalization(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a', masks=['mask'])
    monkeypatch.setattr(BatchRadiomicsExtractor, '_extract_structure_features', lambda *args, **kwargs: {'f': 1})

    extractor = _extractor(
        str(input_dir),
        str(output_dir),
        patient_folders='case_a',
        structures='mask',
        number_of_threads='1',
    )

    assert extractor.plan() == ['case_a']
    result = extractor.run()

    assert result.processed_count == 1
    assert (output_dir / 'radiomics.csv').exists()


@pytest.mark.unit
def test_batch_radiomics_selects_all_patient_folders(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()
    (input_dir / '.hidden').mkdir()
    (input_dir / 'case_a').mkdir()
    (input_dir / 'case_b').mkdir()

    assert _extractor(input_dir, output_dir).plan() == ['case_a', 'case_b']


@pytest.mark.unit
def test_batch_radiomics_selects_explicit_patient_folders(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()

    extractor = _extractor(input_dir, output_dir, patient_folders=['case_b', 'case_a'])

    assert extractor.plan() == ['case_b', 'case_a']


@pytest.mark.unit
def test_batch_radiomics_selects_numeric_folder_range(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()
    for folder in ['1', '2', '3', 'case_a']:
        (input_dir / folder).mkdir()

    extractor = _extractor(input_dir, output_dir, start_folder=2, stop_folder=3)

    assert extractor.plan() == ['2', '3']


@pytest.mark.unit
def test_nifti_radiomics_writes_csv(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a', masks=['mask'])
    monkeypatch.setattr(
        BatchRadiomicsExtractor,
        '_extract_structure_features',
        lambda *args, **kwargs: {'stat_mean': 1.5},
    )

    result = _extractor(input_dir, output_dir).run()

    csv_path = output_dir / 'radiomics.csv'
    assert result.workflow == 'radiomics'
    assert result.processed_count == 1
    assert result.case_results[0].processed_structures == ['mask']
    with open(csv_path, newline='') as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows == [{'pat_id': 'case_a', 'mask_id': 'mask', 'stat_mean': '1.5'}]


@pytest.mark.unit
def test_filtered_nifti_image_is_loaded_when_provided(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    (input_dir / 'case_a').mkdir(parents=True)
    loaded_names = []

    def load_image(self, _case_dir, nifti_name=None):
        loaded_names.append(nifti_name)
        return _make_image()

    monkeypatch.setattr(BatchRadiomicsExtractor, '_load_image', load_image)
    monkeypatch.setattr(BatchRadiomicsExtractor, '_load_mask', lambda *args, **kwargs: _make_image())
    monkeypatch.setattr(BatchRadiomicsExtractor, '_extract_structure_features', lambda *args, **kwargs: {'f': 1})

    _extractor(input_dir, output_dir, nifti_filtered_image_name='filtered').run()

    assert loaded_names == ['image', 'filtered']


@pytest.mark.unit
def test_missing_filtered_nifti_image_is_recorded_as_skipped_case(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a', masks=['mask'])

    result = _extractor(input_dir, output_dir, nifti_filtered_image_name='filtered').run()

    case_result = result.case_results[0]
    assert case_result.status == 'skipped'
    assert case_result.error
    assert result.skipped_count == 1


@pytest.mark.unit
def test_missing_nifti_image_is_recorded_as_skipped_case(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    (input_dir / 'case_a').mkdir(parents=True)

    result = _extractor(input_dir, output_dir).run()

    case_result = result.case_results[0]
    assert case_result.status == 'skipped'
    assert case_result.error
    assert result.skipped_count == 1


@pytest.mark.unit
def test_missing_nifti_mask_is_recorded_as_skipped_structure(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a')

    result = _extractor(input_dir, output_dir).run()

    case_result = result.case_results[0]
    assert case_result.status == 'skipped'
    assert case_result.skipped_structures == ['mask']
    assert case_result.error == "No structures were successfully processed for radiomics extraction."


@pytest.mark.unit
def test_empty_nifti_mask_is_recorded_as_skipped_structure(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a')
    _make_image(np.zeros((3, 3, 3), dtype=np.float64)).save_as_nifti(input_dir / 'case_a' / 'mask.nii.gz')

    result = _extractor(input_dir, output_dir).run()

    case_result = result.case_results[0]
    assert case_result.status == 'skipped'
    assert case_result.skipped_structures == ['mask']
    assert case_result.error == "No structures were successfully processed for radiomics extraction."


@pytest.mark.unit
def test_empty_radiomics_csv_is_created_when_no_feature_rows_are_extracted(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a')

    result = _extractor(input_dir, output_dir).run()

    csv_path = output_dir / 'radiomics.csv'
    assert result.skipped_count == 1
    assert csv_path.exists()
    assert csv_path.read_text() == ''


@pytest.mark.unit
def test_per_structure_failure_skips_only_that_structure(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a', masks=['good', 'bad'])

    def load_mask(self, case_dir, structure_name, image, rtstruct_path):
        if structure_name == 'bad':
            raise DataStructureError("bad mask")
        return _make_image()

    monkeypatch.setattr(BatchRadiomicsExtractor, '_load_mask', load_mask)
    monkeypatch.setattr(BatchRadiomicsExtractor, '_extract_structure_features', lambda *args, **kwargs: {'f': 1})

    result = _extractor(input_dir, output_dir, structures=['good', 'bad']).run()

    case_result = result.case_results[0]
    assert case_result.status == 'processed'
    assert case_result.processed_structures == ['good']
    assert case_result.skipped_structures == ['bad']


@pytest.mark.unit
def test_dicom_radiomics_uses_explicit_structures(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    (input_dir / 'case_a').mkdir(parents=True)
    monkeypatch.setattr(batch_radiomics.Image, 'from_dicom', staticmethod(lambda *args, **kwargs: _make_image()))
    monkeypatch.setattr(batch_radiomics.Image, 'from_dicom_mask', staticmethod(lambda *args, **kwargs: _make_image()))
    monkeypatch.setattr(
        batch_radiomics,
        'get_dicom_files',
        lambda *args, **kwargs: [{'file_path': '/tmp/rtstruct.dcm'}],
    )
    monkeypatch.setattr(BatchRadiomicsExtractor, '_extract_structure_features', lambda *args, **kwargs: {'f': 1})

    result = BatchRadiomicsExtractor(
        input_directory=input_dir,
        output_directory=output_dir,
        input_data_type='dicom',
        modality='CT',
        structures=['GTV'],
        aggregation_dimension='3D',
        aggregation_method='MERG',
        discretization_method='Number of Bins',
        number_of_bins=4,
    ).run()

    case_result = result.case_results[0]
    assert case_result.status == 'processed'
    assert case_result.processed_structures == ['GTV']


@pytest.mark.unit
def test_dicom_radiomics_uses_all_structures(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    (input_dir / 'case_a').mkdir(parents=True)
    monkeypatch.setattr(batch_radiomics.Image, 'from_dicom', staticmethod(lambda *args, **kwargs: _make_image()))
    monkeypatch.setattr(batch_radiomics.Image, 'from_dicom_mask', staticmethod(lambda *args, **kwargs: _make_image()))
    monkeypatch.setattr(
        batch_radiomics,
        'get_dicom_files',
        lambda *args, **kwargs: [{'file_path': '/tmp/rtstruct.dcm'}],
    )
    monkeypatch.setattr(batch_radiomics, 'get_all_structure_names', lambda _rtstruct_path: ['GTV', 'CTV'])
    monkeypatch.setattr(BatchRadiomicsExtractor, '_extract_structure_features', lambda *args, **kwargs: {'f': 1})

    result = BatchRadiomicsExtractor(
        input_directory=input_dir,
        output_directory=output_dir,
        input_data_type='dicom',
        modality='CT',
        use_all_structures=True,
        aggregation_dimension='3D',
        aggregation_method='MERG',
        discretization_method='Number of Bins',
        number_of_bins=4,
    ).run()

    assert result.case_results[0].processed_structures == ['GTV', 'CTV']


@pytest.mark.unit
def test_dicom_missing_rtstruct_is_recorded_as_skipped_case(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    (input_dir / 'case_a').mkdir(parents=True)
    monkeypatch.setattr(batch_radiomics.Image, 'from_dicom', staticmethod(lambda *args, **kwargs: _make_image()))
    monkeypatch.setattr(batch_radiomics, 'get_dicom_files', lambda *args, **kwargs: [])

    result = BatchRadiomicsExtractor(
        input_directory=input_dir,
        output_directory=output_dir,
        input_data_type='dicom',
        modality='CT',
        structures=['GTV'],
        aggregation_dimension='3D',
        aggregation_method='MERG',
        discretization_method='Number of Bins',
        number_of_bins=4,
    ).run()

    case_result = result.case_results[0]
    assert case_result.status == 'skipped'
    assert case_result.skipped_structures == ['GTV']
    assert case_result.error == "No structures were successfully processed for radiomics extraction."


@pytest.mark.unit
def test_dicom_all_structures_without_rtstruct_is_recorded_as_skipped_case(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    (input_dir / 'case_a').mkdir(parents=True)
    monkeypatch.setattr(batch_radiomics.Image, 'from_dicom', staticmethod(lambda *args, **kwargs: _make_image()))
    monkeypatch.setattr(batch_radiomics, 'get_dicom_files', lambda *args, **kwargs: [])

    result = BatchRadiomicsExtractor(
        input_directory=input_dir,
        output_directory=output_dir,
        input_data_type='dicom',
        modality='CT',
        use_all_structures=True,
        aggregation_dimension='3D',
        aggregation_method='MERG',
        discretization_method='Number of Bins',
        number_of_bins=4,
    ).run()

    case_result = result.case_results[0]
    assert case_result.status == 'skipped'
    assert case_result.skipped_structures == []
    assert case_result.error == "No structures were available for radiomics extraction."


@pytest.mark.unit
def test_batch_continues_after_failed_case(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a', masks=['mask'])
    _write_case(input_dir, 'case_b', masks=['mask'])

    def fail_one_case(self, case_dir):
        if case_dir.name == 'case_b':
            raise RuntimeError("case failed")
        return ['mask'], None

    monkeypatch.setattr(BatchRadiomicsExtractor, '_resolve_structures', fail_one_case)
    monkeypatch.setattr(BatchRadiomicsExtractor, '_extract_structure_features', lambda *args, **kwargs: {'f': 1})

    result = _extractor(input_dir, output_dir).run()

    assert result.processed_count == 1
    assert result.failed_count == 1
    assert len(result.errors) == 1
    assert result.errors[0].case_name == 'case_b'


@pytest.mark.unit
def test_sequential_progress_callback_reports_total_case_count(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a', masks=['mask'])
    _write_case(input_dir, 'case_b', masks=['mask'])
    monkeypatch.setattr(BatchRadiomicsExtractor, '_extract_structure_features', lambda *args, **kwargs: {'f': 1})
    progress_steps = []

    _extractor(input_dir, output_dir).run(progress_callback=progress_steps.append)

    assert sum(progress_steps) == 2


@pytest.mark.unit
def test_parallel_progress_callback_reports_total_case_count(monkeypatch, tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a', masks=['mask'])
    _write_case(input_dir, 'case_b', masks=['mask'])
    monkeypatch.setattr(BatchRadiomicsExtractor, '_extract_structure_features', lambda *args, **kwargs: {'f': 1})
    progress_steps = []

    _extractor(
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
        'input_imaging_modality': 'CT',
        'number_of_threads': 4,
        'list_of_patient_folders': ['case_a'],
        'start_folder': None,
        'stop_folder': None,
        'nifti_structures': ['mask'],
        'dicom_structures': ['GTV'],
        'nifti_image_name': 'image',
        'nifti_filtered_image_name': None,
        'use_all_structures': False,
        'aggregation_method': ('2D', 'AVER'),
        'weighting': 'Weighted Mean',
        'discretization': ('Number of Bins', 8, None),
        'intensity_range': [0.0, 100.0],
        'outlier_range': 3.0,
    }
    params.update(kwargs)
    return params


@pytest.mark.unit
def test_gui_mapping_creates_nifti_batch_radiomics_extractor():
    extractor = create_batch_radiomics_extractor_from_input_params(
        _gui_input_params(),
        parallel_backend='threads',
    )

    assert extractor.input_data_type == 'nifti'
    assert extractor.structures == ['mask']
    assert extractor.nifti_image_name == 'image'
    assert extractor.slice_weighting is True
    assert extractor.slice_median is False
    assert extractor.parallel_backend == 'threads'


@pytest.mark.unit
def test_gui_mapping_creates_dicom_batch_radiomics_extractor_with_explicit_structures():
    extractor = create_batch_radiomics_extractor_from_input_params(
        _gui_input_params(input_data_type='dicom', weighting='Mean'),
        parallel_backend='processes',
    )

    assert extractor.input_data_type == 'dicom'
    assert extractor.structures == ['GTV']
    assert extractor.use_all_structures is False
    assert extractor.slice_weighting is False


@pytest.mark.unit
def test_gui_mapping_creates_dicom_batch_radiomics_extractor_with_all_structures():
    extractor = create_batch_radiomics_extractor_from_input_params(
        _gui_input_params(input_data_type='dicom', use_all_structures=True),
        parallel_backend='threads',
    )

    assert extractor.input_data_type == 'dicom'
    assert extractor.structures is None
    assert extractor.use_all_structures is True
