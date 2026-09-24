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


def _make_irregular_roi():
    values = np.arange(1, 217, dtype=np.float64).reshape(6, 6, 6)
    mask = np.zeros_like(values)
    mask[1:5, 1:4, 1:4] = 1
    mask[4, 3, 3] = 0
    return _make_image(values), _make_image(mask)


def _write_case(input_dir, case_name, image_name='image', masks=None, filtered_name=None):
    case_dir = input_dir / case_name
    case_dir.mkdir(parents=True)
    _make_image(np.arange(1, 28, dtype=np.float64).reshape(3, 3, 3)).save_as_nifti(case_dir / f'{image_name}.nii.gz')
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
        ({'ivh_bin_size': 0.25}, "ivh_method"),
        ({'ivh_method': 'unsupported'}, "ivh_method"),
        ({'ivh_method': 'direct', 'ivh_bin_size': 0.25}, "direct IVH"),
        ({'ivh_method': 'fixed_bin_size'}, "ivh_bin_size"),
        ({'ivh_method': 'fixed_bin_size', 'ivh_bin_size': -1}, "ivh_bin_size"),
        ({'ivh_method': 'fixed_bin_size', 'ivh_bin_size': 0.25, 'ivh_number_of_bins': 10}, "ivh_number_of_bins"),
        ({'ivh_method': 'fixed_bin_number'}, "ivh_number_of_bins"),
        ({'ivh_method': 'fixed_bin_number', 'ivh_number_of_bins': 2.5}, "ivh_number_of_bins"),
        ({'ivh_method': 'fixed_bin_number', 'ivh_number_of_bins': 10, 'ivh_bin_size': 0.25}, "ivh_bin_size"),
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
def test_batch_radiomics_normalizes_custom_ivh_settings(tmp_path):
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    extractor = _extractor(
        input_dir, tmp_path / 'output',
        ivh_method=' FIXED_BIN_NUMBER ', ivh_number_of_bins='128',
    )

    extractor.validate()

    assert extractor.ivh_method == 'fixed_bin_number'
    assert extractor.ivh_number_of_bins == 128


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
def test_empty_radiomics_csv_truncates_stale_content(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    _write_case(input_dir, 'case_a')
    output_dir.mkdir()
    csv_path = output_dir / 'radiomics.csv'
    csv_path.write_text('pat_id,mask_id,old_feature\ncase_old,mask,1\n')

    result = _extractor(input_dir, output_dir).run()

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
    assert extractor.ivh_method is None


@pytest.mark.unit
@pytest.mark.parametrize('modality', ['CT', 'PET'])
def test_gui_filtered_image_uses_1000_ivh_bins_with_original_image_range(tmp_path, modality):
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    extractor = create_batch_radiomics_extractor_from_input_params(
        _gui_input_params(
            input_directory=str(input_dir),
            output_directory=str(tmp_path / 'output'),
            input_imaging_modality=modality,
            nifti_filtered_image_name='filtered',
            intensity_range=[50.0, 150.0],
            outlier_range=None,
            aggregation_method=('3D', 'MERG'),
            weighting='Mean',
        ),
        parallel_backend='threads',
    )
    assert extractor.ivh_method == 'fixed_bin_number'
    assert extractor.ivh_number_of_bins == 1000
    extractor.validate()

    image, mask = _make_irregular_roi()
    filtered_image = _make_image(image.array * 0.25 - 60)
    features = extractor._extract_structure_features(image, filtered_image, mask)
    retained = filtered_image.array[
        (mask.array > 0) & (image.array >= 50) & (image.array <= 150)
    ]
    assert features['stat_min'] == retained.min()
    assert features['stat_max'] == retained.max()
    assert 1 <= features['ivh_i10'] <= 1000
    assert 1 <= features['ivh_i90'] <= 1000


@pytest.mark.unit
def test_gui_mapping_ignores_stale_all_structures_for_nifti(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()

    extractor = create_batch_radiomics_extractor_from_input_params(
        _gui_input_params(
            input_directory=str(input_dir),
            output_directory=str(output_dir),
            input_data_type='nifti',
            nifti_structures=['mask_a', 'mask_b'],
            use_all_structures=True,
        ),
        parallel_backend='threads',
    )

    assert extractor.use_all_structures is False
    assert extractor.structures == ['mask_a', 'mask_b']
    extractor.validate()


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


@pytest.mark.unit
def test_gui_workflow_writes_moran_and_geary_columns(tmp_path):
    input_dir, output_dir = tmp_path / 'input', tmp_path / 'output'
    case_dir = _write_case(input_dir, 'case_a', masks=['mask'])
    values = np.arange(1, 217, dtype=float).reshape(6, 6, 6)
    mask = np.zeros_like(values)
    mask[1:5, 1:4, 1:4] = 1
    mask[4, 3, 3] = 0
    _make_image(values).save_as_nifti(case_dir / 'image.nii.gz')
    _make_image(mask).save_as_nifti(case_dir / 'mask.nii.gz')
    extractor = create_batch_radiomics_extractor_from_input_params(
        _gui_input_params(
            input_directory=str(input_dir),
            output_directory=str(output_dir),
            number_of_threads=1,
            intensity_range=None,
            outlier_range=None,
            aggregation_method=('3D', 'MERG'),
            weighting='Mean',
        ),
        parallel_backend='threads',
    )
    result = extractor.run()
    assert result.processed_count == 1
    with (output_dir / 'radiomics.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert np.isfinite(float(rows[0]['morph_moran_i']))
    assert np.isfinite(float(rows[0]['morph_geary_c']))
    assert np.isfinite(float(rows[0]['ivh_v10']))
    assert np.isfinite(float(rows[0]['ivh_i90']))


@pytest.mark.unit
@pytest.mark.parametrize(
    'modality, method, number_of_bins, bin_size',
    [
        ('CT', 'direct', None, None),
        ('PET', 'fixed_bin_size', None, 0.1),
        ('MRI', 'fixed_bin_number', 1000, None),
        ('MG', 'fixed_bin_number', 1000, None),
        ('US', 'fixed_bin_number', 1000, None),
        ('RTDOSE', 'fixed_bin_size', None, 0.1),
    ],
)
def test_batch_ivh_uses_modality_strategy_and_gui_range(
    monkeypatch, tmp_path, modality, method, number_of_bins, bin_size
):
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    extractor = _extractor(
        input_dir,
        tmp_path / 'output',
        modality=modality,
        intensity_range=(50, 150),
    )
    extractor.validate()
    observed = {}
    original_apply = batch_radiomics.IVHIntensityDiscretizer.apply

    def capture_ivh_preparation(discretizer, roi_data):
        observed.update(discretizer.get_params())
        observed['intensity_range'] = roi_data.intensity_range
        return original_apply(discretizer, roi_data)

    monkeypatch.setattr(batch_radiomics.IVHIntensityDiscretizer, 'apply', capture_ivh_preparation)
    image, mask = _make_irregular_roi()
    features = extractor._extract_structure_features(image, None, mask)

    assert observed == {
        'method': method,
        'number_of_bins': number_of_bins,
        'bin_size': bin_size,
        'intensity_range': (50.0, 150.0),
    }
    retained = image.array[(mask.array > 0) & (image.array >= 50) & (image.array <= 150)]
    assert features['stat_min'] == retained.min()
    assert features['stat_max'] == retained.max()
    assert all(
        np.isfinite(features[name])
        for name in (
            'ivh_v10',
            'ivh_v90',
            'ivh_i10',
            'ivh_i90',
            'ivh_diff_v10_v90',
            'ivh_diff_i10_i90',
        )
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    'modality, method, number_of_bins, bin_size',
    [
        ('CT', 'fixed_bin_size', None, 0.25),
        ('PET', 'fixed_bin_number', 128, None),
        ('MRI', 'direct', None, None),
    ],
)
def test_batch_ivh_custom_strategy_overrides_modality(
    monkeypatch, tmp_path, modality, method, number_of_bins, bin_size
):
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    extractor = _extractor(
        input_dir, tmp_path / 'output',
        modality=modality,
        intensity_range=(50, 150),
        ivh_method=method,
        ivh_number_of_bins=number_of_bins,
        ivh_bin_size=bin_size,
    )
    extractor.validate()
    observed = {}
    original_apply = batch_radiomics.IVHIntensityDiscretizer.apply

    def capture_ivh_preparation(discretizer, roi_data):
        observed.update(discretizer.get_params())
        observed['intensity_range'] = roi_data.intensity_range
        return original_apply(discretizer, roi_data)

    monkeypatch.setattr(batch_radiomics.IVHIntensityDiscretizer, 'apply', capture_ivh_preparation)
    image, mask = _make_irregular_roi()
    features = extractor._extract_structure_features(image, None, mask)

    assert observed == {
        'method': method,
        'number_of_bins': number_of_bins,
        'bin_size': bin_size,
        'intensity_range': (50.0, 150.0),
    }
    assert np.isfinite(features['ivh_i10'])


@pytest.mark.unit
def test_batch_custom_ivh_settings_write_features_to_csv(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    case_dir = input_dir / 'case_a'
    case_dir.mkdir(parents=True)
    image, mask = _make_irregular_roi()
    image.save_as_nifti(case_dir / 'image.nii.gz')
    mask.save_as_nifti(case_dir / 'mask.nii.gz')
    extractor = _extractor(
        input_dir, output_dir,
        ivh_method='fixed_bin_number', ivh_number_of_bins=128,
    )

    result = extractor.run()

    assert result.processed_count == 1
    with (output_dir / 'radiomics.csv').open(newline='') as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 1
    assert np.isfinite(float(rows[0]['ivh_i10']))


@pytest.mark.unit
@pytest.mark.parametrize('modality', ['PET', 'RTDOSE'])
def test_batch_single_ivh_bin_keeps_other_features_in_csv(tmp_path, modality):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    case_dir = input_dir / 'case_a'
    case_dir.mkdir(parents=True)
    image, mask = _make_irregular_roi()
    image.array = np.linspace(1.01, 1.09, image.array.size).reshape(image.array.shape)
    image.save_as_nifti(case_dir / 'image.nii.gz')
    mask.save_as_nifti(case_dir / 'mask.nii.gz')

    result = _extractor(input_dir, output_dir, modality=modality).run()

    assert result.processed_count == 1
    assert result.case_results[0].processed_structures == ['mask']
    assert result.case_results[0].skipped_structures == []
    assert result.case_results[0].feature_count > 100
    with (output_dir / 'radiomics.csv').open(newline='') as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 1
    assert rows[0]['pat_id'] == 'case_a'
    assert rows[0]['mask_id'] == 'mask'
    assert float(rows[0]['stat_max']) > float(rows[0]['stat_min'])
    assert np.isfinite(float(rows[0]['morph_volume']))
    assert not any(name.startswith('ivh_') for name in rows[0])


@pytest.mark.unit
@pytest.mark.parametrize(
    'modality, ivh_options',
    [('PET', {}), ('RTDOSE', {}), ('CT', {'ivh_method': 'fixed_bin_size', 'ivh_bin_size': 0.25})],
)
def test_batch_ivh_fixed_width_uses_observed_minimum_without_gui_range(monkeypatch, tmp_path, modality, ivh_options):
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    extractor = _extractor(input_dir, tmp_path / 'output', modality=modality, **ivh_options)
    extractor.validate()
    observed = {}
    original_apply = batch_radiomics.IVHIntensityDiscretizer.apply

    def capture_ivh_preparation(discretizer, roi_data):
        observed['intensity_range'] = roi_data.intensity_range
        return original_apply(discretizer, roi_data)

    monkeypatch.setattr(batch_radiomics.IVHIntensityDiscretizer, 'apply', capture_ivh_preparation)
    image, mask = _make_irregular_roi()
    features = extractor._extract_structure_features(image, None, mask)

    assert observed['intensity_range'] == (44.0, np.inf)
    assert features['stat_min'] == 44
    assert np.isfinite(features['ivh_i10'])


@pytest.mark.unit
@pytest.mark.parametrize('modality', ['CT', 'PET', 'MRI', 'MG', 'US', 'RTDOSE'])
def test_gui_texture_bin_size_does_not_change_ivh_features(tmp_path, modality):
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    image, mask = _make_irregular_roi()
    ivh_results = []
    texture_bin_counts = []

    for texture_bin_size in (5, 10):
        extractor = _extractor(
            input_dir,
            tmp_path / 'output',
            modality=modality,
            discretization_method='Bin Size',
            number_of_bins=None,
            bin_size=texture_bin_size,
            intensity_range=(50, 150),
        )
        extractor.validate()
        features = extractor._extract_structure_features(image, None, mask)
        ivh_results.append({name: value for name, value in features.items() if name.startswith('ivh_')})
        texture_bin_counts.append(features['no_bins'])

    assert ivh_results[0] == ivh_results[1]
    assert texture_bin_counts[0] != texture_bin_counts[1]
