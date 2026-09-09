import csv

import numpy as np
import pytest
from conftest import _prepare_data_dir
from ibsi_helpers import load_references, select_ibsi_i_references
from test_ibsi_1 import ibsi_i_feature_tolerances, ibsi_i_validation
from test_ibsi_2 import ibsi_ii_ph_i_validation, ibsi_ii_ph_ii_validation

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('problem', ['empty', 'duplicate', 'blank', 'nan', 'negative'])
def test_invalid_reference_table_is_rejected(tmp_path, problem):
    path = tmp_path / 'reference.csv'
    rows = [['1.A', 'stat_mean', '1', '0']]
    if problem == 'empty':
        rows = []
    elif problem == 'duplicate':
        rows *= 2
    elif problem == 'blank':
        rows[0][2] = ''
    elif problem == 'nan':
        rows[0][2] = 'nan'
    else:
        rows[0][3] = '-1'
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter=';')
        writer.writerow(['filter_id', 'feature_tag', 'consensus_value', 'tolerance'])
        writer.writerows(rows)
    with pytest.raises(ValueError):
        load_references(path, 'feature_tag', 'consensus_value', delimiter=';', phase='II', config='1.A')


def test_unknown_configuration_is_rejected():
    from test_ibsi_2 import ibsi_ii_feature_tolerances

    with pytest.raises(ValueError, match='Empty'):
        ibsi_ii_feature_tolerances('typo')


@pytest.mark.parametrize('validate', [ibsi_i_validation, ibsi_ii_ph_ii_validation])
def test_empty_comparison_is_rejected(validate):
    with pytest.raises(AssertionError, match='Empty'):
        validate({}, {})


def test_aggregation_selection_does_not_depend_on_results():
    reference = select_ibsi_i_references(ibsi_i_feature_tolerances('config_A'), '2D', 'AVER')
    assert 'cm_joint_max_2D_avg' in reference
    assert 'cm_joint_max_2D_comb' not in reference
    assert 'img_dim_x_init_img' not in reference
    features = {tag: float(row['reference value']) for tag, row in reference.items()}
    del features['cm_joint_max_2D_avg']
    with pytest.raises(pytest.fail.Exception, match='Missing required feature cm_joint_max_2D_avg'):
        ibsi_i_validation(reference, features, config_a=True)


@pytest.mark.parametrize('actual', [np.zeros((1, 2)), np.array([np.nan, 0]), np.array([np.inf, 0])])
def test_response_map_rejects_invalid_arrays(actual):
    with pytest.raises(AssertionError):
        ibsi_ii_ph_i_validation(actual, np.zeros(2), 'example')


def test_response_map_error_includes_diagnostics():
    with pytest.raises(pytest.fail.Exception, match='maximum error=.*tolerance='):
        ibsi_ii_ph_i_validation(np.array([0.0, 2.0]), np.array([0.0, 1.0]), 'example')


@pytest.mark.parametrize('damage', ['missing', 'corrupt', 'new_archive', 'partial'])
def test_extraction_repairs_incomplete_or_stale_data(tmp_path, damage):
    import zipfile

    archive = tmp_path / 'source.zip'
    output = tmp_path / 'output'

    def write_archive(value):
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr('source/image.dat', value)

    write_archive('original')
    _prepare_data_dir(archive, output)
    target = output / 'image.dat'
    expected = 'original'
    if damage == 'missing':
        target.unlink()
    elif damage == 'corrupt':
        target.write_text('corrupt!')
    elif damage == 'partial':
        target.unlink()
        (output / '.extraction_finished.flag').unlink()
    else:
        expected = 'revised!'
        write_archive(expected)
    _prepare_data_dir(archive, output)
    assert target.read_text() == expected


@pytest.mark.parametrize(
    ('config', 'count'),
    [
        ('config_A', 411),
        ('config_B', 411),
        ('config_C', 275),
        ('config_D', 275),
        ('config_E', 275),
        ('digital_phantom', 487),
    ],
)
def test_ibsi_i_reference_inventory(config, count):
    assert len(ibsi_i_feature_tolerances(config)) == count


@pytest.mark.parametrize('config', [f'{number}.{variant}' for number in range(1, 10) for variant in 'AB'])
def test_ibsi_ii_reference_inventory(config):
    from test_ibsi_2 import ibsi_ii_feature_tolerances

    assert len(ibsi_ii_feature_tolerances(config)) == 18


@pytest.mark.parametrize(('value', 'passes'), [(2.1486486, True), (2.16, False), (float('nan'), False)])
def test_digital_phantom_reference_precision(value, passes):
    reference = {'stat_mean': ibsi_i_feature_tolerances('digital_phantom')['stat_mean']}
    if passes:
        ibsi_i_validation(reference, {'stat_mean': value})
    else:
        with pytest.raises(pytest.fail.Exception):
            ibsi_i_validation(reference, {'stat_mean': value})


@pytest.mark.parametrize(('dimension', 'method'), [('typo', 'AVER'), ('3D', 'DIR_MERG'), ('2D', 'MERG')])
def test_invalid_aggregation_cannot_drop_texture_references(dimension, method):
    with pytest.raises(ValueError, match='Invalid IBSI aggregation'):
        select_ibsi_i_references(ibsi_i_feature_tolerances('config_A'), dimension, method)


def test_newly_available_reference_requires_exception_review(tmp_path):
    path = tmp_path / 'reference.csv'
    path.write_text('filter_id;feature_tag;consensus_value;tolerance\n8.B;stat_qcod;1;0.1\n')
    with pytest.raises(ValueError, match='availability changed'):
        load_references(path, 'feature_tag', 'consensus_value', delimiter=';', phase='II', config='8.B')


def test_8b_exception_cannot_hide_available_reference():
    reference = {'stat_qcod': {'filter_id': '8.B', 'consensus_value': '1', 'tolerance': '0'}}
    with pytest.raises(AssertionError, match='availability changed'):
        ibsi_ii_ph_ii_validation(reference, {}, config_8b=True)


def test_8b_exception_cannot_apply_to_another_configuration():
    from test_ibsi_2 import ibsi_ii_feature_tolerances

    reference = {'stat_qcod': ibsi_ii_feature_tolerances('8.A')['stat_qcod']}
    with pytest.raises(AssertionError, match='another configuration'):
        ibsi_ii_ph_ii_validation(reference, {}, config_8b=True)


def test_valid_mode_without_references_cannot_drop_all_textures():
    with pytest.raises(ValueError, match='No texture references'):
        select_ibsi_i_references(ibsi_i_feature_tolerances('config_A'), '3D', 'AVER')
