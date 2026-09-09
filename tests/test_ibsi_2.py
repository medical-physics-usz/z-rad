from pathlib import Path

import numpy as np
import pytest
from ibsi_helpers import load_references, matches_reference

from zrad.filtering import create_filter
from zrad.image import Image
from zrad.preprocessing import (
    ImageResampler,
    IntensityMaskBuilder,
    MaskResampler,
    Resegmenter,
    RoiData,
    TextureDiscretizer,
)
from zrad.radiomics import Radiomics


def _run_ph_i_case(filtering, phantom, filename, config, data_dir):
    filtered_image = filtering.apply(phantom)
    response_map_path = data_dir / 'Ph_I' / 'response_maps' / filename
    response_map = Image.from_nifti(str(response_map_path))
    ibsi_ii_ph_i_validation(filtered_image.array, response_map.array, config)


def ibsi_ii_feature_tolerances(filter_id):
    csv_path = Path(__file__).parent / 'data' / 'ibsi_2_reference_values.csv'
    return load_references(csv_path, 'feature_tag', 'consensus_value', delimiter=';', phase='II', config=filter_id)


def ibsi_ii_ph_i_validation(filtered_image, response_map, config_id):

    assert filtered_image.shape == response_map.shape, f"{config_id}: response-map shape mismatch"
    assert response_map.size > 0, f"{config_id}: empty response map"
    assert np.isfinite(response_map).all(), f"{config_id}: non-finite reference voxels"
    assert np.isfinite(filtered_image).all(), f"{config_id}: non-finite computed voxels"
    tolerance = 0.01 * np.ptp(response_map)
    errors = np.abs(filtered_image - response_map)
    failing = np.count_nonzero(errors > tolerance)
    if failing:
        pytest.fail(
            f"{config_id}: {failing}/{response_map.size} voxels out of tolerance; "
            f"maximum error={errors.max():.8g}, tolerance={tolerance:.8g}"
        )


def ibsi_ii_ph_ii_validation(ibsi_features, features, config_8b=False):

    assert ibsi_features, "Empty IBSI reference selection"
    for raw_tag, feature_info in ibsi_features.items():
        tag = str(raw_tag)
        if config_8b and tag == 'stat_qcod':
            # IBSI II reference manual, Table 7.16: consensus is "none" for
            # 8.B stat_qcod, so IBSI publishes no reference value or tolerance.
            # https://doi.org/10.48550/arXiv.2006.05470
            assert feature_info.get('filter_id') == '8.B', '8.B exception used for another configuration'
            assert feature_info['consensus_value'] == feature_info['tolerance'] == '', (
                'Reference availability changed; review 8.B exception'
            )
            continue

        if tag not in features:
            pytest.fail(f"Missing required feature {tag}")

        if not matches_reference(features[tag], feature_info['consensus_value'], feature_info['tolerance']):
            pytest.fail(
                f"Feature {tag} out of tolerance: computed={features[tag]}, "
                f"reference={feature_info['consensus_value']}, tolerance={feature_info['tolerance']}"
            )


@pytest.mark.unit
@pytest.mark.parametrize('config_8b', [False, True])
@pytest.mark.parametrize('features', [{}, {'stat_mean': 1.0}])
def test_ibsi_ii_requires_all_reference_features(config_8b, features):
    reference = {
        'stat_mean': {'consensus_value': '1', 'tolerance': '0'},
        'stat_var': {'consensus_value': '2', 'tolerance': '0'},
    }
    missing = 'stat_var' if features else 'stat_mean'
    with pytest.raises(pytest.fail.Exception, match=f'Missing required feature {missing}'):
        ibsi_ii_ph_ii_validation(reference, features, config_8b=config_8b)


@pytest.mark.unit
def test_ibsi_ii_8b_excludes_only_qcod():
    reference = ibsi_ii_feature_tolerances('8.B')
    features = {tag: float(info['consensus_value']) for tag, info in reference.items() if tag != 'stat_qcod'}
    ibsi_ii_ph_ii_validation(reference, features, config_8b=True)
    del features['stat_mean']
    with pytest.raises(pytest.fail.Exception, match='Missing required feature stat_mean'):
        ibsi_ii_ph_ii_validation(reference, features, config_8b=True)


@pytest.mark.unit
def test_ibsi_ii_requires_qcod_outside_8b():
    reference = {'stat_qcod': ibsi_ii_feature_tolerances('8.A')['stat_qcod']}
    with pytest.raises(pytest.fail.Exception, match='Missing required feature stat_qcod'):
        ibsi_ii_ph_ii_validation(reference, {})


def _extract_filtered_features(image, filtered_image, mask, aggr_dim='2D', aggr_method='AVER'):
    roi_data = IntensityMaskBuilder().apply(
        RoiData(
            image=image,
            filtered_image=filtered_image,
            morphological_mask=mask,
        )
    )
    roi_data = Resegmenter(intensity_range=[-1000, 400]).apply(roi_data)
    roi_data = TextureDiscretizer(bin_size=25).apply(roi_data)
    return Radiomics(
        aggr_dim=aggr_dim,
        aggr_method=aggr_method,
    ).extract_features(roi_data=roi_data)


@pytest.fixture()
def ct_phantom_image(ibsi_i_data_dir):
    return Image.from_dicom(dicom_dir=str(ibsi_i_data_dir / 'dicom' / 'image'), modality='CT')


@pytest.fixture()
def ct_phantom_mask(ct_phantom_image, ibsi_i_data_dir):
    return Image.from_dicom_mask(
        rtstruct_path=str(ibsi_i_data_dir / 'dicom' / 'mask' / 'DCM_RS_00060.dcm'),
        structure_name='GTV-1',
        reference=ct_phantom_image,
    )


@pytest.fixture()
def res3d_1mm_image_spline(ct_phantom_image):
    preprocessing = ImageResampler(
        resolution=(1, 1, 1),
        method='BSpline',
        intensity_rounding='nearest_integer',
    )
    res_image = preprocessing.apply(ct_phantom_image)

    return res_image


@pytest.fixture()
def res3d_1mm_mask_linear(ct_phantom_mask):
    preprocessing = MaskResampler(
        resolution=(1, 1, 1),
        method='Linear',
        partial_volume_threshold=0.5,
    )
    res_mask = preprocessing.apply(ct_phantom_mask)

    return res_mask


@pytest.fixture()
def checkerboard_phantom(ibsi_ii_data_dir):
    return Image.from_nifti(ibsi_ii_data_dir / 'Ph_I/nifti/checkerboard/image/checkerboard.nii.gz')


@pytest.fixture()
def impulse_phantom(ibsi_ii_data_dir):
    return Image.from_nifti(ibsi_ii_data_dir / 'Ph_I/nifti/impulse/image/impulse.nii.gz')


@pytest.fixture()
def sphere_phantom(ibsi_ii_data_dir):
    return Image.from_nifti(ibsi_ii_data_dir / 'Ph_I/nifti/sphere/image/sphere.nii.gz')


@pytest.fixture()
def pattern_1_phantom(ibsi_ii_data_dir):
    return Image.from_nifti(ibsi_ii_data_dir / 'Ph_I/nifti/pattern_1/image/pattern_1.nii.gz')


@pytest.mark.integration
@pytest.mark.parametrize(
    ('config', 'params_and_images'),
    list(
        {
            '1.a.1': ['constant', '3D', 'checkerboard_phantom', '1_a_1-ValidCRM.nii'],
            '1.a.2': ['nearest', '3D', 'checkerboard_phantom', '1_a_2-ValidCRM.nii'],
            '1.a.3': ['wrap', '3D', 'checkerboard_phantom', '1_a_3-ValidCRM.nii'],
            '1.a.4': ['reflect', '3D', 'checkerboard_phantom', '1_a_4-ValidCRM.nii'],
            '1.b.1': ['constant', '2D', 'impulse_phantom', '1_b_1-ValidCRM.nii'],
        }.items()
    ),
)
def test_ibsi_ii_ph_i_1(ibsi_ii_data_dir, request, config, params_and_images):
    params_and_images = list(params_and_images)
    params_and_images[-2] = request.getfixturevalue(params_and_images[-2])
    filtering = create_filter(
        filtering_method='Mean', padding_type=params_and_images[0], dimensionality=params_and_images[1], support=15
    )
    _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_data_dir)


@pytest.mark.integration
@pytest.mark.parametrize(
    ('config', 'params_and_images'),
    list(
        {
            '2.a': ['constant', '3D', 3.0, 'impulse_phantom', '2_a-ValidCRM.nii'],
            '2.b': ['reflect', '3D', 5.0, 'checkerboard_phantom', '2_b-ValidCRM.nii'],
            '2.c': ['reflect', '2D', 5.0, 'checkerboard_phantom', '2_c-ValidCRM.nii'],
        }.items()
    ),
)
def test_ibsi_ii_ph_i_2(ibsi_ii_data_dir, request, config, params_and_images):
    params_and_images = list(params_and_images)
    params_and_images[-2] = request.getfixturevalue(params_and_images[-2])
    filtering = create_filter(
        filtering_method='Laplacian of Gaussian',
        padding_type=params_and_images[0],
        dimensionality=params_and_images[1],
        sigma_mm=params_and_images[2],
        cutoff=4,
    )
    _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_data_dir)


@pytest.mark.integration
@pytest.mark.parametrize(
    ('config', 'params_and_images'),
    list(
        {
            '3.a.1': ['constant', '3D', 'E5L5S5', False, None, False, 0, 'impulse_phantom', '3_a_1-ValidCRM.nii'],
            '3.a.2': ['constant', '3D', 'E5L5S5', True, 'max', False, 0, 'impulse_phantom', '3_a_2-ValidCRM.nii'],
            '3.a.3': ['constant', '3D', 'E5L5S5', True, 'max', True, 7, 'impulse_phantom', '3_a_3-ValidCRM.nii'],
            '3.b.1': ['reflect', '3D', 'E3W5R5', False, None, False, 0, 'checkerboard_phantom', '3_b_1-ValidCRM.nii'],
            '3.b.2': ['reflect', '3D', 'E3W5R5', True, 'max', False, 0, 'checkerboard_phantom', '3_b_2-ValidCRM.nii'],
            '3.b.3': ['reflect', '3D', 'E3W5R5', True, 'max', True, 7, 'checkerboard_phantom', '3_b_3-ValidCRM.nii'],
            '3.c.1': ['reflect', '2D', 'L5S5', False, None, False, 0, 'checkerboard_phantom', '3_c_1-ValidCRM.nii'],
            '3.c.2': ['reflect', '2D', 'L5S5', True, 'max', False, 0, 'checkerboard_phantom', '3_c_2-ValidCRM.nii'],
            '3.c.3': ['reflect', '2D', 'L5S5', True, 'max', True, 7, 'checkerboard_phantom', '3_c_3-ValidCRM.nii'],
        }.items()
    ),
)
def test_ibsi_ii_ph_i_3(ibsi_ii_data_dir, request, config, params_and_images):
    params_and_images = list(params_and_images)
    params_and_images[-2] = request.getfixturevalue(params_and_images[-2])
    filtering = create_filter(
        filtering_method='Laws Kernels',
        padding_type=params_and_images[0],
        dimensionality=params_and_images[1],
        response_map=params_and_images[2],
        rotation_invariance=params_and_images[3],
        pooling=params_and_images[4],
        energy_map=params_and_images[5],
        distance=params_and_images[6],
    )
    _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_data_dir)


@pytest.mark.integration
@pytest.mark.parametrize(
    "config,padding,res_mm,sigma_mm,lambda_mm,gamma,theta,rot_inv,orth_planes,n_stds,phantom,truth_file",
    [
        (
            "4.a.1",
            "constant",
            2.0,
            10.0,
            4.0,
            1 / 2,
            np.pi / 3,
            False,
            False,
            11,
            "impulse_phantom",
            "4_a_1-ValidCRM.nii",
        ),
        (
            "4.a.2",
            "constant",
            2.0,
            10.0,
            4.0,
            1 / 2,
            np.pi / 4,
            True,
            True,
            11,
            "impulse_phantom",
            "4_a_2-ValidCRM.nii",
        ),
        (
            "4.b.1",
            "reflect",
            2.0,
            20.0,
            8.0,
            5 / 2,
            5 * np.pi / 4,
            False,
            False,
            None,
            "sphere_phantom",
            "4_b_1-ValidCRM.nii",
        ),
        (
            "4.b.2",
            "reflect",
            2.0,
            20.0,
            8.0,
            5 / 2,
            np.pi / 8,
            True,
            True,
            None,
            "sphere_phantom",
            "4_b_2-ValidCRM.nii",
        ),
    ],
    ids=lambda val, *_: val,  # use the config string as the test‐id
)
def test_ibsi_ii_ph_i_4(
    config,
    padding,
    res_mm,
    sigma_mm,
    lambda_mm,
    gamma,
    theta,
    rot_inv,
    orth_planes,
    n_stds,
    phantom,
    truth_file,
    impulse_phantom,
    sphere_phantom,
    ibsi_ii_data_dir,
):
    # pick the right fixture
    phantom_data = {'impulse_phantom': impulse_phantom, 'sphere_phantom': sphere_phantom}[phantom]

    filtering = create_filter(
        filtering_method='Gabor',
        padding_type=padding,
        res_mm=res_mm,
        sigma_mm=sigma_mm,
        lambda_mm=lambda_mm,
        gamma=gamma,
        theta=theta,
        rotation_invariance=rot_inv,
        orthogonal_planes=orth_planes,
        n_stds=n_stds,
    )
    _run_ph_i_case(filtering, phantom_data, truth_file, config, ibsi_ii_data_dir)


@pytest.mark.integration
@pytest.mark.parametrize(
    ('config', 'params_and_images'),
    list(
        {
            '5.a.1': ['constant', 'LHL', False, 'impulse_phantom', '5_a_1-ValidCRM.nii'],
            '5.a.2': ['constant', 'LHL', True, 'impulse_phantom', '5_a_2-ValidCRM.nii'],
        }.items()
    ),
)
def test_ibsi_ii_ph_i_5(ibsi_ii_data_dir, request, config, params_and_images):
    params_and_images = list(params_and_images)
    params_and_images[-2] = request.getfixturevalue(params_and_images[-2])
    filtering = create_filter(
        filtering_method='Wavelets',
        wavelet_type='db2',
        dimensionality='3D',
        padding_type=params_and_images[0],
        response_map=params_and_images[1],
        decomposition_level=1,
        rotation_invariance=params_and_images[2],
    )
    _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_data_dir)


@pytest.mark.integration
@pytest.mark.parametrize(
    ('config', 'params_and_images'),
    list(
        {
            '6.a.1': ['wrap', 'HHL', False, 'sphere_phantom', '6_a_1-ValidCRM.nii'],
            '6.a.2': ['wrap', 'HHL', True, 'sphere_phantom', '6_a_2-ValidCRM.nii'],
        }.items()
    ),
)
def test_ibsi_ii_ph_i_6(ibsi_ii_data_dir, request, config, params_and_images):
    params_and_images = list(params_and_images)
    params_and_images[-2] = request.getfixturevalue(params_and_images[-2])
    filtering = create_filter(
        filtering_method='Wavelets',
        wavelet_type='coif1',
        dimensionality='3D',
        padding_type=params_and_images[0],
        response_map=params_and_images[1],
        decomposition_level=1,
        rotation_invariance=params_and_images[2],
    )
    _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_data_dir)


@pytest.mark.integration
@pytest.mark.parametrize(
    ('config', 'params_and_images'),
    list(
        {
            '7.a.1': ['reflect', 'LLL', False, 'checkerboard_phantom', '7_a_1-ValidCRM.nii'],
            '7.a.2': ['reflect', 'HHH', True, 'checkerboard_phantom', '7_a_2-ValidCRM.nii'],
        }.items()
    ),
)
def test_ibsi_ii_ph_i_7(ibsi_ii_data_dir, request, config, params_and_images):
    params_and_images = list(params_and_images)
    params_and_images[-2] = request.getfixturevalue(params_and_images[-2])
    filtering = create_filter(
        filtering_method='Wavelets',
        wavelet_type='haar',
        dimensionality='3D',
        padding_type=params_and_images[0],
        response_map=params_and_images[1],
        decomposition_level=2,
        rotation_invariance=params_and_images[2],
    )
    _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_data_dir)


@pytest.mark.integration
@pytest.mark.parametrize('level', [1, 2, 3], ids=['8.a.1', '8.a.2', '8.a.3'])
def test_ibsi_ii_ph_i_8(ibsi_ii_data_dir, checkerboard_phantom, level):
    filtering = create_filter(
        filtering_method='Simoncelli', dimensionality='3D', padding_type='wrap', decomposition_level=level
    )
    _run_ph_i_case(filtering, checkerboard_phantom, f'8_a_{level}-ValidCRM.nii', f'8.a.{level}', ibsi_ii_data_dir)


@pytest.mark.integration
@pytest.mark.parametrize(
    ('config', 'phantom', 'order', 'filename'),
    [
        ('9.a', 'impulse_phantom', (1, 0, 0), '9_a-ValidCRM.nii'),
        ('9.b.1', 'sphere_phantom', (0, 2, 0), '9_b_1-ValidCRM.nii'),
    ],
)
def test_ibsi_ii_ph_i_9(ibsi_ii_data_dir, request, config, phantom, order, filename):
    filtering = create_filter(
        filtering_method='Riesz-transformed LoG',
        dimensionality='3D',
        padding_type='constant',
        sigma_mm=3.0,
        cutoff=4,
        riesz_order=order,
    )
    _run_ph_i_case(filtering, request.getfixturevalue(phantom), filename, config, ibsi_ii_data_dir)


@pytest.mark.integration
def test_ibsi_ii_ph_i_10(ibsi_ii_data_dir, pattern_1_phantom):
    filtering = create_filter(
        filtering_method='Simoncelli',
        dimensionality='3D',
        padding_type='nearest',
        decomposition_level=1,
        riesz_order=(0, 2, 0),
    )
    _run_ph_i_case(
        filtering,
        pattern_1_phantom,
        '10_b_1-ValidCRM.nii',
        '10.b.1',
        ibsi_ii_data_dir,
    )


@pytest.mark.integration
def test_ibsi_ii_ph_ii_2a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('2.A')

    filtering = create_filter(filtering_method='Mean', padding_type='reflect', dimensionality='2D', support=5)

    filtered_image = filtering.apply(ct_phantom_image)

    features = _extract_filtered_features(ct_phantom_image, filtered_image, ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_2b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('2.B')

    filtering = create_filter(filtering_method='Mean', padding_type='reflect', dimensionality='3D', support=5)

    filtered_image = filtering.apply(res3d_1mm_image_spline)

    features = _extract_filtered_features(res3d_1mm_image_spline, filtered_image, res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_3a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('3.A')

    filtering = create_filter(
        filtering_method='Laplacian of Gaussian', padding_type='reflect', dimensionality='2D', sigma_mm=1.5, cutoff=4
    )

    filtered_image = filtering.apply(ct_phantom_image)

    features = _extract_filtered_features(ct_phantom_image, filtered_image, ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_3b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('3.B')

    filtering = create_filter(
        filtering_method='Laplacian of Gaussian', padding_type='reflect', dimensionality='3D', sigma_mm=1.5, cutoff=4
    )

    filtered_image = filtering.apply(res3d_1mm_image_spline)

    features = _extract_filtered_features(res3d_1mm_image_spline, filtered_image, res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_4a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('4.A')

    filtering = create_filter(
        filtering_method='Laws Kernels',
        padding_type='reflect',
        dimensionality='2D',
        response_map='L5E5',
        rotation_invariance=True,
        pooling="max",
        energy_map=True,
        distance=7,
    )

    filtered_image = filtering.apply(ct_phantom_image)

    features = _extract_filtered_features(ct_phantom_image, filtered_image, ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_4b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('4.B')

    filtering = create_filter(
        filtering_method='Laws Kernels',
        response_map="L5E5E5",
        padding_type="reflect",
        dimensionality="3D",
        rotation_invariance=True,
        pooling="max",
        energy_map=True,
        distance=7,
    )

    filtered_image = filtering.apply(res3d_1mm_image_spline)

    features = _extract_filtered_features(res3d_1mm_image_spline, filtered_image, res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


# Gabor
@pytest.mark.integration
def test_ibsi_ii_ph_ii_5a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('5.A')

    filtering = create_filter(
        filtering_method='Gabor',
        padding_type='reflect',
        dimensionality='2D',
        res_mm=0.977,
        sigma_mm=5.0,
        lambda_mm=2.0,
        gamma=3 / 2,
        theta=np.pi / 8,
        rotation_invariance=True,
        orthogonal_planes=False,
    )

    filtered_image = filtering.apply(ct_phantom_image)

    features = _extract_filtered_features(ct_phantom_image, filtered_image, ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_5b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('5.B')

    filtering = create_filter(
        filtering_method='Gabor',
        padding_type="reflect",
        dimensionality="3D",
        res_mm=1.0,
        sigma_mm=5.0,
        lambda_mm=2.0,
        gamma=3 / 2,
        theta=np.pi / 8,
        rotation_invariance=True,
        orthogonal_planes=True,
    )

    filtered_image = filtering.apply(res3d_1mm_image_spline)

    features = _extract_filtered_features(res3d_1mm_image_spline, filtered_image, res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_6a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('6.A')

    filtering = create_filter(
        filtering_method='Wavelets',
        wavelet_type="db3",
        dimensionality='2D',
        padding_type="reflect",
        response_map="LH",
        decomposition_level=1,
        rotation_invariance=True,
    )

    filtered_image = filtering.apply(ct_phantom_image)

    features = _extract_filtered_features(ct_phantom_image, filtered_image, ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_6b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('6.B')

    filtering = create_filter(
        filtering_method='Wavelets',
        wavelet_type="db3",
        dimensionality='3D',
        padding_type="reflect",
        response_map="LLH",
        decomposition_level=1,
        rotation_invariance=True,
    )

    filtered_image = filtering.apply(res3d_1mm_image_spline)

    features = _extract_filtered_features(res3d_1mm_image_spline, filtered_image, res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_7a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('7.A')

    filtering = create_filter(
        filtering_method='Wavelets',
        wavelet_type="db3",
        dimensionality='2D',
        padding_type="reflect",
        response_map="HH",
        decomposition_level=2,
        rotation_invariance=True,
    )

    filtered_image = filtering.apply(ct_phantom_image)

    features = _extract_filtered_features(ct_phantom_image, filtered_image, ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_7b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('7.B')

    filtering = create_filter(
        filtering_method='Wavelets',
        wavelet_type="db3",
        dimensionality='3D',
        padding_type="reflect",
        response_map="HHH",
        decomposition_level=2,
        rotation_invariance=True,
    )

    filtered_image = filtering.apply(res3d_1mm_image_spline)

    features = _extract_filtered_features(res3d_1mm_image_spline, filtered_image, res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_8a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('8.A')

    filtering = create_filter(
        filtering_method='Simoncelli', padding_type='periodic', decomposition_level=1, dimensionality='2D'
    )

    filtered_image = filtering.apply(ct_phantom_image)

    features = _extract_filtered_features(ct_phantom_image, filtered_image, ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_8b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('8.B')

    filtering = create_filter(
        filtering_method='Simoncelli', padding_type='periodic', decomposition_level=1, dimensionality='3D'
    )

    filtered_image = filtering.apply(res3d_1mm_image_spline)

    features = _extract_filtered_features(res3d_1mm_image_spline, filtered_image, res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, features, config_8b=True)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_9a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('9.A')

    filtering = create_filter(
        filtering_method='Simoncelli', padding_type='periodic', decomposition_level=2, dimensionality='2D'
    )

    filtered_image = filtering.apply(ct_phantom_image)

    features = _extract_filtered_features(ct_phantom_image, filtered_image, ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_9b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('9.B')

    filtering = create_filter(
        filtering_method='Simoncelli', padding_type='periodic', decomposition_level=2, dimensionality='3D'
    )

    filtered_image = filtering.apply(res3d_1mm_image_spline)

    features = _extract_filtered_features(res3d_1mm_image_spline, filtered_image, res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, features)


@pytest.mark.integration
@pytest.mark.parametrize(
    ('config', 'image_fixture', 'mask_fixture'),
    [
        ('1.A', 'ct_phantom_image', 'ct_phantom_mask'),
        ('1.B', 'res3d_1mm_image_spline', 'res3d_1mm_mask_linear'),
    ],
)
def test_ibsi_ii_ph_ii_unfiltered(request, config, image_fixture, mask_fixture):
    # IBSI II Table 6.3: configurations 1.A/1.B use no filter.
    image = request.getfixturevalue(image_fixture)
    mask = request.getfixturevalue(mask_fixture)
    features = _extract_filtered_features(image, image, mask)
    ibsi_ii_ph_ii_validation(ibsi_ii_feature_tolerances(config), features)
