from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zrad.filtering import Filtering
from zrad.image import Image
from zrad.preprocessing import Preprocessing
from zrad.radiomics import Radiomics


def _run_ph_i_case(filtering, phantom, filename, config, data_dir):
    filtered_image = filtering.apply_filter(phantom)
    response_map = Image()
    response_map_path = data_dir / 'Ph_I' / 'response_maps' / filename
    response_map.read_nifti_image(str(response_map_path))
    ibsi_ii_ph_i_validation(filtered_image.array, response_map.array, config)


def ibsi_ii_feature_tolerances(filter_id):
    csv_path = Path(__file__).parent / 'data' / 'ibsi_2_reference_values.csv'
    df = pd.read_csv(csv_path, delimiter=';')
    return df[df['filter_id'] == filter_id]


def ibsi_ii_ph_i_validation(filtered_image, response_map, config_id):

    tolerance = 0.01 * (np.max(response_map) - np.min(response_map))
    within_tolerance = (filtered_image >= (response_map - tolerance)) & (
                filtered_image <= (response_map + tolerance))

    total_voxels = response_map.size
    voxels_within_tolerance = np.sum(within_tolerance)
    if total_voxels != voxels_within_tolerance:
        print(total_voxels-voxels_within_tolerance)
        pytest.fail(f"Failed {config_id}")


def ibsi_ii_ph_ii_validation(ibsi_features, features):

    for tag in ibsi_features['feature_tag']:

        if str(tag) in features:
            val = float(ibsi_features[ibsi_features['feature_tag'] == tag]['consensus_value'].iloc[0])
            tol = float(ibsi_features[ibsi_features['feature_tag'] == tag]['tolerance'].iloc[0])
            upper_boundary = val + tol
            lower_boundary = val - tol
            if not (lower_boundary <= features[tag] <= upper_boundary):
                pytest.fail(
                    f"Feature {tag} out of tolerance: {features[tag]} not in range ({lower_boundary}, {upper_boundary})")





@pytest.fixture()
def ct_phantom_image(ibsi_i_data_dir):
    image = Image()
    image.read_dicom_image(dicom_dir=str(ibsi_i_data_dir / 'dicom' / 'image'), modality='CT')
    return image


@pytest.fixture()
def ct_phantom_mask(ct_phantom_image, ibsi_i_data_dir):
    mask = Image()
    mask.read_dicom_mask(
        rtstruct_path=str(ibsi_i_data_dir / 'dicom' / 'mask' / 'DCM_RS_00060.dcm'),
        structure_name='GTV-1',
        image=ct_phantom_image
    )
    return mask


@pytest.fixture()
def res3d_1mm_image_spline(ct_phantom_image):
    preprocessing = Preprocessing(input_imaging_modality='CT',
                                  resample_resolution=1,
                                  resample_dimension='3D',
                                  interpolation_method='BSpline')
    res_image = preprocessing.resample(ct_phantom_image, image_type='image')

    return res_image


@pytest.fixture()
def res3d_1mm_mask_linear(ct_phantom_mask):
    preprocessing = Preprocessing(input_imaging_modality='CT',
                                  resample_resolution=1,
                                  resample_dimension='3D',
                                  interpolation_method='Linear',
                                  interpolation_threshold=.5)
    res_mask = preprocessing.resample(ct_phantom_mask, image_type='mask')

    return res_mask


@pytest.fixture()
def checkerboard_phantom():
    checkerboard = Image()
    checkerboard.read_nifti_image('tests/data/IBSI_II/Ph_I/nifti/checkerboard/image/checkerboard.nii.gz')
    return checkerboard


@pytest.fixture()
def impulse_phantom():
    impulse = Image()
    impulse.read_nifti_image('tests/data/IBSI_II/Ph_I/nifti/impulse/image/impulse.nii.gz')
    return impulse


@pytest.fixture()
def sphere_phantom():
    sphere = Image()
    sphere.read_nifti_image('tests/data/IBSI_II/Ph_I/nifti/sphere/image/sphere.nii.gz')
    return sphere


@pytest.mark.integration
def test_ibsi_ii_ph_i_1(ibsi_ii_data_dir, checkerboard_phantom, impulse_phantom):

    for config, params_and_images in {'1.a.1': ['constant', '3D', checkerboard_phantom, '1_a_1-ValidCRM.nii'],
                                      '1.a.2': ['nearest', '3D', checkerboard_phantom, '1_a_2-ValidCRM.nii'],
                                      '1.a.3': ['wrap', '3D', checkerboard_phantom, '1_a_3-ValidCRM.nii'],
                                      '1.a.4': ['reflect', '3D', checkerboard_phantom, '1_a_4-ValidCRM.nii'],
                                      '1.b.1': ['constant', '2D', impulse_phantom, '1_b_1-ValidCRM.nii']}.items():

        filtering = Filtering(filtering_method='Mean',
                              padding_type=params_and_images[0],
                              dimensionality=params_and_images[1],
                              support=15)

        _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_data_dir)


@pytest.mark.integration
def test_ibsi_ii_ph_i_2(ibsi_ii_data_dir, checkerboard_phantom, impulse_phantom):

    for config, params_and_images in {'2.a': ['constant', '3D', 3.0, impulse_phantom, '2_a-ValidCRM.nii'],
                                      '2.b': ['reflect', '3D', 5.0, checkerboard_phantom, '2_b-ValidCRM.nii'],
                                      '2.c': ['reflect', '2D', 5.0, checkerboard_phantom, '2_c-ValidCRM.nii']}.items():

        filtering = Filtering(filtering_method='Laplacian of Gaussian',
                              padding_type=params_and_images[0],
                              dimensionality=params_and_images[1],
                              sigma_mm=params_and_images[2],
                              cutoff=4)

        _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_data_dir)


@pytest.mark.integration
def test_ibsi_ii_ph_i_3(ibsi_ii_data_dir, checkerboard_phantom, impulse_phantom):

    for config, params_and_images in {'3.a.1': ['constant', '3D', 'E5L5S5', False, None, False, 0, impulse_phantom, '3_a_1-ValidCRM.nii'],
                                      '3.a.2': ['constant', '3D', 'E5L5S5', True, 'max', False, 0, impulse_phantom, '3_a_2-ValidCRM.nii'],
                                      '3.a.3': ['constant', '3D', 'E5L5S5', True, 'max', True, 7, impulse_phantom, '3_a_3-ValidCRM.nii'],
                                      '3.b.1': ['reflect', '3D', 'E3W5R5', False, None, False, 0, checkerboard_phantom, '3_b_1-ValidCRM.nii'],
                                      '3.b.2': ['reflect', '3D', 'E3W5R5', True, 'max', False, 0, checkerboard_phantom, '3_b_2-ValidCRM.nii'],
                                      '3.b.3': ['reflect', '3D', 'E3W5R5', True, 'max', True, 7, checkerboard_phantom, '3_b_3-ValidCRM.nii'],
                                      '3.c.1': ['reflect', '2D', 'L5S5', False, None, False, 0, checkerboard_phantom, '3_c_1-ValidCRM.nii'],
                                      '3.c.2': ['reflect', '2D', 'L5S5', True, 'max', False, 0, checkerboard_phantom, '3_c_2-ValidCRM.nii'],
                                      '3.c.3': ['reflect', '2D', 'L5S5', True, 'max', True, 7, checkerboard_phantom, '3_c_3-ValidCRM.nii'],
                                      }.items():

        filtering = Filtering(filtering_method='Laws Kernels',
                              padding_type=params_and_images[0],
                              dimensionality=params_and_images[1],
                              response_map=params_and_images[2],
                              rotation_invariance=params_and_images[3],
                              pooling=params_and_images[4],
                              energy_map=params_and_images[5],
                              distance=params_and_images[6])

        _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_data_dir)


@pytest.mark.integration
@pytest.mark.parametrize(
    "config,padding,res_mm,sigma_mm,lambda_mm,gamma,theta,rot_inv,orth_planes,n_stds,phantom,truth_file",
    [
        ("4.a.1", "constant", 2.0, 10.0,   4.0, 1/2,     np.pi/3, False, False, 11, "impulse_phantom", "4_a_1-ValidCRM.nii"),
        ("4.a.2", "constant", 2.0, 10.0,   4.0, 1/2,     np.pi/4, True,  True,  11, "impulse_phantom", "4_a_2-ValidCRM.nii"),
        ("4.b.1", "reflect",  2.0, 20.0,   8.0, 5/2, 5*np.pi/4, False, False, None, "sphere_phantom",  "4_b_1-ValidCRM.nii"),
        ("4.b.2", "reflect",  2.0, 20.0,   8.0, 5/2,   np.pi/8, True,  True,  None, "sphere_phantom",  "4_b_2-ValidCRM.nii"),
    ],
    ids=lambda val, *_: val  # use the config string as the test‐id
)
def test_ibsi_ii_ph_i_4(
    config, padding, res_mm, sigma_mm, lambda_mm, gamma, theta,
    rot_inv, orth_planes, n_stds, phantom, truth_file,
    impulse_phantom, sphere_phantom, ibsi_ii_data_dir
):
    # pick the right fixture
    phantom_data = {'impulse_phantom': impulse_phantom,
                    'sphere_phantom': sphere_phantom}[phantom]

    filtering = Filtering(
        filtering_method='Gabor',
        padding_type=padding,
        res_mm=res_mm,
        sigma_mm=sigma_mm,
        lambda_mm=lambda_mm,
        gamma=gamma,
        theta=theta,
        rotation_invariance=rot_inv,
        orthogonal_planes=orth_planes,
        n_stds=n_stds
    )
    _run_ph_i_case(filtering, phantom_data, truth_file, config, ibsi_ii_data_dir)


@pytest.mark.integration
def test_ibsi_ii_ph_i_5(ibsi_ii_data_dir, impulse_phantom):

    for config, params_and_images in {'5.a.1': ['constant', 'LHL', False, impulse_phantom, '5_a_1-ValidCRM.nii'],
                                      '5.a.2': ['constant', 'LHL', True, impulse_phantom, '5_a_2-ValidCRM.nii'],
                                      }.items():
        filtering = Filtering(filtering_method='Wavelets',
                              wavelet_type="db2",
                              dimensionality='3D',
                              padding_type=params_and_images[0],
                              response_map=params_and_images[1],
                              decomposition_level=1,
                              rotation_invariance=params_and_images[2])

        _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_data_dir)


@pytest.mark.integration
def test_ibsi_ii_ph_i_6(ibsi_ii_data_dir, sphere_phantom):

    for config, params_and_images in {'6.a.1': ['wrap', 'HHL', False, sphere_phantom, '6_a_1-ValidCRM.nii'],
                                      '6.a.2': ['wrap', 'HHL', True, sphere_phantom, '6_a_2-ValidCRM.nii'],
                                      }.items():
        filtering = Filtering(filtering_method='Wavelets',
                              wavelet_type="coif1",
                              dimensionality='3D',
                              padding_type=params_and_images[0],
                              response_map=params_and_images[1],
                              decomposition_level=1,
                              rotation_invariance=params_and_images[2])

        _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_data_dir)


@pytest.mark.integration
def test_ibsi_ii_ph_i_7(ibsi_ii_data_dir, checkerboard_phantom):

    for config, params_and_images in {'7.a.1': ['reflect', 'LLL', False, checkerboard_phantom, '7_a_1-ValidCRM.nii'],
                                      '7.a.2': ['reflect', 'HHH', True, checkerboard_phantom, '7_a_2-ValidCRM.nii'],
                                      }.items():
        filtering = Filtering(filtering_method='Wavelets',
                              wavelet_type="haar",
                              dimensionality='3D',
                              padding_type=params_and_images[0],
                              response_map=params_and_images[1],
                              decomposition_level=2,
                              rotation_invariance=params_and_images[2])

        _run_ph_i_case(filtering, params_and_images[-2], params_and_images[-1], config, ibsi_ii_data_dir)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_2a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('2.A')

    filtering = Filtering(filtering_method='Mean',
                          padding_type='reflect',
                          dimensionality='2D',
                          support=5)

    filtered_image = filtering.apply_filter(ct_phantom_image)

    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='AVER',
                          intensity_range=[-1000, 400],
                          bin_size=25)

    radiomics.extract_features(image=ct_phantom_image, filtered_image=filtered_image, mask=ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, radiomics.features_)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_2b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('2.B')

    filtering = Filtering(filtering_method='Mean',
                          padding_type='reflect',
                          dimensionality='3D',
                          support=5)

    filtered_image = filtering.apply_filter(res3d_1mm_image_spline)

    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='AVER',
                          intensity_range=[-1000, 400],
                          bin_size=25)

    radiomics.extract_features(image=res3d_1mm_image_spline, filtered_image=filtered_image, mask=res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, radiomics.features_)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_3a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('3.A')

    filtering = Filtering(filtering_method='Laplacian of Gaussian',
                          padding_type='reflect',
                          dimensionality='2D',
                          sigma_mm=1.5,
                          cutoff=4)

    filtered_image = filtering.apply_filter(ct_phantom_image)

    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='AVER',
                          intensity_range=[-1000, 400],
                          bin_size=25)

    radiomics.extract_features(image=ct_phantom_image, filtered_image=filtered_image, mask=ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, radiomics.features_)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_3b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('3.B')

    filtering = Filtering(filtering_method='Laplacian of Gaussian',
                          padding_type='reflect',
                          dimensionality='3D',
                          sigma_mm=1.5,
                          cutoff=4)

    filtered_image = filtering.apply_filter(res3d_1mm_image_spline)

    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='AVER',
                          intensity_range=[-1000, 400],
                          bin_size=25)

    radiomics.extract_features(image=res3d_1mm_image_spline, filtered_image=filtered_image, mask=res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, radiomics.features_)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_4a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('4.A')

    filtering = Filtering(filtering_method='Laws Kernels',
                          padding_type='reflect',
                          dimensionality='2D',
                          response_map='L5E5',
                          rotation_invariance=True,
                          pooling="max",
                          energy_map=True,
                          distance=7)

    filtered_image = filtering.apply_filter(ct_phantom_image)

    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='AVER',
                          intensity_range=[-1000, 400],
                          bin_size=25)

    radiomics.extract_features(image=ct_phantom_image, filtered_image=filtered_image, mask=ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, radiomics.features_)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_4b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('4.B')

    filtering = Filtering(filtering_method='Laws Kernels',
                          response_map="L5E5E5",
                          padding_type="reflect",
                          dimensionality="3D",
                          rotation_invariance=True,
                          pooling="max",
                          energy_map=True,
                          distance=7)

    filtered_image = filtering.apply_filter(res3d_1mm_image_spline)

    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='AVER',
                          intensity_range=[-1000, 400],
                          bin_size=25)

    radiomics.extract_features(image=res3d_1mm_image_spline, filtered_image=filtered_image, mask=res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, radiomics.features_)

# Gabor
@pytest.mark.integration
def test_ibsi_ii_ph_ii_5a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('5.A')

    filtering = Filtering(filtering_method='Gabor',
                          padding_type='reflect',
                          dimensionality='2D',
                          res_mm=0.977,
                          sigma_mm=5.0,
                          lambda_mm=2.0,
                          gamma=3/2,
                          theta=np.pi/8,
                          rotation_invariance=True,
                          orthogonal_planes=False)

    filtered_image = filtering.apply_filter(ct_phantom_image)

    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='AVER',
                          intensity_range=[-1000, 400],
                          bin_size=25)

    radiomics.extract_features(image=ct_phantom_image, filtered_image=filtered_image, mask=ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, radiomics.features_)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_5b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('5.B')

    filtering = Filtering(filtering_method='Gabor',
                          padding_type="reflect",
                          dimensionality="3D",
                          res_mm=1.0,
                          sigma_mm=5.0,
                          lambda_mm=2.0,
                          gamma=3/2,
                          theta=np.pi/8,
                          rotation_invariance=True,
                          orthogonal_planes=True)

    filtered_image = filtering.apply_filter(res3d_1mm_image_spline)

    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='AVER',
                          intensity_range=[-1000, 400],
                          bin_size=25)

    radiomics.extract_features(image=res3d_1mm_image_spline, filtered_image=filtered_image, mask=res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, radiomics.features_)

@pytest.mark.integration
def test_ibsi_ii_ph_ii_6a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('6.A')

    filtering = Filtering(filtering_method='Wavelets',
                          wavelet_type="db3",
                          dimensionality='2D',
                          padding_type="reflect",
                          response_map="LH",
                          decomposition_level=1,
                          rotation_invariance=True)

    filtered_image = filtering.apply_filter(ct_phantom_image)

    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='AVER',
                          intensity_range=[-1000, 400],
                          bin_size=25)

    radiomics.extract_features(image=ct_phantom_image, filtered_image=filtered_image, mask=ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, radiomics.features_)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_6b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('6.B')

    filtering = Filtering(filtering_method='Wavelets',
                          wavelet_type="db3",
                          dimensionality='3D',
                          padding_type="reflect",
                          response_map="LLH",
                          decomposition_level=1,
                          rotation_invariance=True)

    filtered_image = filtering.apply_filter(res3d_1mm_image_spline)

    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='AVER',
                          intensity_range=[-1000, 400],
                          bin_size=25)

    radiomics.extract_features(image=res3d_1mm_image_spline, filtered_image=filtered_image, mask=res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, radiomics.features_)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_7a(ct_phantom_image, ct_phantom_mask):
    ibsi_features = ibsi_ii_feature_tolerances('7.A')

    filtering = Filtering(filtering_method='Wavelets',
                          wavelet_type="db3",
                          dimensionality='2D',
                          padding_type="reflect",
                          response_map="HH",
                          decomposition_level=2,
                          rotation_invariance=True)

    filtered_image = filtering.apply_filter(ct_phantom_image)

    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='AVER',
                          intensity_range=[-1000, 400],
                          bin_size=25)

    radiomics.extract_features(image=ct_phantom_image, filtered_image=filtered_image, mask=ct_phantom_mask)
    ibsi_ii_ph_ii_validation(ibsi_features, radiomics.features_)


@pytest.mark.integration
def test_ibsi_ii_ph_ii_7b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
    ibsi_features = ibsi_ii_feature_tolerances('7.B')

    filtering = Filtering(filtering_method='Wavelets', wavelet_type="db3",
                          dimensionality='3D',
                          padding_type="reflect",
                          response_map="HHH",
                          decomposition_level=2,
                          rotation_invariance=True)

    filtered_image = filtering.apply_filter(res3d_1mm_image_spline)

    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='AVER',
                          intensity_range=[-1000, 400],
                          bin_size=25)

    radiomics.extract_features(image=res3d_1mm_image_spline, filtered_image=filtered_image, mask=res3d_1mm_mask_linear)
    ibsi_ii_ph_ii_validation(ibsi_features, radiomics.features_)
