from pathlib import Path

import pandas as pd
import pytest

from zrad.image import Image
from zrad.preprocessing import Preprocessing
from zrad.radiomics import Radiomics


def ibsi_i_feature_tolerances(sheet_name):
    csv_path = Path(__file__).parent / 'data' / f'ibsi_1_reference_values_{sheet_name}.csv'
    return pd.read_csv(csv_path)


def ibsi_i_validation(ibsi_features, features, config_a=False):
    for tag in ibsi_features['tag']:
        if config_a and tag == 'ih_qcod':
            continue

        if str(tag) in features:
            val = float(ibsi_features[ibsi_features['tag'] == tag]['reference value'].iloc[0])
            tol = float(ibsi_features[ibsi_features['tag'] == tag]['tolerance'].iloc[0])
            upper_boundary = val + tol
            lower_boundary = val - tol

            if not (lower_boundary <= features[tag] <= upper_boundary):
                pytest.fail(f"Feature {tag} out of tolerance: {features[tag]} not in range ({lower_boundary}, {upper_boundary})")

@pytest.fixture()
def dcm_ct_phantom_image(ibsi_i_data_dir):
    image = Image()
    image.read_dicom_image(dicom_dir=ibsi_i_data_dir / 'dicom' / 'image', modality='CT')
    return image

@pytest.fixture()
def dcm_ct_phantom_mask(dcm_ct_phantom_image):

    image = Image()
    image.read_dicom_mask(image=dcm_ct_phantom_image,
                          rtstruct_path='tests/data/IBSI_I/dicom/mask/DCM_RS_00060.dcm',
                          structure_name='GTV-1')
    return image

@pytest.fixture()
def nii_ct_phantom_image(ibsi_i_data_dir):
    image = Image()
    image.read_nifti_image(str(ibsi_i_data_dir / 'nifti' / 'image' / 'phantom.nii.gz'))
    return image


@pytest.fixture()
def nii_ct_phantom_mask(nii_ct_phantom_image):

    mask = Image()
    mask.read_nifti_mask(image=nii_ct_phantom_image, mask_path='tests/data/IBSI_I/nifti/mask/mask.nii.gz')
    return mask


@pytest.fixture()
def res2d_2mm_image_linear(nii_ct_phantom_image):

    preprocessing = Preprocessing(input_imaging_modality='CT',
                                  resample_resolution=2,
                                  resample_dimension='2D',
                                  interpolation_method='Linear')
    res_image = preprocessing.resample(nii_ct_phantom_image, image_type='image')

    return res_image


@pytest.fixture()
def res2d_2mm_mask_linear(nii_ct_phantom_mask):

    preprocessing = Preprocessing(input_imaging_modality='CT',
                                  resample_resolution=2,
                                  resample_dimension='2D',
                                  interpolation_method='Linear',
                                  interpolation_threshold=.5)
    res_mask = preprocessing.resample(nii_ct_phantom_mask, image_type='mask')

    return res_mask


@pytest.fixture()
def res3d_2mm_image_linear(nii_ct_phantom_image):

    preprocessing = Preprocessing(input_imaging_modality='CT',
                                  resample_resolution=2,
                                  resample_dimension='3D',
                                  interpolation_method='Linear')
    res_image = preprocessing.resample(nii_ct_phantom_image, image_type='image')

    return res_image


@pytest.fixture()
def res3d_2mm_mask_linear(nii_ct_phantom_mask):

    preprocessing = Preprocessing(input_imaging_modality='CT',
                                  resample_resolution=2,
                                  resample_dimension='3D',
                                  interpolation_method='Linear',
                                  interpolation_threshold=.5)
    res_mask = preprocessing.resample(nii_ct_phantom_mask, image_type='mask')

    return res_mask


@pytest.fixture()
def res3d_2mm_image_spline(dcm_ct_phantom_image):

    preprocessing = Preprocessing(input_imaging_modality='CT',
                                  resample_resolution=2,
                                  resample_dimension='3D',
                                  interpolation_method='BSpline')
    res_image = preprocessing.resample(dcm_ct_phantom_image, image_type='image')

    return res_image


@pytest.mark.integration
def test_ibsi_i_config_a(dcm_ct_phantom_image, dcm_ct_phantom_mask):
    ibsi_features = ibsi_i_feature_tolerances('config_A')

    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='AVER',
                          intensity_range=[-500, 400],
                          bin_size=25,
                          calc_ivh_features=True)

    radiomics.extract_features(image=dcm_ct_phantom_image, mask=dcm_ct_phantom_mask)
    ibsi_i_validation(ibsi_features, radiomics.features_, True)

    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='SLICE_MERG',
                          intensity_range=[-500, 400],
                          bin_size=25,
                          calc_ivh_features=True)

    radiomics.extract_features(image=dcm_ct_phantom_image, mask=dcm_ct_phantom_mask)
    ibsi_i_validation(ibsi_features, radiomics.features_, True)

    radiomics = Radiomics(aggr_dim='2.5D',
                          aggr_method='DIR_MERG',
                          intensity_range=[-500, 400],
                          bin_size=25,
                          calc_ivh_features=True)

    radiomics.extract_features(image=dcm_ct_phantom_image, mask=dcm_ct_phantom_mask)
    ibsi_i_validation(ibsi_features, radiomics.features_, True)

    radiomics = Radiomics(aggr_dim='2.5D',
                          aggr_method='MERG',
                          intensity_range=[-500, 400],
                          bin_size=25,
                          calc_ivh_features=True)

    radiomics.extract_features(image=dcm_ct_phantom_image, mask=dcm_ct_phantom_mask)
    ibsi_i_validation(ibsi_features, radiomics.features_, True)


@pytest.mark.integration
def test_ibsi_i_config_b(res2d_2mm_image_linear, res2d_2mm_mask_linear):
    ibsi_features = ibsi_i_feature_tolerances('config_B')

    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='AVER',
                          intensity_range=[-500, 400],
                          number_of_bins=32)

    radiomics.extract_features(image=res2d_2mm_image_linear, mask=res2d_2mm_mask_linear)
    ibsi_i_validation(ibsi_features, radiomics.features_)
    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='SLICE_MERG',
                          intensity_range=[-500, 400],
                          number_of_bins=32,
                          calc_ivh_features=True)

    radiomics.extract_features(image=res2d_2mm_image_linear, mask=res2d_2mm_mask_linear)
    ibsi_i_validation(ibsi_features, radiomics.features_)

    radiomics = Radiomics(aggr_dim='2.5D',
                          aggr_method='DIR_MERG',
                          intensity_range=[-500, 400],
                          number_of_bins=32,
                          calc_ivh_features=True)

    radiomics.extract_features(image=res2d_2mm_image_linear, mask=res2d_2mm_mask_linear)
    ibsi_i_validation(ibsi_features, radiomics.features_)

    radiomics = Radiomics(aggr_dim='2.5D',
                          aggr_method='MERG',
                          intensity_range=[-500, 400],
                          number_of_bins=32,
                          calc_ivh_features=True)

    radiomics.extract_features(image=res2d_2mm_image_linear, mask=res2d_2mm_mask_linear)
    ibsi_i_validation(ibsi_features, radiomics.features_)


@pytest.mark.integration
def test_ibsi_i_config_c(res3d_2mm_image_linear, res3d_2mm_mask_linear):
    ibsi_features = ibsi_i_feature_tolerances('config_C')

    radiomics = Radiomics(aggr_dim='3D',
                          aggr_method='AVER',
                          intensity_range=[-1000, 400],
                          bin_size=25,
                          calc_ivh_features=True,
                          ivh_bin_size=2.5
                          )

    radiomics.extract_features(image=res3d_2mm_image_linear, mask=res3d_2mm_mask_linear)
    ibsi_i_validation(ibsi_features, radiomics.features_)

    radiomics = Radiomics(aggr_dim='3D',
                          aggr_method='MERG',
                          intensity_range=[-1000, 400],
                          bin_size=25,
                          calc_ivh_features=True,
                          ivh_bin_size=2.5
                          )

    radiomics.extract_features(image=res3d_2mm_image_linear, mask=res3d_2mm_mask_linear)
    ibsi_i_validation(ibsi_features, radiomics.features_)


@pytest.mark.integration
def test_ibsi_i_config_d(res3d_2mm_image_linear, res3d_2mm_mask_linear):
    ibsi_features = ibsi_i_feature_tolerances('config_D')

    radiomics = Radiomics(aggr_dim='3D',
                          aggr_method='AVER',
                          outlier_range=3,
                          number_of_bins=32,
                          calc_ivh_features=True)

    radiomics.extract_features(image=res3d_2mm_image_linear, mask=res3d_2mm_mask_linear)
    ibsi_i_validation(ibsi_features, radiomics.features_)

    radiomics = Radiomics(aggr_dim='3D',
                          aggr_method='MERG',
                          outlier_range=3,
                          number_of_bins=32,
                          calc_ivh_features=True)

    radiomics.extract_features(image=res3d_2mm_image_linear, mask=res3d_2mm_mask_linear)
    ibsi_i_validation(ibsi_features, radiomics.features_)


@pytest.mark.integration
def test_ibsi_i_config_e(res3d_2mm_image_spline, res3d_2mm_mask_linear):
    ibsi_features = ibsi_i_feature_tolerances('config_E')

    radiomics = Radiomics(aggr_dim='3D',
                          aggr_method='AVER',
                          intensity_range=[-1000, 400],
                          outlier_range=3,
                          number_of_bins=32,
                          calc_ivh_features=True,
                          ivh_number_of_bins=1000)

    radiomics.extract_features(image=res3d_2mm_image_spline, mask=res3d_2mm_mask_linear)
    ibsi_i_validation(ibsi_features, radiomics.features_)

    radiomics = Radiomics(aggr_dim='3D',
                          aggr_method='MERG',
                          intensity_range=[-1000, 400],
                          outlier_range=3,
                          number_of_bins=32,
                          calc_ivh_features=True,
                          ivh_number_of_bins=1000)

    radiomics.extract_features(image=res3d_2mm_image_spline, mask=res3d_2mm_mask_linear)
    ibsi_i_validation(ibsi_features, radiomics.features_)
