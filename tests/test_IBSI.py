import pytest
import pandas as pd

from zrad.image import Image
from zrad.preprocessing import Preprocessing
from zrad.radiomics import Radiomics


def ibsi_i_features(sheet_name):
    url = 'https://ibsi.radiomics.hevs.ch/assets/IBSI-1-submission-table.xlsx'
    return pd.read_excel(url, sheet_name=sheet_name)


def validation(ibsi_features, features, config_a=False):
    for tag in ibsi_features['tag']:
        if config_a and tag == 'ih_qcod':
            pass
        else:
            if str(tag) in features:
                val = float(ibsi_features[ibsi_features['tag'] == tag]['reference value'].iloc[0])
                tol = float(ibsi_features[ibsi_features['tag'] == tag]['tolerance'].iloc[0])
                upper_boundary = val + tol
                lower_boundary = val - tol
                if not (lower_boundary <= features[tag] <= upper_boundary):
                    assert False


@pytest.fixture()
def orig_image():
    image = Image()
    image.read_nifti_image(
        r'C:\Users\User\Desktop\data_sets-master\data_sets-master\ibsi_1_ct_radiomics_phantom\nifti\phantom.nii.gz')
    return image


@pytest.fixture()
def orig_mask():
    mask = Image()
    mask.read_nifti_image(
        r'C:\Users\User\Desktop\data_sets-master\data_sets-master\ibsi_1_ct_radiomics_phantom\nifti\mask.nii.gz')
    return mask


@pytest.fixture()
def res2d_2mm_image_linear(orig_image):

    preprocessing = Preprocessing(input_imaging_modality='CT',
                                  resample_resolution=2,
                                  resample_dimension='2D',
                                  interpolation_method='Linear')
    res_image = preprocessing.resample(orig_image, image_type='image')

    return res_image


@pytest.fixture()
def res2d_2mm_mask_linear(orig_mask):

    preprocessing = Preprocessing(input_imaging_modality='CT',
                                  resample_resolution=2,
                                  resample_dimension='2D',
                                  interpolation_method='Linear',
                                  interpolation_threshold=.5)
    res_mask = preprocessing.resample(orig_mask, image_type='mask')

    return res_mask


@pytest.fixture()
def res3d_2mm_image_linear(orig_image):

    preprocessing = Preprocessing(input_imaging_modality='CT',
                                  resample_resolution=2,
                                  resample_dimension='3D',
                                  interpolation_method='Linear')
    res_image = preprocessing.resample(orig_image, image_type='image')

    return res_image


@pytest.fixture()
def res3d_2mm_mask_linear(orig_mask):

    preprocessing = Preprocessing(input_imaging_modality='CT',
                                  resample_resolution=2,
                                  resample_dimension='3D',
                                  interpolation_method='Linear',
                                  interpolation_threshold=.5)
    res_mask = preprocessing.resample(orig_mask, image_type='mask')

    return res_mask


@pytest.fixture()
def res3d_2mm_image_spline(orig_image):

    preprocessing = Preprocessing(input_imaging_modality='CT',
                                  resample_resolution=2,
                                  resample_dimension='3D',
                                  interpolation_method='BSpline')
    res_image = preprocessing.resample(orig_image, image_type='image')

    return res_image


def test_ibsi_i_config_a(orig_image, orig_mask):
    ibsi_features = ibsi_i_features('config A')
    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='AVER',
                          intensity_range=[-500, 400],
                          bin_size=25)

    radiomics.extract_features(image=orig_image, mask=orig_mask)
    validation(ibsi_features, radiomics.features_, True)


def test_ibsi_i_config_b(res2d_2mm_image_linear, res2d_2mm_mask_linear):
    ibsi_features = ibsi_i_features('config B')
    radiomics = Radiomics(aggr_dim='2D',
                          aggr_method='AVER',
                          intensity_range=[-500, 400],
                          number_of_bins=32)

    radiomics.extract_features(image=res2d_2mm_image_linear, mask=res2d_2mm_mask_linear)
    validation(ibsi_features, radiomics.features_)


def test_ibsi_i_config_c(res3d_2mm_image_linear, res3d_2mm_mask_linear):
    ibsi_features = ibsi_i_features('config C')
    radiomics = Radiomics(aggr_dim='3D',
                          aggr_method='AVER',
                          intensity_range=[-1000, 400],
                          bin_size=25)

    radiomics.extract_features(image=res3d_2mm_image_linear, mask=res3d_2mm_mask_linear)
    validation(ibsi_features, radiomics.features_)


def test_ibsi_i_config_d(res3d_2mm_image_linear, res3d_2mm_mask_linear):
    ibsi_features = ibsi_i_features('config D')
    radiomics = Radiomics(aggr_dim='3D',
                          aggr_method='AVER',
                          outlier_range=3,
                          number_of_bins=32)

    radiomics.extract_features(image=res3d_2mm_image_linear, mask=res3d_2mm_mask_linear)
    validation(ibsi_features, radiomics.features_)


def test_ibsi_i_config_e(res3d_2mm_image_spline, res3d_2mm_mask_linear):
    ibsi_features = ibsi_i_features('config E')
    radiomics = Radiomics(aggr_dim='3D',
                          aggr_method='AVER',
                          intensity_range=[-1000, 400],
                          outlier_range=3,
                          number_of_bins=32)

    radiomics.extract_features(image=res3d_2mm_image_spline, mask=res3d_2mm_mask_linear)
    validation(ibsi_features, radiomics.features_)
