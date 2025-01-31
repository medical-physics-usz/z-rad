import os

import pandas as pd
import pytest

from zrad.image import Image
from zrad.preprocessing import Preprocessing
from zrad.filtering import Filtering
from zrad.radiomics import Radiomics
from zrad.toolbox_logic import load_ibsi_phantom

def ibsi_ii_feature_tolerances(filter_id):
    url = "https://raw.githubusercontent.com/theibsi/ibsi_2_reference_data/main/reference_feature_values/reference_values.csv"
    df = pd.read_csv(url, delimiter=';')
    return df[df['filter_id'] == filter_id]

def ibsi_ii_validation(ibsi_features, features):

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
def ct_phantom_image():
    if not os.path.isdir('tests/test_data/IBSI_I'):
        load_ibsi_phantom(chapter=1, phantom='ct_radiomics', imaging_format="nifti", save_path='tests/test_data/IBSI_I')

    image = Image()
    image.read_nifti_image('tests/test_data/IBSI_I/image/phantom.nii.gz')
    return image


@pytest.fixture()
def ct_phantom_mask():
    mask = Image()
    mask.read_nifti_image('tests/test_data/IBSI_I/mask/mask.nii.gz')
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

def test_ibsi_ii_config_2a(ct_phantom_image, ct_phantom_mask):
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
    ibsi_ii_validation(ibsi_features, radiomics.features_)

def test_ibsi_ii_config_2b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
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
    ibsi_ii_validation(ibsi_features, radiomics.features_)

def test_ibsi_ii_config_3a(ct_phantom_image, ct_phantom_mask):
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
    ibsi_ii_validation(ibsi_features, radiomics.features_)

def test_ibsi_ii_config_3b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
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
    ibsi_ii_validation(ibsi_features, radiomics.features_)

def test_ibsi_ii_config_4a(ct_phantom_image, ct_phantom_mask):
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
    ibsi_ii_validation(ibsi_features, radiomics.features_)

def test_ibsi_ii_config_4b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
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
    ibsi_ii_validation(ibsi_features, radiomics.features_)

def test_ibsi_ii_config_6a(ct_phantom_image, ct_phantom_mask):
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
    ibsi_ii_validation(ibsi_features, radiomics.features_)

def test_ibsi_ii_config_6b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
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
    ibsi_ii_validation(ibsi_features, radiomics.features_)

def test_ibsi_ii_config_7a(ct_phantom_image, ct_phantom_mask):
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
    ibsi_ii_validation(ibsi_features, radiomics.features_)

def test_ibsi_ii_config_7b(res3d_1mm_image_spline, res3d_1mm_mask_linear):
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
    ibsi_ii_validation(ibsi_features, radiomics.features_)