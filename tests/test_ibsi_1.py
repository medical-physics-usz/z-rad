import csv
from pathlib import Path

import pytest

from zrad.image import Image
from zrad.preprocessing import (
    ImageResampler,
    IntensityMaskBuilder,
    IVHIntensityDiscretizer,
    MaskResampler,
    Resegmenter,
    RoiData,
    TextureDiscretizer,
)
from zrad.radiomics import Radiomics


def ibsi_i_feature_tolerances(sheet_name):
    csv_path = Path(__file__).parent / 'data' / f'ibsi_1_reference_values_{sheet_name}.csv'
    with open(csv_path, newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        return {row['tag']: row for row in reader}


def ibsi_i_validation(ibsi_features, features, config_a=False):
    for raw_tag, feature_info in ibsi_features.items():
        tag = str(raw_tag)
        if config_a and tag == 'ih_qcod':
            continue

        if tag in features:
            val = float(feature_info['reference value'])
            tol = float(feature_info['tolerance'])
            upper_boundary = val + tol
            lower_boundary = val - tol

            if not (lower_boundary <= features[tag] <= upper_boundary):
                pytest.fail(
                    f"Feature {tag} out of tolerance: {features[tag]} not in range ({lower_boundary}, {upper_boundary})"
                )


@pytest.fixture()
def dcm_ct_phantom_image(ibsi_i_data_dir):
    return Image.from_dicom(dicom_dir=ibsi_i_data_dir / 'dicom' / 'image', modality='CT')


@pytest.fixture()
def dcm_ct_phantom_mask(dcm_ct_phantom_image):
    return Image.from_dicom_mask(
        reference=dcm_ct_phantom_image,
        rtstruct_path='tests/data/IBSI_I/dicom/mask/DCM_RS_00060.dcm',
        structure_name='GTV-1',
    )


@pytest.fixture()
def nii_ct_phantom_image(ibsi_i_data_dir):
    return Image.from_nifti(str(ibsi_i_data_dir / 'nifti' / 'image' / 'phantom.nii.gz'))


@pytest.fixture()
def nii_ct_phantom_mask(nii_ct_phantom_image):
    return Image.from_nifti_mask(
        reference=nii_ct_phantom_image,
        mask_path='tests/data/IBSI_I/nifti/mask/mask.nii.gz',
    )


def _resolution(image, value, dimension):
    if dimension == '2D':
        return (value, value, image.spacing[2])
    return (value, value, value)


def _prepare_roi_data(
    image,
    mask,
    intensity_range=None,
    outlier_range=None,
    number_of_bins=None,
    bin_size=None,
    ivh_method=None,
    ivh_number_of_bins=None,
    ivh_bin_size=None,
):
    roi_data = IntensityMaskBuilder().apply(
        RoiData(
            image=image,
            morphological_mask=mask,
        )
    )
    roi_data = Resegmenter(
        intensity_range=intensity_range,
        outlier_range=outlier_range,
    ).apply(roi_data)
    if number_of_bins is not None or bin_size is not None:
        roi_data = TextureDiscretizer(
            number_of_bins=number_of_bins,
            bin_size=bin_size,
        ).apply(roi_data)
    if ivh_method is not None:
        roi_data = IVHIntensityDiscretizer(
            method=ivh_method,
            number_of_bins=ivh_number_of_bins,
            bin_size=ivh_bin_size,
        ).apply(roi_data)
    return roi_data


def _extract_features(image, mask, aggr_dim, aggr_method, **prep_kwargs):
    return Radiomics(
        aggr_dim=aggr_dim,
        aggr_method=aggr_method,
    ).extract_features(
        roi_data=_prepare_roi_data(image, mask, **prep_kwargs),
    )


@pytest.fixture()
def res2d_2mm_image_linear(nii_ct_phantom_image):

    preprocessing = ImageResampler(
        resolution=_resolution(nii_ct_phantom_image, 2, '2D'),
        method='Linear',
        intensity_rounding='nearest_integer',
    )
    res_image = preprocessing.apply(nii_ct_phantom_image)

    return res_image


@pytest.fixture()
def res2d_2mm_mask_linear(nii_ct_phantom_mask):

    preprocessing = MaskResampler(
        resolution=_resolution(nii_ct_phantom_mask, 2, '2D'),
        method='Linear',
        partial_volume_threshold=0.5,
    )
    res_mask = preprocessing.apply(nii_ct_phantom_mask)

    return res_mask


@pytest.fixture()
def res3d_2mm_image_linear(nii_ct_phantom_image):

    preprocessing = ImageResampler(
        resolution=_resolution(nii_ct_phantom_image, 2, '3D'),
        method='Linear',
        intensity_rounding='nearest_integer',
    )
    res_image = preprocessing.apply(nii_ct_phantom_image)

    return res_image


@pytest.fixture()
def res3d_2mm_mask_linear(nii_ct_phantom_mask):

    preprocessing = MaskResampler(
        resolution=_resolution(nii_ct_phantom_mask, 2, '3D'),
        method='Linear',
        partial_volume_threshold=0.5,
    )
    res_mask = preprocessing.apply(nii_ct_phantom_mask)

    return res_mask


@pytest.fixture()
def res3d_2mm_image_spline(dcm_ct_phantom_image):

    preprocessing = ImageResampler(
        resolution=_resolution(dcm_ct_phantom_image, 2, '3D'),
        method='BSpline',
        intensity_rounding='nearest_integer',
    )
    res_image = preprocessing.apply(dcm_ct_phantom_image)

    return res_image


@pytest.mark.integration
def test_ibsi_i_config_a(dcm_ct_phantom_image, dcm_ct_phantom_mask):
    ibsi_features = ibsi_i_feature_tolerances('config_A')

    features = _extract_features(
        dcm_ct_phantom_image,
        dcm_ct_phantom_mask,
        aggr_dim='2D',
        aggr_method='AVER',
        intensity_range=[-500, 400],
        bin_size=25,
        ivh_method='direct',
    )
    ibsi_i_validation(ibsi_features, features, True)

    features = _extract_features(
        dcm_ct_phantom_image,
        dcm_ct_phantom_mask,
        aggr_dim='2D',
        aggr_method='SLICE_MERG',
        intensity_range=[-500, 400],
        bin_size=25,
        ivh_method='direct',
    )
    ibsi_i_validation(ibsi_features, features, True)

    features = _extract_features(
        dcm_ct_phantom_image,
        dcm_ct_phantom_mask,
        aggr_dim='2.5D',
        aggr_method='DIR_MERG',
        intensity_range=[-500, 400],
        bin_size=25,
        ivh_method='direct',
    )
    ibsi_i_validation(ibsi_features, features, True)

    features = _extract_features(
        dcm_ct_phantom_image,
        dcm_ct_phantom_mask,
        aggr_dim='2.5D',
        aggr_method='MERG',
        intensity_range=[-500, 400],
        bin_size=25,
        ivh_method='direct',
    )
    ibsi_i_validation(ibsi_features, features, True)


@pytest.mark.integration
def test_ibsi_i_config_b(res2d_2mm_image_linear, res2d_2mm_mask_linear):
    ibsi_features = ibsi_i_feature_tolerances('config_B')

    features = _extract_features(
        res2d_2mm_image_linear,
        res2d_2mm_mask_linear,
        aggr_dim='2D',
        aggr_method='AVER',
        intensity_range=[-500, 400],
        number_of_bins=32,
    )
    ibsi_i_validation(ibsi_features, features)
    features = _extract_features(
        res2d_2mm_image_linear,
        res2d_2mm_mask_linear,
        aggr_dim='2D',
        aggr_method='SLICE_MERG',
        intensity_range=[-500, 400],
        number_of_bins=32,
        ivh_method='direct',
    )
    ibsi_i_validation(ibsi_features, features)

    features = _extract_features(
        res2d_2mm_image_linear,
        res2d_2mm_mask_linear,
        aggr_dim='2.5D',
        aggr_method='DIR_MERG',
        intensity_range=[-500, 400],
        number_of_bins=32,
        ivh_method='direct',
    )
    ibsi_i_validation(ibsi_features, features)

    features = _extract_features(
        res2d_2mm_image_linear,
        res2d_2mm_mask_linear,
        aggr_dim='2.5D',
        aggr_method='MERG',
        intensity_range=[-500, 400],
        number_of_bins=32,
        ivh_method='direct',
    )
    ibsi_i_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_i_config_c(res3d_2mm_image_linear, res3d_2mm_mask_linear):
    ibsi_features = ibsi_i_feature_tolerances('config_C')

    features = _extract_features(
        res3d_2mm_image_linear,
        res3d_2mm_mask_linear,
        aggr_dim='3D',
        aggr_method='AVER',
        intensity_range=[-1000, 400],
        bin_size=25,
        ivh_method='fixed_bin_size',
        ivh_bin_size=2.5,
    )
    ibsi_i_validation(ibsi_features, features)

    features = _extract_features(
        res3d_2mm_image_linear,
        res3d_2mm_mask_linear,
        aggr_dim='3D',
        aggr_method='MERG',
        intensity_range=[-1000, 400],
        bin_size=25,
        ivh_method='fixed_bin_size',
        ivh_bin_size=2.5,
    )
    ibsi_i_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_i_config_d(res3d_2mm_image_linear, res3d_2mm_mask_linear):
    ibsi_features = ibsi_i_feature_tolerances('config_D')

    features = _extract_features(
        res3d_2mm_image_linear,
        res3d_2mm_mask_linear,
        aggr_dim='3D',
        aggr_method='AVER',
        outlier_range=3,
        number_of_bins=32,
        ivh_method='direct',
    )
    ibsi_i_validation(ibsi_features, features)

    features = _extract_features(
        res3d_2mm_image_linear,
        res3d_2mm_mask_linear,
        aggr_dim='3D',
        aggr_method='MERG',
        outlier_range=3,
        number_of_bins=32,
        ivh_method='direct',
    )
    ibsi_i_validation(ibsi_features, features)


@pytest.mark.integration
def test_ibsi_i_config_e(res3d_2mm_image_spline, res3d_2mm_mask_linear):
    ibsi_features = ibsi_i_feature_tolerances('config_E')

    features = _extract_features(
        res3d_2mm_image_spline,
        res3d_2mm_mask_linear,
        aggr_dim='3D',
        aggr_method='AVER',
        intensity_range=[-1000, 400],
        outlier_range=3,
        number_of_bins=32,
        ivh_method='fixed_bin_number',
        ivh_number_of_bins=1000,
    )
    ibsi_i_validation(ibsi_features, features)

    features = _extract_features(
        res3d_2mm_image_spline,
        res3d_2mm_mask_linear,
        aggr_dim='3D',
        aggr_method='MERG',
        intensity_range=[-1000, 400],
        outlier_range=3,
        number_of_bins=32,
        ivh_method='fixed_bin_number',
        ivh_number_of_bins=1000,
    )
    ibsi_i_validation(ibsi_features, features)
