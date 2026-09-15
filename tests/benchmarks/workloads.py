"""One workload definition for timing and memory; constructors perform setup.

Operations return new results. Input arrays are read-only to detect accidental
mutation. Filter metadata preparation is part of apply; output caching is absent.
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

from zrad.filtering import Gabor, Laws, LoG, Mean, RieszLoG, Simoncelli, Wavelets2D, Wavelets3D
from zrad.image import Image
from zrad.preprocessing import (
    ImageResampler,
    IntensityMaskBuilder,
    IVHIntensityDiscretizer,
    MaskResampler,
    Pipeline,
    Resegmenter,
    RoiData,
    TextureDiscretizer,
)
from zrad.radiomics import Radiomics

IMAGE_SHAPES = {'small': (32, 96, 96), 'medium': (64, 128, 128), 'large': (96, 192, 192)}
ROI_SHAPES = {'small': (16, 24, 24), 'medium': (24, 40, 40), 'large': (40, 64, 64)}
DATA = Path(__file__).resolve().parents[1] / 'data'


@dataclass
class Workload:
    identifier: str
    operation: Callable
    validate: Callable
    metadata: dict
    rounds: int = 7
    setup: Callable | None = None


def image_from_array(array, spacing=(1.0, 1.0, 2.0)):
    return Image(
        array=array,
        origin=(0.0, 0.0, 0.0),
        spacing=spacing,
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        shape=array.shape[::-1],
    )


def synthetic_pair(shape):
    rng = np.random.default_rng(20260911)
    axes = np.ogrid[tuple(slice(0, size) for size in shape)]
    distance = sum(((axis - (size - 1) / 2) / (size * 0.38)) ** 2 for axis, size in zip(axes, shape))
    # Smooth anatomical-scale structure plus nonconstant texture; no huge bin range.
    array = np.rint(80 * np.cos(distance * 2) + rng.normal(0, 15, shape)).astype(np.float64)
    mask = (distance < 1).astype(np.float64)
    array.flags.writeable = mask.flags.writeable = False
    return image_from_array(array), image_from_array(mask)


def metadata(image, mask=None, **parameters):
    info = {
        'shape_zyx': list(image.array.shape),
        'voxel_count': int(image.array.size),
        'spacing_xyz_mm': list(image.spacing),
        'dtype': str(image.array.dtype),
        'dimensionality': '3D',
        'seed': 20260911,
        **parameters,
    }
    if mask is not None:
        info.update(roi_voxels=int(np.count_nonzero(mask.array)), roi_fraction=float(np.mean(mask.array > 0)))
    return info


def validate_image(result, expected_shape=None):
    assert isinstance(result, Image)
    assert result.array.size and np.isfinite(result.array).all()
    if expected_shape is not None:
        assert result.array.shape == expected_shape


def validate_features(result):
    assert result and all(np.isfinite(value) for value in result.values())


def prepare_roi(image, mask):
    roi = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))
    roi = Resegmenter(intensity_range=(-1000, 400)).apply(roi)
    roi = TextureDiscretizer(number_of_bins=32).apply(roi)
    return IVHIntensityDiscretizer(method='direct').apply(roi)


def resampling(size, method='Linear', target=False, mask=False, dimension='3D'):
    image, roi_mask = synthetic_pair(IMAGE_SHAPES[size])
    source = roi_mask if mask else image
    if dimension == '3D':
        resolution = (1.5, 1.5, 1.5)
    elif dimension == '2D':
        resolution = (1.5, 1.5, source.spacing[2])
    else:
        raise ValueError(f'Unsupported resample dimension {dimension!r}.')
    resampler = (MaskResampler if mask else ImageResampler)(resolution=resolution, method=method)
    if target:
        if dimension != '3D':
            raise ValueError('Target-grid benchmark uses the 3D resampling grid.')
        reference = resampler.apply(image)
        operation = partial(image.resample_to_target, reference)
        identifier = f'image/target_linear/{size}'
        shape = reference.array.shape
    else:
        operation = partial(resampler.apply, source)
        name = f'{"mask" if mask else "image"}_{method.lower()}'
        if dimension == '2D':
            name += '_in_plane'
        identifier = f'preprocessing/{name}/{size}'
        shape = tuple(np.ceil(np.array(source.shape) * np.array(source.spacing) / resolution).astype(int)[::-1])

    def validate(result):
        validate_image(result, shape)
        assert tuple(result.spacing) == resolution
        if mask:
            assert set(np.unique(result.array)) <= {0, 1}
            assert np.any(result.array > 0)
        if dimension == '2D':
            assert result.array.shape[0] == source.array.shape[0]

    return Workload(
        identifier,
        operation,
        validate,
        metadata(
            source,
            roi_mask,
            interpolation=method,
            resample_dimension=dimension,
            output_spacing_xyz_mm=list(resolution),
            output_shape_zyx=[int(axis) for axis in shape],
            output_voxel_count=int(np.prod(shape)),
        ),
        rounds=5 if size == 'large' else 7,
    )


def target_grid_alignment(size):
    image, roi_mask = synthetic_pair(IMAGE_SHAPES[size])
    spacing = (1.5, 1.5, 1.5)
    source_extent = np.array(image.shape) * np.array(image.spacing)
    target_shape = np.ceil(source_extent / spacing * (1.1, 0.9, 1.0)).astype(int)
    target_origin = tuple(source_extent * (0.15, -0.1, 0.2))
    target_array = np.zeros(tuple(target_shape[::-1]), dtype=np.float64)
    target_array.flags.writeable = False
    target = Image(
        array=target_array,
        origin=target_origin,
        spacing=spacing,
        direction=image.direction,
        shape=tuple(target_shape),
    )
    background = float(np.nanmin(image.array))

    def validate(result):
        validate_image(result, target_array.shape)
        assert tuple(result.origin) == target.origin
        assert tuple(result.spacing) == target.spacing
        assert tuple(result.direction) == target.direction
        assert tuple(result.shape) == target.shape
        assert result.array[0, 0, 0] == background
        assert result.array[-1, -1, -1] == background
        assert result.array[tuple(dimension // 2 for dimension in result.array.shape)] > background

    return Workload(
        f'image/target_partial_overlap_linear/{size}',
        partial(image.resample_to_target, target),
        validate,
        metadata(
            image,
            roi_mask,
            interpolation='Linear',
            target_shape_zyx=list(target_array.shape),
            target_origin_xyz_mm=list(target_origin),
            target_spacing_xyz_mm=list(spacing),
        ),
        rounds=5 if size == 'large' else 7,
    )


def preprocessing(operation):
    image, mask = synthetic_pair(IMAGE_SHAPES['medium'])
    roi = RoiData(image=image, morphological_mask=mask)
    step = IntensityMaskBuilder()
    if operation != 'roi':
        roi = step.apply(roi)
        step = (
            TextureDiscretizer(number_of_bins=32)
            if operation == 'discretize'
            else Resegmenter(intensity_range=(-50, 100), outlier_range=3)
        )

    def validate(result):
        values = (result.texture_discretized_image if operation == 'discretize' else result.intensity_mask).array
        assert np.isfinite(values).any() and not np.isinf(values).any()
        if operation == 'discretize':
            assert np.nanmin(values) == 1 and np.nanmax(values) == 32

    return Workload(
        f'preprocessing/{operation}/medium',
        partial(step.apply, roi),
        validate,
        metadata(image, mask, operation=operation, number_of_bins=32 if operation == 'discretize' else None),
    )


def texture_fixed_bin_size():
    image, mask = synthetic_pair(IMAGE_SHAPES['medium'])
    roi = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))
    roi = Resegmenter(intensity_range=(-50, 100)).apply(roi)
    step = TextureDiscretizer(bin_size=25)

    def validate(result):
        values = result.texture_discretized_image.array
        assert result.intensity_range == (-50.0, 100.0)
        assert np.isfinite(values).any() and not np.isinf(values).any()
        assert np.nanmin(values) == 1 and np.nanmax(values) == 7

    return Workload(
        'preprocessing/texture_fixed_bin_size_25/medium',
        partial(step.apply, roi),
        validate,
        metadata(
            image,
            mask,
            operation='texture_fixed_bin_size',
            bin_size=25,
            intensity_range=list(roi.intensity_range),
            prepared_roi_voxels=int(np.isfinite(roi.intensity_mask.array).sum()),
        ),
    )


def filtering(kind, size):
    image, _ = synthetic_pair(IMAGE_SHAPES[size])
    filters = {
        'mean': lambda: Mean(padding_type='reflect', support=5, dimensionality='3D'),
        'log': lambda: LoG(padding_type='reflect', sigma_mm=2.0, cutoff=4, dimensionality='3D'),
        'wavelet': lambda: Wavelets3D(
            wavelet_type='db3',
            padding_type='reflect',
            response_map='HHL',
            decomposition_level=1,
            rotation_invariance=False,
        ),
        'riesz': lambda: RieszLoG(
            padding_type='reflect', sigma_mm=2.0, cutoff=4, dimensionality='3D', riesz_order=(1, 0, 0)
        ),
        'laws': lambda: Laws(
            response_map='E3W5R5',
            padding_type='reflect',
            dimensionality='3D',
            rotation_invariance=True,
            pooling='max',
            energy_map=True,
            distance=7,
        ),
        'gabor': lambda: Gabor(
            padding_type='reflect',
            res_mm=1.0,
            sigma_mm=5.0,
            lambda_mm=2.0,
            gamma=1.5,
            theta=np.pi / 8,
            rotation_invariance=True,
            orthogonal_planes=False,
        ),
        'wavelet_2d': lambda: Wavelets2D(
            wavelet_type='db3',
            padding_type='reflect',
            response_map='LH',
            decomposition_level=2,
            rotation_invariance=True,
        ),
        'simoncelli': lambda: Simoncelli(
            padding_type='periodic', decomposition_level=2, dimensionality='3D'
        ),
        'riesz_simoncelli': lambda: Simoncelli(
            padding_type='periodic', decomposition_level=1, dimensionality='3D', riesz_order=(0, 2, 0)
        ),
    }
    flt = filters[kind]()
    return Workload(
        f'filtering/{kind}/{size}',
        partial(flt.apply, image),
        partial(validate_image, expected_shape=image.array.shape),
        metadata(image, filter=flt.get_params()),
        rounds=5 if size == 'large' or kind in {'laws', 'gabor'} else 7,
    )


def clear_radiomics_result_caches():
    """Per-round setup only: exclude retained image results from extraction."""
    from zrad.radiomics.intensity import _LOCAL_MEANS_CACHE

    # This is the only module-level radiomics result cache. New contexts and
    # feature groups alone do not reset it because its key includes array identity.
    _LOCAL_MEANS_CACHE.clear()


def radiomics(size, family='all', aggregation=('3D', 'MERG')):
    image, mask = synthetic_pair(ROI_SHAPES[size])
    roi = prepare_roi(image, mask)
    for value in vars(roi).values():
        if isinstance(value, Image):
            value.array.flags.writeable = False
    extractor = Radiomics(aggr_dim=aggregation[0], aggr_method=aggregation[1])
    if family == 'spatial':
        operation = partial(extractor.extract_features, roi_data=roi, features=['morph_moran_i', 'morph_geary_c'])
    else:
        families = ['glcm', 'glrlm', 'glszm', 'gldzm', 'ngtdm', 'ngldm'] if family == 'texture' else family
        operation = partial(extractor.extract_features, roi_data=roi, families=families)

    def validate(result):
        validate_features(result)
        if family == 'all':
            assert len(result) > 150 and 'morph_moran_i' in result and 'ivh_v10' in result
        if family == 'spatial':
            assert set(result) == {'morph_moran_i', 'morph_geary_c'}

    family_id = 'all_fresh' if family == 'all' else family
    aggregation_id = f'/{aggregation[0].lower().replace(".", "_")}/{aggregation[1].lower()}'
    if aggregation == ('3D', 'MERG'):
        aggregation_id = ''
    return Workload(
        f'radiomics/{family_id}/{size}{aggregation_id}',
        operation,
        validate,
        metadata(
            image,
            mask,
            feature_group=family,
            number_of_bins=32,
            aggregation='/'.join(aggregation),
            **({'cache_policy': 'fresh_image_results'} if family == 'all' else {}),
        ),
        rounds=5,
        setup=clear_radiomics_result_caches if family in ('all', 'local_intensity') else None,
    )


def pipeline():
    image, mask = synthetic_pair(ROI_SHAPES['large'])
    flt = LoG(padding_type='reflect', sigma_mm=2.0, cutoff=4, dimensionality='3D')
    steps = Pipeline(
        [
            ('image', ImageResampler(resolution=(1.5, 1.5, 1.5))),
            ('mask', MaskResampler(resolution=(1.5, 1.5, 1.5))),
            ('filter', flt),
            ('roi', IntensityMaskBuilder()),
            ('range', Resegmenter(intensity_range=(-1000, 400))),
            ('texture', TextureDiscretizer(number_of_bins=32)),
            ('ivh', IVHIntensityDiscretizer(method='direct')),
        ]
    )
    roi = RoiData(image=image, morphological_mask=mask)
    extractor = Radiomics(aggr_dim='3D', aggr_method='MERG')

    def operation():
        return extractor.extract_features(roi_data=steps.apply(roi), families='all')

    return Workload(
        'pipeline/log_radiomics/large',
        operation,
        validate_features,
        metadata(image, mask, steps=steps.get_params()),
        rounds=3,
    )


def load_ct(directory):
    image = Image.from_dicom(directory / 'dicom/image', modality='CT')
    mask = Image.from_dicom_mask(
        reference=image, rtstruct_path=directory / 'dicom/mask/DCM_RS_00060.dcm', structure_name='GTV-1'
    )
    image.array.flags.writeable = mask.array.flags.writeable = False
    return image, mask


def ibsi(phase, pair):
    # Reuse published reference selection/tolerances without invoking pytest tests.
    from ibsi_helpers import load_references, matches_reference, select_ibsi_i_references

    image, mask = pair
    if phase == 'i':
        references = select_ibsi_i_references(
            load_references(
                DATA / 'ibsi_1_reference_data/ibsi_1_reference_values_config_C.csv',
                'tag',
                'reference value',
                delimiter=',',
                phase='I',
            ),
            '3D',
            'MERG',
        )
        value_key = 'reference value'
        steps = Pipeline(
            [
                ('image', ImageResampler(resolution=(2, 2, 2), method='Linear', intensity_rounding='nearest_integer')),
                ('mask', MaskResampler(resolution=(2, 2, 2), method='Linear', partial_volume_threshold=0.5)),
                ('roi', IntensityMaskBuilder()),
                ('range', Resegmenter(intensity_range=(-1000, 400))),
                ('texture', TextureDiscretizer(bin_size=25)),
                ('ivh', IVHIntensityDiscretizer(method='fixed_bin_size', bin_size=2.5)),
            ]
        )
        extractor = Radiomics(aggr_dim='3D', aggr_method='MERG')
        identifier = 'ibsi/i_config_c'
    else:
        references = load_references(
            DATA / 'ibsi_2_reference_data/reference_feature_values/reference_values.csv',
            'feature_tag',
            'consensus_value',
            delimiter=';',
            phase='II',
            config='3.B',
        )
        value_key = 'consensus_value'
        steps = Pipeline(
            [
                ('image', ImageResampler(resolution=(1, 1, 1), method='BSpline', intensity_rounding='nearest_integer')),
                ('mask', MaskResampler(resolution=(1, 1, 1), method='Linear', partial_volume_threshold=0.5)),
                ('filter', LoG(padding_type='reflect', dimensionality='3D', sigma_mm=1.5, cutoff=4)),
                ('roi', IntensityMaskBuilder()),
                ('range', Resegmenter(intensity_range=(-1000, 400))),
                ('texture', TextureDiscretizer(bin_size=25)),
            ]
        )
        # Matches test_ibsi_ii_ph_ii_3b: 3D filtering, 2D/AVER feature aggregation.
        extractor = Radiomics(aggr_dim='2D', aggr_method='AVER')
        identifier = 'ibsi/ii_config_3b'
    roi = RoiData(image=image, morphological_mask=mask)

    def operation():
        return extractor.extract_features(roi_data=steps.apply(roi))

    def validate(result):
        assert references
        for tag, row in references.items():
            assert tag in result, tag
            assert matches_reference(result[tag], row[value_key], row['tolerance']), (tag, result[tag], row)

    return Workload(
        identifier,
        operation,
        validate,
        metadata(
            image,
            mask,
            dataset='IBSI CT radiomics phantom',
            seed=None,
            steps=steps.get_params(),
            aggregation=f'{extractor.aggr_dim}/{extractor.aggr_method}',
        ),
        rounds=3,
    )


def pet_suv(directory, io=False):
    from pydicom import dcmread

    from zrad.io.pet_suv import _enhanced_suv_array, is_enhanced_pet

    mask = Image.from_nifti(directory / 'DRO_error_5_0/mask/DRO_mask.nii.gz').array > 0
    if io:
        source = directory / 'DRO_0_0/PT'
        operation = partial(Image.from_dicom, dicom_dir=source, modality='PET')
        identifier = 'pet_suv/dicom_load_and_convert/dro_0_0'
        shape = mask.shape
    else:
        source = directory / 'DRO_7_0_0/PT'
        paths = sorted(path for path in source.rglob('*') if path.is_file() and not path.name.startswith('.'))
        assert len(paths) == 1, paths
        ds = dcmread(paths[0])
        assert is_enhanced_pet(ds)
        pixels = ds.pixel_array  # decode/cache explicitly outside timing
        pixels.flags.writeable = False
        operation = partial(_enhanced_suv_array, ds)
        identifier = 'pet_suv/enhanced_conversion/dro_7_0_0'
        shape = pixels.shape

    def validate(result):
        array = result.array if isinstance(result, Image) else result
        actual = tuple(round(float(fn(array[mask])), 2) for fn in (np.min, np.median, np.max))
        assert actual == (0.2, 1.0, 4.0)

    return Workload(
        identifier,
        operation,
        validate,
        {
            'dataset': source.parent.name,
            'shape_zyx': list(shape),
            'roi_voxels': int(mask.sum()),
            'io_included': io,
            'pixel_decode_included': io,
            'cache_policy': 'warm decoded pixels / OS cache',
        },
        rounds=5,
    )
