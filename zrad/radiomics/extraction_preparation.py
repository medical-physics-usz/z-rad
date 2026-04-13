import numpy as np

from ..exceptions import DataStructureError
from ..image import Image
from .extraction_context import PreparedExtractionData, PreparedMaskSet
from .intensity import build_ivh_mask


METADATA_REQUIREMENTS = frozenset({'analysis_masks', 'discretized_intensity_image'})


def collect_requirements(groups, *, include_metadata=False):
    required = set()
    for group in groups:
        required.update(group.requirements)
    if include_metadata:
        required.update(METADATA_REQUIREMENTS)
    return required


def prepare_extraction_data(context, groups, *, include_metadata=False):
    required = collect_requirements(groups, include_metadata=include_metadata)

    base_masks = None
    if 'base_masks' in required:
        base_masks = prepare_mask_set(context, validation_dim=None if context.is_slice_2d else '3D')

    analysis_masks = None
    if 'analysis_masks' in required:
        if context.is_slice_2d or context.aggr_dim == '3D':
            analysis_masks = base_masks or prepare_mask_set(
                context,
                validation_dim=None if context.is_slice_2d else '3D',
            )
        else:
            analysis_masks = prepare_mask_set(context, validation_dim=context.aggr_dim)

    discretized_intensity_image = None
    if 'discretized_intensity_image' in required:
        discretized_intensity_image = discretize_intensity_image(
            context,
            analysis_masks.intensity_mask,
        )

    ivh_intensity_image = None
    if 'ivh_intensity_image' in required:
        ivh_intensity_image = build_ivh_mask(
            context,
            analysis_masks.intensity_mask,
            bin_number_discretize=bin_number_discr,
            bin_size_discretize=bin_size_discr,
        )

    return PreparedExtractionData(
        base_masks=base_masks,
        analysis_masks=analysis_masks,
        discretized_intensity_image=discretized_intensity_image,
        ivh_intensity_image=ivh_intensity_image,
    )


def prepare_mask_set(context, validation_dim):
    validated_mask = context.mask.copy()
    if validation_dim is not None:
        validated_mask = validate_mask(validated_mask, validation_dim)

    morphological_mask = validated_mask.copy()
    morphological_mask.array = morphological_mask.array.astype(np.int8)

    intensity_mask = validated_mask.copy()
    intensity_mask.array = np.where(validated_mask.array > 0, context.feature_image.array, np.nan)
    intensity_mask = apply_intensity_filters(
        intensity_mask=intensity_mask,
        morphological_mask=morphological_mask,
        original_image=context.image,
        context=context,
    )

    return PreparedMaskSet(
        morphological_mask=morphological_mask,
        intensity_mask=intensity_mask,
    )


def validate_mask(mask, aggr_dim):
    masked_array = mask.array
    min_box_size = 3
    min_voxel_number_3d = 27
    min_voxel_number_2d = 9

    if aggr_dim == '3D':
        valid_coords = np.where(masked_array != 0)
        if len(valid_coords[0]) == 0:
            raise DataStructureError("No valid voxels in 3D array.")

        zmin, zmax = valid_coords[0].min(), valid_coords[0].max() + 1
        ymin, ymax = valid_coords[1].min(), valid_coords[1].max() + 1
        xmin, xmax = valid_coords[2].min(), valid_coords[2].max() + 1

        bbox_shape = (zmax - zmin, ymax - ymin, xmax - xmin)
        no_valid_voxels = len(valid_coords[0])

        if min(bbox_shape) < min_box_size:
            raise DataStructureError(f"3D bounding box dimension < {min_box_size}.")
        if no_valid_voxels < min_voxel_number_3d:
            raise DataStructureError(f"Valid voxel count < {min_voxel_number_3d}.")
    else:
        n_slices = masked_array.shape[0]
        for z_idx in range(n_slices):
            slice_arr = masked_array[z_idx, :, :]
            if not np.any(slice_arr):
                continue

            valid_coords = np.where(slice_arr != 0)
            no_valid_voxels = len(valid_coords[0])
            if no_valid_voxels == 0:
                continue

            ymin, ymax = valid_coords[0].min(), valid_coords[0].max() + 1
            xmin, xmax = valid_coords[1].min(), valid_coords[1].max() + 1
            height = ymax - ymin
            width = xmax - xmin

            if min(height, width) < min_box_size or no_valid_voxels < min_voxel_number_2d:
                slice_arr[:, :] = 0

        if not np.any(masked_array):
            raise DataStructureError(
                "Not a single slice meets the minimum 2D/2.5D requirements. "
                "Consider finer resampling or check the data."
            )

        mask.array = masked_array

    return mask


def apply_intensity_filters(intensity_mask, morphological_mask, original_image, context):
    result = intensity_mask

    if context.intensity_range is not None:
        intensity_range_mask = np.where(
            (original_image.array <= context.intensity_range[1])
            & (original_image.array >= context.intensity_range[0]),
            1,
            0,
        )
        result = Image(
            array=np.where(
                (intensity_range_mask > 0) & (~np.isnan(result.array)),
                result.array,
                np.nan,
            ),
            origin=result.origin,
            spacing=result.spacing,
            direction=result.direction,
            shape=result.shape,
        )

    if context.outlier_range is not None and str(context.outlier_range).strip().replace('.', '').isdigit():
        flattened_image = np.where(morphological_mask.array > 0, original_image.array, np.nan).ravel()
        valid_values = flattened_image[~np.isnan(flattened_image)]
        mean = np.mean(valid_values)
        std = np.std(valid_values)
        outlier_mask = np.where(
            (original_image.array <= mean + context.outlier_range * std)
            & (original_image.array >= mean - context.outlier_range * std)
            & (~np.isnan(result.array)),
            1,
            0,
        )

        result = Image(
            array=np.where((outlier_mask > 0) & (~np.isnan(result.array)), result.array, np.nan),
            origin=result.origin,
            spacing=result.spacing,
            direction=result.direction,
            shape=result.shape,
        )

    return result


def bin_size_discr(image, min_val, bin_size):
    return Image(
        array=np.floor((image.array - min_val) / bin_size) + 1,
        origin=image.origin,
        spacing=image.spacing,
        direction=image.direction,
        shape=image.shape,
    )


def bin_number_discr(image, bin_number):
    return Image(
        array=np.where(
            image.array != np.nanmax(image.array),
            np.floor(
                bin_number * (image.array - np.nanmin(image.array)) / (np.nanmax(image.array) - np.nanmin(image.array))
            ) + 1,
            bin_number,
        ),
        origin=image.origin,
        spacing=image.spacing,
        direction=image.direction,
        shape=image.shape,
    )


def discretize_intensity_image(context, intensity_mask):
    discretized = intensity_mask.copy()
    if context.bin_size is not None:
        if context.intensity_range is not None:
            min_val = context.intensity_range[0]
        else:
            min_val = np.nanmin(discretized.array)
        discretized = bin_size_discr(discretized, min_val, context.bin_size)
    if context.number_of_bins is not None:
        discretized = bin_number_discr(discretized, context.number_of_bins)
    return discretized


def build_extraction_metadata(prepared_data):
    morph_mask = prepared_data.require_analysis_masks().morphological_mask.array
    intensity_array = prepared_data.require_discretized_intensity_image().array
    bounding_box_min, no_voxels = calc_bounding_box_and_voxels(morph_mask)
    no_bins = calc_number_of_bins(intensity_array)
    return {
        'bounding_box_min': bounding_box_min,
        'no_voxels': no_voxels,
        'no_bins': no_bins,
    }


def calc_bounding_box_and_voxels(mask_array):
    if mask_array is None:
        return 0, 0

    valid_coords = np.argwhere(mask_array > 0)
    if valid_coords.size == 0:
        return 0, 0

    min_coords = valid_coords.min(axis=0)
    max_coords = valid_coords.max(axis=0)
    bbox_dims = max_coords - min_coords + 1
    return int(bbox_dims.min()), int(valid_coords.shape[0])


def calc_number_of_bins(intensity_array):
    if intensity_array is None:
        return 0
    valid_values = intensity_array[~np.isnan(intensity_array)]
    if valid_values.size == 0:
        return 0
    return int(np.unique(valid_values).size)
