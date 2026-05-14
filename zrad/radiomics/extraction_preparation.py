import numpy as np

from ..preprocessing import RoiData, RoiMaskValidator
from .extraction_context import PreparedExtractionData


METADATA_REQUIREMENTS = frozenset({'analysis_masks'})


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
    needs_analysis_masks = bool(
        {'analysis_masks', 'discretized_intensity_image', 'ivh_intensity_image'} & required
    )
    if needs_analysis_masks:
        if context.is_slice_2d or context.aggr_dim == '3D':
            analysis_masks = base_masks or prepare_mask_set(
                context,
                validation_dim=None if context.is_slice_2d else '3D',
            )
        else:
            analysis_masks = prepare_mask_set(context, validation_dim=context.aggr_dim)

    discretized_intensity_image = None
    if 'discretized_intensity_image' in required:
        discretized_intensity_image = analysis_masks.texture_discretized_image

    ivh_intensity_image = None
    ivh_min_intensity = None
    ivh_max_intensity = None
    ivh_discretization_step = 1
    if 'ivh_intensity_image' in required:
        ivh_intensity_image = analysis_masks.ivh_intensity_image
        ivh_min_intensity, ivh_max_intensity, ivh_discretization_step = _derive_ivh_parameters(analysis_masks)

    return PreparedExtractionData(
        base_masks=base_masks,
        analysis_masks=analysis_masks,
        discretized_intensity_image=discretized_intensity_image,
        ivh_intensity_image=ivh_intensity_image,
        ivh_min_intensity=ivh_min_intensity,
        ivh_max_intensity=ivh_max_intensity,
        ivh_discretization_step=ivh_discretization_step,
    )


def prepare_mask_set(context, validation_dim):
    validated_mask = RoiMaskValidator(validation_dim).apply(context.mask)
    return _with_validated_morphological_mask(context.roi_data, validated_mask)


def _with_validated_morphological_mask(roi_data, validated_mask):
    def apply_validated_mask(image):
        if image is None:
            return None
        masked = image.copy()
        masked.array = np.where(
            (validated_mask.array > 0) & (~np.isnan(masked.array)),
            masked.array,
            np.nan,
        )
        return masked

    return RoiData(
        image=roi_data.image,
        filtered_image=roi_data.filtered_image,
        morphological_mask=validated_mask,
        intensity_mask=apply_validated_mask(roi_data.intensity_mask),
        texture_discretized_image=apply_validated_mask(roi_data.texture_discretized_image),
        ivh_intensity_image=apply_validated_mask(roi_data.ivh_intensity_image),
        intensity_range=roi_data.intensity_range,
        ivh_discretization_method=roi_data.ivh_discretization_method,
        ivh_discretization_step=roi_data.ivh_discretization_step,
    )


def _derive_ivh_parameters(roi_data):
    method = roi_data.ivh_discretization_method
    step = roi_data.ivh_discretization_step
    if roi_data.ivh_intensity_image is None or method is None or step is None:
        raise RuntimeError('IVH intensity image and metadata were not prepared for this extraction.')

    valid_values = roi_data.ivh_intensity_image.array[~np.isnan(roi_data.ivh_intensity_image.array)]
    if valid_values.size == 0:
        raise RuntimeError('IVH intensity image contains no valid values.')

    if method == 'fixed_bin_number':
        return np.nanmin(valid_values), np.nanmax(valid_values), step

    observed_min = np.nanmin(valid_values)
    observed_max = np.nanmax(valid_values)
    if roi_data.intensity_range is None:
        return observed_min, observed_max, step

    lower, upper = roi_data.intensity_range
    if method == 'fixed_bin_size':
        minimum = lower + 0.5 * step
        maximum = observed_max if not np.isfinite(upper) else upper - 0.5 * step
        return minimum, maximum, step

    if method == 'direct':
        maximum = observed_max if not np.isfinite(upper) else upper
        return lower, maximum, step

    raise RuntimeError(f"Unsupported IVH discretization method: {method}.")


def build_extraction_metadata(prepared_data):
    morph_mask = prepared_data.require_analysis_masks().morphological_mask.array
    intensity_image = prepared_data.discretized_intensity_image
    bounding_box_min, no_voxels = calc_bounding_box_and_voxels(morph_mask)
    no_bins = 0 if intensity_image is None else calc_number_of_bins(intensity_image.array)
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
