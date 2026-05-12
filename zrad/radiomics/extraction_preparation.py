import numpy as np

from ..preprocessing import Resegmenter, RoiData, RoiMaskValidator
from ..preprocessing.discretization import ImageDiscretizer, IntensityVolumeHistogramDiscretizer
from .extraction_context import PreparedExtractionData


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
        discretized_intensity_image = discretize_intensity_image(
            context,
            analysis_masks.intensity_mask,
        )

    ivh_intensity_image = None
    ivh_min_intensity = None
    ivh_max_intensity = None
    ivh_discretization_step = 1
    if 'ivh_intensity_image' in required:
        (
            ivh_intensity_image,
            ivh_min_intensity,
            ivh_max_intensity,
            ivh_discretization_step,
        ) = prepare_ivh_intensity_image(context, analysis_masks.intensity_mask)

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
    roi_data = _with_validated_morphological_mask(context.roi_data, validated_mask)
    if not context.resegment_roi_data:
        return roi_data
    return Resegmenter(
        intensity_range=context.intensity_range,
        outlier_range=context.outlier_range,
    ).apply(roi_data)


def _with_validated_morphological_mask(roi_data, validated_mask):
    intensity_mask = roi_data.intensity_mask.copy()
    intensity_mask.array = np.where(
        (validated_mask.array > 0) & (~np.isnan(intensity_mask.array)),
        intensity_mask.array,
        np.nan,
    )
    return RoiData(
        image=roi_data.image,
        filtered_image=roi_data.filtered_image,
        morphological_mask=validated_mask,
        intensity_mask=intensity_mask,
    )


def discretize_intensity_image(context, intensity_mask):
    minimum = context.intensity_range[0] if context.intensity_range is not None else None
    return ImageDiscretizer(
        number_of_bins=context.number_of_bins,
        bin_size=context.bin_size,
        minimum=minimum,
    ).apply(intensity_mask)


def prepare_ivh_intensity_image(context, intensity_mask):
    """Return the image and axis settings used for IVH features."""
    if context.ivh_method == 'direct':
        if context.intensity_range is not None:
            min_intensity = context.intensity_range[0]
            max_intensity = (
                context.intensity_range[1]
                if np.isfinite(context.intensity_range[1])
                else np.nanmax(intensity_mask.array)
            )
        else:
            min_intensity = np.nanmin(intensity_mask.array)
            max_intensity = np.nanmax(intensity_mask.array)
        return intensity_mask.copy(), min_intensity, max_intensity, 1

    minimum = context.intensity_range[0] if context.intensity_range is not None else None
    ivh_intensity_image = IntensityVolumeHistogramDiscretizer(
        number_of_bins=context.ivh_number_of_bins if context.ivh_method == 'fixed_bin_number' else None,
        bin_size=context.ivh_bin_size if context.ivh_method == 'fixed_bin_size' else None,
        minimum=minimum,
    ).apply(intensity_mask)
    if context.ivh_method == 'fixed_bin_number':
        return (
            ivh_intensity_image,
            np.nanmin(ivh_intensity_image.array),
            np.nanmax(ivh_intensity_image.array),
            1,
        )

    min_intensity = np.nanmin(ivh_intensity_image.array)
    max_intensity = np.nanmax(ivh_intensity_image.array)
    if context.intensity_range is not None:
        min_intensity = context.intensity_range[0] + 0.5 * context.ivh_bin_size
        if np.isfinite(context.intensity_range[1]):
            max_intensity = context.intensity_range[1] - 0.5 * context.ivh_bin_size
    return ivh_intensity_image, min_intensity, max_intensity, context.ivh_bin_size


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
