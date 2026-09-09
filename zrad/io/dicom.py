import os
import warnings

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.errors import InvalidDicomError
from skimage import draw

from ..exceptions import DataStructureError, DataStructureWarning
from .pet_suv import apply_suv_correction, is_enhanced_pet, validate_pet_dicom_tags


def read_dicom_image(dicom_dir, modality):
    """Read a DICOM image series as a SimpleITK image."""
    dicom_files = get_dicom_files(directory=dicom_dir, modality=modality)
    if len(dicom_files) == 0:
        raise DataStructureError(f"No {modality} data found in {dicom_dir}. Patient skipped.")

    image = None
    if modality == "PET":
        validate_pet_dicom_tags(dicom_files)
    if modality in ["CT", "MRI", "PET"]:
        validate_z_spacing(dicom_files)
    if modality == "US":
        validate_ultrasound_dicom_tags(dicom_files)
    if modality in ["CT", "MRI", "PET", "MG", "US"]:
        enhanced_pet = modality == "PET" and all(is_enhanced_pet(dcm_file["ds"]) for dcm_file in dicom_files)
        image = process_dicom_series(
            dicom_files,
            modality,
            reorder_enhanced_frames=not enhanced_pet,
        )
    if modality == "PET":
        image = apply_suv_correction(dicom_files, image)
        if enhanced_pet:
            image = _reorder_enhanced_image_frames(dicom_files, image)
    if modality == "RTDOSE":
        image = read_dicom_dose(dicom_files[0]["file_path"])
    if image is None:
        raise DataStructureError(f"Unsupported DICOM modality {modality}.")
    return image


def read_dicom_mask(rtstruct_path, structure_name, reference_image, dicom_dir=None):
    """Read an RTSTRUCT or DICOM SEG mask aligned to a reference image."""
    dicom_data = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
    if getattr(dicom_data, "Modality", None) == "SEG":
        return extract_dicom_seg_mask(rtstruct_path, structure_name, reference_image, dicom_dir)
    return extract_dicom_mask(rtstruct_path, structure_name, reference_image)


def read_dicom_dose(rtdose_path):
    ds = pydicom.dcmread(rtdose_path)
    if ds.DoseUnits != "GY":
        raise DataStructureError(f"Only dose in Gy is supported. Provided {ds.DoseUnits}. Patient skipped")
    if ds.DoseType != "PHYSICAL":
        raise DataStructureError(f"Only physical dose is supported. Provided {ds.DoseType}. Patient skipped.")
    raw_dose_image = sitk.ReadImage(rtdose_path)
    dose_array = sitk.GetArrayFromImage(raw_dose_image) * ds.DoseGridScaling
    dose_image = sitk.GetImageFromArray(dose_array)
    dose_image.SetOrigin(raw_dose_image.GetOrigin())
    dose_image.SetSpacing(raw_dose_image.GetSpacing())
    dose_image.SetDirection(raw_dose_image.GetDirection())

    return dose_image


def remove_duplicate_slices(dicom_files_info):
    """Remove duplicate slices with identical ImagePositionPatient."""
    cleaned = []
    seen_ipps = set()
    duplicates = 0

    for info in dicom_files_info:
        ds = info["ds"]

        if "ImagePositionPatient" in ds:
            ipp = tuple(map(float, ds.ImagePositionPatient))
        else:
            ipp = None

        if ipp is None:
            cleaned.append(info)
            continue

        if ipp in seen_ipps:
            duplicates += 1
            continue

        seen_ipps.add(ipp)
        cleaned.append(info)
    if duplicates > 0:
        warnings.warn(
            f"Removed {duplicates} duplicate slice(s) with identical ImagePositionPatient.",
            DataStructureWarning,
        )

    return cleaned


def sort_by_geometric_position(dicom_files_info):
    """Sort slices by physical position along the slice normal."""

    def slice_distance(ds):
        iop = np.array(ds.ImageOrientationPatient, dtype=float)
        row = iop[:3]
        col = iop[3:]
        normal = np.cross(row, col)
        ipp = np.array(ds.ImagePositionPatient, dtype=float)
        return float(np.dot(ipp, normal))

    return sorted(dicom_files_info, key=lambda x: slice_distance(x["ds"]))


def get_dicom_files(directory, modality):
    modality_dicom = modality_mapping(modality)
    dicom_files_info = []
    if modality_dicom in ["CT", "PT", "MR"]:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(directory)
        selected_series = None
        for sid in series_ids:
            files = reader.GetGDCMSeriesFileNames(directory, sid)
            dcm = pydicom.dcmread(os.path.join(directory, files[0]), stop_before_pixels=True)
            if dcm.Modality == modality_dicom:
                selected_series = sid
                break
        if selected_series is None:
            raise DataStructureError(f"No {modality_dicom} series found for {directory}. Patient skipped")
        all_files = reader.GetGDCMSeriesFileNames(directory, selected_series)
    else:
        all_files = [os.path.join(directory, i) for i in os.listdir(directory)]
    for file_path in all_files:
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            if ds.Modality == modality_dicom:
                if hasattr(ds, "ImageType") and (
                    "LOCALIZER" in ds.ImageType or any("MIP" in entry for entry in ds.ImageType)
                ):
                    continue
                dicom_files_info.append({"file_path": file_path, "ds": ds})

        except InvalidDicomError:
            continue
        except Exception as e:
            warning_msg = f"An error occurred while processing file {file_path}: {str(e)}"
            warnings.warn(warning_msg, DataStructureWarning)

    enhanced_flags = [is_enhanced_pet(item["ds"]) for item in dicom_files_info] if modality_dicom == "PT" else []
    if any(enhanced_flags) and not all(enhanced_flags):
        raise DataStructureError("A PET series cannot mix Enhanced and conventional PET instances.")
    enhanced_series = bool(enhanced_flags) and all(enhanced_flags)

    if len(dicom_files_info) > 1 and not enhanced_series:
        signatures = []
        for item in dicom_files_info:
            ds = item["ds"]

            pix = tuple(ds.PixelSpacing) if hasattr(ds, "PixelSpacing") else None
            thick = getattr(ds, "SliceThickness", None)
            space = getattr(ds, "SpacingBetweenSlices", None)
            kernel = getattr(ds, "ConvolutionKernel", None)

            signatures.append((pix, thick, space, kernel))

        unique_keys = []
        for sig in signatures:
            if sig not in unique_keys:
                unique_keys.append(sig)

        group_counts = []
        for key in unique_keys:
            count = 0
            for sig in signatures:
                if sig == key:
                    count += 1
            group_counts.append(count)

        best_key = unique_keys[group_counts.index(max(group_counts))]

        filtered = []
        for item, sig in zip(dicom_files_info, signatures):
            if sig == best_key:
                filtered.append(item)

        if len(filtered) != len(dicom_files_info):
            warnings.warn(
                "Series contains mixed geometries; keeping the largest consistent set.",
                DataStructureWarning,
            )

        dicom_files_info = filtered
    if modality_dicom in ["CT", "PT", "MR"] and not enhanced_series:
        dicom_files_info = remove_duplicate_slices(dicom_files_info)
        dicom_files_info = sort_by_geometric_position(dicom_files_info)
    elif enhanced_series:
        dicom_files_info = _sort_enhanced_instances(dicom_files_info)
    return dicom_files_info


def _enhanced_functional_group_sequence(ds, frame_index, sequence_keyword):
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if per_frame is not None and frame_index < len(per_frame):
        frame_group = per_frame[frame_index]
        if hasattr(frame_group, sequence_keyword):
            sequence = getattr(frame_group, sequence_keyword)
            if sequence:
                return sequence
    shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if shared and hasattr(shared[0], sequence_keyword):
        return getattr(shared[0], sequence_keyword)
    return None


def _enhanced_frame_geometry(ds, frame_index):
    orientation_sequence = _enhanced_functional_group_sequence(ds, frame_index, "PlaneOrientationSequence")
    position_sequence = _enhanced_functional_group_sequence(ds, frame_index, "PlanePositionSequence")
    orientation_source = orientation_sequence[0] if orientation_sequence else ds
    position_source = position_sequence[0] if position_sequence else ds
    try:
        orientation = np.asarray(orientation_source.ImageOrientationPatient, dtype=float)
        position = np.asarray(position_source.ImagePositionPatient, dtype=float)
    except (AttributeError, IndexError, TypeError, ValueError):
        raise DataStructureError("Enhanced PET frame geometry is missing or invalid.")
    if orientation.shape != (6,) or position.shape != (3,) or not np.all(np.isfinite(position)):
        raise DataStructureError("Enhanced PET frame geometry has invalid dimensions or values.")

    row = orientation[:3]
    column = orientation[3:]
    row_norm = np.linalg.norm(row)
    column_norm = np.linalg.norm(column)
    if (
        not np.all(np.isfinite(orientation))
        or row_norm == 0
        or column_norm == 0
        or not np.isclose(np.dot(row / row_norm, column / column_norm), 0.0, rtol=0, atol=1e-4)
    ):
        raise DataStructureError("Enhanced PET Image Orientation (Patient) is invalid.")
    row = row / row_norm
    column = column / column_norm
    normal = np.cross(row, column)
    normal_norm = np.linalg.norm(normal)
    if not np.isfinite(normal_norm) or normal_norm == 0:
        raise DataStructureError("Enhanced PET Image Orientation (Patient) is invalid.")
    return np.concatenate((row, column)), position, normal / normal_norm


def _enhanced_frame_distances(dicom_files):
    distances = []
    reference_orientation = None
    reference_position = None
    reference_frame_of_reference_uid = None
    for dcm_file in dicom_files:
        ds = dcm_file["ds"]
        frame_of_reference_uid = str(getattr(ds, "FrameOfReferenceUID", "")).strip()
        if not frame_of_reference_uid:
            raise DataStructureError("Enhanced PET Frame of Reference UID (0020,0052) is missing or empty.")
        if reference_frame_of_reference_uid is None:
            reference_frame_of_reference_uid = frame_of_reference_uid
        elif frame_of_reference_uid != reference_frame_of_reference_uid:
            raise DataStructureError("Enhanced PET instances use inconsistent Frame of Reference UIDs.")
        try:
            frame_count = int(ds.NumberOfFrames)
        except (AttributeError, TypeError, ValueError):
            raise DataStructureError("Number of Frames (0028,0008) is missing or invalid for Enhanced PET.")
        groups = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
        if groups is None or len(groups) != frame_count:
            raise DataStructureError(
                "Per-Frame Functional Groups Sequence (5200,9230) does not match the number of Enhanced PET frames."
            )

        for frame_index in range(frame_count):
            orientation, position, normal = _enhanced_frame_geometry(ds, frame_index)
            if reference_orientation is None:
                reference_orientation = orientation
                reference_position = position
            elif not np.allclose(orientation, reference_orientation, rtol=0, atol=1e-6):
                raise DataStructureError("Enhanced PET frame orientations are inconsistent.")
            else:
                displacement = position - reference_position
                if (
                    abs(np.dot(displacement, reference_orientation[:3])) > 1e-4
                    or abs(np.dot(displacement, reference_orientation[3:])) > 1e-4
                ):
                    raise DataStructureError("Enhanced PET frames have inconsistent in-plane origins.")
            distances.append(float(np.dot(position, normal)))
    return distances


def _validated_enhanced_slice_spacing(distances):
    differences = np.diff(np.sort(np.asarray(distances, dtype=float)))
    if len(differences) == 0:
        return None, None
    if np.any(np.abs(differences) <= 1e-6):
        raise DataStructureError("Enhanced PET contains multiple frames at the same spatial position.")

    slice_spacings = differences
    reference_spacing = slice_spacings[0]
    spacing_deviations = np.abs(slice_spacings - reference_spacing)
    spacing_threshold = 0.1
    if np.any(spacing_deviations > spacing_threshold):
        maximum_deviation = float(np.max(spacing_deviations))
        raise DataStructureError(
            f"Inconsistent z-spacing. Absolute deviation is {maximum_deviation:.3f} which is greater than "
            f"{spacing_threshold:.3f} mm."
        )
    return float(np.median(slice_spacings)), 1


def _enhanced_frame_references(dicom_files):
    references = []
    for dcm_file in dicom_files:
        for frame_index in range(_enhanced_instance_frame_count(dcm_file)):
            references.append((dcm_file, frame_index))
    return references


def _enhanced_frame_order(dicom_files):
    distances = np.asarray(_enhanced_frame_distances(dicom_files), dtype=float)
    return np.argsort(distances, kind="stable")


def _reorder_enhanced_image_frames(dicom_files, image):
    frame_order = _enhanced_frame_order(dicom_files)
    frames = sitk.GetArrayFromImage(image)
    if frames.ndim == 2:
        frames = frames[np.newaxis, ...]
    if frames.ndim != 3 or frames.shape[0] != len(frame_order):
        raise DataStructureError("Enhanced PET pixel data dimensions do not match its frame geometry.")
    if np.array_equal(frame_order, np.arange(len(frame_order))):
        return image
    reordered = sitk.GetImageFromArray(frames[frame_order])
    reordered.CopyInformation(image)
    return reordered


def _enhanced_instance_frame_count(dcm_file):
    try:
        frame_count = int(dcm_file["ds"].NumberOfFrames)
    except (AttributeError, TypeError, ValueError):
        raise DataStructureError("Number of Frames (0028,0008) is missing or invalid for Enhanced PET.")
    if frame_count <= 0:
        raise DataStructureError("Number of Frames (0028,0008) must be positive for Enhanced PET.")
    return frame_count


def _sort_enhanced_instances(dicom_files):
    concatenation_uids = [str(getattr(item["ds"], "ConcatenationUID", "")).strip() for item in dicom_files]
    offsets = [getattr(item["ds"], "ConcatenationFrameOffsetNumber", None) for item in dicom_files]
    has_uid = [bool(value) for value in concatenation_uids]
    has_offset = [value not in [None, ""] for value in offsets]

    if any(has_uid) and not all(has_uid):
        raise DataStructureError("Enhanced PET instances cannot mix concatenation and non-concatenation objects.")
    if all(has_uid) and len(set(concatenation_uids)) != 1:
        raise DataStructureError("Enhanced PET instances have inconsistent Concatenation UIDs (0020,9161).")
    if any(has_offset) and not all(has_uid):
        raise DataStructureError(
            "Enhanced PET Concatenation Frame Offset Numbers require a consistent Concatenation UID."
        )
    if any(has_offset) and not all(has_offset):
        raise DataStructureError("Enhanced PET instances have incomplete Concatenation Frame Offset Numbers.")

    if all(has_offset):
        parsed_instances = []
        for item, offset in zip(dicom_files, offsets):
            try:
                offset = int(offset)
            except (TypeError, ValueError):
                raise DataStructureError("Concatenation Frame Offset Number (0020,9228) is invalid.")
            if offset < 0:
                raise DataStructureError("Concatenation Frame Offset Number (0020,9228) cannot be negative.")
            parsed_instances.append((offset, item))

        parsed_instances.sort(key=lambda entry: entry[0])
        expected_offset = 0
        for position, (offset, item) in enumerate(parsed_instances, start=1):
            if offset != expected_offset:
                raise DataStructureError("Enhanced PET concatenation has missing or overlapping frames.")
            expected_offset += _enhanced_instance_frame_count(item)

            in_concatenation_number = getattr(item["ds"], "InConcatenationNumber", None)
            if in_concatenation_number not in [None, ""]:
                try:
                    if int(in_concatenation_number) != position:
                        raise DataStructureError("Enhanced PET In-concatenation Numbers are inconsistent.")
                except (TypeError, ValueError):
                    raise DataStructureError("In-concatenation Number (0020,9162) is invalid.")

            total = getattr(item["ds"], "InConcatenationTotalNumber", None)
            if total not in [None, ""]:
                try:
                    if int(total) != len(dicom_files):
                        raise DataStructureError("Enhanced PET In-concatenation Total Number is inconsistent.")
                except (TypeError, ValueError):
                    raise DataStructureError("In-concatenation Total Number (0020,9163) is invalid.")
        return [item for _offset, item in parsed_instances]

    if any(has_uid):
        raise DataStructureError("Enhanced PET concatenation is missing Concatenation Frame Offset Numbers.")

    instance_distances = []
    for item in dicom_files:
        distances = _enhanced_frame_distances([item])
        instance_distances.append((min(distances), item))

    return [item for _distance, item in sorted(instance_distances, key=lambda entry: entry[0])]


def _dimension_index_values(frame_content, expected_count):
    raw_values = getattr(frame_content, "DimensionIndexValues", None)
    if raw_values in [None, ""]:
        raise DataStructureError("Enhanced PET frame is missing Dimension Index Values (0020,9157).")
    try:
        values = tuple(raw_values)
    except TypeError:
        values = (raw_values,)
    if len(values) != expected_count:
        raise DataStructureError("Enhanced PET Dimension Index Values do not match Dimension Index Sequence.")
    try:
        return tuple(int(value) for value in values)
    except (TypeError, ValueError):
        raise DataStructureError("Enhanced PET Dimension Index Values are invalid.")


def _validate_enhanced_spatial_dimensions(dicom_files):
    spatial_pointers = {0x00200032, 0x00209057}  # Image Position (Patient), In-Stack Position Number
    reference_pointers = None
    reference_non_spatial_values = None
    reference_stack_id = None
    reference_temporal_position = None

    for dcm_file in dicom_files:
        ds = dcm_file["ds"]
        organization_type = str(getattr(ds, "DimensionOrganizationType", "")).strip().upper()
        if organization_type not in {"", "3D", "3D_TEMPORAL"}:
            raise DataStructureError(
                f"Enhanced PET Dimension Organization Type '{organization_type}' cannot be represented as one 3D volume."
            )

        dimension_sequence = getattr(ds, "DimensionIndexSequence", None)
        if not dimension_sequence:
            raise DataStructureError("Enhanced PET Dimension Index Sequence (0020,9222) is missing or empty.")
        try:
            pointers = tuple(int(item.DimensionIndexPointer) for item in dimension_sequence)
        except (AttributeError, TypeError, ValueError):
            raise DataStructureError("Enhanced PET Dimension Index Sequence contains an invalid pointer.")
        if reference_pointers is None:
            reference_pointers = pointers
        elif pointers != reference_pointers:
            raise DataStructureError("Enhanced PET instances use inconsistent Dimension Index Sequences.")

        non_spatial_indices = tuple(index for index, pointer in enumerate(pointers) if pointer not in spatial_pointers)
        for frame_index in range(_enhanced_instance_frame_count(dcm_file)):
            frame_content_sequence = _enhanced_functional_group_sequence(ds, frame_index, "FrameContentSequence")
            if not frame_content_sequence or len(frame_content_sequence) != 1:
                raise DataStructureError("Enhanced PET frame must contain one Frame Content Sequence item.")
            frame_content = frame_content_sequence[0]
            index_values = _dimension_index_values(frame_content, len(pointers))
            non_spatial_values = tuple(index_values[index] for index in non_spatial_indices)
            if reference_non_spatial_values is None:
                reference_non_spatial_values = non_spatial_values
            elif non_spatial_values != reference_non_spatial_values:
                raise DataStructureError("Enhanced PET contains multiple non-spatial frame dimensions.")

            stack_id = getattr(frame_content, "StackID", None)
            if stack_id not in [None, ""]:
                stack_id = str(stack_id)
                if reference_stack_id is None:
                    reference_stack_id = stack_id
                elif stack_id != reference_stack_id:
                    raise DataStructureError("Enhanced PET contains multiple frame stacks.")

            temporal_position = getattr(frame_content, "TemporalPositionIndex", None)
            if temporal_position not in [None, ""]:
                try:
                    temporal_position = int(temporal_position)
                except (TypeError, ValueError):
                    raise DataStructureError("Enhanced PET Temporal Position Index (0020,9128) is invalid.")
                if reference_temporal_position is None:
                    reference_temporal_position = temporal_position
                elif temporal_position != reference_temporal_position:
                    raise DataStructureError("Enhanced PET contains multiple temporal positions.")


def _enhanced_image_geometry(dicom_files):
    _validate_enhanced_spatial_dimensions(dicom_files)
    frame_references = _enhanced_frame_references(dicom_files)
    frame_order = _enhanced_frame_order(dicom_files)
    first_dcm_file, first_frame_index = frame_references[int(frame_order[0])]
    orientation, position, normal = _enhanced_frame_geometry(first_dcm_file["ds"], first_frame_index)
    pixel_spacing = None
    pixel_measures = None
    for dcm_file in dicom_files:
        ds = dcm_file["ds"]
        for frame_index in range(_enhanced_instance_frame_count(dcm_file)):
            pixel_measures_sequence = _enhanced_functional_group_sequence(ds, frame_index, "PixelMeasuresSequence")
            frame_pixel_measures = pixel_measures_sequence[0] if pixel_measures_sequence else ds
            try:
                frame_pixel_spacing = np.asarray(frame_pixel_measures.PixelSpacing, dtype=float)
            except (AttributeError, TypeError, ValueError):
                raise DataStructureError("Enhanced PET Pixel Spacing (0028,0030) is missing or invalid.")
            if (
                frame_pixel_spacing.shape != (2,)
                or not np.all(np.isfinite(frame_pixel_spacing))
                or np.any(frame_pixel_spacing <= 0)
            ):
                raise DataStructureError("Enhanced PET Pixel Spacing (0028,0030) must contain two positive values.")
            if pixel_spacing is None:
                pixel_spacing = frame_pixel_spacing
                pixel_measures = frame_pixel_measures
            elif not np.allclose(frame_pixel_spacing, pixel_spacing, rtol=0, atol=1e-6):
                raise DataStructureError("Enhanced PET frames have inconsistent in-plane pixel spacing.")

    distances = _enhanced_frame_distances(dicom_files)
    if len(distances) > 1:
        z_spacing, _direction_sign = _validated_enhanced_slice_spacing(distances)
    else:
        spacing_between_slices = getattr(pixel_measures, "SpacingBetweenSlices", None)
        slice_thickness = getattr(pixel_measures, "SliceThickness", None)
        try:
            z_spacing = float(spacing_between_slices if spacing_between_slices not in [None, ""] else slice_thickness)
        except (TypeError, ValueError):
            raise DataStructureError("Enhanced PET slice spacing is missing or invalid.")
        if not np.isfinite(z_spacing) or z_spacing <= 0:
            raise DataStructureError("Enhanced PET slice spacing must be positive.")

    direction = np.vstack((orientation[:3], orientation[3:], normal)).flatten(order="F")
    spacing = (float(pixel_spacing[1]), float(pixel_spacing[0]), z_spacing)
    return tuple(position), spacing, tuple(direction)


def validate_z_spacing(dicom_files):
    if dicom_files and all(is_enhanced_pet(dcm_file["ds"]) for dcm_file in dicom_files):
        slice_z_origin = _enhanced_frame_distances(dicom_files)
        _validated_enhanced_slice_spacing(slice_z_origin)
        return
    else:
        slice_z_origin = []
        for dcm_file in dicom_files:
            slice_z_origin.append(float(dcm_file["ds"].ImagePositionPatient[2]))
        slice_z_origin = sorted(slice_z_origin)
        slice_thickness = [abs(slice_z_origin[i] - slice_z_origin[i + 1]) for i in range(len(slice_z_origin) - 1)]
    for i in range(len(slice_thickness) - 1):
        spacing_difference = abs(slice_thickness[i] - slice_thickness[i + 1])
        spacing_threshold = 0.1
        if spacing_difference > spacing_threshold:
            error_msg = f"Inconsistent z-spacing. Absolute deviation is {spacing_difference:.3f} which is greater than {spacing_threshold:.3f} mm."
            raise DataStructureError(error_msg)


def validate_ultrasound_dicom_tags(dicom_files):
    if len(dicom_files) != 1:
        error_msg = "Ultrasound volume should be stored as one dicom file, image excluded."
        raise DataStructureError(error_msg)

    ds = dicom_files[0]["ds"]
    if ds.Modality != "US":
        error_msg = f'Ultrasound DICOM modality should be "US", but {ds.Modality} is provided, image excluded.'
        raise DataStructureError(error_msg)

    if "PixelSpacing" not in ds:
        raise DataStructureError("Ultrasound volume does not have PixelSpacing specified, image excluded.")

    if "SliceThickness" not in ds:
        raise DataStructureError("Ultrasound volume does not have slice SliceThickness specified, image excluded.")


def modality_mapping(modality):
    modality_map = {
        "PET": "PT",
        "CT": "CT",
        "MRI": "MR",
        "RTSTRUCT": "RTSTRUCT",
        "SEG": "SEG",
        "US": "US",
        "MG": "MG",
        "RTDOSE": "RTDOSE",
    }
    return modality_map[modality]


def process_dicom_series(dicom_files, modality, reorder_enhanced_frames=True):
    if modality == "PET" and dicom_files and all(is_enhanced_pet(item["ds"]) for item in dicom_files):
        origin, spacing, direction = _enhanced_image_geometry(dicom_files)
        enhanced_images = [sitk.ReadImage(item["file_path"]) for item in dicom_files]
        reference = enhanced_images[0]
        arrays = []
        for image in enhanced_images:
            if image.GetSize()[:2] != reference.GetSize()[:2]:
                raise DataStructureError("Enhanced PET instances have incompatible in-plane dimensions.")
            arrays.append(sitk.GetArrayFromImage(image))
        frames = np.concatenate(arrays, axis=0)
        frame_order = _enhanced_frame_order(dicom_files)
        if frames.ndim != 3 or frames.shape[0] != len(frame_order):
            raise DataStructureError("Enhanced PET pixel data dimensions do not match its frame geometry.")
        if reorder_enhanced_frames:
            frames = frames[frame_order]
        image = sitk.GetImageFromArray(frames)
        image.SetOrigin(origin)
        image.SetSpacing(spacing)
        image.SetDirection(direction)
        return image

    if modality in ["CT", "MRI", "PET", "MG"]:
        reader = sitk.ImageSeriesReader()
        file_names = [i["file_path"] for i in dicom_files]
        reader.SetFileNames(file_names)
        image = reader.Execute()

    elif modality == "US":
        image = sitk.ReadImage(dicom_files[0]["file_path"])

    slice_z_origin = []
    direction = None
    for dicom_file in dicom_files:
        ds = dicom_file["ds"]

        if ds.Modality in ["CT", "PT", "MR"]:
            pixel_spacing = ds.PixelSpacing

            iop = np.array(ds.ImageOrientationPatient, dtype=float)
            row_cosines = iop[:3]
            col_cosines = iop[3:]
            normal = np.cross(row_cosines, col_cosines)
            if direction is None:
                direction = np.vstack([row_cosines, col_cosines, normal]).flatten(order="F")
            distance_along_normal = np.dot(np.array(ds.ImagePositionPatient, dtype=float), normal)
            slice_z_origin.append(distance_along_normal)

        elif ds.Modality == "MG":
            pixel_spacing = ds.ImagerPixelSpacing
            slice_z_origin.append(ds.BodyPartThickness)
            image.SetOrigin([0, 0, 0])
            direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]

        elif ds.Modality == "US":
            pixel_spacing = ds.PixelSpacing
            slice_z_origin.append(ds.SliceThickness)
            image.SetOrigin([0, 0, 0])
            direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    slice_z_origin = sorted(slice_z_origin)
    if len(slice_z_origin) > 1:
        slice_thickness = np.median(np.abs(np.diff(np.asarray(slice_z_origin, float))))

    elif len(slice_z_origin) == 1:
        slice_thickness = slice_z_origin[0]

    # DICOM PixelSpacing and ImagerPixelSpacing are (row, column); SimpleITK uses (x, y, z).
    image.SetSpacing((float(pixel_spacing[1]), float(pixel_spacing[0]), float(slice_thickness)))
    image.SetDirection(direction)
    # SimpleITK applies DICOM rescaling. Entirely nonnegative decoded CT values
    # are valid; their sign cannot establish whether conversion succeeded.

    return image


def _normalized_contour_type(contour):
    return contour["type"].upper().replace("_", "").strip()


def _generate_rtstruct_mask_array(contours, sitk_image):
    width, height, depth = sitk_image.GetSize()
    mask_array = np.zeros((depth, height, width), dtype=np.uint8)
    skipped_outside_fov = 0
    supported_contours = 0

    for contour in contours:
        contour_type = _normalized_contour_type(contour)
        if contour_type not in ["CLOSEDPLANAR", "INTERPOLATEDPLANAR", "CLOSEDPLANARXOR"]:
            continue

        supported_contours += 1
        points = contour["points"]
        num_points = len(points["x"])
        if num_points == 0:
            skipped_outside_fov += 1
            continue

        transformed_points = np.array(
            [
                sitk_image.TransformPhysicalPointToContinuousIndex((points["x"][i], points["y"][i], points["z"][i]))
                for i in range(num_points)
            ]
        )

        z = int(np.rint(np.median(transformed_points[:, 2])))
        if z < 0 or z >= depth:
            skipped_outside_fov += 1
            continue

        polygon_coords = np.column_stack((transformed_points[:, 1], transformed_points[:, 0]))
        mask_layer = draw.polygon2mask((height, width), polygon_coords)
        if not mask_layer.any():
            skipped_outside_fov += 1
            continue

        if contour_type == "CLOSEDPLANARXOR":
            mask_array[z] = np.logical_xor(mask_array[z], mask_layer).astype(np.uint8)
        else:
            mask_array[z] = np.logical_or(mask_array[z], mask_layer).astype(np.uint8)

    return mask_array, skipped_outside_fov, supported_contours


def extract_dicom_mask(rtstruct_path, roi_name, image):
    def get_contour_data(file_path, selected_roi):
        def get_contour_coord(metadata, current_roi_sequence, skip_contours_bool=False):
            contour_info = {
                "name": getattr(metadata[current_roi_sequence.ReferencedROINumber], "ROIName", "unknown"),
                "roi_number": current_roi_sequence.ReferencedROINumber,
                "display_color": getattr(current_roi_sequence, "ROIDisplayColor", []),
            }

            if not skip_contours_bool and hasattr(current_roi_sequence, "ContourSequence"):
                contour_info["sequence"] = [
                    {
                        "type": getattr(contour, "ContourGeometricType", "unknown"),
                        "points": {
                            "x": [contour.ContourData[i] for i in range(0, len(contour.ContourData), 3)],
                            "y": [contour.ContourData[i + 1] for i in range(0, len(contour.ContourData), 3)],
                            "z": [contour.ContourData[i + 2] for i in range(0, len(contour.ContourData), 3)],
                        },
                    }
                    for contour in current_roi_sequence.ContourSequence
                ]

            return contour_info

        dicom_data = pydicom.dcmread(file_path)
        if not hasattr(dicom_data, "StructureSetROISequence"):
            raise InvalidDicomError()

        contour_data = None
        metadata_map = {data.ROINumber: data for data in dicom_data.StructureSetROISequence}
        for roi_sequence in dicom_data.ROIContourSequence:
            current_roi_name = getattr(metadata_map[roi_sequence.ReferencedROINumber], "ROIName", "unknown")
            if current_roi_name == selected_roi:
                contour_data = get_contour_coord(metadata_map, roi_sequence)
                break

        return contour_data

    from ..image import Image

    rt_struct = get_contour_data(rtstruct_path, roi_name)
    if rt_struct and "sequence" in rt_struct:
        mask, skipped_outside_fov, _supported_contours = _generate_rtstruct_mask_array(rt_struct["sequence"], image)
        if skipped_outside_fov > 0 and np.any(mask):
            warnings.warn(
                f"Skipped {skipped_outside_fov} RTSTRUCT contour(s) for ROI '{roi_name}' because they fall outside "
                "the target image field of view.",
                DataStructureWarning,
            )
        if not np.any(mask):
            warnings.warn(
                f"RTSTRUCT ROI '{roi_name}' has no overlap with the target image field of view. ROI skipped.",
                DataStructureWarning,
            )
            return Image()
        return Image(
            array=mask,
            origin=image.GetOrigin(),
            spacing=np.array(image.GetSpacing()),
            direction=image.GetDirection(),
            shape=image.GetSize(),
        )
    return Image()


def _segment_labels(dicom_data):
    """Return a SegmentNumber-to-SegmentLabel mapping for a DICOM SEG."""
    if not hasattr(dicom_data, "SegmentSequence"):
        raise InvalidDicomError("The DICOM file does not contain a SegmentSequence.")
    return {
        int(segment.SegmentNumber): str(getattr(segment, "SegmentLabel", "unknown"))
        for segment in dicom_data.SegmentSequence
    }


def _seg_functional_group_sequence(seg, functional_group, sequence_keyword):
    sequence = getattr(functional_group, sequence_keyword, None)
    if sequence:
        return sequence
    shared = getattr(seg, "SharedFunctionalGroupsSequence", None)
    if shared and hasattr(shared[0], sequence_keyword):
        sequence = getattr(shared[0], sequence_keyword)
        if sequence:
            return sequence
    return None


def _seg_geometry_descriptor(
    orientation,
    position,
    pixel_spacing,
    z_spacing,
    frame_of_reference_uid=None,
):
    try:
        orientation = np.asarray(orientation, dtype=float)
        position = np.asarray(position, dtype=float)
        pixel_spacing = np.asarray(pixel_spacing, dtype=float)
        z_spacing = float(z_spacing)
    except (TypeError, ValueError) as exc:
        raise DataStructureError("DICOM SEG frame geometry is invalid.") from exc
    if (
        orientation.shape != (6,)
        or position.shape != (3,)
        or pixel_spacing.shape != (2,)
        or not np.all(np.isfinite(orientation))
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(pixel_spacing))
        or np.any(pixel_spacing <= 0)
        or not np.isfinite(z_spacing)
        or z_spacing <= 0
    ):
        raise DataStructureError("DICOM SEG frame geometry is invalid.")

    row = orientation[:3]
    column = orientation[3:]
    row_norm = np.linalg.norm(row)
    column_norm = np.linalg.norm(column)
    if (
        row_norm == 0
        or column_norm == 0
        or not np.isclose(np.dot(row / row_norm, column / column_norm), 0.0, rtol=0, atol=1e-4)
    ):
        raise DataStructureError("DICOM SEG frame orientation is invalid.")
    row = row / row_norm
    column = column / column_norm
    normal = np.cross(row, column)
    normal_norm = np.linalg.norm(normal)
    if not np.isfinite(normal_norm) or normal_norm == 0:
        raise DataStructureError("DICOM SEG frame orientation is invalid.")
    normal /= normal_norm
    direction = np.vstack((row, column, normal)).flatten(order="F")
    return {
        "origin": tuple(position),
        "spacing": (float(pixel_spacing[1]), float(pixel_spacing[0]), z_spacing),
        "direction": tuple(direction),
        "normal": normal,
        "frame_of_reference_uid": frame_of_reference_uid,
    }


def _pixel_measures_geometry(pixel_measures, fallback_z_spacing):
    pixel_spacing = getattr(pixel_measures, "PixelSpacing", None)
    spacing_between_slices = getattr(pixel_measures, "SpacingBetweenSlices", None)
    slice_thickness = getattr(pixel_measures, "SliceThickness", None)
    z_spacing = spacing_between_slices if spacing_between_slices not in [None, ""] else slice_thickness
    if z_spacing in [None, ""]:
        z_spacing = fallback_z_spacing
    return pixel_spacing, z_spacing


def _source_frame_geometries(dicom_dir, image):
    """Map source SOP Instance UID and frame number to physical frame geometry."""
    geometries = {}
    if not dicom_dir:
        return geometries
    fallback_z_spacing = float(image.GetSpacing()[2])
    for filename in (os.path.join(dicom_dir, name) for name in os.listdir(dicom_dir)):
        if not os.path.isfile(filename):
            continue
        try:
            source = pydicom.dcmread(filename, stop_before_pixels=True)
            if not hasattr(source, "SOPInstanceUID"):
                continue
            uid = str(source.SOPInstanceUID)
            frame_of_reference_uid = str(getattr(source, "FrameOfReferenceUID", "")).strip() or None
            if is_enhanced_pet(source):
                frame_count = _enhanced_instance_frame_count({"ds": source})
                for frame_index in range(frame_count):
                    orientation, position, _normal = _enhanced_frame_geometry(source, frame_index)
                    pixel_measures_sequence = _enhanced_functional_group_sequence(
                        source,
                        frame_index,
                        "PixelMeasuresSequence",
                    )
                    pixel_measures = pixel_measures_sequence[0] if pixel_measures_sequence else source
                    pixel_spacing, z_spacing = _pixel_measures_geometry(pixel_measures, fallback_z_spacing)
                    geometries[(uid, frame_index + 1)] = _seg_geometry_descriptor(
                        orientation,
                        position,
                        pixel_spacing,
                        z_spacing,
                        frame_of_reference_uid,
                    )
            elif all(
                hasattr(source, keyword)
                for keyword in ("ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing")
            ):
                pixel_spacing, z_spacing = _pixel_measures_geometry(source, fallback_z_spacing)
                geometry = _seg_geometry_descriptor(
                    source.ImageOrientationPatient,
                    source.ImagePositionPatient,
                    pixel_spacing,
                    z_spacing,
                    frame_of_reference_uid,
                )
                geometries[(uid, None)] = geometry
                geometries[(uid, 1)] = geometry
        except (InvalidDicomError, AttributeError, IndexError, TypeError, ValueError, RuntimeError, DataStructureError):
            continue
    return geometries


def _referenced_frame_numbers(source_image):
    raw_frame_numbers = getattr(source_image, "ReferencedFrameNumber", None)
    if raw_frame_numbers in [None, ""]:
        return (None,)
    if isinstance(raw_frame_numbers, (str, bytes)) or not hasattr(raw_frame_numbers, "__iter__"):
        raw_frame_numbers = (raw_frame_numbers,)
    try:
        return tuple(int(value) for value in raw_frame_numbers)
    except (TypeError, ValueError) as exc:
        raise DataStructureError("SEG Referenced Frame Number (0008,1160) is invalid.") from exc


def _seg_referenced_geometries(functional_group, source_geometries):
    geometries = []
    missing_uids = set()
    derivation_images = getattr(functional_group, "DerivationImageSequence", None) or []
    for derivation_image in derivation_images:
        for source_image in getattr(derivation_image, "SourceImageSequence", None) or []:
            referenced_uid = str(getattr(source_image, "ReferencedSOPInstanceUID", "")).strip()
            if not referenced_uid:
                continue
            for frame_number in _referenced_frame_numbers(source_image):
                geometry = source_geometries.get((referenced_uid, frame_number))
                if geometry is None and frame_number is None:
                    matches = [
                        value for (uid, _frame_number), value in source_geometries.items() if uid == referenced_uid
                    ]
                    unique_matches = []
                    for match in matches:
                        if not any(match is existing for existing in unique_matches):
                            unique_matches.append(match)
                    if len(unique_matches) == 1:
                        geometry = unique_matches[0]
                if geometry is None:
                    missing_uids.add(referenced_uid)
                elif not any(geometry is existing for existing in geometries):
                    geometries.append(geometry)
    return geometries, missing_uids


def _reference_geometry_descriptor(image, position):
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    spacing = image.GetSpacing()
    orientation = np.concatenate((direction[:, 0], direction[:, 1]))
    return _seg_geometry_descriptor(
        orientation,
        position,
        (spacing[1], spacing[0]),
        spacing[2],
    )


def _seg_frame_geometry(seg, functional_group, source_geometries, image, frame_shape):
    pixel_measures_sequence = _seg_functional_group_sequence(seg, functional_group, "PixelMeasuresSequence")
    orientation_sequence = _seg_functional_group_sequence(seg, functional_group, "PlaneOrientationSequence")
    position_sequence = _seg_functional_group_sequence(seg, functional_group, "PlanePositionSequence")
    position = None
    if position_sequence:
        position = getattr(position_sequence[0], "ImagePositionPatient", None)
    elif hasattr(functional_group, "ImagePositionPatient"):
        position = functional_group.ImagePositionPatient

    referenced_geometries, missing_uids = _seg_referenced_geometries(functional_group, source_geometries)
    seg_frame_of_reference_uid = str(getattr(seg, "FrameOfReferenceUID", "")).strip() or None
    for referenced_geometry in referenced_geometries:
        source_uid = referenced_geometry["frame_of_reference_uid"]
        if seg_frame_of_reference_uid and source_uid and seg_frame_of_reference_uid != source_uid:
            raise DataStructureError("DICOM SEG and its referenced source image use different Frame of Reference UIDs.")

    if pixel_measures_sequence and orientation_sequence and position is not None:
        pixel_spacing, z_spacing = _pixel_measures_geometry(
            pixel_measures_sequence[0],
            image.GetSpacing()[2],
        )
        return (
            _seg_geometry_descriptor(
                orientation_sequence[0].ImageOrientationPatient,
                position,
                pixel_spacing,
                z_spacing,
                seg_frame_of_reference_uid,
            ),
            None,
        )

    if len(referenced_geometries) == 1:
        return referenced_geometries[0], None
    if len(referenced_geometries) > 1:
        raise DataStructureError(
            "SEG frame references multiple source frames but does not provide independent spatial geometry."
        )
    if position is not None:
        if frame_shape != (image.GetSize()[1], image.GetSize()[0]):
            raise DataStructureError(
                "SEG frame dimensions differ from the image and the SEG does not provide complete spatial geometry."
            )
        return _reference_geometry_descriptor(image, position), None
    if missing_uids:
        return None, sorted(missing_uids)[0]
    raise DataStructureError("SEG frame has neither usable spatial geometry nor a source-image reference.")


def _resample_seg_frame(frame, geometry, image):
    source = sitk.GetImageFromArray((frame > 0).astype(np.uint8)[np.newaxis, ...])
    source.SetOrigin(geometry["origin"])
    source.SetSpacing(geometry["spacing"])
    source.SetDirection(geometry["direction"])

    width, height, depth = image.GetSize()
    target_direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    target_normal = target_direction[:, 2]
    if abs(float(np.dot(geometry["normal"], target_normal))) >= 1.0 - 1e-4:
        continuous_index = image.TransformPhysicalPointToContinuousIndex(geometry["origin"])
        z_index = int(np.rint(continuous_index[2]))
        if z_index < 0 or z_index >= depth:
            return None, None
        target = sitk.Image((width, height, 1), sitk.sitkUInt8)
        target.SetOrigin(image.TransformIndexToPhysicalPoint((0, 0, z_index)))
        target.SetSpacing(image.GetSpacing())
        target.SetDirection(image.GetDirection())
        resampled = sitk.Resample(
            source,
            target,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        return z_index, sitk.GetArrayFromImage(resampled)[0]

    resampled = sitk.Resample(
        source,
        image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    return None, sitk.GetArrayFromImage(resampled)


def extract_dicom_seg_mask(seg_path, segment_name, image, dicom_dir=None):
    """Extract one named segment from a DICOM SEG onto ``image``'s grid."""
    from ..image import Image

    seg = pydicom.dcmread(seg_path)
    labels = _segment_labels(seg)
    matching_numbers = {number for number, label in labels.items() if label == segment_name}
    if not matching_numbers:
        return Image()

    raw_segmentation_type = getattr(seg, "SegmentationType", None)
    segmentation_type = str(raw_segmentation_type).strip().upper() if raw_segmentation_type is not None else ""
    if segmentation_type != "BINARY":
        displayed_type = segmentation_type or "missing"
        raise DataStructureError(
            f"Unsupported DICOM SEG SegmentationType '{displayed_type}'. Only BINARY segmentations are supported."
        )

    frames = np.asarray(seg.pixel_array)
    if frames.ndim == 2:
        frames = frames[np.newaxis, ...]
    if frames.ndim != 3:
        raise DataStructureError("DICOM SEG pixel data must contain two-dimensional frames.")
    width, height, depth = image.GetSize()
    groups = getattr(seg, "PerFrameFunctionalGroupsSequence", None)
    if groups is None or len(groups) != len(frames):
        raise DataStructureError("SEG PerFrameFunctionalGroupsSequence does not match its pixel frames.")

    source_geometries = _source_frame_geometries(dicom_dir, image)
    volume = np.zeros((depth, height, width), dtype=np.uint8)
    missing_uids = set()
    for frame, group in zip(frames, groups):
        try:
            segment_number = int(group.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise DataStructureError("SEG frame does not identify its referenced segment.") from exc
        if segment_number not in matching_numbers:
            continue
        geometry, missing_uid = _seg_frame_geometry(seg, group, source_geometries, image, frame.shape)
        if missing_uid is not None:
            missing_uids.add(missing_uid)
            continue
        z_index, resampled = _resample_seg_frame(frame, geometry, image)
        if resampled is None:
            continue
        if z_index is None:
            volume = np.maximum(volume, resampled)
        else:
            volume[z_index] = np.maximum(volume[z_index], resampled)

    if missing_uids:
        raise DataStructureError(
            f"{len(missing_uids)} referenced SEG source image(s) were not found in the selected DICOM series."
        )
    if not np.any(volume):
        warnings.warn(
            f"DICOM SEG segment '{segment_name}' has no overlap with the target image field of view. Segment skipped.",
            DataStructureWarning,
        )
        return Image()
    return Image(
        array=volume,
        origin=image.GetOrigin(),
        spacing=np.array(image.GetSpacing()),
        direction=image.GetDirection(),
        shape=image.GetSize(),
    )


def get_all_structure_names(rtstruct_path):
    """Extract all structure names from an RTSTRUCT or DICOM SEG file."""
    dicom_data = pydicom.dcmread(rtstruct_path)

    if getattr(dicom_data, "Modality", None) == "SEG":
        return list(_segment_labels(dicom_data).values())

    if not hasattr(dicom_data, "StructureSetROISequence"):
        raise InvalidDicomError(f"The DICOM file at {rtstruct_path} is not a valid RTSTRUCT file.")

    metadata_map = {roi_data.ROINumber: roi_data for roi_data in dicom_data.StructureSetROISequence}

    structure_names = []
    if hasattr(dicom_data, "ROIContourSequence"):
        for roi_sequence in dicom_data.ROIContourSequence:
            roi_number = roi_sequence.ReferencedROINumber
            roi_name = getattr(metadata_map.get(roi_number, {}), "ROIName", "unknown")
            structure_names.append(roi_name)

    return structure_names
