import os
import warnings

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.errors import InvalidDicomError
from skimage import draw

from ..exceptions import DataStructureError, DataStructureWarning
from .pet_suv import apply_suv_correction, validate_pet_dicom_tags


def read_dicom_image(dicom_dir, modality):
    """Read a DICOM image series as a SimpleITK image."""
    dicom_files = get_dicom_files(directory=dicom_dir, modality=modality)
    if len(dicom_files) == 0:
        raise DataStructureError(f"No {modality} data found in {dicom_dir}. Patient skipped.")

    image = None
    if modality in ["CT", "MRI", "PET"]:
        validate_z_spacing(dicom_files)
    if modality in ["CT", "MRI", "PET", "MG"]:
        image = process_dicom_series(dicom_files)
    if modality == "PET":
        validate_pet_dicom_tags(dicom_files)
        image = apply_suv_correction(dicom_files, image)
    if modality == "RTDOSE":
        image = read_dicom_dose(dicom_files[0]["file_path"])
    if image is None:
        raise DataStructureError(f"Unsupported DICOM modality {modality}.")
    return image


def read_dicom_mask(rtstruct_path, structure_name, reference_image):
    """Read an RTSTRUCT mask as an Image aligned to a reference SimpleITK image."""
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

    if len(dicom_files_info) > 1:
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
    if modality_dicom in ["CT", "PT", "MR"]:
        dicom_files_info = remove_duplicate_slices(dicom_files_info)
        dicom_files_info = sort_by_geometric_position(dicom_files_info)
    return dicom_files_info


def validate_z_spacing(dicom_files):
    slice_z_origin = []
    for dcm_file in dicom_files:
        slice_z_origin.append(float(dcm_file["ds"].ImagePositionPatient[2]))
    slice_z_origin = sorted(slice_z_origin)
    slice_thickness = [abs(slice_z_origin[i] - slice_z_origin[i + 1]) for i in range(len(slice_z_origin) - 1)]
    for i in range(len(slice_thickness) - 1):
        spacing_difference = abs((slice_thickness[i] - slice_thickness[i + 1]))
        spacing_threshold = 0.1
        if spacing_difference > spacing_threshold:
            error_msg = f"Inconsistent z-spacing. Absolute deviation is {spacing_difference:.3f} which is greater than {spacing_threshold:.3f} mm."
            raise DataStructureError(error_msg)


def modality_mapping(modality):
    modality_map = {
        "PET": "PT",
        "CT": "CT",
        "MRI": "MR",
        "RTSTRUCT": "RTSTRUCT",
        "MG": "MG",
        "RTDOSE": "RTDOSE",
    }
    return modality_map[modality]


def process_dicom_series(dicom_files):
    reader = sitk.ImageSeriesReader()
    file_names = [i["file_path"] for i in dicom_files]
    reader.SetFileNames(file_names)
    image = reader.Execute()

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

    slice_z_origin = sorted(slice_z_origin)
    if len(slice_z_origin) > 1:
        slice_thickness = np.median(np.abs(np.diff(np.asarray(slice_z_origin, float))))

    elif len(slice_z_origin) == 1:
        slice_thickness = slice_z_origin[0]

    image.SetSpacing((float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness)))
    image.SetDirection(direction)
    if dicom_files[0]["ds"].Modality == "CT" and np.min(sitk.GetArrayFromImage(image)) >= 0:
        error_msg = f'Non-negative CT intensity. SITK failed to convert CT into HU for {dicom_files[0]["file_path"]}. The patient is excluded from analysis'
        raise DataStructureError(error_msg)

    return image


def extract_dicom_mask(rtstruct_path, roi_name, image):
    def generate_mask_array(contours, sitk_image):
        width, height, depth = sitk_image.GetSize()
        mask_array = np.zeros((depth, height, width), dtype=np.uint8)

        for contour in contours:
            contour_type = contour["type"].upper().replace("_", "").strip()
            if contour_type not in ["CLOSEDPLANAR", "INTERPOLATEDPLANAR", "CLOSEDPLANARXOR"]:
                continue

            points = contour["points"]
            num_points = len(points["x"])
            transformed_points = np.array(
                [
                    sitk_image.TransformPhysicalPointToContinuousIndex((points["x"][i], points["y"][i], points["z"][i]))
                    for i in range(num_points)
                ]
            )

            z_indices = np.rint(transformed_points[:, 2]).astype(int)
            polygon_coords = np.column_stack((transformed_points[:, 1], transformed_points[:, 0]))
            mask_layer = draw.polygon2mask((height, width), polygon_coords)

            for z in z_indices:
                if contour_type == "CLOSEDPLANARXOR":
                    mask_array[z] = np.logical_xor(mask_array[z], mask_layer).astype(np.uint8)
                else:
                    mask_array[z] = np.logical_or(mask_array[z], mask_layer).astype(np.uint8)

        return mask_array

    def get_contour_data(file_path, selected_roi):
        def get_contour_coord(metadata, current_roi_sequence, skip_contours_bool=False):
            contour_info = {
                "name": getattr(metadata[current_roi_sequence.ReferencedROINumber], "ROIName", "unknown"),
                "roi_number": current_roi_sequence.ReferencedROINumber,
                "referenced_frame": getattr(
                    metadata[current_roi_sequence.ReferencedROINumber],
                    "ReferencedFrameOfReferenceUID",
                    "unknown",
                ),
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
        mask = generate_mask_array(rt_struct["sequence"], image)
        return Image(
            array=mask,
            origin=image.GetOrigin(),
            spacing=np.array(image.GetSpacing()),
            direction=image.GetDirection(),
            shape=image.GetSize(),
        )
    return Image()


def get_all_structure_names(rtstruct_path):
    """Extract all structure names from an RTSTRUCT DICOM file."""
    dicom_data = pydicom.dcmread(rtstruct_path)

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
