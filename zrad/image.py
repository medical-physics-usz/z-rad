import copy
import os
import warnings
from datetime import datetime

import SimpleITK as sitk
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from skimage import draw

from .exceptions import DataStructureWarning, DataStructureError


def parse_time(time_str):
    """
    Parse a time string into a datetime object using various possible formats.

    Args:
        time_str (str or bytes): The time string to parse.

    Returns:
        datetime: Parsed datetime object.

    Raises:
        ValueError: If the time string does not match any expected formats.
    """
    if isinstance(time_str, bytes):
        time_str = time_str.decode('utf-8').strip()

    for fmt in ('%H%M%S.%f', '%H%M%S', '%Y%m%d%H%M%S.%f', '%Y%m%d%H%M%S'):
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
        except TypeError:
            continue
    raise ValueError(f"Time data '{time_str}' does not match expected formats")


class Image:
    def __init__(self, array=None, origin=None, spacing=None, direction=None, shape=None):
        self.sitk_image = None
        self.array = array
        self.origin = origin
        self.spacing = spacing
        self.direction = direction
        self.shape = shape

    def copy(self):
        return Image(
            array=copy.deepcopy(self.array),
            origin=copy.deepcopy(self.origin),
            spacing=copy.deepcopy(self.spacing),
            direction=copy.deepcopy(self.direction),
            shape=copy.deepcopy(self.shape)
        )

    def save_as_nifti(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        img = sitk.GetImageFromArray(self.array)
        img.SetOrigin(self.origin)
        img.SetSpacing(self.spacing)
        img.SetDirection(self.direction)
        sitk.WriteImage(img, output_path)

    def read_nifti_image(self, image_path):
        sitk_reader = sitk.ImageFileReader()
        sitk_reader.SetImageIO("NiftiImageIO")
        sitk_reader.SetFileName(image_path)
        image = sitk_reader.Execute()
        array = sitk.GetArrayFromImage(image)
        self.sitk_image = image
        self.array = array.astype(np.float64)
        self.origin = image.GetOrigin()
        self.spacing = np.array(image.GetSpacing())
        self.direction = image.GetDirection()
        self.shape = image.GetSize()

    def read_nifti_mask(self, image, mask_path):
        sitk_reader = sitk.ImageFileReader()
        sitk_reader.SetImageIO("NiftiImageIO")
        sitk_reader.SetFileName(mask_path)
        mask = sitk_reader.Execute()
        array = sitk.GetArrayFromImage(mask)
        self.sitk_image = image
        self.array = array.astype(np.float64)
        self.origin = image.origin
        self.spacing = image.spacing
        self.direction = image.direction
        self.shape = image.shape

    def read_dicom_image(self, dicom_dir, modality):
        dicom_files = get_dicom_files(directory=dicom_dir, modality=modality)
        if len(dicom_files) == 0:
            raise DataStructureError(f"No {modality} data found in {dicom_dir}. Patient skipped.")
        if modality in ["CT", "MRI", "PET"]:
            validate_z_spacing(dicom_files)
        if modality in ["CT", "MRI", "PET", "MG"]:
            image = process_dicom_series(dicom_files)
        if modality == 'PET':
            validate_pet_dicom_tags(dicom_files)
            image = apply_suv_correction(dicom_files, image)
        if modality == 'RTDOSE':
            file_path = dicom_files[0]['file_path']
            image = self.read_dicom_dose(file_path)
        if image:
            array = sitk.GetArrayFromImage(image)
            self.sitk_image = image
            self.array = array.astype(np.float64)
            self.origin = image.GetOrigin()
            self.spacing = np.array(image.GetSpacing())
            self.direction = image.GetDirection()
            self.shape = image.GetSize()

    def read_dicom_mask(self, rtstruct_path, structure_name, image):
        mask = extract_dicom_mask(rtstruct_path, structure_name, image.sitk_image)
        self.array = mask.array
        self.origin = mask.origin
        self.spacing = mask.spacing
        self.direction = mask.direction
        self.shape = mask.shape

    def read_dicom_dose(self, rtdose_path):
        ds = pydicom.dcmread(rtdose_path)
        if ds.DoseUnits != 'GY':
            raise DataStructureError(f"Only dose in Gy is supported. Provided {ds.DoseUnits}. Patient skipped")
        if ds.DoseType != 'PHYSICAL':
            raise DataStructureError(f"Only physical dose is supported. Provided {ds.DoseType}. Patient skipped.")
        raw_dose_image = sitk.ReadImage(rtdose_path)
        dose_array = sitk.GetArrayFromImage(raw_dose_image) * ds.DoseGridScaling
        dose_image = sitk.GetImageFromArray(
            dose_array)  # dose_array is a NumPy array that contains the dose values (after applying DoseGridScaling). This line converts dose_array back into a SITK image.
        dose_image.SetOrigin(
            raw_dose_image.GetOrigin())  # Copies the origin (spatial reference of the image in the coordinate system) from the original DICOM image (raw_dose_image) to dose_image.
        dose_image.SetSpacing(
            raw_dose_image.GetSpacing())  # Ensures that the voxel spacing (physical size of each pixel in millimeters) is retained from the original dose DICOM file.
        dose_image.SetDirection(
            raw_dose_image.GetDirection())  # Copies the image orientation (how the image is aligned in 3D space) from the original dose DICOM file.

        return dose_image


def get_dicom_files(directory, modality):
    modality_dicom = modality_mapping(modality)
    dicom_files_info = []
    if modality_dicom in ['CT', 'PT', 'MR']:
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(directory)
        selected_series = None
        for sid in series_IDs:
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
            # Try to read the DICOM file without loading pixel data
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)

            # Check if the DICOM file's Modality matches the desired modality
            if ds.Modality == modality_dicom:
                if hasattr(ds, 'ImageType') and ('LOCALIZER' in ds.ImageType or any('MIP' in entry for entry in ds.ImageType)):
                    continue
                dicom_files_info.append({'file_path': file_path, 'ds': ds})
        except InvalidDicomError:
            # File is not a valid DICOM; skip it
            continue
        except Exception as e:
            # Handle any other unexpected exceptions
            warning_msg = f"An error occurred while processing file {file_path}: {str(e)}"
            warnings.warn(warning_msg, DataStructureWarning)
    return dicom_files_info


def validate_z_spacing(dicom_files):
    slice_z_origin = []
    for dcm_file in dicom_files:
        slice_z_origin.append(float(dcm_file['ds'].ImagePositionPatient[2]))
    slice_z_origin = sorted(slice_z_origin)
    slice_thickness = [abs(slice_z_origin[i] - slice_z_origin[i + 1]) for i in range(len(slice_z_origin) - 1)]
    for i in range(len(slice_thickness) - 1):
        spacing_difference = abs((slice_thickness[i] - slice_thickness[i + 1]))
        spacing_threshold = 0.1
        if spacing_difference > spacing_threshold:
            error_msg = f'Inconsistent z-spacing. Absolute deviation is {spacing_difference:.3f} which is greater than {spacing_threshold:.3f} mm.'
            raise DataStructureError(error_msg)


def calc_elapsed_time(ds, decay_constant, acquisition_time, injection_time):
    frame_reference_time = float(ds.FrameReferenceTime) / 1000
    decay_during_frame = decay_constant * ds.ActualFrameDuration / 1000
    avg_count_rate_time = (1 / decay_constant) * np.log(
        decay_during_frame / (1 - np.exp(-decay_during_frame)))

    return (acquisition_time - injection_time).total_seconds() + avg_count_rate_time - frame_reference_time


def validate_pet_dicom_tags(dicom_files):
    for dcm_file in dicom_files:
        ds = dcm_file['ds']
        image_id = dcm_file['file_path']

        try:
            pat_weight = ds[(0x0010, 0x1030)].value
            if float(pat_weight) < 1:
                warning_msg = f"For patient's {image_id} image, patient's weight tag (0071, 1022) contains weight < 1kg. Patient is excluded from the analysis."
                warnings.warn(warning_msg, DataStructureWarning)

        except (KeyError, TypeError):
            warning_msg = f"For patient's {image_id} image, patient's weight tag (0071, 1022) is not present. Patient is excluded from the analysis."
            warnings.warn(warning_msg, DataStructureWarning)
        if 'DECY' not in ds[(0x0028, 0x0051)].value or 'ATTN' not in ds[(0x0028, 0x0051)].value:
            warning_msg = f"For patient's {image_id} image, in DICOM tag (0028, 0051) either no 'DECY' (decay correction) or 'ATTN' (attenuation correction). Patient is excluded from the analysis."
            warnings.warn(warning_msg, DataStructureWarning)

        if ds.Units == 'BQML':
            injection_time = parse_time(
                ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime)
            half_life = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
            decay_constant = np.log(2) / half_life

            if ds.DecayCorrection == 'START':
                if 'PHILIPS' in ds.Manufacturer.upper():

                    acquisition_time = parse_time(ds.AcquisitionTime).replace(year=injection_time.year,
                                                                              month=injection_time.month,
                                                                              day=injection_time.day)

                    elapsed_time = calc_elapsed_time(ds, decay_constant, acquisition_time, injection_time)
                elif 'SIEMENS' in ds.Manufacturer.upper() or 'CPS' in ds.Manufacturer.upper():
                    try:
                        elapsed_time = (
                                    parse_time(ds[(0x0071, 0x1022)].value).replace(year=injection_time.year,
                                                                                   month=injection_time.month,
                                                                                   day=injection_time.day)
                                    - injection_time).total_seconds()

                    except (KeyError, TypeError):
                        acquisition_time = parse_time(ds.AcquisitionTime).replace(year=injection_time.year,
                                                                                  month=injection_time.month,
                                                                                  day=injection_time.day)

                        elapsed_time = calc_elapsed_time(ds, decay_constant, acquisition_time, injection_time)
                elif 'GE' in ds.Manufacturer.upper():
                    try:
                        elapsed_time = (
                                    parse_time(ds[(0x0009, 0x100d)].value).replace(year=injection_time.year,
                                                                                   month=injection_time.month,
                                                                                   day=injection_time.day)
                                    - injection_time).total_seconds()
                    except (KeyError, TypeError):
                        acquisition_time = parse_time(ds.AcquisitionTime).replace(year=injection_time.year,
                                                                                  month=injection_time.month,
                                                                                  day=injection_time.day)
                        frame_reference_time = float(ds.FrameReferenceTime) / 1000

                        elapsed_time = (acquisition_time - injection_time).total_seconds() - frame_reference_time
                else:
                    warning_msg = f"For patient's {image_id} image, an unknown PET scaner manufacturer is present. Z-Rad only supports Philips, Siemens, and GE."
                    raise DataStructureError(warning_msg)

            elif ds.DecayCorrection == 'ADMIN':
                elapsed_time = 0
            else:
                warning_msg = f"For patient's {image_id} image, An unsupported Decay Correction {ds.DecayCorrection} is present. Only supported are 'START' and 'ADMIN'. Patient is excluded from the analysis."
                raise DataStructureError(warning_msg)
            if elapsed_time < 0:
                error_msg = f"For patient's {image_id} image, patient is excluded from the analysis due to the negative time difference in the decay factor."
                raise DataStructureError(error_msg)
            elif elapsed_time > 0 and abs(elapsed_time) < 1800 and ds.DecayCorrection != 'ADMIN':
                warning_msg = f"Only {abs(elapsed_time) / 60} minutes after the injection."
                warnings.warn(warning_msg, DataStructureWarning)
        elif ds.Units == 'CNTS' and 'PHILIPS' in ds.Manufacturer.upper():
            if not (((0x7053, 0x1009) in ds and ds[(0x7053, 0x1009)].value != 0) or ((0x7053, 0x1000) in ds and ds[(0x7053, 0x1000)].value != 0)):
                error_msg = f"For patient's {image_id} image, patient is excluded, Philips scale factors not present (PET units CNTS)"
                raise DataStructureError(error_msg)
        elif ds.Units == 'GML':
            if (0x0054, 0x1006) in ds:
                if ds[0x0054, 0x1006].value != 'BW':
                    error_msg = f"For patient's {image_id} image, patient is excluded, SUV Type is not BW (GML units)"
                    raise DataStructureError(error_msg)
        else:
            error_msg = f"For patient's {image_id} image, patient is excluded, only supported PET Units are BQML for Philips, Siemens and GE or CNTS for Philips"
            raise DataStructureError(error_msg)


def apply_suv_correction(dicom_files, suv_image):

    def process_single_slice(dicom_file_path):

        def process_gml(pixel_array_units):
            return pixel_array_units

        def process_bqml(activity_concentration, ds):
            injection_time = parse_time(ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime)
            patient_weight = float(ds.PatientWeight)
            injected_dose = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
            half_life = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
            decay_constant = np.log(2) / half_life
            manufacturer = ds.Manufacturer.upper()

            if ds.DecayCorrection == 'START':
                if 'PHILIPS' in manufacturer:
                    acquisition_time = parse_time(ds.AcquisitionTime).replace(year=injection_time.year,
                                                                              month=injection_time.month,
                                                                              day=injection_time.day)

                    elapsed_time = calc_elapsed_time(ds, decay_constant, acquisition_time, injection_time)

                elif 'SIEMENS' in manufacturer or 'CPS' in manufacturer:
                    try:
                        elapsed_time = (
                                parse_time(ds[(0x0071, 0x1022)].value).replace(year=injection_time.year,
                                                                               month=injection_time.month,
                                                                               day=injection_time.day)
                                - injection_time).total_seconds()

                    except (KeyError, TypeError):
                        acquisition_time = parse_time(ds.AcquisitionTime).replace(year=injection_time.year,
                                                                                  month=injection_time.month,
                                                                                  day=injection_time.day)

                        elapsed_time = calc_elapsed_time(ds, decay_constant, acquisition_time, injection_time)
                elif 'GE' in manufacturer:
                    try:
                        elapsed_time = (
                                parse_time(ds[(0x0009, 0x100d)].value).replace(year=injection_time.year,
                                                                               month=injection_time.month,
                                                                               day=injection_time.day)
                                - injection_time).total_seconds()
                    except (KeyError, TypeError):
                        acquisition_time = parse_time(ds.AcquisitionTime).replace(year=injection_time.year,
                                                                                  month=injection_time.month,
                                                                                  day=injection_time.day)
                        frame_reference_time = float(ds.FrameReferenceTime) / 1000

                        elapsed_time = (acquisition_time - injection_time).total_seconds() - frame_reference_time
                else:
                    error_msg = f"Vendor {ds.Manufacturer} is not supported with BQML units!"
                    raise DataStructureError(error_msg)
            elif ds.DecayCorrection == 'ADMIN':
                elapsed_time = 0
            else:
                error_msg = f"Decay correction {ds.DecayCorrection} is not supported!"
                raise DataStructureError(error_msg)

            decay_factor = np.exp(-1 * ((np.log(2) * elapsed_time) / half_life))
            decay_corrected_dose = injected_dose * decay_factor
            suv = activity_concentration / (decay_corrected_dose / (patient_weight * 1000))

            return suv

        def process_cnts(pixel_array_units, ds):
            if 'PHILIPS' in ds.Manufacturer.upper():

                if (0x7053, 0x1009) in ds and ds[(0x7053, 0x1009)].value != 0:
                    activity_concentration_bqml = pixel_array_units * ds[(0x7053, 0x1009)].value
                    suv = process_bqml(activity_concentration_bqml, ds)

                    return suv

                elif (0x7053, 0x1000) in ds and ds[(0x7053, 0x1000)].value != 0:
                    suv = pixel_array_units * ds[(0x7053, 0x1000)].value

                    return suv

                else:
                    error_msg = f"Philips-specific scaling factors not present!"
                    raise DataStructureError(error_msg)

            else:
                error_msg = f"Vendor {ds.Manufacturer} is not supported with CNTS units!"
                raise DataStructureError(error_msg)

        ds = pydicom.dcmread(dicom_file_path)
        units = ds.Units
        pixel_array_units = (ds.pixel_array * ds.RescaleSlope) + ds.RescaleIntercept

        if units == 'GML':
            if (0x0054, 0x1006) in ds:
                if ds[0x0054, 0x1006].value == 'BW':
                    suv = process_gml(pixel_array_units)
                else:
                    error_msg = f"GML with {ds[0x0054, 0x1006].value} SUV normalizatoin is not supported!"
                    raise DataStructureError(error_msg)
            else:
                suv = process_gml(pixel_array_units)
        elif units == 'BQML':
            suv = process_bqml(pixel_array_units, ds)
        elif units == 'CNTS':
            suv = process_cnts(pixel_array_units, ds)
        else:
            error_msg = f"Units {units} are not supported!"
            raise DataStructureError(error_msg)

        return suv.T

    intensity_array = np.zeros(suv_image.GetSize())

    dicom_files = sorted(dicom_files, key=lambda f: float(f['ds'].ImagePositionPatient[2]))
    for z_slice_id, dicom_file in enumerate(dicom_files):
        intensity_array[:, :, z_slice_id] = process_single_slice(dicom_file['file_path'])

    # Flip from head to toe
    intensity_image = sitk.GetImageFromArray(intensity_array.T)
    intensity_image.SetOrigin(suv_image.GetOrigin())
    intensity_image.SetSpacing(np.array(suv_image.GetSpacing()))
    intensity_image.SetDirection(np.array(suv_image.GetDirection()))
    return intensity_image

def modality_mapping(modality):
    modality_map = {'PET': 'PT', 'CT': 'CT', 'MRI': 'MR', 'RTSTRUCT': 'RTSTRUCT', 'MG': 'MG', 'RTDOSE': 'RTDOSE'}
    return modality_map[modality]

def process_dicom_series(dicom_files):
    reader = sitk.ImageSeriesReader()
    file_names = [i['file_path'] for i in dicom_files]
    reader.SetFileNames(file_names)
    image = reader.Execute()

    slice_z_origin = []
    direction = None
    for dicom_file in dicom_files:
        ds = dicom_file['ds']

        if ds.Modality in ['CT', 'PT', 'MR']:
            pixel_spacing = ds.PixelSpacing

            # Use full 3D position
            iop = np.array(ds.ImageOrientationPatient, dtype=float)  # 6 values
            row_cosines = iop[:3]
            col_cosines = iop[3:]
            normal = np.cross(row_cosines, col_cosines)  # image plane normal
            if direction is None:
                direction = np.vstack([row_cosines, col_cosines, normal]).flatten(order="F")
            # Project position onto normal vector
            distance_along_normal = np.dot(np.array(ds.ImagePositionPatient, dtype=float), normal)
            slice_z_origin.append(distance_along_normal)

        elif ds.Modality == 'MG':
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
    if dicom_files[0]['ds'].Modality == 'CT' and np.min(sitk.GetArrayFromImage(image)) >= 0:
        error_msg = f'Non-negative CT intensity. SITK failed to convert CT into HU for {dicom_files[0]["file_path"]}. The patient is excluded from analysis'
        raise DataStructureError(error_msg)

    return image


def extract_dicom_mask(rtstruct_path, roi_name, image):
    def generate_mask_array(contours, sitk_image):

        # Get image dimensions: (width, height, depth)
        width, height, depth = sitk_image.GetSize()
        # Create an empty 3D mask with shape (depth, height, width)
        mask_array = np.zeros((depth, height, width), dtype=np.uint8)

        for contour in contours:
            # Normalize contour type string
            contour_type = contour['type'].upper().replace('_', '').strip()
            if contour_type not in ['CLOSEDPLANAR', 'INTERPOLATEDPLANAR', 'CLOSEDPLANARXOR']:
                continue

            points = contour['points']
            num_points = len(points['x'])
            # Transform all physical points to continuous index coordinates
            transformed_points = np.array([
                sitk_image.TransformPhysicalPointToContinuousIndex(
                    (points['x'][i], points['y'][i], points['z'][i])
                )
                for i in range(num_points)
            ])

            # Compute rounded z indices for each point
            z_indices = np.rint(transformed_points[:, 2]).astype(int)

            # Prepare polygon coordinates for x and y. Note: polygon2mask expects (row, col) = (y, x)
            polygon_coords = np.column_stack((transformed_points[:, 1], transformed_points[:, 0]))
            mask_layer = draw.polygon2mask((height, width), polygon_coords)

            # Combine the mask on each relevant slice:
            # Use XOR for "CLOSEDPLANARXOR" contours; for others, simply add (logical OR).
            for z in z_indices:
                if contour_type == 'CLOSEDPLANARXOR':
                    mask_array[z] = np.logical_xor(mask_array[z], mask_layer).astype(np.uint8)
                else:
                    mask_array[z] = np.logical_or(mask_array[z], mask_layer).astype(np.uint8)

        return mask_array

    def get_contour_data(file_path, selected_roi):
        def get_contour_coord(metadata, current_roi_sequence, skip_contours_bool=False):
            contour_info = {
                'name': getattr(metadata[current_roi_sequence.ReferencedROINumber], 'ROIName', 'unknown'),
                'roi_number': current_roi_sequence.ReferencedROINumber,
                'referenced_frame': getattr(metadata[current_roi_sequence.ReferencedROINumber],
                                            'ReferencedFrameOfReferenceUID', 'unknown'),
                'display_color': getattr(current_roi_sequence, 'ROIDisplayColor', [])
            }

            if not skip_contours_bool and hasattr(current_roi_sequence, 'ContourSequence'):
                contour_info['sequence'] = [{
                    'type': getattr(contour, 'ContourGeometricType', 'unknown'),
                    'points': {
                        'x': [contour.ContourData[i] for i in range(0, len(contour.ContourData), 3)],
                        'y': [contour.ContourData[i + 1] for i in range(0, len(contour.ContourData), 3)],
                        'z': [contour.ContourData[i + 2] for i in range(0, len(contour.ContourData), 3)]
                    }
                } for contour in current_roi_sequence.ContourSequence]

            return contour_info

        dicom_data = pydicom.dcmread(file_path)
        if not hasattr(dicom_data, 'StructureSetROISequence'):
            raise InvalidDicomError()

        contour_data = None
        metadata_map = {data.ROINumber: data for data in dicom_data.StructureSetROISequence}
        for roi_sequence in dicom_data.ROIContourSequence:
            roi_name = getattr(metadata_map[roi_sequence.ReferencedROINumber], 'ROIName', 'unknown')
            if roi_name == selected_roi:
                contour_data = get_contour_coord(metadata_map, roi_sequence)
                break

        return contour_data

    rt_struct = get_contour_data(rtstruct_path, roi_name)
    if rt_struct:
        mask = generate_mask_array(rt_struct['sequence'], image)
        return Image(array=mask,
                     origin=image.GetOrigin(),
                     spacing=np.array(image.GetSpacing()),
                     direction=image.GetDirection(),
                     shape=image.GetSize())
    else:
        return Image()


def get_all_structure_names(rtstruct_path):
    """Extracts all structure names from an RTSTRUCT DICOM file.

    Args:
        rtstruct_path (str): Path to the RTSTRUCT DICOM file.

    Returns:
        list: A list of structure names.

    Raises:
        InvalidDicomError: If the DICOM file does not have a StructureSetROISequence attribute.
    """
    # Read the DICOM file
    dicom_data = pydicom.dcmread(rtstruct_path)

    # Check if the file contains a StructureSetROISequence
    if not hasattr(dicom_data, 'StructureSetROISequence'):
        raise InvalidDicomError(f"The DICOM file at {rtstruct_path} is not a valid RTSTRUCT file.")

    # Map ROI numbers to metadata for quick lookup
    metadata_map = {roi_data.ROINumber: roi_data for roi_data in dicom_data.StructureSetROISequence}

    # Extract structure names
    structure_names = []
    if hasattr(dicom_data, 'ROIContourSequence'):
        for roi_sequence in dicom_data.ROIContourSequence:
            roi_number = roi_sequence.ReferencedROINumber
            roi_name = getattr(metadata_map.get(roi_number, {}), 'ROIName', 'unknown')
            structure_names.append(roi_name)

    return structure_names
