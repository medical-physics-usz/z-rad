import json
import logging
import os
import sys
import time
import urllib.request
from datetime import datetime

import SimpleITK as sitk
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from skimage import draw


class Image:
    def __init__(self, array, origin, spacing, direction, shape):
        self.array = array
        self.origin = origin
        self.spacing = spacing
        self.direction = direction
        self.shape = shape

    def save_as_nifti(self, instance, key):
        if key.startswith('MASK_'):
            key = key[5:]
        output_path = os.path.join(instance.output_dir, instance.patient_number, key + '.nii.gz')
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        img = sitk.GetImageFromArray(self.array)
        img.SetOrigin(self.origin)
        img.SetSpacing(self.spacing)
        img.SetDirection(self.direction)
        sitk.WriteImage(img, output_path)


def list_folders_in_defined_range(folder_to_start, folder_to_stop, directory_path):
    list_of_folders = []
    for folder in os.listdir(directory_path):
        if folder.isdigit():
            folder = int(folder)
            if int(folder_to_start) <= folder <= int(folder_to_stop):
                list_of_folders.append(str(folder))

    return list_of_folders


def extract_nifti_image(instance, sitk_reader, key=None):

    if key is None or key == 'IMAGE':
        if os.path.isfile(os.path.join(instance.patient_folder, instance.nifti_image + '.nii.gz')):
            sitk_reader.SetFileName(os.path.join(instance.patient_folder, instance.nifti_image + '.nii.gz'))
        elif os.path.isfile(os.path.join(instance.patient_folder, instance.nifti_image + '.nii')):
            sitk_reader.SetFileName(os.path.join(instance.patient_folder, instance.nifti_image + '.nii'))
    elif key == 'ORIG_IMAGE':
        if os.path.isfile(os.path.join(instance.patient_folder, instance.nifti_image_orig + '.nii.gz')):
            sitk_reader.SetFileName(os.path.join(instance.patient_folder, instance.nifti_image_orig + '.nii.gz'))
        elif os.path.isfile(os.path.join(instance.patient_folder, instance.nifti_image_orig + '.nii')):
            sitk_reader.SetFileName(os.path.join(instance.patient_folder, instance.nifti_image_orig + '.nii'))
    image = sitk_reader.Execute()
    array = sitk.GetArrayFromImage(image)

    return Image(array=array.astype(np.float64),
                 origin=image.GetOrigin(),
                 spacing=np.array(image.GetSpacing()),
                 direction=image.GetDirection(),
                 shape=image.GetSize())


def extract_nifti_mask(instance, sitk_reader, mask, patient_image):
    file_found = False

    if os.path.isfile(os.path.join(instance.patient_folder, mask + '.nii.gz')):
        file_found = True
        sitk_reader.SetFileName(os.path.join(instance.patient_folder, mask + '.nii.gz'))
    elif os.path.isfile(os.path.join(instance.patient_folder, mask + '.nii')):
        file_found = True
        sitk_reader.SetFileName(os.path.join(instance.patient_folder, mask + '.nii'))
    if file_found:
        image = sitk_reader.Execute()
        array = sitk.GetArrayFromImage(image)

        return file_found, Image(array=array.astype(np.float64),
                                 origin=patient_image.origin,
                                 spacing=patient_image.spacing,
                                 direction=patient_image.direction,
                                 shape=patient_image.shape
                                 )
    else:
        return file_found, None


def extract_dicom(dicom_dir, rtstract, modality, rtstruct_file='', selected_structures=None):
    def parse_time(time_str):
        for fmt in ('%H%M%S.%f', '%H%M%S', '%Y%m%d%H%M%S.%f'):
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"time data '{time_str}' does not match expected formats")

    def generate(contours, dcm_im):
        dimensions = dcm_im.GetSize()

        initial_mask = sitk.Image(dimensions, sitk.sitkUInt8)
        initial_mask.CopyInformation(dcm_im)

        mask_array = sitk.GetArrayFromImage(initial_mask)
        mask_array.fill(0)

        for contour in contours:  # Removed tqdm from this line
            if contour['type'].upper().replace('_', '').strip() not in ['CLOSEDPLANAR', 'INTERPOLATEDPLANAR',
                                                                        'CLOSEDPLANARXOR']:
                continue

            points = contour['points']
            transformed_points = np.array(
                [dcm_im.TransformPhysicalPointToContinuousIndex((points['x'][i], points['y'][i], points['z'][i]))
                 for i in range(len(points['x']))])

            z_coord = int(round(transformed_points[0, 2]))
            mask_layer = draw.polygon2mask([dimensions[0], dimensions[1]][::-1],
                                           np.column_stack((transformed_points[:, 1], transformed_points[:, 0])))
            updated_mask = np.logical_xor(mask_array[z_coord, :, :], mask_layer)
            mask_array[z_coord, :, :] = np.where(updated_mask, 1, 0)

        return mask_array

    def process_rt_struct(file_path, selected_ROIs):  # , skip_contours=False):

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

        dicom_data = pydicom.read_file(file_path)
        if not hasattr(dicom_data, 'StructureSetROISequence'):
            raise InvalidDicomError()

        contours_data = []
        metadata_map = {data.ROINumber: data for data in dicom_data.StructureSetROISequence}

        for roi_sequence in dicom_data.ROIContourSequence:
            if 'ExtractAllMasks' in selected_ROIs:
                contours_data.append(get_contour_coord(metadata_map, roi_sequence))

            elif ('ExtractAllMasks' not in selected_ROIs
                  and getattr(metadata_map[roi_sequence.ReferencedROINumber], 'ROIName', 'unknown')
                  in selected_ROIs):

                contours_data.append(get_contour_coord(metadata_map, roi_sequence))

        return contours_data

    def process_dicom_series(directory, imaging_modality):

        # Function to check if a DICOM file is not an RT struct file
        def is_not_rt_struct(dicom_path, uid):
            try:
                dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                return dcm.SOPClassUID != uid
            except InvalidDicomError:
                return False

        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(directory)
        if not series_ids:
            raise ValueError("No DICOM series found in the directory.")
        file_names = reader.GetGDCMSeriesFileNames(directory, series_ids[0])

        dicom_files = [f for f in os.listdir(directory) if (os.path.splitext(f)[-1] == '' or f.endswith('.dcm')) and
                       is_not_rt_struct(os.path.join(directory, f), series_ids[0])]

        ct_raw_intensity = []
        slice_z_origin = []
        HU_intercept = []
        HU_slope = []

        for dcm_file in dicom_files:
            dicom_file_path = os.path.join(directory, dcm_file)
            dicom = pydicom.dcmread(dicom_file_path)
            if dicom.Modality in ['CT', 'PT', 'MR']:
                pixel_spacing = dicom.PixelSpacing
                slice_z_origin.append(float(dicom.ImagePositionPatient[2]))
                if dicom.Modality == 'CT' and dicom.PixelRepresentation == 0:
                    ct_raw_intensity.append(True)
                    HU_intercept.append(float(dicom.RescaleIntercept))
                    HU_slope.append(float(dicom.RescaleSlope))

        slice_z_origin = sorted(slice_z_origin)
        slice_thickness = abs(np.median([slice_z_origin[i] - slice_z_origin[i + 1]
                                         for i in range(len(slice_z_origin) - 1)]))

        reader.SetFileNames(file_names)
        image = reader.Execute()

        image.SetSpacing((float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness)))

        return image

    def calculate_suv(dicom_file_path, min_acquisition_time):

        ds = pydicom.dcmread(dicom_file_path)
        units = ds.Units
        injection_time = parse_time(ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime)

        if ds.DecayCorrection == 'START':
            if 'PHILIPS' in ds.Manufacturer.upper():
                acquisition_time = min_acquisition_time
            elif 'SIEMENS' in ds.Manufacturer.upper() and units == 'BQML':
                try:
                    acquisition_time = parse_time(ds[(0x0071, 0x1022)].value).replace(year=injection_time.year,
                                                                                      month=injection_time.month,
                                                                                      day=injection_time.day)
                except KeyError:
                    acquisition_time = min_acquisition_time
            elif 'GE' in ds.Manufacturer.upper() and units == 'BQML':
                try:
                    acquisition_time = parse_time(ds[(0x0009, 0x100d)].value).replace(year=injection_time.year,
                                                                                      month=injection_time.month,
                                                                                      day=injection_time.day)
                except KeyError:
                    acquisition_time = min_acquisition_time
        elif ds.DecayCorrection == 'ADMIN':
            acquisition_time = injection_time

        patient_weight = float(ds.PatientWeight)
        injected_dose = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)

        elapsed_time = (acquisition_time - injection_time).total_seconds()

        half_life = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
        decay_factor = np.exp(-1 * ((np.log(2) * elapsed_time) / half_life))

        decay_corrected_dose = injected_dose * decay_factor

        image_data = ds.pixel_array
        activity_concentration = (image_data * ds.RescaleSlope) + ds.RescaleIntercept

        if 'PHILIPS' in ds.Manufacturer.upper() and units == 'CNTS':
            activity_concentration = activity_concentration * ds[(0x7053, 0x1009)].value

        suv = activity_concentration / (decay_corrected_dose / (patient_weight * 1000))

        suv = suv.T

        return suv

    def process_pet_dicom(pat_folder, suv_image):
        intensity_array = np.zeros(suv_image.GetSize())
        acquisition_time_list = []
        for dicom_file in os.listdir(pat_folder):
            dcm_data = pydicom.dcmread(os.path.join(pat_folder, dicom_file))
            if dcm_data.Modality == 'PT':
                acquisition_time_list.append(parse_time(dcm_data.AcquisitionTime))
        min_acquisition_time = np.min(acquisition_time_list)
        for dicom_file in os.listdir(pat_folder):
            dcm_data = pydicom.dcmread(os.path.join(pat_folder, dicom_file))
            if dcm_data.Modality == 'PT':
                z_slice_id = int(dcm_data.InstanceNumber) - 1
                intensity_array[:, :, z_slice_id] = calculate_suv(os.path.join(pat_folder, dicom_file),
                                                                  min_acquisition_time)

        # Flip from head to toe
        intensity_array = np.flip(intensity_array, axis=2)

        intensity_image = sitk.GetImageFromArray(intensity_array.T)
        intensity_image.SetOrigin(suv_image.GetOrigin())
        intensity_image.SetSpacing(np.array(suv_image.GetSpacing()))
        intensity_image.SetDirection(np.array(suv_image.GetDirection()))

        return intensity_image

    if modality in ['CT', 'MR']:
        dicom_image = process_dicom_series(dicom_dir, modality)

    elif modality == 'PT':
        dicom_image = process_pet_dicom(dicom_dir, process_dicom_series(dicom_dir, modality))

    if rtstract:
        rt_structs = process_rt_struct(rtstruct_file, selected_structures)

        masks = {}
        for rt_struct in rt_structs:
            if 'ExtractAllMasks' in selected_structures:
                if 'sequence' not in rt_struct:
                    continue

                mask = generate(rt_struct['sequence'], dicom_image)
                masks[rt_struct['name']] = Image(array=mask,
                                                 origin=dicom_image.GetOrigin(),
                                                 spacing=np.array(dicom_image.GetSpacing()),
                                                 direction=dicom_image.GetDirection(),
                                                 shape=dicom_image.GetSize())

            else:
                if rt_struct['name'] in selected_structures:
                    if 'sequence' not in rt_struct:
                        continue

                    mask = generate(rt_struct['sequence'], dicom_image)
                    masks[rt_struct['name']] = Image(array=mask,
                                                     origin=dicom_image.GetOrigin(),
                                                     spacing=np.array(dicom_image.GetSpacing()),
                                                     direction=dicom_image.GetDirection(),
                                                     shape=dicom_image.GetSize())

        return masks
    else:
        return Image(array=sitk.GetArrayFromImage(dicom_image),
                     origin=dicom_image.GetOrigin(),
                     spacing=np.array(dicom_image.GetSpacing()),
                     direction=dicom_image.GetDirection(),
                     shape=dicom_image.GetSize())


def check_dicom_tags(directory, pat_index, logger, image_vol='3D'):
    def parse_time(time_str):
        for fmt in ('%H%M%S.%f', '%H%M%S', '%Y%m%d%H%M%S.%f'):
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        logger.error(f"Time data '{time_str}' does not match expected formats")
        raise ValueError(f"Time data '{time_str}' does not match expected formats")

    def is_not_rt_struct(dicom_path):
        try:
            dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            return dcm.Modality != 'RTSTRUCT'
        except InvalidDicomError:
            return False

    for f in os.listdir(directory):
        if os.path.splitext(f)[-1] != '':
            if not f.endswith('.dcm'):
                logger.warning(f'Patient {pat_index} contains not only DICOM files '
                               f'but also other extentions, e.g. {os.path.splitext(f)[-1]}. '
                               'Excluded from the analysis.')
                return True

    dicom_files = [f for f in os.listdir(directory) if (os.path.splitext(f)[-1] == '' or f.endswith('.dcm'))
                   and is_not_rt_struct(os.path.join(directory, f))]

    if image_vol == '3D':
        slice_z_origin = []
        for dcm_file in dicom_files:
            dicom_file_path = os.path.join(directory, dcm_file)
            dicom = pydicom.dcmread(dicom_file_path)
            if dicom.Modality in ['CT', 'PT', 'MR']:
                slice_z_origin.append(float(dicom.ImagePositionPatient[2]))
        slice_z_origin = sorted(slice_z_origin)
        slice_thickness = [abs(slice_z_origin[i] - slice_z_origin[i + 1])
                           for i in range(len(slice_z_origin) - 1)]
        for i in range(len(slice_thickness) - 1):
            if abs((slice_thickness[i] - slice_thickness[i + 1])) > 10 ** (-3):
                logger.warning(f'Patient {pat_index} is excluded from the analysis'
                               ' due to the inconsistent z-spacing. Absolute deviation is more than 0.001 mm.')
                return True

    acquisition_time_list = []
    for dcm_file in dicom_files:
        dicom_file_path = os.path.join(directory, dcm_file)
        dicom = pydicom.dcmread(dicom_file_path)
        if dicom.Modality == 'PT':
            acquisition_time_list.append(parse_time(dicom.AcquisitionTime))
    time_mismatch = False
    for dcm_file in dicom_files:
        dicom_file_path = os.path.join(directory, dcm_file)
        dicom = pydicom.dcmread(dicom_file_path)
        if dicom.Modality == 'PT':
            try:
                pat_weight = dicom[(0x0010, 0x1030)].value
                if float(pat_weight) < 1:
                    logger.warning(
                        f"For the patient {pat_index} the patient's weight tag (0071, 1022) contains weight < 1kg."
                        'Patient is excluded from the analysis.')
                    return True

            except KeyError:
                logger.warning(
                    f"For the patient {pat_index} the patient's weight tag (0071, 1022) is not present." 
                    'Patient is excluded from the analysis.')
                return True
            if 'DECY' not in dicom[(0x0028, 0x0051)].value or 'ATTN' not in dicom[(0x0028, 0x0051)].value:
                logger.warning(
                    f"For the patient {pat_index}, in DICOM tag (0028, 0051) either no "
                    "'DECY' (decay correction) or 'ATTN' (attenuation correction) "
                    'Patient is excluded from the analysis.')
                return True

            if dicom.Units == 'BQML':
                injection_time = parse_time(
                    dicom.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime)
                if dicom.DecayCorrection == 'START':
                    if 'PHILIPS' in dicom.Manufacturer.upper():
                        acquisition_time = np.min(acquisition_time_list)
                    elif 'SIEMENS' in dicom.Manufacturer.upper():
                        try:
                            acquisition_time = parse_time(dicom[(0x0071, 0x1022)].value).replace(
                                year=injection_time.year,
                                month=injection_time.month,
                                day=injection_time.day)
                            if (acquisition_time != np.min(acquisition_time_list)
                                    or acquisition_time != parse_time(dicom.SeriesTime) and not time_mismatch):
                                time_mismatch = True
                                logger.warning(f'For the patient {pat_index} there is a mismatch between the earliest '
                                               'acquisition time, series time and Siemens private tag (0071, 1022). '
                                               'Time from the Siemens private tag was used.')
                        except KeyError:
                            acquisition_time = np.min(acquisition_time_list)
                            if not time_mismatch:
                                time_mismatch = True
                                logger.warning(f'For the patient {pat_index} private Siemens tag (0071, 1022)'
                                               ' is not present. The earliest of all acquisition times was used.')
                            if acquisition_time != parse_time(dicom.SeriesTime) and not time_mismatch:
                                time_mismatch = True
                                logger.warning(f'For the patient {pat_index} a mismatch present between the earliest '
                                               'acquisition time and series time. Earliest acquisition time was used.')
                    elif 'GE' in dicom.Manufacturer.upper():
                        try:
                            acquisition_time = parse_time(dicom[(0x0009, 0x100d)].value).replace(
                                year=injection_time.year,
                                month=injection_time.month,
                                day=injection_time.day)
                            if (acquisition_time != np.min(acquisition_time_list)
                                    or acquisition_time != parse_time(dicom.SeriesTime) and not time_mismatch):
                                time_mismatch = True
                                logger.warning(f'For the patient {pat_index} a mismatch present between '
                                               'the earliest acquisition time, series time, and GE private tag. '
                                               'Time from the GE private tag was used.')
                        except KeyError:
                            acquisition_time = np.min(acquisition_time_list)
                            if not time_mismatch:
                                time_mismatch = True
                                logger.warning(f'For the patient {pat_index} private GE tag (0009, 100d)'
                                               ' is not present. The earliest of all acquisition times was used.')
                            if acquisition_time != parse_time(dicom.SeriesTime) and not time_mismatch:
                                time_mismatch = True
                                logger.warning(f'For the patient {pat_index} a mismatch present between the earliest '
                                               'acquisition time and series time. Earliest acquisition time was used.')
                    else:
                        logger.warning(f'For the patient {pat_index} the unknown PET scaner manufacturer is present. '
                                       'Z-Rad only supports Philips, Siemens, and GE.')
                        return True

                elif dicom.DecayCorrection == 'ADMIN':
                    acquisition_time = injection_time

                else:
                    logger.warning(
                        f'For the patient {pat_index} the unsupported Decay Correction {dicom.DecayCorrection} '
                        'is present. Only supported are "START" and "ADMIN". '
                        'Patient is excluded from the analysis.')
                    return True
                elapsed_time = (acquisition_time - injection_time).total_seconds()
                if elapsed_time < 0:
                    logger.warning(f'Patient {pat_index} is excluded from the analysis '
                                   'due to the negative time difference in the decay factor.')
                    return True
                elif elapsed_time > 0 and abs(elapsed_time) < 1800 and dicom.DecayCorrection != 'ADMIN':
                    logger.warning(f'Only {abs(elapsed_time) / 60} minutes after the injection')
            elif dicom.Units == 'CNTS' and 'PHILIPS' in dicom.Manufacturer.upper():
                try:
                    activity_scale_factor = dicom[(0x7053, 0x1009)].value
                    print(activity_scale_factor)
                    if activity_scale_factor == 0.0:
                        logger.warning(
                            f'Patient {pat_index} is excluded, Philips private activity scale factor (7053, 1009) = 0.'
                            '(PET units CNTS)')
                        return True
                except KeyError:
                    logger.warning(
                        f'Patient {pat_index} is excluded, Philips private activity scale factor '
                        '(7053, 1009) is missing. (PET units CNTS)')
                    return True
            else:
                logger.warning(f'Patient {pat_index} is excluded, only supported PET Units are "BQML" for Philips, '
                               'Siemens and GE or "CNTS" for Philips')
                return True
    return False


def get_logger(logger_date_time):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # File handler with UTF-8 encoding
        if not os.path.exists(os.path.join(os.getcwd(), 'Log files')):
            os.makedirs(os.path.join(os.getcwd(), 'Log files'))
        file_handler = logging.FileHandler(os.path.join(os.getcwd(), 'Log files', f'{logger_date_time}.log'),
                                           encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger = logging.getLogger()
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    logging.shutdown()


def close_all_loggers():
    # Close all handlers of the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    # Iterate over all loggers and close their handlers
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def download_file(url, local_filename, verbose=False):
    """
    Downloads a file from the specified URL and saves it to the local file system.

    Args:
        url (str): The URL of the file to download.
        local_filename (str): The local path where the file should be saved.
        verbose (bool): If True, prints download status messages.

    Raises:
        URLError: If there's an issue with the network or URL.
        IOError: If there's an issue writing the file to the local system.
    """
    try:
        with urllib.request.urlopen(url) as response:
            with open(local_filename, 'wb') as out_file:
                out_file.write(response.read())
        if verbose:
            print(f"Downloaded {local_filename}")
    except urllib.error.URLError as e:
        print(f"Failed to download {url}. Error: {e}")
        raise
    except IOError as e:
        print(f"Failed to save {local_filename}. Error: {e}")
        raise


def fetch_github_directory_files(owner, repo, directory_path, save_path=None, token=None, verbose=False):
    """
    Fetches all files from a specified GitHub repository directory and saves them locally.

    Args:
        owner (str): The GitHub username or organization that owns the repository.
        repo (str): The name of the GitHub repository.
        directory_path (str): The path to the directory within the repository.
        save_path (str, optional): The local directory where the downloaded files will be saved.
        token (str, optional): GitHub personal access token for authentication.
        verbose (bool, optional): If True, print progress messages.

    Raises:
        ValueError: If an unsupported imaging format or chapter is specified.
        URLError: If there's an issue with the network or GitHub API.
        IOError: If there's an issue writing files to the local system.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{directory_path}"
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'

    try:
        with urllib.request.urlopen(urllib.request.Request(api_url, headers=headers)) as response:
            items = json.loads(response.read().decode())

            for item in items:
                item_path = item['path']  # Full path of the item in the repo
                relative_path = os.path.relpath(item_path,
                                                directory_path)  # Relative path within the specified directory

                if item['type'] == 'file':
                    download_url = item['download_url']
                    if save_path:
                        local_path = os.path.join(save_path, relative_path)
                    else:
                        local_path = os.path.join(directory_path, relative_path)

                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    if verbose:
                        print(f"Downloading {relative_path}...")
                    download_file(download_url, local_path, verbose)
                    if verbose:
                        print(f"{relative_path} downloaded.")
                elif item['type'] == 'dir':
                    # Recursive call to handle subdirectories
                    new_save_path = os.path.join(save_path, relative_path) if save_path else None
                    fetch_github_directory_files(owner, repo, item_path, new_save_path, token, verbose)

    except urllib.error.HTTPError as e:
        if e.code == 403:
            print("Rate limit exceeded. Waiting for 60 seconds before retrying...")
            time.sleep(60)  # Wait for 60 seconds before retrying
            fetch_github_directory_files(owner, repo, directory_path, save_path, token, verbose)
        else:
            print(f"Failed to fetch directory contents from {api_url}. Error: {e}")
            raise
    except urllib.error.URLError as e:
        print(f"Failed to fetch directory contents from {api_url}. Error: {e}")
        raise
    except IOError as e:
        print(f"Failed to create directories or save files to {save_path}. Error: {e}")
        raise


def load_ibsi_phantom(chapter=1, phantom='ct_radiomics', imaging_format="dicom", save_path=None):
    """
    Downloads a specified IBSI Phantom dataset in the chosen imaging format and chapter from the IBSI GitHub repository.

    Args:
        chapter (int): The chapter number of the IBSI dataset. Supported values are 1 and 2.
        phantom (str): The type of phantom dataset to download. Options are "ct_radiomics" and "digital".
        imaging_format (str): The imaging format to download. Options are "dicom" and "nifti".
        save_path (str, optional): The local directory where the dataset will be saved.
                                   If None, the dataset is saved under the original directory structure.

    Raises:
        ValueError: If an unsupported chapter, phantom type, or imaging format is specified.
    """
    owner = "theibsi"
    repo = "data_sets"
    supported_chapters = [1, 2]
    supported_phantoms = ['ct_radiomics', 'digital']
    supported_formats = ["dicom", "nifti"]

    if chapter not in supported_chapters:
        raise ValueError(f"Unsupported chapter '{chapter}'. Supported chapters are: {supported_chapters}")

    if phantom not in supported_phantoms:
        raise ValueError(f"Unsupported phantom '{phantom}'. Supported phantoms are: {supported_phantoms}")

    if imaging_format.lower() not in supported_formats:
        raise ValueError(f"Unsupported imaging format '{imaging_format}'. Supported formats are: {supported_formats}")

    if chapter == 1 and phantom == 'digital' and imaging_format == "dicom":
        raise ValueError(f"The DICOM mask was deprecated due to incorrect image spacing. The phantom is available in NIfTI format and consists of the image itself (image) and its segmentation (mask).")

    directory_path = f"ibsi_{chapter}_{phantom}_phantom/{imaging_format.lower()}"
    fetch_github_directory_files(owner, repo, directory_path, save_path)
