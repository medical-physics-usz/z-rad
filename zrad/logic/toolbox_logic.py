import os
import random
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
        output_path = os.path.join(instance.save_dir, instance.patient_number, key + '.nii.gz')
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


def extract_nifti_image(instance, sitk_reader):

    sitk_reader.SetFileName(os.path.join(instance.patient_folder, instance.nifti_image))
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

    def process_rt_struct(file_path, skip_contours=False):
        dicom_data = pydicom.read_file(file_path)
        if not hasattr(dicom_data, 'StructureSetROISequence'):
            raise InvalidDicomError()

        contours_data = []
        metadata_map = {data.ROINumber: data for data in dicom_data.StructureSetROISequence}

        for roi_sequence in dicom_data.ROIContourSequence:
            contour_info = {
                'name': getattr(metadata_map[roi_sequence.ReferencedROINumber], 'ROIName', 'unknown'),
                'roi_number': roi_sequence.ReferencedROINumber,
                'referenced_frame': getattr(metadata_map[roi_sequence.ReferencedROINumber],
                                            'ReferencedFrameOfReferenceUID', 'unknown'),
                'display_color': getattr(roi_sequence, 'ROIDisplayColor', [])
            }

            if not skip_contours and hasattr(roi_sequence, 'ContourSequence'):
                contour_info['sequence'] = [{
                    'type': getattr(contour, 'ContourGeometricType', 'unknown'),
                    'points': {
                        'x': [contour.ContourData[i] for i in range(0, len(contour.ContourData), 3)],
                        'y': [contour.ContourData[i + 1] for i in range(0, len(contour.ContourData), 3)],
                        'z': [contour.ContourData[i + 2] for i in range(0, len(contour.ContourData), 3)]
                    }
                } for contour in roi_sequence.ContourSequence]

            contours_data.append(contour_info)

        return contours_data

    def process_dicom_series(directory):

        # Function to check if a DICOM file is not an RT struct file
        def is_not_rt_struct(dicom_path, uid):
            try:
                dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                return dcm.SOPClassUID != uid
            except Exception:
                return False

        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(directory)
        if not series_ids:
            raise ValueError("No DICOM series found in the directory.")
        file_names = reader.GetGDCMSeriesFileNames(directory, series_ids[0])

        # List all DICOM files in the directory that are not RT struct files
        dicom_files = [f for f in os.listdir(directory) if
                       f.endswith('.dcm') and is_not_rt_struct(os.path.join(directory, f), series_ids[0])]

        # Randomly select a DICOM file from the filtered list
        if dicom_files:
            dicom_file_path = os.path.join(directory, random.choice(dicom_files))
            dicom = pydicom.dcmread(dicom_file_path)

            # Retrieve pixel spacing (Row Spacing, Column Spacing)
            pixel_spacing = dicom.PixelSpacing

            # Retrieve slice thickness if available
            slice_thickness = dicom.get('SliceThickness', 'Not available')
        else:
            print("No suitable DICOM file found in the directory.")

        reader.SetFileNames(file_names)

        image = reader.Execute()
        image.SetSpacing((float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness)))

        return image

    dicom_image = process_dicom_series(dicom_dir)

    def calculate_suv(dicom_file_path):

        def parse_time(time_str):

            for fmt in ('%H%M%S.%f', '%H%M%S'):
                try:
                    return datetime.strptime(time_str, fmt)
                except ValueError:
                    continue
            raise ValueError(f"time data '{time_str}' does not match expected formats")

        ds = pydicom.dcmread(dicom_file_path)

        unit = ds.Units

        if unit != 'BQML':
            raise ValueError(f"Expected unit to be 'BQML', but got {unit}")

        patient_weight = float(ds.PatientWeight)
        injected_dose = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
        injection_time_str = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
        acquisition_time_str = ds.AcquisitionTime

        injection_time = parse_time(injection_time_str)
        acquisition_time = parse_time(acquisition_time_str)
        elapsed_time = (acquisition_time - injection_time).total_seconds()

        half_life = 109.8 * 60
        decay_factor = np.exp(-1 * ((np.log(2) * elapsed_time) / half_life))

        decay_corrected_dose = injected_dose * decay_factor

        image_data = ds.pixel_array
        activity_concentration = (image_data * ds.RescaleSlope) + ds.RescaleIntercept
        suv = activity_concentration / (decay_corrected_dose / (patient_weight * 1000))

        # 90 degree rotation
        suv = suv.T

        return suv

    def process_pet_dicom(pat_folder, suv_image):
        intensity_array = np.zeros(suv_image.GetSize())
        for dicom_file in os.listdir(pat_folder):
            dcm_data = pydicom.dcmread(os.path.join(pat_folder, dicom_file))

            if dcm_data.Modality != 'RTSTRUCT':
                z_slice_id = int(dcm_data.InstanceNumber) - 1
                intensity_array[:, :, z_slice_id] = calculate_suv(os.path.join(pat_folder, dicom_file))

        # Flip from head to toe
        intensity_array = np.flip(intensity_array, axis=2)

        intensity_image = sitk.GetImageFromArray(intensity_array.T)
        intensity_image.SetOrigin(suv_image.GetOrigin())
        intensity_image.SetSpacing(np.array(suv_image.GetSpacing()))
        intensity_image.SetDirection(np.array(suv_image.GetDirection()))

        return intensity_image

    if modality in ['CT', 'MR']:
        dicom_image = process_dicom_series(dicom_dir)

    elif modality == 'PT':
        dicom_image = process_pet_dicom(dicom_dir, process_dicom_series(dicom_dir))

    if rtstract:
        rt_structs = process_rt_struct(rtstruct_file)

        masks = {}
        for rt_struct in rt_structs:
            if selected_structures == 'ALL':
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
