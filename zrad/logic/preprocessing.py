import os

import SimpleITK as sitk
import numpy as np
import pydicom
from joblib import Parallel, delayed, cpu_count

from .toolbox_logic import Image, extract_nifti_image, extract_nifti_mask, list_folders_in_defined_range, extract_dicom


class Preprocessing:

    def __init__(self, load_dir, save_dir,
                 input_data_type, input_imaging_mod,
                 resample_resolution, resample_dimension,
                 image_interpolation_method, mask_interpolation_method, mask_interpolation_threshold=.5,
                 start_folder=None, stop_folder=None, list_of_patient_folders=None,
                 structure_set=None, nifti_image=None,
                 number_of_threads=1):

        if os.path.exists(load_dir):
            self.load_dir = load_dir
        else:
            print(f"Load directory '{load_dir}' does not exist. Aborted!")
            return

        if os.path.exists(save_dir):
            self.save_dir = save_dir
        else:
            print(f"Save directory '{save_dir}' does not exist. Aborted!")
            return

        if list_of_patient_folders is None and start_folder is not None and stop_folder is not None:
            self.list_of_patient_folders = list_folders_in_defined_range(start_folder, stop_folder, self.load_dir)
        elif list_of_patient_folders is not None and list_of_patient_folders not in [[], ['']]:
            self.list_of_patient_folders = list_of_patient_folders
        else:
            print('Incorrectly selected patient folders. Aborted!')
            return

        if input_data_type in ['DICOM', 'NIFTI']:
            self.input_data_type = input_data_type
        else:
            print(f"Wrong input data type '{input_data_type}', available types: 'DICOM', 'NIFTI'. Aborted!")
            return

        if input_imaging_mod in ['CT', 'PT', 'MR']:
            self.input_imaging_mod = input_imaging_mod
        else:
            print(f"Wrong input imaging type '{input_imaging_mod}', available types: 'CT', 'PT', 'MR'. Aborted!")
            return

        if self.input_data_type == 'NIFTI':
            if nifti_image is not None:
                image_exists = True
                for folder in self.list_of_patient_folders:
                    if not os.path.exists(os.path.join(load_dir, str(folder), nifti_image)):
                        image_exists = False
                        if not image_exists:
                            print(f"The NIFTI image file does not exist "
                                  f"'{os.path.join(load_dir, str(folder), nifti_image)}'")
                if image_exists:
                    self.nifti_image = nifti_image
            else:
                print('Select the NIFTI image file. Aborted!')
                return

        if (mask_interpolation_method in ['NN', 'Linear', 'BSpline', 'Gaussian']
                and image_interpolation_method in ['NN', 'Linear', 'BSpline', 'Gaussian']):
            self.mask_interpolation_method = mask_interpolation_method
            self.image_interpolation_method = image_interpolation_method
            if mask_interpolation_method == 'NN':
                self.mask_threshold = 1.0
            else:
                if isinstance(mask_interpolation_threshold, (int, float)) and 0 <= mask_interpolation_threshold <= 1:
                    self.mask_threshold = float(mask_interpolation_threshold)
                else:
                    print(f"Selected threshold '{mask_interpolation_threshold} not in range'"
                          f'[0,1] or not float/integer data type')
        else:
            print(f"Wrong mask and/or image interpolation method "
                  f"(mask: {mask_interpolation_threshold}, image {image_interpolation_method}), "
                  "available methods: 'NN', 'Linear', 'BSpline', 'Gaussian'")
            return

        if isinstance(number_of_threads, (int, float)) and 0 < number_of_threads <= cpu_count():
            self.number_of_threads = number_of_threads
        else:
            print('Number of threads is not an integer or selected nubmer is greater than maximum number of available'
                  f'CPU. (Max available {cpu_count()} units)')
            return

        if structure_set is not None:
            self.structure_set = structure_set
        else:
            self.structure_set = ['']

        if type(resample_resolution) in [float, int] and resample_resolution > 0:
            self.resample_resolution = resample_resolution
        else:
            print(f'Resample resolution {resample_resolution} is not int/float or non-positive. Aborted!')
            return

        if resample_dimension in ['3D', '2D']:
            self.resample_dimension = resample_dimension
        else:
            print(f"Resample dimension '{resample_dimension}' is not '2D' or '3D'. Aborted!")
            return

        self.patient_folder = None
        self.patient_number = None

    def resample(self):
        Parallel(n_jobs=self.number_of_threads)(
            delayed(self._load_patient)(patient_folder) for patient_folder in self.list_of_patient_folders)

    def _load_patient(self, patient_number):

        self.patient_number = str(patient_number)
        self.patient_folder = os.path.join(self.load_dir, self.patient_number)
        self.pat_original_image_and_masks = {}
        self.pat_resampled_image_and_masks = {}

        if self.input_data_type == 'NIFTI':
            self._process_nifti_files()
        else:
            self._process_dicom_files()

        self._resampling()
        self._save_as_nifti()

    def _process_nifti_files(self):
        def extract_nifti(key, mask_file_name=None):

            reader = sitk.ImageFileReader()
            reader.SetImageIO("NiftiImageIO")
            if key == 'IMAGE':
                return extract_nifti_image(self, reader)

            elif key.startswith('MASK'):
                return extract_nifti_mask(self, reader, mask_file_name, self.pat_original_image_and_masks['IMAGE'])

        self.pat_original_image_and_masks['IMAGE'] = extract_nifti('IMAGE')
        if self.structure_set != ['']:
            for nifti_file_name in self.structure_set:
                instance_key = 'MASK_' + nifti_file_name
                file_found, mask = extract_nifti(instance_key, nifti_file_name)
                if file_found:
                    self.pat_original_image_and_masks[instance_key] = mask

    def _process_dicom_files(self):
        self.pat_original_image_and_masks['IMAGE'] = extract_dicom(dicom_dir=self.patient_folder, rtstract=False,
                                                                   modality=self.input_imaging_mod)

        if self.structure_set != ['']:
            for dicom_file in os.listdir(self.patient_folder):
                dcm_data = pydicom.dcmread(os.path.join(self.patient_folder, dicom_file))
                if dcm_data.Modality == 'RTSTRUCT':
                    masks = extract_dicom(dicom_dir=self.patient_folder, rtstract=True,
                                          rtstruct_file=os.path.join(self.patient_folder, dicom_file),
                                          selected_structures=self.structure_set,
                                          modality=self.input_imaging_mod)
                    for instance_key in masks.keys():
                        self.pat_original_image_and_masks['MASK_'+instance_key] = masks[instance_key]

    def _resampling(self):

        def interpolator(key):
            interpolators = {
                'Linear': sitk.sitkLinear,
                'NN': sitk.sitkNearestNeighbor,
                'BSpline': sitk.sitkBSpline,
                'Gaussian': sitk.sitkGaussian
            }
            if key.startswith('IMAGE'):
                return interpolators.get(self.image_interpolation_method)
            elif key.startswith('MASK'):
                return interpolators.get(self.mask_interpolation_method)

        def resampled_origin(initial_shape, initial_spacing, resulted_spacing, initial_origin, axis=0):
            """
            https://arxiv.org/pdf/1612.07003.pdf QCY4
            """
            n_a = float(initial_shape[axis])
            s_a = initial_spacing[axis]
            s_b = resulted_spacing[axis]
            n_b = np.ceil((n_a * s_a) / s_b)
            x_b = initial_origin[axis] + (s_a * (n_a - 1) - s_b * (n_b - 1)) / 2
            return x_b

        input_spacing = self.pat_original_image_and_masks['IMAGE'].spacing
        input_origin = self.pat_original_image_and_masks['IMAGE'].origin
        input_direction = self.pat_original_image_and_masks['IMAGE'].direction
        input_shape = self.pat_original_image_and_masks['IMAGE'].shape

        if self.resample_dimension == '3D':
            output_spacing = [self.resample_resolution] * 3
        elif self.resample_dimension == '2D':
            output_spacing = [self.resample_resolution] * 2 + [self.pat_original_image_and_masks['IMAGE'].spacing[2]]
        else:
            raise ValueError(f"Unsupported resize_dim: {self.resample_dimension}")

        output_origin = [
            resampled_origin(input_shape, input_spacing, output_spacing, input_origin, axis) for axis in range(3)
        ]

        output_shape = np.ceil((np.array(input_shape) * (input_spacing / output_spacing))).astype(int)

        for instance_key in list(self.pat_original_image_and_masks.keys()):

            sitk_image = sitk.GetImageFromArray(self.pat_original_image_and_masks[instance_key].array)
            sitk_image.SetSpacing(input_spacing)
            sitk_image.SetOrigin(input_origin)
            sitk_image.SetDirection(input_direction)

            resample = sitk.ResampleImageFilter()
            resample.SetOutputSpacing(output_spacing)
            resample.SetOutputOrigin(output_origin)
            resample.SetOutputDirection(input_direction)
            resample.SetSize(output_shape.tolist())
            resample.SetOutputPixelType(sitk.sitkFloat64)
            resample.SetInterpolator(interpolator(instance_key))
            resampled_sitk_image = resample.Execute(sitk_image)

            resampled_array = None

            if instance_key.startswith('IMAGE'):
                if self.input_imaging_mod == 'CT':
                    resampled_sitk_image = sitk.Round(resampled_sitk_image)
                    resampled_sitk_image = sitk.Cast(resampled_sitk_image, sitk.sitkInt16)
                elif self.input_imaging_mod in ['MR', 'PT']:
                    resampled_sitk_image = sitk.Cast(resampled_sitk_image, sitk.sitkFloat64)
                resampled_array = sitk.GetArrayFromImage(resampled_sitk_image)

            elif instance_key.startswith('MASK'):
                resampled_array = np.where(sitk.GetArrayFromImage(resampled_sitk_image) >= self.mask_threshold, 1, 0)
                resampled_array = resampled_array.astype(np.int16)

            self.pat_resampled_image_and_masks[instance_key] = Image(array=resampled_array,
                                                                     origin=output_origin,
                                                                     spacing=output_spacing,
                                                                     direction=input_direction,
                                                                     shape=output_shape)
            del self.pat_original_image_and_masks[instance_key]

    def _save_as_nifti(self):
        for key, img in self.pat_resampled_image_and_masks.items():
            img.save_as_nifti(instance=self, key=key)
