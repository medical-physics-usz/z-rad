import os
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count

import SimpleITK as sitk
import numpy as np
import pydicom

from .toolbox_logic import Image, extract_nifti_image, extract_nifti_mask, list_folders_in_defined_range, \
    extract_dicom, check_dicom_tags, get_logger, handle_uncaught_exception, close_all_loggers

sys.excepthook = handle_uncaught_exception


class Preprocessing:

    def __init__(self, input_dir, output_dir,
                 input_data_type, input_imaging_modality,
                 structure_set=None,
                 just_save_as_nifti=False,
                 resample_resolution=1.0, resample_dimension='3D',
                 image_interpolation_method='Linear',
                 mask_interpolation_method='Linear', mask_interpolation_threshold=.5,
                 start_folder=None, stop_folder=None, list_of_patient_folders=None,
                 nifti_image=None,
                 number_of_threads=1):

        close_all_loggers()
        self.logger_date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.logger = get_logger(self.logger_date_time+'_Preprocessing')
        self.logger.info("Preliminary Data Check Started")
        if os.path.exists(input_dir):
            self.input_dir = input_dir
        else:
            self.logger.error(f"Load directory '{input_dir}' does not exist.")
            raise ValueError(f"Load directory '{input_dir}' does not exist.")

        if os.path.exists(output_dir):
            self.output_dir = output_dir
        else:
            os.makedirs(output_dir)
            self.output_dir = output_dir

        if start_folder is not None and stop_folder is not None:
            self.list_of_patient_folders = list_folders_in_defined_range(start_folder, stop_folder, self.input_dir)
        elif list_of_patient_folders is not None and list_of_patient_folders not in [[], ['']]:
            self.list_of_patient_folders = list_of_patient_folders
        elif list_of_patient_folders is None and start_folder is None and stop_folder is None:
            self.list_of_patient_folders = [
                e for e in os.listdir(input_dir)
                if not e.startswith('.') and os.path.isdir(os.path.join(input_dir, e))
            ]
        else:
            self.logger.error('Incorrectly selected patient folders.')
            raise ValueError('Incorrectly selected patient folders.')

        if input_data_type in ['DICOM', 'NIfTI']:
            self.input_data_type = input_data_type
        else:
            raise ValueError(f"Wrong input data type '{input_data_type}', available types: 'DICOM', 'NIfTI'.")

        if input_imaging_modality in ['CT', 'PT', 'MR']:
            self.input_imaging_modality = input_imaging_modality
        else:
            self.logger.error(f"Wrong input imaging type '{input_imaging_modality}', "
                              "available types: 'CT', 'PT', 'MR'.")
            raise ValueError(f"Wrong input imaging type '{input_imaging_modality}', available types: 'CT', 'PT', 'MR'.")

        if self.input_data_type == 'DICOM':
            list_pat_id_to_del = []
            for pat_index, pat_path in enumerate(self.list_of_patient_folders):
                if check_dicom_tags(os.path.join(input_dir, pat_path), pat_path, self.logger, resample_dimension):

                    list_pat_id_to_del.append(pat_path)
            for pat_to_del in np.unique(list_pat_id_to_del):
                self.list_of_patient_folders.remove(pat_to_del)

        if self.input_data_type == 'NIfTI':
            if nifti_image is not None:
                image_exists = True
                for folder in self.list_of_patient_folders:
                    if (not os.path.isfile(os.path.join(input_dir, str(folder), nifti_image + '.nii.gz'))
                       and not os.path.isfile(os.path.join(input_dir, str(folder), nifti_image + '.nii'))):
                        image_exists = False
                        if not image_exists:
                            self.logger.info(f"The NIfTI image file does not exist "
                                             f"'{os.path.join(input_dir, str(folder), nifti_image)}'. "
                                             f"The patient {folder} is skipped")
                            self.list_of_patient_folders.remove(folder)
                if image_exists:
                    self.nifti_image = nifti_image
            else:
                self.logger.error('Select the NIfTI image file.')
                raise ValueError('Select the NIfTI image file.')

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
                    self.logger.error(f"Selected threshold '{mask_interpolation_threshold} not in range'"
                                      f'[0,1] or not float/integer data type')
                    raise ValueError(f"Selected threshold '{mask_interpolation_threshold} not in range'"
                                     f'[0,1] or not float/integer data type')
        else:
            self.logger.error(f"Wrong mask and/or image interpolation method (mask: {mask_interpolation_threshold}, "
                              f"image {image_interpolation_method}), available methods: "
                              "'NN', 'Linear', 'BSpline', 'Gaussian'")
            raise ValueError(f"Wrong mask and/or image interpolation method (mask: {mask_interpolation_threshold}, "
                             f"image {image_interpolation_method}), available methods: "
                             "'NN', 'Linear', 'BSpline', 'Gaussian'")

        if isinstance(number_of_threads, (int, float)) and 0 < number_of_threads <= cpu_count():
            self.number_of_threads = number_of_threads
        else:
            self.logger.error(f'Provided number of threads: {number_of_threads}, is not an integer or selected number '
                              f'is greater than maximum number of available CPU. (Max available {cpu_count()} units)')
            raise ValueError(f'Provided number of threads: {number_of_threads}, is not an integer or selected number '
                             f'is greater than maximum number of available CPU. (Max available {cpu_count()} units)')

        if structure_set is not None:
            self.structure_set = structure_set
        else:
            self.structure_set = ['']

        if type(resample_resolution) in [float, int] and resample_resolution > 0:
            self.resample_resolution = resample_resolution
        else:
            self.logger.error(f'Resample resolution {resample_resolution} is not int/float or non-positive.')
            raise ValueError(f'Resample resolution {resample_resolution} is not int/float or non-positive.')

        if resample_dimension in ['3D', '2D']:
            self.resample_dimension = resample_dimension
        else:
            self.logger.error(f"Resample dimension '{resample_dimension}' is not '2D' or '3D'.")
            raise ValueError(f"Resample dimension '{resample_dimension}' is not '2D' or '3D'.")

        self.just_save_as_nifti = just_save_as_nifti
        self.patient_folder = None
        self.patient_number = None

        self.logger.info("Preliminary Data Check Completed")

    def resample(self):

        if len(self.list_of_patient_folders) > 0:
            try:
                if self.just_save_as_nifti:
                    self.logger.info('Conversion Started')
                else:
                    self.logger.info('Resampling Started')
                with Pool(self.number_of_threads) as pool:
                    pool.map(self._load_patient, sorted(self.list_of_patient_folders))
                if self.just_save_as_nifti:
                    self.logger.info('Conversion Completed')
                else:
                    self.logger.info('Resampling Completed')
            except Exception:
                self.logger.error("Caught an exception", exc_info=True)
        else:
            self.logger.info('No patients to resample.')

    def _load_patient(self, patient_number):
        self.patient_logger = get_logger(self.logger_date_time+'_Preprocessing')
        self.patient_logger.info(f'Working on patient: {patient_number}')
        self.patient_number = str(patient_number)
        self.patient_folder = os.path.join(self.input_dir, self.patient_number)
        self.pat_original_image_and_masks = {}
        self.pat_resampled_image_and_masks = {}
        if not self.just_save_as_nifti:
            self.image_resampled = False
        if self.input_data_type == 'NIfTI':
            self._process_nifti_files()
        else:
            self._process_dicom_files()
        self.patient_logger.info(f'Completed patient: {patient_number}')

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
                    self._resampling()
                    self._save_as_nifti()
                    del self.pat_original_image_and_masks[instance_key]
        else:
            self._resampling()
            self._save_as_nifti()

    def _process_dicom_files(self):
        self.pat_original_image_and_masks['IMAGE'] = extract_dicom(dicom_dir=self.patient_folder, rtstract=False,
                                                                   modality=self.input_imaging_modality)

        if self.structure_set != ['']:
            for dicom_file in os.listdir(self.patient_folder):
                dcm_data = pydicom.dcmread(os.path.join(self.patient_folder, dicom_file))
                if dcm_data.Modality == 'RTSTRUCT':
                    masks = extract_dicom(dicom_dir=self.patient_folder, rtstract=True,
                                          rtstruct_file=os.path.join(self.patient_folder, dicom_file),
                                          selected_structures=self.structure_set,
                                          modality=self.input_imaging_modality)
                    for instance_key in masks.keys():
                        self.pat_original_image_and_masks['MASK_'+instance_key] = masks[instance_key]
                        if not self.just_save_as_nifti:
                            self._resampling()
                        self._save_as_nifti()
                        del self.pat_original_image_and_masks['MASK_'+instance_key]

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

        output_origin = [
            resampled_origin(input_shape, input_spacing, output_spacing, input_origin, axis) for axis in range(3)
        ]

        output_shape = np.ceil((np.array(input_shape) * (input_spacing / output_spacing))).astype(int)

        for instance_key in list(self.pat_original_image_and_masks.keys()):

            if (instance_key == 'IMAGE' and not self.image_resampled) or instance_key != 'IMAGE':

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
                    if self.input_imaging_modality == 'CT':
                        resampled_sitk_image = sitk.Round(resampled_sitk_image)
                        resampled_sitk_image = sitk.Cast(resampled_sitk_image, sitk.sitkInt16)
                    elif self.input_imaging_modality in ['MR', 'PT']:
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

    def _save_as_nifti(self):
        if self.just_save_as_nifti:
            save_pat_image_and_masks = self.pat_original_image_and_masks
        else:
            save_pat_image_and_masks = self.pat_resampled_image_and_masks
        for key, img in save_pat_image_and_masks.items():
            if (key == 'IMAGE' and not self.image_resampled) or key != 'IMAGE':
                img.save_as_nifti(instance=self, key=key)
