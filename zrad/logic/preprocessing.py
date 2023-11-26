import os

import multiprocess
import SimpleITK as sitk
import numpy as np
import pydicom
from rt_utils import RTStructBuilder

from zrad.logic.toolbox_logic import (Image, extract_dicom, extract_nifti_image,
                                      extract_nifti_mask, list_folders_in_range)


class Preprocessing:

    def __init__(self, load_dir, folder_prefix, start_folder, stop_folder, list_of_patient_folders,
                 input_data_type, dicom_structures, nifti_image, nifti_structures, mask_interpolation_method,
                 mask_interpolation_threshold, image_interpolation_method, resample_resolution,
                 resample_dimension, save_dir, output_data_type, output_imaging_type, number_of_threads):

        self.load_dir = load_dir
        self.folder_prefix = folder_prefix
        self.list_of_patient_folders = (
            [self.folder_prefix + str(patient) for patient in list_of_patient_folders]
            if list_of_patient_folders else
            list_folders_in_range(
                self.folder_prefix + str(start_folder),
                self.folder_prefix + str(stop_folder),
                self.load_dir
            )
        )

        self.number_of_threads = number_of_threads
        self.save_dir = save_dir
        self.input_data_type = input_data_type
        self.output_data_type = output_data_type

        # ------NIFTI specific-------------
        self.nifti_image = nifti_image
        self.nifti_structures = nifti_structures

        # ------DICOM specific-------------
        self.dicom_structures = dicom_structures
        self.output_imaging_type = output_imaging_type

        # ------Resampling specific-------------
        self.resample_resolution = resample_resolution
        self.image_interpolation_method = image_interpolation_method
        self.resample_dimension = resample_dimension
        self.mask_interpolation_method = mask_interpolation_method
        if self.mask_interpolation_method in ["Linear", "BSpline", "Gaussian"]:
            self.mask_threshold = float(mask_interpolation_threshold)
        elif self.mask_interpolation_method == "NN":
            self.mask_threshold = 1.0

        # ------------Patient specific parameters-----------------

        self.patient_folder = None
        self.patient_number = None

    def resample(self):
        with multiprocess.Pool(self.number_of_threads) as pool:
            pool.map(self._load_patient, self.list_of_patient_folders)

    def _load_patient(self, patient_number):
        self.patient_number = patient_number
        self.patient_folder = os.path.join(self.load_dir, self.patient_number)
        self.pat_original_image_and_masks = {}
        self.pat_resampled_image_and_masks = {}

        if self.input_data_type == 'NIFTI':
            self._process_nifti_files()
        elif self.input_data_type == 'DICOM':
            self._process_dicom_files()

        self._resampling()

        if self.output_data_type == 'NIFTI':
            self._save_as_nifti()
        elif self.output_data_type == 'DICOM':
            self._save_as_dicom()

    # ------------------NIFTI pypeline--------------------------
    def _process_nifti_files(self):

        def extract_nifti(key, mask_file_name=None):

            reader = sitk.ImageFileReader()
            reader.SetImageIO("NiftiImageIO")
            if key == 'IMAGE':
                return extract_nifti_image(self, reader)

            elif key.startswith('MASK'):
                return extract_nifti_mask(self, reader, mask_file_name)

        self.pat_original_image_and_masks['IMAGE'] = extract_nifti('IMAGE')
        if self.nifti_structures != ['']:
            for nifti_file_name in self.nifti_structures:
                instance_key = 'MASK_' + nifti_file_name
                self.pat_original_image_and_masks[instance_key] = extract_nifti(instance_key, nifti_file_name)

    # -----------------DICOM pypeline-----------------------------

    def _process_dicom_files(self):
        self.pat_original_image_and_masks['IMAGE'] = extract_dicom(self)
        if self.dicom_structures != ['']:
            for dicom_file in os.listdir(self.patient_folder):
                dcm_data = pydicom.dcmread(os.path.join(self.patient_folder, dicom_file))
                if dcm_data.Modality == 'RTSTRUCT':
                    rtstruct = RTStructBuilder.create_from(dicom_series_path=self.patient_folder,
                                                           rt_struct_path=os.path.join(self.patient_folder,
                                                                                       dicom_file
                                                                                       )
                                                           )
                    for ROI in self.dicom_structures:
                        instance_key = 'MASK_' + ROI
                        mask_roi = rtstruct.get_roi_mask_by_name(ROI)
                        mask_roi = mask_roi * 1
                        mask_roi = mask_roi.transpose(2, 0, 1)
                        self.pat_original_image_and_masks[instance_key] = Image(
                            array=mask_roi.tobytes(),
                            origin=self.pat_original_image_and_masks['IMAGE'].origin,
                            spacing=self.pat_original_image_and_masks['IMAGE'].spacing,
                            direction=self.pat_original_image_and_masks['IMAGE'].direction,
                            shape=self.pat_original_image_and_masks['IMAGE'].shape,
                            dtype=mask_roi.dtype
                        )

    # -----------------Preprocessing pypeline------------------------

    def _resampling(self):

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

            sitk_image = sitk.GetImageFromArray(
                np.frombuffer(
                    self.pat_original_image_and_masks[instance_key].array,
                    dtype=self.pat_original_image_and_masks[instance_key].dtype).reshape(input_shape[::-1])
            )
            sitk_image.SetSpacing(input_spacing)
            sitk_image.SetOrigin(input_origin)
            sitk_image.SetDirection(input_direction)

            resample = sitk.ResampleImageFilter()
            resample.SetOutputSpacing(output_spacing)
            resample.SetOutputOrigin(output_origin)
            resample.SetOutputDirection(input_direction)
            resample.SetSize(output_shape.tolist())
            resample.SetOutputPixelType(sitk.sitkFloat64)
            resample.SetInterpolator(self._get_interpolator(instance_key))
            resampled_sitk_image = resample.Execute(sitk_image)

            resampled_array = None

            if instance_key.startswith('IMAGE'):
                resampled_sitk_image = sitk.Round(resampled_sitk_image)
                resampled_sitk_image = sitk.Cast(resampled_sitk_image, sitk.sitkInt16)
                resampled_array = sitk.GetArrayFromImage(resampled_sitk_image)
            elif instance_key.startswith('MASK'):
                resampled_array = np.where(sitk.GetArrayFromImage(resampled_sitk_image) >= self.mask_threshold, 1, 0)

            self.pat_resampled_image_and_masks[instance_key] = Image(array=resampled_array,
                                                                     origin=output_origin,
                                                                     spacing=output_spacing,
                                                                     direction=input_direction,
                                                                     shape=output_shape,
                                                                     dtype=None)
            del self.pat_original_image_and_masks[instance_key]

    def _get_interpolator(self, instance_key):
        interpolators = {
            'Linear': sitk.sitkLinear,
            'NN': sitk.sitkNearestNeighbor,
            'BSpline': sitk.sitkBSpline,
            'Gaussian': sitk.sitkGaussian
        }
        if instance_key.startswith('IMAGE'):
            return interpolators.get(self.image_interpolation_method)
        elif instance_key.startswith('MASK'):
            return interpolators.get(self.mask_interpolation_method)

    # -----------------------Saving pypeline-----------------------------
    def _save_as_nifti(self):
        for key, img in self.pat_resampled_image_and_masks.items():
            img.save_as_nifti(instance=self, key=key)

    def _save_as_dicom(self):
        self.pat_resampled_image_and_masks['IMAGE'].save_image_as_dicom(instance=self)

        rtstruct_save = RTStructBuilder.create_new(dicom_series_path=self.save_dir + '/' + self.patient_number)

        for instance_key, img in self.pat_resampled_image_and_masks.items():
            if instance_key.startswith('MASK'):
                img.save_mask_as_dicom(key=instance_key, rtstruct_file=rtstruct_save)

        rtstruct_save.save(os.path.join(self.save_dir + '/' + self.patient_number, 'rt-struct'))
