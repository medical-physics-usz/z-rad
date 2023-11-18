import os
import re

import SimpleITK as sitk
import multiprocess
import numpy as np

from zrad.logic.toolbox_logic import Image


class Filtering:

    def __init__(self, load_dir, folder_prefix, start_folder, stop_folder, list_of_patient_folders, input_data_type,
                 nifti_image, filter_type, my_filter, save_dir, number_of_threads):

        # --------Load/Save data part-------
        def extract_elements_between(start, stop, pat_list):
            result = []

            start_number = int(re.search(r'\d+', start).group())
            stop_number = int(re.search(r'\d+', stop).group())

            for element in pat_list:
                element_number = int(re.search(r'\d+', element).group())
                if start_number <= element_number <= stop_number:
                    result.append(element)
                elif element_number > stop_number:
                    break

            return result

        self.load_dir = load_dir
        self.folder_prefix = folder_prefix
        self.list_of_patient_folders = [self.folder_prefix + str(pat) for pat in list_of_patient_folders] if list_of_patient_folders else \
            extract_elements_between(self.folder_prefix + str(start_folder), self.folder_prefix + str(stop_folder),
                                     os.listdir(self.load_dir))
        self.number_of_threads = int(number_of_threads)
        self.save_dir = save_dir
        self.input_data_type = input_data_type
        self.filter_type = filter_type
        self.filter = my_filter
        # ------NIFTI specific-------------
        self.nifti_image = nifti_image
        # ------------Patient specific parameters-----------------------
        self.patient_folder = None
        self.patient_number = None

    def filtering(self):
        print('START')
        with multiprocess.Pool(self.number_of_threads) as pool:
            pool.map(self.load_patient, self.list_of_patient_folders)

    def load_patient(self, patient_number):
        self.patient_number = patient_number
        self.patient_folder = os.path.join(self.load_dir, self.patient_number)
        self.pat_image = None
        self.filtered_image = None
        if self.input_data_type == 'NIFTI':
            self.process_nifti_files()
        elif self.input_data_type == 'DICOM':
            self.process_dicom_files()
        self.apply_filter()
        self.save_as_nifti()
        print('STOPPED')

        # ------------------NIFTI pypeline--------------------------

    def process_nifti_files(self):
        self.pat_image = self.extract_nifti('IMAGE')

    def extract_nifti(self, key):
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        if key == 'IMAGE':
            reader.SetFileName(os.path.join(self.patient_folder, self.nifti_image))
            image = reader.Execute()
            array = sitk.GetArrayFromImage(image).astype(np.float64)
            return Image(array=array,
                         origin=image.GetOrigin(),
                         spacing=np.array(image.GetSpacing()),
                         direction=image.GetDirection(),
                         shape=image.GetSize(),
                         dtype=array.dtype)

        # -----------------DICOM pypeline-----------------------------

    def process_dicom_files(self):
        self.pat_image = self.extract_dicom()

    def extract_dicom(self):
        reader = sitk.ImageSeriesReader()
        reader.SetImageIO("GDCMImageIO")
        dicom_series = reader.GetGDCMSeriesFileNames(self.patient_folder)
        reader.SetFileNames(dicom_series)
        image = reader.Execute()
        array = sitk.GetArrayFromImage(image).astype(np.float64)
        return Image(array=array,
                     origin=image.GetOrigin(),
                     spacing=np.array(image.GetSpacing()),
                     direction=image.GetDirection(),
                     shape=image.GetSize(),
                     dtype=array.dtype)

    def apply_filter(self):
        if self.filter_type == 'Laplacian of Gaussian':
            self.filter.res_mm = float(self.pat_image.spacing[0])
        self.filtered_image = self.filter.implement(self.pat_image.array.transpose(1, 2, 0))
        self.filtered_image = self.filtered_image.transpose(2, 0, 1)

    def save_as_nifti(self):
        output_path = os.path.join(self.save_dir, self.patient_number,
                                   'Filtered_with_'+self.filter_type+'_Image' + '.nii.gz')
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        img = sitk.GetImageFromArray(self.filtered_image)
        img.SetOrigin(self.pat_image.origin)
        img.SetSpacing(self.pat_image.spacing)
        img.SetDirection(self.pat_image .direction)
        sitk.WriteImage(img, output_path)
