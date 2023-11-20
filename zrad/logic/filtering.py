import os
import re

import SimpleITK as sitk

from zrad.logic.toolbox_logic import start_multiprocessing, nifti_save_with_sitk, extract_dicom, process_nifti_image


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
        start_multiprocessing(self)

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

        # ------------------NIFTI pypeline--------------------------

    def process_nifti_files(self):
        self.pat_image = self.extract_nifti('IMAGE')

    def extract_nifti(self, key):
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        if key == 'IMAGE':
            return process_nifti_image(self, reader)

        # -----------------DICOM pypeline-----------------------------

    def process_dicom_files(self):
        self.pat_image = extract_dicom(self)

    def apply_filter(self):
        if self.filter_type == 'Laplacian of Gaussian':
            self.filter.res_mm = float(self.pat_image.spacing[0])
        self.filtered_image = self.filter.implement(self.pat_image.array.transpose(1, 2, 0))
        self.filtered_image = self.filtered_image.transpose(2, 0, 1)

    def save_as_nifti(self):
        nifti_save_with_sitk(self, image=self.pat_image, image_array=self.filtered_image,
                             key='Filtered_with_'+self.filter_type+'_Image')
