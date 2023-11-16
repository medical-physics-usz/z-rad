import numpy as np
import os
import SimpleITK as sitk
import re
import multiprocess


class Image:
    def __init__(self, array, origin, spacing, direction, shape, key):
        self.array = array
        self.origin = origin
        self.spacing = spacing
        self.direction = direction
        self.shape = shape
        self.key = key

class Radiomics:

    def __init__(self, load_dir, folder_prefix, start_folder, stop_folder, list_of_patient_folders,input_data_type,
                 dicom_structures, nifti_image, nifti_structures, save_dir, number_of_threads):
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

        # ------NIFTI specific-------------

        self.nifti_image = nifti_image


        # ------------Patient specific parameters-----------------------

        self.patient_folder = None
        self.patient_number = None
        self.patient_im_characteristics = None
        self.patient_save_rtdicom_path = None

        # --------------RUN----------------------------------------

        print('enter file specific functions')
        self.load_data()

    def load_data(self):
        print('start_folder POOL')
        print(self.list_of_patient_folders)
        with multiprocess.Pool(self.number_of_threads) as pool:
            pool.map(self.load_patient, self.list_of_patient_folders)

    def load_patient(self, patient_number):
        print('perform specific patient calculation')
        self.patient_number = patient_number
        self.patient_folder = os.path.join(self.load_dir, self.patient_number)
        self.pat_image = None
        if self.input_data_type == 'NIFTI':
            self.process_nifti_files()
        elif self.input_data_type == 'DICOM':
            print('Start DICOM')
            self.process_dicom_files()
        print(self.pat_image.array.shape)
        #self.save_as_dicom()
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
            return Image(array=sitk.GetArrayFromImage(image).astype(np.float64),
                                       origin=image.GetOrigin(),
                                       spacing=np.array(image.GetSpacing()),
                                       direction=image.GetDirection(),
                                       shape=image.GetSize(),
                                       key=key)

        # -----------------DICOM pypeline-----------------------------

    def process_dicom_files(self):
        print('DICOM')
        self.pat_image = self.extract_dicom()


    def extract_dicom(self):
        print('Extract dicom')
        reader = sitk.ImageSeriesReader()
        reader.SetImageIO("GDCMImageIO")
        dicom_series = reader.GetGDCMSeriesFileNames(self.patient_folder)
        reader.SetFileNames(dicom_series)
        image = reader.Execute()

        return Image(array=sitk.GetArrayFromImage(image).astype(np.float64),
                                   origin=image.GetOrigin(),
                                   spacing=np.array(image.GetSpacing()),
                                   direction=image.GetDirection(),
                                   shape=image.GetSize(),
                                   key='IMAGE')

