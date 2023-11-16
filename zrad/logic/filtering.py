import numpy as np
import os
import SimpleITK as sitk
import re
import multiprocess
from logic.filters import Mean, LoG, Wavelets2D, Wavelets3D, Laws


class Image:
    def __init__(self, array, origin, spacing, direction, shape, key):
        self.array = array
        self.origin = origin
        self.spacing = spacing
        self.direction = direction
        self.shape = shape
        self.key = key

class Filtering:

    def __init__(self, load_dir, folder_prefix, start_folder, stop_folder, list_of_patient_folders, input_data_type,
                 nifti_image, filter_type, filter_dimension, filter_padding_type, mean_filter_support,
                 log_filter_sigma, log_filter_cutoff, #laws_filter_response_map, laws_filter_rot_inv,
                 wavelet_filter_response_map, wavelet_filter_type,
                 wavelet_filter_decomposition_lvl, wavelet_filter_pseudo_rot_inv, save_dir, number_of_threads):
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
        self.filter_type=filter_type

        # ------NIFTI specific-------------

        self.nifti_image = nifti_image


        # ------------Patient specific parameters-----------------------

        self.patient_folder = None
        self.patient_number = None
        self.patient_im_characteristics = None
        self.patient_save_rtdicom_path = None

        #------------- Filtering: Mean-----------------
        self.filtered_image = None
        self.filter_padding_type = filter_padding_type
        self.filter_dimension = filter_dimension
        if self.filter_type == 'Mean':
            self.mean_filter_support = int(mean_filter_support)
        elif self.filter_type == 'Laplacian of Gaussian':
            self.LoG_sigma = float(log_filter_sigma)
            self.LoG_cutoff = float(log_filter_cutoff)
        elif self.filter_type == 'Wavelets':
            self.wavelet_responce_map = wavelet_filter_response_map
            self.wavelet_type = wavelet_filter_type
            self.wavelet_decomposition_lvl = wavelet_filter_decomposition_lvl
            self.wavelet_pseudo_rot_inv = wavelet_filter_pseudo_rot_inv
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
        self.filtering()
        self.save_as_nifti()
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
    def filtering(self):
        print('Filtering')
        if self.filter_type == 'Mean':
            print('Filtering_mean')
            mean_filter = Mean(padding_type=self.filter_padding_type, support=self.mean_filter_support, dimensionality=self.filter_dimension)
            print('Filter_defined')
            self.filtered_image = mean_filter.filter(self.pat_image.array.transpose(1, 2, 0))
            self.filtered_image = self.filtered_image.transpose(2, 0, 1)
            print(self.filtered_image)
            print('Filter_performed')
            print(self.filtered_image.shape)
        elif self.filter_type == 'Laplacian of Gaussian':
            LoG_filter = LoG(padding_type=self.filter_padding_type, sigma_mm=self.LoG_sigma, cutoff=self.LoG_cutoff, res_mm=self.pat_image.spacing[0], dimensionality=self.filter_dimension)
            self.filtered_image = LoG_filter.filter(self.pat_image.array.transpose(1, 2, 0))
            self.filtered_image = self.filtered_image.transpose(2, 0, 1)
            print(self.filtered_image)
            print('Filter_performed')
            print(self.filtered_image.shape)

        elif self.filter_type == 'Wavelets' and self.filter_dimension == '2D':
            Wavelet_2D_filter = Wavelets2D(wavelet_type=self.wavelet_type, padding_type=self.filter_padding_type,
                                           response_map=self.wavelet_responce_map, decomposition_level=self.wavelet_decomposition_lvl,
                                           rotation_invariance=self.wavelet_pseudo_rot_inv)
            self.filtered_image = Wavelet_2D_filter.filter(self.pat_image.array.transpose(1, 2, 0))
            print(self.pat_image.array.transpose(1, 2, 0).shape)
            self.filtered_image = self.filtered_image.transpose(2, 0, 1)
            print(self.filtered_image)
            print('Filter_performed')
            print(self.filtered_image.shape)

        elif self.filter_type == 'Wavelets' and self.filter_dimension == '3D':
            Wavelet_filter = Wavelets3D(wavelet_type=self.wavelet_type, padding_type=self.filter_padding_type,
                                        response_map=self.wavelet_responce_map, decomposition_level=self.wavelet_decomposition_lvl,
                                        rotation_invariance=self.wavelet_pseudo_rot_inv)
            self.filtered_image = Wavelet_filter.filter(self.pat_image.array.transpose(1, 2, 0))
            print(self.pat_image.array.transpose(1, 2, 0).shape)
            self.filtered_image = self.filtered_image.transpose(2, 0, 1)
            print(self.filtered_image)
            print('Filter_performed')
            print(self.filtered_image.shape)
        elif self.filter_type == 'Laws':
            Laws_filter = Laws(response_map="L5E5E5", padding_type=self.filter_padding_type, dimensionality=self.filter_dimension,
                               rotation_invariance=True, pooling="max", energy_map=True, distance=7)
            self.filtered_image = Laws_filter.filter(self.pat_image.array.transpose(1, 2, 0))
            print(self.pat_image.array.transpose(1, 2, 0).shape)
            self.filtered_image = self.filtered_image.transpose(2, 0, 1)
            print(self.filtered_image)
            print('Filter_performed')
            print(self.filtered_image.shape)


    def save_as_nifti(self):
        print('Start to save NIFTI?')

        output_path = os.path.join(self.save_dir, self.patient_number, 'Filtered_Image' + '.nii.gz')
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        img = sitk.GetImageFromArray(self.filtered_image)
        img.SetOrigin(self.pat_image.origin)
        img.SetSpacing(self.pat_image.spacing)
        img.SetDirection(self.pat_image .direction)
        print('Image was saved as: ', output_path)
        sitk.WriteImage(img, output_path)
