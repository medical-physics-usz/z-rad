import numpy as np
import SimpleITK as sitk
import time
import os
import multiprocess
import pydicom
from rt_utils import RTStructBuilder


class Image:
    def __init__(self, array, origin, spacing, direction, shape, dtype):
        self.array = array
        self.origin = origin
        self.spacing = spacing
        self.direction = direction
        self.shape = shape
        self.dtype = dtype


class Preprocessing:

    def __init__(self, load_dir, folder_prefix, start_folder, stop_folder, list_of_patient_folders,
                 input_data_type, dicom_structures, nifti_image, nifti_structures, mask_interpolation_method,
                 mask_interpolation_threshold, image_interpolation_method, resample_resolution,
                 resample_dimension, save_dir, output_data_type, output_imaging_type, number_of_threads):

        def list_folders_in_range(folder_to_start, folder_to_stop, directory_path):

            items = [item for item in sorted(os.listdir(directory_path)) if
                     os.path.isdir(os.path.join(directory_path, item))]

            try:
                start_index = items.index(folder_to_start)
                stop_index = items.index(folder_to_stop)
            except ValueError as e:
                raise ValueError(f"Start or stop folder not found in the directory: {e}")

            return items[start_index:stop_index + 1]

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

        self.number_of_threads = int(number_of_threads)
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
        print('STARTED')
        with multiprocess.Pool(self.number_of_threads) as pool:
            pool.map(self.load_patient, self.list_of_patient_folders)

    def load_patient(self, patient_number):
        self.patient_number = patient_number
        self.patient_folder = os.path.join(self.load_dir, self.patient_number)
        self.pat_original_image_and_masks = {}
        self.pat_resampled_image_and_masks = {}

        if self.input_data_type == 'NIFTI':
            self.process_nifti_files()
        elif self.input_data_type == 'DICOM':
            self.process_dicom_files()

        self.resampling()

        if self.output_data_type == 'NIFTI':
            self.save_as_nifti()
        elif self.output_data_type == 'DICOM':
            self.save_as_dicom()

        print('STOPPED')

    # ------------------NIFTI pypeline--------------------------
    def process_nifti_files(self):

        self.pat_original_image_and_masks['IMAGE'] = self.extract_nifti('IMAGE')
        if self.nifti_structures != ['']:
            for nifti_file in self.nifti_structures:
                instance_key = 'MASK_' + nifti_file
                self.pat_original_image_and_masks[instance_key] = self.extract_nifti(instance_key, nifti_file)

    def extract_nifti(self, instance_key, mask_file=None):

        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        if instance_key == 'IMAGE':
            reader.SetFileName(os.path.join(self.patient_folder, self.nifti_image))
            image = reader.Execute()
            array = sitk.GetArrayFromImage(image)

            return Image(array=array.tobytes(),
                         origin=image.GetOrigin(),
                         spacing=np.array(image.GetSpacing()),
                         direction=image.GetDirection(),
                         shape=image.GetSize(),
                         dtype=array.dtype
                         )

        elif instance_key.startswith('MASK'):
            if os.path.isfile(os.path.join(self.patient_folder, mask_file + '.nii.gz')):
                reader.SetFileName(os.path.join(self.patient_folder, mask_file + '.nii.gz'))
            elif os.path.isfile(os.path.join(self.patient_folder, mask_file + '.nii')):
                reader.SetFileName(os.path.join(self.patient_folder, mask_file + '.nii'))
            image = reader.Execute()
            array = sitk.GetArrayFromImage(image)

            return Image(array=array.tobytes(),
                         origin=self.pat_original_image_and_masks['IMAGE'].origin,
                         spacing=self.pat_original_image_and_masks['IMAGE'].spacing,
                         direction=self.pat_original_image_and_masks['IMAGE'].direction,
                         shape=self.pat_original_image_and_masks['IMAGE'].shape,
                         dtype=array.dtype
                         )

    # -----------------DICOM pypeline-----------------------------

    def process_dicom_files(self):
        self.pat_original_image_and_masks['IMAGE'] = self.extract_dicom()
        if self.dicom_structures != ['']:
            for dicom_file in os.listdir(self.patient_folder):
                dcm_data = pydicom.dcmread(os.path.join(self.patient_folder, dicom_file))

                if dcm_data.Modality == 'RTSTRUCT':
                    rtstruct = RTStructBuilder.create_from(
                        dicom_series_path=self.patient_folder,
                        rt_struct_path=os.path.join(self.patient_folder, dicom_file))

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

    def extract_dicom(self):
        reader = sitk.ImageSeriesReader()
        reader.SetImageIO("GDCMImageIO")
        dicom_series = reader.GetGDCMSeriesFileNames(self.patient_folder)
        reader.SetFileNames(dicom_series)
        image = reader.Execute()
        array = sitk.GetArrayFromImage(image)
        return Image(array=array.tobytes(),
                     origin=image.GetOrigin(),
                     spacing=np.array(image.GetSpacing()),
                     direction=image.GetDirection(),
                     shape=image.GetSize(),
                     dtype=array.dtype
                     )

    # -----------------Preprocessing pypeline------------------------

    def resampling(self):

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
            resample.SetInterpolator(self.get_interpolator(instance_key))
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

    def get_interpolator(self, instance_key):
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
    def save_as_nifti(self):

        for instance_key, Img in self.pat_resampled_image_and_masks.items():
            output_path = os.path.join(self.save_dir, self.patient_number, instance_key + '.nii.gz')
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            img = sitk.GetImageFromArray(Img.array)
            img.SetOrigin(Img.origin)
            img.SetSpacing(Img.spacing)
            img.SetDirection(Img.direction)
            sitk.WriteImage(img, output_path)

    def save_as_dicom(self):

        resampled_image = self.pat_resampled_image_and_masks['IMAGE']

        res_image = sitk.GetImageFromArray(resampled_image.array)
        res_image.SetOrigin(resampled_image.origin)
        res_image.SetDirection(resampled_image.direction)
        res_image = sitk.Cast(res_image, sitk.sitkInt16)

        def write_slices(series_tags, new_img, i, spacing_list):
            image_slice = new_img[:, :, i]
            list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tags))

            image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
            image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
            image_slice.SetMetaData("0008|0060", self.output_imaging_type)

            # Setting the Image Position (Patient) tag using provided spacing and z_origin
            x, y, _ = new_img.TransformIndexToPhysicalPoint((0, 0, 0))
            # Compute z using provided spacing and z_origin
            z = self.pat_resampled_image_and_masks['IMAGE'].origin[2] + i * spacing_list[2]

            image_slice.SetMetaData("0020|0032", f"{x}\\{y}\\{z}")
            image_slice.SetMetaData("0010|0010", 'P'+str(self.patient_number))
            image_slice.SetMetaData("0010|0020", str(self.patient_number))
            image_slice.SetMetaData("0020,0013", str(i))
            writer.SetFileName(os.path.join(self.save_dir + '/' + self.patient_number, f'slice_{i}.dcm'))
            writer.Execute(image_slice)

        # Ensure save directory exists
        if not os.path.exists(self.save_dir + '/' + self.patient_number):
            os.makedirs(self.save_dir + '/' + self.patient_number)

        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()

        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")

        direction = res_image.GetDirection()
        series_tag_values = [("0008|0031", modification_time),
                             ("0008|0021", modification_date),
                             ("0008|0008", "DERIVED\\SECONDARY"),
                             (
                                 "0020|000e",
                                 "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),
                             ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],
                                                               direction[1], direction[4], direction[7])))),
                             ("0008|103e", "Created with Z-RAD")]

        # Set the new spacing for the res_image
        res_image.SetSpacing(resampled_image.spacing)

        list(map(lambda i: write_slices(series_tag_values, res_image, i, resampled_image.spacing),
                 range(res_image.GetDepth())))

        rtstruct_save = RTStructBuilder.create_new(dicom_series_path=self.save_dir + '/' + self.patient_number)

        for instance_key, Img in self.pat_resampled_image_and_masks.items():
            if instance_key.startswith('MASK'):
                roi_mask = Img.array
                roi_mask = np.array(roi_mask, dtype=bool)
                roi_mask = roi_mask.transpose(1, 2, 0)
                rtstruct_save.add_roi(mask=roi_mask, name=instance_key[5:])

        rtstruct_save.save(os.path.join(self.save_dir + '/' + self.patient_number, 'rt-struct'))
