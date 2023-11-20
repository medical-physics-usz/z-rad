import os

import SimpleITK as sitk
import multiprocess
import numpy as np


class Image:
    def __init__(self, array, origin, spacing, direction, shape, dtype):
        self.array = array
        self.origin = origin
        self.spacing = spacing
        self.direction = direction
        self.shape = shape
        self.dtype = dtype


def save_as_nifti(instance):
    for instance_key, Img in instance.pat_resampled_image_and_masks.items():
        output_path = os.path.join(instance.save_dir, instance.patient_number, instance_key + '.nii.gz')
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        img = sitk.GetImageFromArray(Img.array)
        img.SetOrigin(Img.origin)
        img.SetSpacing(Img.spacing)
        img.SetDirection(Img.direction)
        sitk.WriteImage(img, output_path)


def extract_dicom(instance):
    reader = sitk.ImageSeriesReader()
    reader.SetImageIO("GDCMImageIO")
    dicom_series = reader.GetGDCMSeriesFileNames(instance.patient_folder)
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


def nifti_save_with_sitk(instance, image, image_array, key):
    output_path = os.path.join(instance.save_dir, instance.patient_number, key + '.nii.gz')
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    img = sitk.GetImageFromArray(image_array)
    img.SetOrigin(image.origin)
    img.SetSpacing(image.spacing)
    img.SetDirection(image.direction)
    sitk.WriteImage(img, output_path)


def start_multiprocessing(instance):
    print('STARTED')
    with multiprocess.Pool(instance.number_of_threads) as pool:
        pool.map(instance.load_patient, instance.list_of_patient_folders)
    print('STOPPED')


def process_nifti_image(instance, sitk_reader):

    sitk_reader.SetFileName(os.path.join(instance.patient_folder, instance.nifti_image))
    image = sitk_reader.Execute()
    array = sitk.GetArrayFromImage(image)

    return Image(array=array.tobytes(),
                 origin=image.GetOrigin(),
                 spacing=np.array(image.GetSpacing()),
                 direction=image.GetDirection(),
                 shape=image.GetSize(),
                 dtype=array.dtype
                 )


def process_nifti_mask(instance, sitk_reader, mask):

    if os.path.isfile(os.path.join(instance.patient_folder, mask + '.nii.gz')):
        sitk_reader.SetFileName(os.path.join(instance.patient_folder, mask + '.nii.gz'))
    elif os.path.isfile(os.path.join(instance.patient_folder, mask + '.nii')):
        sitk_reader.SetFileName(os.path.join(instance.patient_folder, mask + '.nii'))
    image = sitk_reader.Execute()
    array = sitk.GetArrayFromImage(image)

    return Image(array=array.tobytes(),
                 origin=instance.pat_original_image_and_masks['IMAGE'].origin,
                 spacing=instance.pat_original_image_and_masks['IMAGE'].spacing,
                 direction=instance.pat_original_image_and_masks['IMAGE'].direction,
                 shape=instance.pat_original_image_and_masks['IMAGE'].shape,
                 dtype=array.dtype
                 )
