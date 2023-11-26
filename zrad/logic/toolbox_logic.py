import os
import time
import SimpleITK as sitk
import numpy as np


class Image:
    def __init__(self, array, origin, spacing, direction, shape, dtype):
        self.array = array
        self.origin = origin
        self.spacing = spacing
        self.direction = direction
        self.shape = shape
        self.dtype = dtype

    def save_as_nifti(self, instance, key):
        output_path = os.path.join(instance.save_dir, instance.patient_number, key + '.nii.gz')
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        img = sitk.GetImageFromArray(self.array)
        img.SetOrigin(self.origin)
        img.SetSpacing(self.spacing)
        img.SetDirection(self.direction)
        sitk.WriteImage(img, output_path)

    def save_image_as_dicom(self, instance):
        res_image = sitk.GetImageFromArray(self.array)
        res_image.SetOrigin(self.origin)
        res_image.SetDirection(self.direction)
        res_image = sitk.Cast(res_image, sitk.sitkInt16)

        def write_slices(series_tags, new_img, i, spacing_list):
            image_slice = new_img[:, :, i]
            list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tags))

            image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
            image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
            image_slice.SetMetaData("0008|0060", instance.output_imaging_type)

            # Setting the Image Position (Patient) tag using provided spacing and z_origin
            x, y, _ = new_img.TransformIndexToPhysicalPoint((0, 0, 0))
            # Compute z using provided spacing and z_origin
            z = instance.pat_resampled_image_and_masks['IMAGE'].origin[2] + i * spacing_list[2]

            image_slice.SetMetaData("0020|0032", f"{x}\\{y}\\{z}")
            image_slice.SetMetaData("0010|0010", 'P' + str(instance.patient_number))
            image_slice.SetMetaData("0010|0020", str(instance.patient_number))
            image_slice.SetMetaData("0020,0013", str(i))
            writer.SetFileName(os.path.join(instance.save_dir + '/' + instance.patient_number, f'slice_{i}.dcm'))
            writer.Execute(image_slice)

        # Ensure save directory exists
        if not os.path.exists(instance.save_dir + '/' + instance.patient_number):
            os.makedirs(instance.save_dir + '/' + instance.patient_number)

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
        res_image.SetSpacing(self.spacing)

        list(map(lambda i: write_slices(series_tag_values, res_image, i, self.spacing),
                 range(res_image.GetDepth())))

    def save_mask_as_dicom(self, key, rtstruct_file):
        roi_mask = self.array
        roi_mask = np.array(roi_mask, dtype=bool)
        roi_mask = roi_mask.transpose(1, 2, 0)
        rtstruct_file.add_roi(mask=roi_mask, name=key[5:])


def list_folders_in_range(folder_to_start, folder_to_stop, directory_path):
    items = [item for item in sorted(os.listdir(directory_path)) if
             os.path.isdir(os.path.join(directory_path, item))]

    try:
        start_index = items.index(folder_to_start)
        stop_index = items.index(folder_to_stop)
    except ValueError as e:
        raise ValueError(f"Start or stop folder not found in the directory: {e}")

    return items[start_index:stop_index + 1]


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


# NIFTI extraction

def extract_nifti_image(instance, sitk_reader):

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


def extract_nifti_mask(instance, sitk_reader, mask):

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
