import copy
import os

import SimpleITK as sitk
import numpy as np

from .io import dicom, nifti


class Image:
    def __init__(self, array=None, origin=None, spacing=None, direction=None, shape=None):
        self.sitk_image = None
        self.array = array
        self.origin = origin
        self.spacing = spacing
        self.direction = direction
        self.shape = shape

    @classmethod
    def from_nifti(cls, image_path):
        """Create an image from a NIfTI file."""
        return cls._from_sitk_image(nifti.read_nifti_image(image_path))

    @classmethod
    def from_nifti_mask(cls, mask_path, reference):
        """Create a NIfTI mask aligned to a reference image."""
        mask = nifti.read_nifti_mask(mask_path, reference.sitk_image)
        image = cls._from_sitk_image(mask)
        image.origin = reference.origin
        image.spacing = reference.spacing
        image.direction = reference.direction
        image.shape = reference.shape
        return image

    @classmethod
    def from_dicom(cls, dicom_dir, modality):
        """Create an image from a DICOM series."""
        return cls._from_sitk_image(dicom.read_dicom_image(dicom_dir, modality))

    @classmethod
    def from_dicom_mask(cls, rtstruct_path, structure_name, reference):
        """Create a DICOM RTSTRUCT mask aligned to a reference image."""
        return dicom.read_dicom_mask(rtstruct_path, structure_name, reference.sitk_image)

    @classmethod
    def _from_sitk_image(cls, image):
        array = sitk.GetArrayFromImage(image)
        result = cls(
            array=array.astype(np.float64),
            origin=image.GetOrigin(),
            spacing=np.array(image.GetSpacing()),
            direction=image.GetDirection(),
            shape=image.GetSize(),
        )
        result.sitk_image = image
        return result

    def copy(self):
        return Image(
            array=copy.deepcopy(self.array),
            origin=copy.deepcopy(self.origin),
            spacing=copy.deepcopy(self.spacing),
            direction=copy.deepcopy(self.direction),
            shape=copy.deepcopy(self.shape)
        )

    def save_as_nifti(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        img = sitk.GetImageFromArray(self.array)
        img.SetOrigin(self.origin)
        img.SetSpacing(self.spacing)
        img.SetDirection(self.direction)
        sitk.WriteImage(img, output_path)
