import copy
import os

import numpy as np
import SimpleITK as sitk

from .io import dicom, nifti


class Image:
    """Image volume with voxel data and physical geometry metadata.

    ``Image`` stores arrays in NumPy order while preserving the origin, spacing,
    direction, and size used by SimpleITK. The class is used throughout
    preprocessing, filtering, and radiomics to keep image data aligned with ROI
    masks.

    Parameters
    ----------
    array : numpy.ndarray or None, optional
        Voxel array. Image data are typically stored in ``(z, y, x)`` order.
    origin : sequence of float or None, optional
        Physical origin of the image in SimpleITK ``(x, y, z)`` order.
    spacing : sequence of float or None, optional
        Physical voxel spacing in SimpleITK ``(x, y, z)`` order.
    direction : sequence of float or None, optional
        Flattened 3D direction cosine matrix.
    shape : sequence of int or None, optional
        Image size in SimpleITK ``(x, y, z)`` order.
    """

    def __init__(self, array=None, origin=None, spacing=None, direction=None, shape=None):
        self.sitk_image = None
        self.array = array
        self.origin = origin
        self.spacing = spacing
        self.direction = direction
        self.shape = shape

    @classmethod
    def from_nifti(cls, image_path):
        """Create an image from a NIfTI file.

        Parameters
        ----------
        image_path : str or path-like
            Path to the NIfTI image file.

        Returns
        -------
        image : Image
            Image populated with voxel data and geometry read from the file.
        """
        return cls._from_sitk_image(nifti.read_nifti_image(image_path))

    @classmethod
    def from_nifti_mask(cls, mask_path, reference):
        """Create a NIfTI mask aligned to a reference image.

        Parameters
        ----------
        mask_path : str or path-like
            Path to the NIfTI mask file.
        reference : Image
            Reference image that defines the target grid and geometry.

        Returns
        -------
        mask : Image
            Binary mask image resampled onto the reference geometry.
        """
        mask = nifti.read_nifti_mask(mask_path, reference.sitk_image)
        image = cls._from_sitk_image(mask)
        image.origin = reference.origin
        image.spacing = reference.spacing
        image.direction = reference.direction
        image.shape = reference.shape
        return image

    @classmethod
    def from_dicom(cls, dicom_dir, modality):
        """Create an image from a DICOM series.

        Parameters
        ----------
        dicom_dir : str or path-like
            Directory containing the DICOM series.
        modality : str
            Imaging modality used by the DICOM reader.

        Returns
        -------
        image : Image
            Image populated with voxel data and geometry read from the series.
        """
        return cls._from_sitk_image(dicom.read_dicom_image(dicom_dir, modality))

    @classmethod
    def from_dicom_mask(cls, rtstruct_path, structure_name, reference):
        """Create a DICOM RTSTRUCT mask aligned to a reference image.

        Parameters
        ----------
        rtstruct_path : str or path-like
            Path to the DICOM RTSTRUCT file.
        structure_name : str
            Name of the ROI structure to rasterize.
        reference : Image
            Reference image that defines the target grid and geometry.

        Returns
        -------
        mask : Image
            Binary mask image aligned to the reference geometry.
        """
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
        """Return a deep copy of the image data and geometry.

        Returns
        -------
        image : Image
            New image with copied array, origin, spacing, direction, and shape.
        """
        return Image(
            array=copy.deepcopy(self.array),
            origin=copy.deepcopy(self.origin),
            spacing=copy.deepcopy(self.spacing),
            direction=copy.deepcopy(self.direction),
            shape=copy.deepcopy(self.shape),
        )

    def save_as_nifti(self, output_path):
        """Write the image to a NIfTI file.

        Parameters
        ----------
        output_path : str or path-like
            Destination file path for the written NIfTI image.
        """
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        img = sitk.GetImageFromArray(self.array)
        img.SetOrigin(self.origin)
        img.SetSpacing(self.spacing)
        img.SetDirection(self.direction)
        sitk.WriteImage(img, output_path)
