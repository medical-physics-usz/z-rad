import logging
import os
from glob import glob
from os import makedirs
from os.path import isdir, isfile, join

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from read import ReadImageStructure
from utils import tqdm_joblib
import SimpleITK as sitk
from utils import Image, Mask, ROI
from datetime import datetime


class ResizeNifti(object):
    """Class to resize images and listed structures to a resolution defined by user and saved the results as dicom file
    inp_resolution – resolution defined by user
    inp_struct – list of structure names to be resized
    inp_mypath_load – path with the data to be resized
    inp_mypath_save – path to save resized data
    image_type – image modality
    begin – start number
    stop – stop number
    dim - dimension for resizing
    """

    def __init__(self, inp_resolution, interpolation_type, inp_struct, labels, inp_mypath_load, inp_mypath_save, image_type,
                 begin, stop, n_jobs):
        
        self.logger = logging.getLogger("Resize_Nifti")
        self.interpolation_alg = interpolation_type
        self.resolution = float(inp_resolution)
        self.list_structure = inp_struct
        self.labels = labels
        if inp_mypath_load[-1] != os.sep:
            inp_mypath_load += os.sep
        self.mypath_load = inp_mypath_load
        if inp_mypath_save[-1] != os.sep:
            inp_mypath_save += os.sep
        self.mypath_s = inp_mypath_save

        pat_range = [str(f) for f in range(begin, stop + 1)]
        pat_dirs = glob(self.mypath_load + os.sep + "*[0-9]*")
        list_dir_candidates = [e.split(os.sep)[-1] for e in pat_dirs if e.split(os.sep)[-1].split("_")[0] in pat_range]
        self.list_dir = sorted(list_dir_candidates)
        self.n_jobs = n_jobs
        self.modality = image_type
        self.resize()
        
    def resize(self):
        
        def parfor(name):
            mypath_image = self.mypath_load + name + os.sep

            image_name = 'image.nii.gz'
            contour_name = self.list_structure[0] + '.nii.gz'
            img_path = mypath_image + image_name
            mask_path = mypath_image + contour_name
            new_spacing = [self.resolution, self.resolution, self.resolution]

            def reorient_image(arr):
                arr = arr.transpose(2, 1, 0)
                arr = np.rot90(arr, axes=(0, 1))
                arr = np.flipud(arr)
                return arr

            try:
                makedirs(self.mypath_s + name + os.sep)
            except OSError:
                if not isdir(self.mypath_s + name + os.sep):
                    raise

            interpolated_image = Image(modality=self.modality)
            print('Reading image ' + str(datetime.now().strftime('%H:%M:%S')))
            interpolated_image.read_nifti(img_path)
            print('Interpolating image ' + str(datetime.now().strftime('%H:%M:%S')))
            # interpolated_image.interpolate(new_spacing=new_spacing, method=sitk.sitkBSpline)
            interpolated_image.interpolate(new_spacing=new_spacing, method=sitk.sitkLinear)
            if self.modality == 'CT':
                interpolated_image.round_intensities()
            print('Reorienting image ' + str(datetime.now().strftime('%H:%M:%S')))
            interpolated_image = reorient_image(sitk.GetArrayFromImage(interpolated_image.image))
            interpolated_image = np.rot90(np.flipud(interpolated_image), axes=(1, 0))
            print('Saving image ' + str(datetime.now().strftime('%H:%M:%S')))
            interpolated_image = nib.Nifti1Image(interpolated_image, affine=np.eye(4))
            interpolated_image.header['pixdim'][1:4] = new_spacing
            nib.save(interpolated_image, self.mypath_s + name + os.sep + image_name)
            del interpolated_image

            interpolated_mask = Mask()
            print('Reading mask ' + str(datetime.now().strftime('%H:%M:%S')))
            interpolated_mask.read_nifti(mask_path)
            print('Interpolating mask ' + str(datetime.now().strftime('%H:%M:%S')))
            interpolated_mask.interpolate(new_spacing=new_spacing, method=sitk.sitkLinear)
            print('Rounding intensities ' + str(datetime.now().strftime('%H:%M:%S')))
            interpolated_mask.round_intensities()
            print('Reorienting mask ' + str(datetime.now().strftime('%H:%M:%S')))
            interpolated_mask = reorient_image(sitk.GetArrayFromImage(interpolated_mask.image))
            interpolated_mask = np.rot90(np.flipud(interpolated_mask), axes=(1, 0))
            print('Saving mask ' + str(datetime.now().strftime('%H:%M:%S')))
            interpolated_mask = nib.Nifti1Image(interpolated_mask, affine=np.eye(4))
            interpolated_mask.header['pixdim'][1:4] = new_spacing
            nib.save(interpolated_mask, self.mypath_s + name + os.sep + contour_name)
            del interpolated_mask

        with tqdm_joblib(tqdm(desc="Resizing texture nifti", total=len(self.list_dir))):
            Parallel(n_jobs=self.n_jobs)(delayed(parfor)(name) for name in self.list_dir)
