import logging
import os
from glob import glob
from os import makedirs
from os.path import isdir, isfile, join

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interpn
from tqdm import tqdm

from read import ReadImageStructure
from utils import tqdm_joblib


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
        if self.resolution < 1.:  # set a round factor for slice position
            self.round_factor = 3
        else:
            self.round_factor = 3
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
        
        self.resize()
        
    def resize(self):
        
        def parfor(name):
            mypath_image = self.mypath_load + name + os.sep
            read = ReadImageStructure('nifti', 'UID', mypath_image, self.list_structure, False, '3D', False)
            
            if read.stop_calc != '':
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Issue segmentation: patient: ' + name)
                print(read.stop_calc)
                
            else:
                image_name = ''
                contour_name = ''
                for f in read.onlyfiles:
                    if isfile(join(mypath_image, f)) and self.list_structure[0] in f: #nifti only handles one ROI
                        contour_name = f
                    else: 
                        image_name = f
                img = nib.load(mypath_image + image_name)
                contour = nib.load(mypath_image + contour_name)      
                slope = img.header['scl_slope']
                intercept = img.header['scl_inter']
#                    print(slope, intercept)
                if np.isnan(slope):
                    slope = 1.
                if np.isnan(intercept):
                    intercept = 0
                IM_matrix = img.get_fdata().transpose(2, 1, 0) * slope + intercept
                contour_matrix = contour.get_fdata().transpose(2, 1, 0)
                for lab in self.labels:
                    ind = np.where(contour_matrix == lab)
                    contour_matrix[ind] = 100
                ind = np.where(contour_matrix != 100)
                contour_matrix[ind] = 0
                ind = np.where(contour_matrix == 100)
                contour_matrix[ind] = 1
                xCTspace = read.xCTspace
                yCTspace = read.yCTspace
                zCTspace = read.zCTspace               
                
                x = np.arange(0, len(contour_matrix[0][0])*xCTspace, xCTspace)
                y = np.arange(0, len(contour_matrix[0])*yCTspace, yCTspace)
                z = np.arange(0, len(contour_matrix)*zCTspace, zCTspace)
                #
                xn = len(np.arange(0, x[-1], self.resolution, dtype = float))
                yn = len(np.arange(0, y[-1], self.resolution, dtype = float))
                zn = len(np.arange(0, z[-1], self.resolution, dtype = float))
                
                new_points = []
                for xi in np.arange(0, x[-1], self.resolution):
                    for yi in np.arange(0, y[-1], self.resolution):
                        for zi in np.arange(0, z[-1], self.resolution):
                            new_points.append([zi, yi, xi])
                new_points = np.array(new_points)
                
                new_values_image = interpn((z,y,x), IM_matrix, new_points, method = 'linear')
                new_values_contour = interpn((z,y,x), contour_matrix, new_points, method = self.interpolation_alg)
                
                new_contour_matrix = np.zeros((zn, yn, xn))
                new_image_matrix = np.zeros((zn, yn, xn))
                
                for pni, pn in enumerate(new_points):
                    new_contour_matrix[int(round(pn[0] / self.resolution, 0)), int(round(pn[1] / self.resolution, 0)), int(round(pn[2] / self.resolution, 0))] = new_values_contour[pni]
                    new_image_matrix[int(round(pn[0] / self.resolution, 0)), int(round(pn[1] / self.resolution, 0)), int(round(pn[2] / self.resolution, 0))] = new_values_image[pni]
                    
                ind = np.where(new_contour_matrix >= 0.5)
                new_contour_matrix[ind] = 1.
                ind = np.where(new_contour_matrix < 0.5)
                new_contour_matrix[ind] = 0
                
                affine_trans = np.zeros((4,4))
                affine_trans[0, 0] = -1 * self.resolution
                affine_trans[1, 1] = -1 * self.resolution
                affine_trans[2, 2] = self.resolution
                affine_trans[3, 3] = 1.
                affine_trans[0, 3] = 0
                affine_trans[1, 3] = 0
                affine_trans[2, 3] = 0

                IM_matrix_nifti = nib.Nifti1Image(new_image_matrix.transpose(2,1,0), affine = affine_trans)
                contour_matrix_nifti = nib.Nifti1Image(new_contour_matrix.transpose(2,1,0), affine = affine_trans)
                
                try:
                    makedirs(self.mypath_s + name + os.sep)
                except OSError:
                    if not isdir(self.mypath_s + name + os.sep):
                        raise
                
                nib.save(IM_matrix_nifti, self.mypath_s + name + os.sep + image_name)
                nib.save(contour_matrix_nifti, self.mypath_s + name + os.sep + contour_name)
        
        with tqdm_joblib(tqdm(desc="Resizing texture nifti", total=len(self.list_dir))):
            Parallel(n_jobs=self.n_jobs)(delayed(parfor)(name) for name in self.list_dir)
