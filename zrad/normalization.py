# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:42:21 2021

@author: marta
"""
import logging

import pydicom as dc
import numpy as np
from os.path import isdir, isfile, join
from os import makedirs, sep

from read import ReadImageStructure
from structure import Structures

#normalization libraries
import nibabel as nib
from hdbet.run import run_hd_bet
import matplotlib.pyplot as plt

class Normalization(object):
    """3 methods for image intensity normalization"""
    def normalization_linear(self, struct_norm1, struct_norm2, mypath_image, rs, slices, x_ct, y_ct, xCTspace, onlyfiles, rows, columns):
        """Get the normalization coefficients for MR image based on two normal structures (for example muscle and white
        matter) and linear function.
        """
        # False for the wavelet and local argument, none for dim
        struct1 = Structures(rs, [struct_norm1], slices, x_ct, y_ct,
                             xCTspace, len(slices), False, None, False)
        norm1_Xcontour = struct1.Xcontour
        norm1_Ycontour = struct1.Ycontour

        struct2 = Structures(rs, [struct_norm2], slices, x_ct, y_ct, xCTspace,
                             len(slices), False, None,
                             False)  # False for the wavelet and local  argument, none for dim
        norm2_Xcontour = struct2.Xcontour
        norm2_Ycontour = struct2.Ycontour

        # to read only slices where there is a contour
        ind = []
        for f in range(len(norm1_Xcontour)):
            if norm1_Xcontour[f] != []:
                ind.append(f)
        for f in range(len(norm2_Xcontour)):
            if norm2_Xcontour[f] != []:
                ind.append(f)

        zmin = np.min(ind)
        zmax = np.max(ind)

        onlyfiles = onlyfiles[zmin:zmax + 1]
        slices = slices[zmin:zmax + 1]
        norm1_Xcontour = norm1_Xcontour[zmin:zmax + 1]
        norm1_Ycontour = norm1_Ycontour[zmin:zmax + 1]
        norm2_Xcontour = norm2_Xcontour[zmin:zmax + 1]
        norm2_Ycontour = norm2_Ycontour[zmin:zmax + 1]

        IM_matrix = []  # list containing the images matrix
        for i in onlyfiles:
            data = dc.read_file(mypath_image + i).PixelData
            data16 = np.array(np.fromstring(data, dtype=np.int16))  # converting to decimal
            # recalculating for rows x columns
            a = []
            for j in range(rows):
                a.append(data16[j * columns:(j + 1) * columns])
            a = np.array(a)
            IM_matrix.append(np.array(a))

        IM_matrix = np.array(IM_matrix)

        v1 = []  # values for structure 1
        v2 = []  # values for structure 2

        for i in range(len(norm1_Xcontour)):  # slices
            for j in range(len(norm1_Xcontour[i])):  # sub-structures in the slice
                for k in range(len(norm1_Xcontour[i][j])):
                    v1.append(IM_matrix[i][norm1_Ycontour[i][j][k]][norm1_Xcontour[i][j][k]])

        for i in range(len(norm2_Xcontour)):  # slices
            for j in range(len(norm2_Xcontour[i])):  # sub-structures in the slice
                for k in range(len(norm2_Xcontour[i][j])):
                    v2.append(IM_matrix[i][norm2_Ycontour[i][j][k]][norm2_Xcontour[i][j][k]])

        f1 = np.mean(v1)
        f2 = np.mean(v2)

        fa = (800 - 300) / (f1 - f2)  # find coefficients of a linear function
        fb = 800 - f1 * fa

        return fa, fb
    
    def mask_ROI(self, norm_Xcontour, norm_Ycontour, matrix_shape, outer_value):
        mask_data = np.zeros(matrix_shape)  # creating the matrix to fill it with points of the structure, y rows, x columns
        mask_data[:, :, :] = outer_value
        for i in range(len(norm_Xcontour)): # slices
            for j in range(len(norm_Xcontour[i])): #sub-structres in the slice
                for k in range(len(norm_Xcontour[i][j])):
                    mask_data[i][norm_Ycontour[i][j][k]][norm_Xcontour[i][j][k]] = 1
        return mask_data
    
    def mask_brain(self, patPos,xCTspace, x_ct, y_ct, z_ct, IM_matrix, path_skull, ImName, path_save):
        if path_skull == '':
            if patPos == 'HFS':
                affine_trans = np.zeros((4,4))
                affine_trans[0, 0] = -1 * xCTspace
                affine_trans[1, 1] = -1 * xCTspace
                affine_trans[2, 2] = xCTspace
                affine_trans[3, 3] = 1.
                affine_trans[0, 3] = -1 * x_ct
                affine_trans[1, 3] = -1 * y_ct
                affine_trans[2, 3] = z_ct
                
                mask_data = self.skull_stripping(IM_matrix, affine_trans, path_save, ImName)
            else:
                self.logger = logging.getLogger("MR read")
                self.logger.info("Scan position not HFS, the skull stripping was not tested")
                raise TypeError
        elif path_skull != '':
            try:
                mask = nib.load(path_skull + sep + str(ImName) + '_mask.nii.gz')
            except WindowsError:
                mask = nib.load(path_skull + sep + str(ImName) + '_mask.nii')
            mask_data = mask.get_fdata().transpose(2,1,0) 

            mask_data[np.where(mask_data == 0)] = np.nan
            
        return mask_data
    
    def skull_stripping(self, IM_matrix, transform, path_save, name):
        IM_matrix_nifti = nib.Nifti1Image(IM_matrix, affine = transform)
        try:
            makedirs(path_save + sep+ 'nifti' + sep)
        except OSError:
            if not isdir(path_save + sep + 'nifti' + sep):
                raise
        nib.save(IM_matrix_nifti, path_save + sep + 'nifti' + sep + str(name) + '.nii.gz')  
        del IM_matrix_nifti
        
        input_files = [path_save + sep + 'nifti' + sep + str(name) + '.nii.gz']
        output_files = [path_save + sep + 'brain_mask' + sep + str(name) + '.nii.gz']
        
        try:
            makedirs(path_save + sep + 'brain_mask' + sep)
        except OSError:
            if not isdir(path_save + sep + 'brain_mask' + sep):
                raise
        
        #settings as from https://github.com/MIC-DKFZ/HD-BET/blob/master/HD_BET/hd-bet
        mode= 'fast' #fast\' or \'accurate
        config_file= 'config_hd_bet.py' #D:\\radiomics\\SPHN\\normalization\\HD-BET-master\\HD_BET\\config.py'
        device= 'cpu'
        pp= True
        tta=  True
        save_mask= True 
        overwrite_existing = False
        
        run_hd_bet(input_files, output_files, mode, config_file, device, pp, tta, save_mask, overwrite_existing)
        
        try:
            mask = nib.load(path_save + sep + 'brain_mask' + sep + str(name) + '_mask.nii.gz')
        except WindowsError:
            mask = nib.load(path_save + sep + 'brain_mask' + sep + str(name) + '_mask.nii')
        mask_data = mask.get_fdata().transpose(2,1,0)
        
        mask_data[np.where(mask_data == 0)] = np.nan
        
        a = IM_matrix * mask_data
        ###test segmentation
        for n in range(len(a)//8+1):
            fig = plt.figure(10, figsize=(20, 20))
            fig.text(0.5, 0.95, name)
            for j in range(8):
                axes = fig.add_subplot(3,3, j+1)
                axes.set_title(8*n+j)
                try:
                    im = axes.imshow(a[8 * n + j], cmap=plt.get_cmap('jet'), vmin=0, vmax=np.nanmax(a))
                except IndexError:
                    break
            axes = fig.add_subplot(3, 3, 9)
            try:
                fig.colorbar(im)
            except UnboundLocalError:
                pass
            try:
                makedirs(path_save+ sep +'brain_mask_png' + sep + name + sep)
            except OSError:
                if not isdir(path_save+ sep+'brain_mask_png'+ sep + name + sep):
                    raise
            fig.savefig(path_save+sep+'brain_mask_png'+sep+name+sep+str(n+1)+'.png')
            plt.close()
        
        return mask_data

    def hist_matching(self, newBrain, name, path_save, path_template):
        #matched = match_histograms(newBrain, standardBrain, multichannel=False)
        """
        Return modified source array so that the cumulative density function of
        its values matches the cumulative density function of the template.
        """
        src_values, src_unique_indices, src_counts = np.unique(newBrain.ravel(), return_inverse=True, return_counts=True)
        #src_values - unique values
        #src_unique_indices - indices per voxel to given unique value, eash nan voxel is considered a unique value and has to be handeled separately
        #src_counts - counts how many time thie unique value occurs
#        example 
#        a = array([[nan,  2.,  3.,  4.,  4.],
#        [ 4.,  4.,  4.,  5., nan]])
#        np.unique(a.ravel(), return_inverse=True, return_counts=True)
#        (array([ 2.,  3.,  4.,  5., nan, nan]),
#         array([4, 0, 1, 2, 2, 2, 2, 2, 3, 5], dtype=int64),
#         array([1, 1, 5, 1, 1, 1], dtype=int64))
        
        #read template
        standardBrain = np.load(path_template)

        tmpl_values, tmpl_counts = np.unique(standardBrain.ravel(), return_counts=True)
        
        #check for nan
        src_nan = np.min(np.where(np.isnan(src_values)))
        src_counts = src_counts[ : src_nan]
        
        
        tmpl_nan = np.min(np.where(np.isnan(tmpl_values)))
        tmpl_counts = tmpl_counts[ : tmpl_nan]
        tmpl_values = tmpl_values[ : tmpl_nan]
    
        # calculate normalized quantiles for each array
        src_quantiles = np.cumsum(src_counts) / (newBrain.size - len(np.where(np.isnan(newBrain))[0]))
        tmpl_quantiles = np.cumsum(tmpl_counts) / (standardBrain.size - len(np.where(np.isnan(standardBrain))[0]))
    
        interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
        
        src_new_values = np.zeros(len(src_unique_indices))
        src_new_values[ : ] = np.nan 
        for i, sind in enumerate(src_unique_indices):
            if sind < src_nan:
                src_new_values[i] = interp_a_values[sind]
                
        a=src_new_values.reshape(newBrain.shape)
        ###test segmentation
        for n in range(len(a)//8+1):
            fig = plt.figure(10, figsize=(20, 20))
            fig.text(0.5, 0.95, name)
            for j in range(8):
                axes = fig.add_subplot(3,3, j+1)
                axes.set_title(8*n+j)
                try:
                    im = axes.imshow(a[8 * n + j], cmap=plt.get_cmap('jet'))
                except IndexError:
                    break
                    pass
            axes = fig.add_subplot(3, 3, 9)
            try:
                fig.colorbar(im)
            except UnboundLocalError:
                pass
            try:
                makedirs(path_save+sep+'brain_mask_hist'+sep+name+sep)
            except OSError:
                if not isdir(path_save+sep+'brain_mask_hist'+sep+name+sep):
                    raise
            fig.savefig(path_save+sep+'brain_mask_hist'+sep+name+sep+str(n+1)+'.png')
            plt.close()
            
        affine_trans = np.zeros((4,4))
        affine_trans[0, 0] = -1 * 0.5
        affine_trans[1, 1] = -1 * 0.5
        affine_trans[2, 2] = 0.5
        affine_trans[3, 3] = 1.
        affine_trans[0, 3] = -1 * 100
        affine_trans[1, 3] = -1 * 100
        affine_trans[2, 3] = 100
        
        IM_matrix_nifti = nib.Nifti1Image(a, affine = affine_trans)
        nib.save(IM_matrix_nifti, path_save+sep+'brain_mask_hist'+sep+name+sep+'norm.nii.gz')        
             
        return src_new_values.reshape(newBrain.shape)
    
    def zscore_norm(self, ROI):
        mean = np.nanmean(ROI)
        std = np.nanstd(ROI)
        
        return mean, std

    #__________________________________________Nyul Normalization______________________________________________________

    # This function uses piecewise linear transformation to normalize the image intensities to a template histogram
    # function Args:
       # Img (array): Image matrix
       # temp_img (str): path to the mask of template image
       # save_path (str): Directory to save the normalized image
       # name (str) : the name of image
    # Returns:
       # Normalized Image (array) 

    def Nyul_norm(self,Img,temp_Img, name, path_save):

        i_min = 1 # minimum percentile to consider in the image
        i_max = 99 # maximum percentile to consider on the image
        i_s_min = 1 # minimum percentile on the standard scale
        i_s_max = 100 # maximum percentile on standard scale
        l_percentile = 10 # middle percentile lower band
        U_percentile = 90 # middle percentile upper band
        steps = 10

        # define the percentiles ---------------------------------------------------------------------------------------
        percs1 = np.concatenate(([i_min], np.arange(l_percentile, U_percentile, steps), [i_max])) # Array of percentiles on the image
        percs2 = np.concatenate(([i_s_min], np.arange(l_percentile, U_percentile, steps), [i_s_max])) # Array of percentiles on the template image

        # load template Image ------------------------------------------------------------------------------------------
        temp_data = np.load(temp_Img)

        # Remove NaNs from Image and template array --------------------------------------------------------------------
        temp_NaNs = np.isnan(temp_data)
        temp_non_NaN = temp_data[~temp_NaNs]

        Img_NaNs = np.isnan(Img)
        Img_non_NaN = Img[~Img_NaNs]

        # Calculate Image and Template Landmarks -----------------------------------------------------------------------
        standard_scale = self.get_landmarks(temp_non_NaN.ravel(),percs2)
        img_landmarks = self.get_landmarks(Img_non_NaN.ravel(),percs1)

        # normalizing the Image ----------------------------------------------------------------------------------------
        normalized_intensities = interp1d(img_landmarks, standard_scale,fill_value='extrapolate')
        img_normed = normalized_intensities(Img)

        # Export the Normalized Image in Nifti Format ------------------------------------------------------------------
        affine_trans = np.zeros((4,4))
        affine_trans[0,0] = -1 * 0.5
        affine_trans[1,1] = -1 * 0.5
        affine_trans[2,2] = 0.5
        affine_trans[3,3] = 1.
        affine_trans[0,3] = -1 * 100
        affine_trans[1,3] = -1 * 100
        affine_trans[2,3] = 100
        Img_nifti = nib.Nifti1Image(img_normed, affine=affine_trans)
        try:
            os.mkdir(path_save + 'normalized_img_nyul')
        except FileExistsError:
            pass

        nib.save(Img_nifti, path_save + 'normalized_img_nyul' + sep + name + 'norm.nii.gz')

        return img_normed

    # Define the function for getting the landmarks --------------------------------------------------------------------
    def get_landmarks(self,Img,perc):

        landmarks = np.percentile(Img,perc)
        return landmarks



    
