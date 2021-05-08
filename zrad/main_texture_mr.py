import logging
import os

import numpy as np
import pydicom as dc
from joblib import Parallel, delayed

from os.path import isdir, isfile, join
from os import makedirs

from export import Export
from read import ReadImageStructure
from structure import Structures
from texture import Texture

#normalization libraries
import nibabel as nib
from HD_BET.run import run_hd_bet
import pylab as plt


class main_texture_mr(object):
    """
    Main class to handle MR images, reads images and structures, normalize MR image according to two specified ROI,
    calls radiomics calculation and export class to export results
    Type: object
    Attributes:
    sb - Status bar in the frame
    path_image - path to the patients subfolders
    path_save - path to save radiomics results
    structure - list of structures to be analysed
    pixNr number of analyzed bins, if not specified  = none
    binSize - bin size for the analysis, if not specified = none
    l_ImName - list of patients subfolders (here are data to be analysed)
    save_as - name of text files to save the radiomics results
    Dim - string variable of value 2D or 3D for dimensionality of calculation
    struct_norm1 - ROI1 for the normalization coefficients for MR image based on two normal structures (for example
    muscle and white matter) and linear function
    struct_norm2 - ROI2 for the normalization
    normROI_advanced - ROI to be used in one of the advanced normalization modes (z-score or histogram matching)
    path_skull - path to files with brain segmentations
    norm_type - type of the normalization (z-score or histogram matching)
    path_template - template for histogram matching
    wv - bool, calculate wavelet
    exportList - list of matrices/features to be calculated and exported
    """

    def __init__(self, sb, path_image, path_save, structures, pixNr, binSize, l_ImName, save_as, dim, struct_norm1, struct_norm2, normROI_advanced, path_skull, 
                 norm_type, path_template, wv, local, cropStructure, exportList, n_jobs):

        self.logger = logging.getLogger(__name__)
        self.logger.info("Start")
        self.n_jobs = n_jobs
        image_modality = ['MR']
        dicomProblem = []
        MR_UID = ['1.2.840.10008.5.1.4.1.1.4']  # MR
        meanWV = False #caluclated modified WV transform

        def parfor(ImName):
            self.logger.info("Patient " + ImName)
            mypath_image = path_image + ImName + os.sep
            to_return_3d = list()

            for structure in structures:
                try:
                    read = ReadImageStructure(MR_UID, mypath_image, [structure], wv, dim, local)
                    dicomProblem.append([ImName, read.listDicomProblem])

                    # MR intensities normalization
                    if struct_norm1 != '':
                        norm_slope, norm_inter = self.normalization_linear(struct_norm1, struct_norm2, read, mypath_image)
                    else:
                        norm_slope = 1  # to allow for calculation with no normalization
                        norm_inter = 0

                    bitsRead = str(dc.read_file(mypath_image + read.onlyfiles[1]).BitsAllocated)
                    sign = int(dc.read_file(mypath_image + read.onlyfiles[1]).PixelRepresentation)
                    if sign == 1:
                        bitsRead = 'int' + bitsRead
                    elif sign == 0:
                        bitsRead = 'uint' + bitsRead

                    IM_matrix = []  # list containing the images matrix
                    for f in read.onlyfiles:
                        data = dc.read_file(mypath_image + f).PixelData
                        data16 = np.array(np.fromstring(data, dtype=bitsRead))  # converting to decimal
                        data16 = data16 * norm_slope + norm_inter
                        # recalculating for rows x columns
                        a = []
                        for j in range(read.rows):
                            a.append(data16[j * read.columns:(j + 1) * read.columns])
                        a = np.array(a)
                        IM_matrix.append(np.array(a))
                    IM_matrix = np.array(IM_matrix)
                    
                    #advanced normalization
                    if norm_type == 'z-score' or norm_type == 'histogram matching' and read.stop_calc == '':
                        #normalization ROI
                        if normROI_advanced == 'brain':
                            patPos = dc.read_file(mypath_image + f).PatientPosition
                            brain_mask = self.maskBrain(patPos, read.xCTspace, read.x_ct, read.y_ct, np.min(read.slices), IM_matrix, path_skull, ImName, path_save)
                            IM_matrix_masked = IM_matrix * brain_mask  
                        elif normROI_advanced == 'ROI':
                            structROI = Structures(read.rs, [read.structure_f], read.slices, read.x_ct, read.y_ct, read.xCTspace, len(read.slices), False, None, False) #False for the wavelet and local  argument, none for dim
                            norm_Xcontour = structROI.Xcontour
                            norm_Ycontour = structROI.Ycontour
                            ROI_mask = self.maskROI(norm_Xcontour, norm_Ycontour, IM_matrix.shape, np.nan)
                            IM_matrix_masked = IM_matrix * ROI_mask                            
                        elif normROI_advanced == 'brain-ROI':
                            patPos = dc.read_file(mypath_image + f).PatientPosition
                            brain_mask = self.maskBrain(patPos, read.xCTspace, read.x_ct, read.y_ct, np.min(read.slices), IM_matrix, path_skull, ImName, path_save)
                            structROI = Structures(read.rs, [read.structure_f], read.slices, read.x_ct, read.y_ct, read.xCTspace, len(read.slices), False, None, False) #False for the wavelet and local  argument, none for dim
                            norm_Xcontour = structROI.Xcontour
                            norm_Ycontour = structROI.Ycontour
                            ROI_mask = self.maskROI(norm_Xcontour, norm_Ycontour, IM_matrix.shape, 0)
                            ROI_mask += 1
                            ROI_mask[np.where(ROI_mask==2)] = np.nan
                            brainMinusROI_mask = brain_mask * ROI_mask
                            IM_matrix_masked = IM_matrix * brainMinusROI_mask
                        elif normROI_advanced == 'none':
                            self.logger = logging.getLogger("MR read")
                            self.logger.info("Advanced normalization ROI not defined")
                            break
                        
                        #normaliztion technique
                        if norm_type == 'histogram matching':
                            IM_matrix = self.HistMatching(IM_matrix_masked, ImName, path_save, path_template)   
                            meanWV = True #caluclated modified WV transform
                        elif norm_type == 'z-score':
                            mean, std = self.ZscoreNorm(IM_matrix_masked)      
                            IM_matrix = (IM_matrix - mean)/std


                except OSError:  # error if there is not directory
                    continue
                except IndexError:  # empty folder
                    continue

                    # Texture(arguments).ret() -> function for texture calculation
                    # arguments: image, structure name, image corner x, image corner x, columns, pixelSpacing, HFS or FFS, structure file, list of slice positions, patient number, path to save the textutre maps, map name (eg. AIF1), pixel discretization, site
                    # function returns: number of removed points, minimum values, maximum values, structre used for calculations, mean,std, cov, skewness, kurtosis, enenrgy, entropy, contrast, corrrelation, homogenity, coarseness, neighContrast, busyness, complexity, intensity varation, size variation, fractal dimension, number of points used in the calculations, histogram (values bigger/smaller than median)
                if dim == '3D':
                    lista_results = Texture([IM_matrix], read.structure_f, read.columns, read.rows, read.xCTspace,
                                            read.slices, path_save, ImName, pixNr, binSize, image_modality, wv, local,
                                            cropStructure, read.stop_calc, read.Xcontour, read.Xcontour_W, read.Ycontour,
                                            read.Ycontour_W, read.Xcontour_Rec, read.Ycontour_Rec).ret()

                #                elif dim == '2D': #not working
                #                    lista_results = Texture2D(sb,IM_matrix, structure, x_ct,y_ct, columns, rows, xCTspace, patientPos, rs, slices, path_save, ImName, pixNr, prefix).ret()

                features_3d = [[ImName, lista_results[2], lista_results[:2], lista_results[3:-1], lista_results[-1]]]
                to_return_3d.append(features_3d)
            return to_return_3d

        out = Parallel(n_jobs=self.n_jobs, verbose=20)(delayed(parfor)(ImName) for ImName in l_ImName)

        final_file, wave_names, par_names = Export().Preset(exportList, wv, local, path_save, save_as, image_modality,
                                                            path_image)

        feature_vectors = [feature_vec for batch in out for feature_vec in batch]
        for feature_vec in feature_vectors:
            final_file = Export().ExportResults(feature_vec, final_file, par_names, image_modality,
                                                wave_names, wv, local)
        final_file.close()

    def normalization_linear(self, struct_norm1, struct_norm2, read, mypath_image):
        """Get the normalization coefficients for MR image based on two normal structures (for example muscle and white
        matter) and linear function.
        """
        # False for the wavelet and local argument, none for dim
        struct1 = Structures(read.rs, [struct_norm1], read.slices, read.x_ct, read.y_ct,
                             read.xCTspace, len(read.slices), False, None, False)
        norm1_Xcontour = struct1.Xcontour
        norm1_Ycontour = struct1.Ycontour

        struct2 = Structures(read.rs, [struct_norm2], read.slices, read.x_ct, read.y_ct, read.xCTspace,
                             len(read.slices), False, None,
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

        onlyfiles = read.onlyfiles[zmin:zmax + 1]
        slices = read.slices[zmin:zmax + 1]
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
            for j in range(read.rows):
                a.append(data16[j * read.columns:(j + 1) * read.columns])
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
    
    def maskROI(self, norm_Xcontour, norm_Ycontour, matrix_shape, outer_value):
        mask_data = np.zeros(matrix_shape)  # creating the matrix to fill it with points of the structure, y rows, x columns
        mask_data[:, :, :] = outer_value
        for i in range(len(norm_Xcontour)): # slices
            for j in range(len(norm_Xcontour[i])): #sub-structres in the slice
                for k in range(len(norm_Xcontour[i][j])):
                    mask_data[i][norm_Ycontour[i][j][k]][norm_Xcontour[i][j][k]] = 1
        return mask_data
    
    def maskBrain(self, patPos,xCTspace, x_ct, y_ct, z_ct, IM_matrix, path_skull, ImName, path_save):
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
                
                mask_data = self.SkullStripping(IM_matrix, affine_trans, path_save, ImName)
            else:
                self.logger = logging.getLogger("MR read")
                self.logger.info("Scan position not HFS, the skull stripping was not tested")
                raise TypeError
        elif path_skull != '':
            try:
                mask = nib.load(path_skull + os.sep + str(ImName) + '_mask.nii.gz')
            except WindowsError:
                mask = nib.load(path_skull + os.sep + str(ImName) + '_mask.nii')
            mask_data = mask.get_fdata()

            mask_data[np.where(mask_data == 0)] = np.nan
            
        return mask_data
    
    def SkullStripping(self, IM_matrix, transform, path_save, name):
        IM_matrix_nifti = nib.Nifti1Image(IM_matrix, affine = transform)
        try:
            makedirs(path_save + os.sep+ 'nifti' + os.sep)
        except OSError:
            if not isdir(path_save + os.sep + 'nifti' + os.sep):
                raise
        nib.save(IM_matrix_nifti, path_save + os.sep + 'nifti' + os.sep + str(name) + '.nii.gz')  
        del IM_matrix_nifti
        
        input_files = [path_save + os.sep + 'nifti' + os.sep + str(name) + '.nii.gz']
        output_files = [path_save + os.sep + 'brain_mask' + os.sep + str(name) + '.nii.gz']
        
        try:
            makedirs(path_save + os.sep + 'brain_mask' + os.sep)
        except OSError:
            if not isdir(path_save + os.sep + 'brain_mask' + os.sep):
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
            mask = nib.load(path_save + os.sep + 'brain_mask' + os.sep + str(name) + '_mask.nii.gz')
        except WindowsError:
            mask = nib.load(path_save + os.sep + 'brain_mask' + os.sep + str(name) + '_mask.nii')
        mask_data = mask.get_fdata()
        
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
                    pass
            axes = fig.add_subplot(3, 3, 9)
            try:
                fig.colorbar(im)
            except UnboundLocalError:
                pass
            try:
                makedirs(path_save+ os.sep +'brain_mask_png' +os.sep+name+os.sep)
            except OSError:
                if not isdir(path_save+os.sep+'brain_mask_png'+os.sep+name+os.sep):
                    raise
            fig.savefig(path_save+os.sep+'brain_mask_png'+os.sep+name+os.sep+str(n+1)+'.png')
            plt.close()
        
        return mask_data

    def HistMatching(self, newBrain, name, path_save, path_template):
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
                makedirs(path_save+os.sep+'brain_mask_hist'+os.sep+name+os.sep)
            except OSError:
                if not isdir(path_save+os.sep+'brain_mask_hist'+os.sep+name+os.sep):
                    raise
            fig.savefig(path_save+os.sep+'brain_mask_hist'+os.sep+name+os.sep+str(n+1)+'.png')
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
        nib.save(IM_matrix_nifti, path_save+os.sep+'brain_mask_hist'+os.sep+name+os.sep+'norm.nii.gz')        
             
        return src_new_values.reshape(newBrain.shape)
    
    def ZscoreNorm(self, ROI):
        mean = np.nanmean(ROI)
        std = np.nanstd(ROI)
        
        return mean, std
    


