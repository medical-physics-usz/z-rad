import logging
import os
from os.path import isfile, join

import nibabel as nib
import numpy as np
import pydicom as dc
from joblib import Parallel, delayed

import normalization
from export import Export
from normalization import linear_norm, zscore_norm, hist_matching_norm
from read import ReadImageStructure
from structure import Structures
from texture import Texture


class main_texture_mr(object):
    """
    Main class to handle MR images, reads images and structures, normalize MR image according to two specified ROI,
    calls radiomics calculation and export class to export results
    Type: object
    Attributes:
    sb - Status bar in the frame
    file_type - dicom or nifti, influences reading in the files
    path_image - path to the patients subfolders
    path_save - path to save radiomics results
    structure - list of structures to be analysed
    labels - label number in the nifti file, list of numbers, each number corresponds to different contour
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

    def __init__(self, sb, file_type, path_image, path_save, structures, labels, pixNr, binSize, l_ImName, save_as, dim, struct_norm1, struct_norm2, normROI_advanced, path_skull,
                 norm_type, path_template, wv, local, cropStructure, exportList, n_jobs):

        self.logger = logging.getLogger(__name__)
        self.logger.info("Start")
        self.n_jobs = n_jobs
        image_modality = ['MR']
        dicomProblem = []
        MR_UID = ['1.2.840.10008.5.1.4.1.1.4']  # MR

        def parfor(ImName):
            self.logger.info("Patient " + ImName)
            mypath_image = path_image + ImName + os.sep
            to_return_3d = list()
            meanWV = False  # caluclated modified WV transform

            for structure in structures:
                try:
                    read = ReadImageStructure(file_type, MR_UID, mypath_image, [structure], wv, dim, local)
                    dicomProblem.append([ImName, read.listDicomProblem])

                    if file_type == 'dicom':
                        # MR intensities normalization
                        if norm_type == 'linear':
                            norm_slope, norm_inter = linear_norm(struct_norm1, struct_norm2, mypath_image, read.rs, read.slices, read.x_ct, read.y_ct, read.xCTspace, read.onlyfiles, read.rows, read.columns)
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
                        contour_matrix = ''  # only for nifti

                    elif file_type == 'nifti':  # nifti file type assumes that conversion to eg HU or SUV has already been performed
                        # the folder should contain two file, one contour mask with the name corresponding to the name given in GUI and the other file with the image
                        if read.stop_calc == '':
                            image_name = ''
                            contour_name = ''
                            for f in read.onlyfiles:
                                if isfile(join(mypath_image, f)) and structure in f:  # nifti only handles one ROI
                                    contour_name = f
                                else:
                                    image_name = f
                            img = nib.load(mypath_image + image_name)
                            contour = nib.load(mypath_image + contour_name)
                            slope = img.header['scl_slope']
                            intercept = img.header['scl_inter']
                            if np.isnan(slope):
                                slope = 1.
                            if np.isnan(intercept):
                                intercept = 0
                            IM_matrix = img.get_fdata().transpose(2, 1, 0) * slope + intercept
                            contour_matrix = contour.get_fdata().transpose(2, 1, 0)
                            for lab in labels:
                                ind = np.where(contour_matrix == lab)
                                contour_matrix[ind] = 100
                            ind = np.where(contour_matrix != 100)
                            contour_matrix[ind] = 0
                            ind = np.where(contour_matrix == 100)
                            contour_matrix[ind] = 1
                        else:
                            IM_matrix = ''
                            contour_matrix = ''

                    # advanced normalization
                    if norm_type == 'z-score' or norm_type == 'histogram matching' and read.stop_calc == '':
                        # normalization ROI
                        if normROI_advanced == 'brain':
                            patPos = dc.read_file(mypath_image + f).PatientPosition
                            brain_mask = normalization.mask_brain(patPos, read.xCTspace, read.x_ct, read.y_ct, np.min(read.slices), IM_matrix, path_skull, ImName, path_save)
                            IM_matrix_masked = IM_matrix * brain_mask
                        elif normROI_advanced == 'ROI' and file_type == 'dicom':
                            structROI = Structures(read.rs, [read.structure_f], read.slices, read.x_ct, read.y_ct, read.xCTspace, len(read.slices), False, None, False) #False for the wavelet and local  argument, none for dim
                            norm_Xcontour = structROI.Xcontour
                            norm_Ycontour = structROI.Ycontour
                            ROI_mask = normalization.mask_roi(norm_Xcontour, norm_Ycontour, IM_matrix.shape, np.nan)
                            IM_matrix_masked = IM_matrix * ROI_mask
                        elif normROI_advanced == 'ROI' and file_type == 'nifti':
                            c_temp = contour_matrix.copy()
                            c_temp[np.where(c_temp == 0)] = np.nan
                            IM_matrix_masked = IM_matrix * c_temp
                            del c_temp
                        elif normROI_advanced == 'brain-ROI' and file_type == 'dicom':
                            patPos = dc.read_file(mypath_image + f).PatientPosition
                            brain_mask = normalization.mask_brain(patPos, read.xCTspace, read.x_ct, read.y_ct, np.min(read.slices), IM_matrix, path_skull, ImName, path_save)
                            structROI = Structures(read.rs, [read.structure_f], read.slices, read.x_ct, read.y_ct, read.xCTspace, len(read.slices), False, None, False) #False for the wavelet and local  argument, none for dim
                            norm_Xcontour = structROI.Xcontour
                            norm_Ycontour = structROI.Ycontour
                            ROI_mask = normalization.mask_roi(norm_Xcontour, norm_Ycontour, IM_matrix.shape, 0)
                            ROI_mask += 1
                            ROI_mask[np.where(ROI_mask == 2)] = np.nan
                            brainMinusROI_mask = brain_mask * ROI_mask
                            IM_matrix_masked = IM_matrix * brainMinusROI_mask
                        elif normROI_advanced == 'none':
                            self.logger = logging.getLogger("MR read")
                            self.logger.info("Advanced normalization ROI not defined")
                            break

                        # normaliztion technique
                        if norm_type == 'histogram matching':
                            IM_matrix = hist_matching_norm(IM_matrix_masked, ImName, path_save, path_template)
                            meanWV = True  # calculated modified WV transform
                        elif norm_type == 'z-score':
                            mean, std = zscore_norm(IM_matrix_masked)
                            IM_matrix = (IM_matrix - mean)/std

                except OSError:  # error if there is not directory
                    continue
                except IndexError:  # empty folder
                    continue

                    # Texture(arguments).ret() -> function for texture calculation
                    # arguments: image, structure name, image corner x, image corner x, columns, pixelSpacing, HFS or FFS, structure file, list of slice positions, patient number, path to save the textutre maps, map name (eg. AIF1), pixel discretization, site
                    # function returns: number of removed points, minimum values, maximum values, structre used for calculations, mean,std, cov, skewness, kurtosis, enenrgy, entropy, contrast, corrrelation, homogenity, coarseness, neighContrast, busyness, complexity, intensity varation, size variation, fractal dimension, number of points used in the calculations, histogram (values bigger/smaller than median)
                if dim == '3D':
                    lista_results = Texture(file_type, [IM_matrix], contour_matrix, read.structure_f, read.columns, read.rows, read.xCTspace,
                                            read.slices, path_save, ImName, pixNr, binSize, image_modality, wv, local,
                                            cropStructure, read.stop_calc, meanWV, read.Xcontour, read.Xcontour_W, read.Ycontour,
                                            read.Ycontour_W, read.Xcontour_Rec, read.Ycontour_Rec).ret()

                #                elif dim == '2D': #not working
                #                    lista_results = Texture2D(sb,IM_matrix, structure, x_ct,y_ct, columns, rows, xCTspace, patientPos, rs, slices, path_save, ImName, pixNr, prefix).ret()

                features_3d = [[ImName, lista_results[2], lista_results[:2], lista_results[3:-1], lista_results[-1]]]
                to_return_3d.append(features_3d)
            return to_return_3d

        out = Parallel(n_jobs=self.n_jobs, verbose=20)(delayed(parfor)(ImName) for ImName in l_ImName)

        final_file, wave_names, par_names = Export().Preset(exportList, wv, local, path_save, save_as, image_modality)

        feature_vectors = [feature_vec for batch in out for feature_vec in batch]
        for feature_vec in feature_vectors:
            final_file = Export().ExportResults(feature_vec, final_file, par_names, image_modality,
                                                wave_names, wv, local)
        final_file.close()

