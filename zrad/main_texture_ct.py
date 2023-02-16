import logging
import os
from os.path import isfile, join

import numpy as np
import pandas as pd
import pydicom as dc
import nibabel as nib
from joblib import Parallel, delayed
from tqdm import tqdm

from features2d import Features2D
from read import ReadImageStructure
from texture import Texture
from utils import tqdm_joblib
from export import export_results, preset


class main_texture_ct(object):
    """Main class to handle CT images, reads images and structures, calls radiomics calculation and export class to export results
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
    HUmin - HU range min
    HUmax - HU range max
    outlier - bool, correct for outliers
    wv - bool, calculate wavelet
    exportList - list of matrices/features to be calculated and exported
    """

    def __init__(self, file_type, path_image, path_save, structures, labels, pixNr, binSize, l_ImName, save_as, dim, HUmin, HUmax,
                 outlier_corr, wv, local, cropInput, n_jobs):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Start")
        self.n_jobs = n_jobs
        image_modality = ['CT']
        
        meanWV = False  # calculate modified WV transform

        def parfor(ImName):
            self.logger.info("Patient " + ImName)
            # create dataframe to save results for patient on a line
            # be sure to always have same order of feature names: load feature names out of text file
            with open("feature_names_2D.txt", "r") as f:
                feature_names_2d_list = f.read().split()
            to_return_2d = pd.DataFrame(columns=feature_names_2d_list)
            to_return_3d = list()
            meanWV = False  # calculate modified WV transform
                                
            for structure in structures:
                self.logger.info("Structure " + structure)
                structure = [structure]
                try:
                    mypath_image = path_image + ImName + os.sep
                    # CT and contrast-enhanced CT
                    CT_UID = ['1.2.840.10008.5.1.4.1.1.2', '1.2.840.10008.5.1.4.1.1.2.1', 'CT Image Storage']
                    read = ReadImageStructure(file_type, CT_UID, mypath_image, structure, wv, dim, local)

                    if file_type == 'dicom':
                        # parameters to recalculate intensities HU
        
                        inter = float(dc.read_file(mypath_image + read.onlyfiles[0]).RescaleIntercept)
                        slope = float(dc.read_file(mypath_image + read.onlyfiles[0]).RescaleSlope)
        
                        bitsRead = str(dc.read_file(mypath_image + read.onlyfiles[0]).BitsAllocated)
                        sign = int(dc.read_file(mypath_image + read.onlyfiles[0]).PixelRepresentation)
        
                        if sign == 1:
                            bitsRead = 'int' + bitsRead
                        elif sign == 0:
                            bitsRead = 'uint' + bitsRead
        
                        IM_matrix = []  # list containing the images matrix
                        for f in read.onlyfiles:
                            data = dc.read_file(mypath_image + f).PixelData
                            data16 = np.array(np.fromstring(data, dtype=bitsRead))  # converting to decimal
                            data16 = data16 * slope + inter
                            # recalculating for rows x columns
                            a = np.reshape(data16, (read.rows, read.columns))
                            IM_matrix.append(np.array(a))
                        IM_matrix = np.array(IM_matrix)
                        contour_matrix = ''  #only for nifti
                    
                    elif file_type == 'nifti':  # nifti file type assumes that conversion to eg HU or SUV has already been performed
                        # the folder should contain two file, one contour mask with the name corresponding to the name given in GUI and the other file with the image
                        if read.stop_calc == '':
                            image_name = ''
                            contour_name = ''
                            for f in read.onlyfiles:
                                if isfile(join(mypath_image, f)) and structure[0] in f: #nifti only handles one ROI
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
                except OSError:  # error if there is no directory
                    continue
                except IndexError:  # empty folder
                    continue

                if dim == "2D" or dim == "2D_singleSlice":
                    dict_features = Features2D(dim, [IM_matrix], read.structure_f, read.columns, read.rows,
                                               read.xCTspace, read.zCTspace, read.slices, path_save, ImName, pixNr,
                                               binSize, image_modality, wv, local, cropInput, read.stop_calc, read.Xcontour,
                                               read.Xcontour_W, read.Ycontour, read.Ycontour_W, read.Xcontour_Rec,
                                               read.Ycontour_Rec, HUmin, HUmax, outlier_corr).ret()
                else:
                    lista_results = Texture(file_type, [IM_matrix], contour_matrix, read.structure_f, read.columns, read.rows, read.xCTspace,
                                            read.slices, path_save, ImName, pixNr, binSize, image_modality, wv, local,
                                            cropInput, read.stop_calc, meanWV, read.Xcontour, read.Xcontour_W, read.Ycontour,
                                            read.Ycontour_W, read.Xcontour_Rec, read.Ycontour_Rec, HUmin, HUmax,
                                            outlier_corr).ret()

                # final list contains of the sublist for each patient, sublist contains of
                # [patient number, structure used for calculations, list of texture parameters,
                # number of points used for calculations]
                if dim == "2D" or dim == "2D_singleSlice":
                    dp_export = pd.DataFrame(dict_features, index=[ImName])
                    to_return_2d = to_return_2d.append(dp_export, sort=False)
                else:
                    features_3d = [[ImName, lista_results[2], lista_results[:2], lista_results[3:-1],
                                    lista_results[-1]]]
                    to_return_3d.append(features_3d)

            if dim == "2D" or dim == "2D_singleSlice":
                to_return = to_return_2d
            else:
                to_return = to_return_3d
            return to_return

        with tqdm_joblib(tqdm(desc="Extracting intensity and texture features", total=len(l_ImName))):
            out = Parallel(n_jobs=self.n_jobs)(delayed(parfor)(ImName) for ImName in l_ImName)

        if dim == "2D" or dim == "2D_singleSlice":
            df_features_all = pd.concat(out).reset_index().rename(columns={'index': 'patient'})
            # create dictionary with important parameters
            dict_parameters = {"image_modality": image_modality,
                               "structures": str.join(', ', structures),
                               "pixelNr": pixNr,
                               "bin_size": binSize,
                               "Dimension": dim,
                               "HUmin": HUmin,
                               "HUmax": HUmax,
                               "outlier_corr": outlier_corr,
                               "wv": wv}
            df_parameters = pd.DataFrame.from_dict(dict_parameters)
            # save excel sheet
            with pd.ExcelWriter(path_save + save_as + ".xlsx") as writer:
                df_features_all.to_excel(writer, sheet_name="Features")
                df_parameters.to_excel(writer, sheet_name="Parameters")
        else:
            final_file, wave_names, par_names = preset(wv, path_save, save_as, image_modality)
            feature_vectors = [feature_vec for batch in out for feature_vec in batch]
            for feature_vec in feature_vectors:
                final_file = export_results(feature_vec, final_file, image_modality, wave_names, wv)
            final_file.close()
