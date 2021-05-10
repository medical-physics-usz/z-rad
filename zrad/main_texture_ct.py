import logging
import os

import numpy as np
import pandas as pd
import pydicom as dc
from joblib import Parallel, delayed
from tqdm import tqdm

from export import Export
from features2d import Features2D
from read import ReadImageStructure
from texture import Texture
from utils import tqdm_joblib


class main_texture_ct(object):
    """Main class to handle CT images, reads images and structures, calls radiomics calculation and export class to export results
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
    HUmin - HU range min
    HUmax - HU range max
    outlier - bool, correct for outliers
    wv - bool, calculate wavelet
    exportList - list of matrices/features to be calculated and exported
    """

    def __init__(self, sb, path_image, path_save, structures, pixNr, binSize, l_ImName, save_as, dim, HUmin, HUmax,
                 outlier_corr, wv, local, cropInput, exportList, n_jobs):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Start")
        self.n_jobs = n_jobs
        image_modality = ['CT']
        meanWV = False

        def parfor(ImName):
            self.logger.info("Patient " + ImName)
            # create dataframe to save results for patient on a line
            # be sure to always have same order of feature names: load feature names out of text file
            with open("feature_names_2D.txt", "r") as f:
                feature_names_2d_list = f.read().split()
            to_return_2d = pd.DataFrame(columns=feature_names_2d_list)
            to_return_3d = list()
            meanWV = False #caluclated modified WV transform
                                
            for structure in structures:
                self.logger.info("Structure " + structure)
                structure = [structure]
                try:
                    mypath_image = path_image + ImName + os.sep
                    # CT and contrast-enhanced CT
                    CT_UID = ['1.2.840.10008.5.1.4.1.1.2', '1.2.840.10008.5.1.4.1.1.2.1', 'CT Image Storage']
                    read = ReadImageStructure(CT_UID, mypath_image, structure, wv, dim, local)

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
                        data16 = np.array(np.frombuffer(data, dtype=bitsRead))  # converting to decimal
                        data16 = data16 * slope + inter
                        # recalculating for rows x columns
                        a = np.reshape(data16, (read.rows, read.columns))
                        IM_matrix.append(np.array(a))
                    IM_matrix = np.array(IM_matrix)
                except OSError:  # error if there is not directory
                    continue
                except IndexError:  # empty folder
                    continue

                stop_calc = ''  # in case something would be wrong with the image tags
                if dim == "2D" or dim == "2D_singleSlice":
                    dict_features = Features2D(dim, [IM_matrix], read.structure_f, read.columns, read.rows,
                                               read.xCTspace, read.zCTspace, read.slices, path_save, ImName, pixNr,
                                               binSize, image_modality, wv, local, cropInput, stop_calc, read.Xcontour,
                                               read.Xcontour_W, read.Ycontour, read.Ycontour_W, read.Xcontour_Rec,
                                               read.Ycontour_Rec, HUmin, HUmax, outlier_corr).ret()
                else:
                    lista_results = Texture([IM_matrix], read.structure_f, read.columns, read.rows, read.xCTspace,
                                            read.slices, path_save, ImName, pixNr, binSize, image_modality, wv, local,
                                            cropInput, stop_calc, meanWV, read.Xcontour, read.Xcontour_W, read.Ycontour,
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
            final_file, wave_names, par_names = Export().Preset(exportList, wv, local, path_save, save_as,
                                                                image_modality, path_image)
            feature_vectors = [feature_vec for batch in out for feature_vec in batch]
            for feature_vec in feature_vectors:
                final_file = Export().ExportResults(feature_vec, final_file, par_names, image_modality, wave_names, wv,
                                                    local)
            final_file.close()
