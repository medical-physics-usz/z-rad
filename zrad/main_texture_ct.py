# -*- coding: cp1252 -*-

#import libraries
import pydicom as dc # dicom library
import numpy as np
import pandas as pd

# own classes
# import class to calculate texture parameters
from texture import Texture
from read import ReadImageStructure
from export import Export
from features2d import Features2D
import logging


class main_texture_ct(object):
    """Main class to handle CT images, reads images and structures, calls radiomics calculation and export class to export results
    Type: object
    Attributes:
    sb � Status bar in the frame
    path_image � path to the patients subfolders
    path_save � path to save radiomics results
    structure � list of structures to be analysed
    pixNr number of analyzed bins, if not specified  = none
    binSize � bin size for the analysis, if not specified = none
    l_ImName � list of patients subfolders (here are data to be analysed)
    save_as � name of text files to save the radiomics results
    Dim � string variable of value 2D or 3D for dimensionality of calculation
    HUmin - HU range min
    HUmax � HU range max
    outlier � bool, correct for outliers
    wv � bool, calculate wavelet
    exportList � list of matrices/features to be calculated and exported
    """

    def __init__(self, sb, path_image, path_save, structure, pixNr, binSize, l_ImName, save_as, dim, HUmin, HUmax,
                 outlier_corr, wv, local, cropInput, exportList):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Start")
        final = []  # list with results
        image_modality = ['CT']
        dicomProblem = []

        final_file, wave_names, par_names = Export().Preset(exportList, wv, local, path_save, save_as, image_modality,
                                                            path_image)

        # create dataframe to save results for patient on a line
        df_features_all = pd.DataFrame()
        for ImName in l_ImName:
            self.logger.info("Patient " + ImName)
            try:
                mypath_image = path_image + ImName + '\\'
                CT_UID = ['1.2.840.10008.5.1.4.1.1.2', '1.2.840.10008.5.1.4.1.1.2.1',
                          'CT Image Storage']  # CT and contrast-enhanced CT
                read = ReadImageStructure(CT_UID, mypath_image, structure, wv, dim, local)
                dicomProblem.append([ImName, read.listDicomProblem])

                # parameters to recalculate intensities HU
                if dim == "2D_singleSlice":
                    number = 0
                else:
                    number = 1

                inter = float(dc.read_file(mypath_image + read.onlyfiles[number]).RescaleIntercept)
                slope = float(dc.read_file(mypath_image + read.onlyfiles[number]).RescaleSlope)

                bitsRead = str(dc.read_file(mypath_image + read.onlyfiles[number]).BitsAllocated)
                sign = int(dc.read_file(mypath_image + read.onlyfiles[number]).PixelRepresentation)

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

            except WindowsError:  # error if there is not directory
                continue
            except IndexError:  # empty folder
                continue
            # Texture(arguments).ret() -> function for texture calculation arguments: status bar, image,
            # stucture name, columns,rows, pixelSpacing, list of slice positions,  path to save the texture maps,
            # patient number, pixel discretization number or size, image modality, waveltes, contours, HU range,
            # outlier_corr function returns: number of removed points, minimum values, maximum values, structure used
            # for calculations, mean,std, cov, skewness, kurtosis, enenrgy, entropy, contrast, correlation,
            # homogenity, coarseness, neighContrast, busyness, complexity, intensity variation, size variation,
            # fractal dimension, number of points used in the calculations, histogram (values bigger/smaller than
            # median)
            stop_calc = ''  # in case something would be wrong with the image tags
            if dim == "2D" or dim == "2D_singleSlice":
                dict_features = Features2D(dim, sb, [IM_matrix], read.structure_f, read.columns, read.rows, read.xCTspace, read.zCTspace,
                                          read.slices, path_save, ImName, pixNr, binSize, image_modality, wv, local,
                                          cropInput, stop_calc, read.Xcontour, read.Xcontour_W, read.Ycontour,
                                          read.Ycontour_W, read.Xcontour_Rec, read.Ycontour_Rec, HUmin, HUmax,
                                          outlier_corr).ret()
            else:
                lista_results = Texture(sb, [IM_matrix], read.structure_f, read.columns, read.rows, read.xCTspace,
                                        read.slices, path_save, ImName, pixNr, binSize, image_modality, wv, local,
                                        cropInput, stop_calc, read.Xcontour, read.Xcontour_W, read.Ycontour,
                                        read.Ycontour_W, read.Xcontour_Rec, read.Ycontour_Rec, HUmin, HUmax,
                                        outlier_corr).ret()

            # final list contains of the sublist for each patient, sublist contains of
            # [patient number, structure used for calculations, list of texture parameters,
            # number of points used for calculations]
            if dim == "2D" or dim == "2D_singleSlice":
                # if True:    # comment this out if no phantom calculation
                #     df_phantom = pd.DataFrame.from_dict(dict_features)
                #     df_features_all = df_phantom
                #     continue
                dp = pd.DataFrame.from_dict(dict_features)
                # dp.index = ["original", "LL", "HH", "HL", "LH"]
                # dp_stacked = dp.stack()
                # dp_export = pd.DataFrame(dp_stacked).transpose()
                # add patientID as index to the data frame
                dp_export_labeled = dp.rename(index={0: ImName})
                df_features_all = df_features_all.append(dp_export_labeled)  # ignore index only if not defined before..., ignore_index=True)
            else:
                final = [[ImName, lista_results[2], lista_results[:2], lista_results[3:-1], lista_results[-1]]]
                final_file = Export().ExportResults(final, final_file, par_names, image_modality, wave_names, wv, local)

        if dim == "2D" or dim == "2D_singleSlice":
            # create dictionary with important parameters
            dict_parameters = {"image_modality": image_modality,
                               "structure": structure,
                               "pixelNr": pixNr,
                               "bin_size": binSize,
                               "Dimension": dim,
                               "HUmin": HUmin,
                               "HUmax": HUmax,
                               "outlier_corr": outlier_corr,
                               "wv": wv}
            df_parameters = pd.DataFrame.from_dict(dict_parameters)
            # save excel sheet
            with pd.ExcelWriter(path_save+"features_2D.xlsx") as writer:
                df_features_all.to_excel(writer, sheet_name="Features")
                df_parameters.to_excel(writer, sheet_name="Parameters")
        else:
            final_file.close()
