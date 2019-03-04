# -*- coding: cp1252 -*-

#import libraries
try:
    import pydicom as dc # dicom library
except ImportError:
    import dicom as dc # dicom library
import numpy as np # numerical computation
from numpy import arange, floor
import pylab as py # drawing plots
from os import listdir, makedirs # managing files
from os.path import isfile, join, isdir

from scipy.stats import norm # statistical analysis
import scipy.optimize as optimization
import matplotlib
from scipy import ndimage
from time import gmtime

#own classes
#import class to calculate texture parameters
from texture import Texture
from exception import MyException
from read import ReadImageStructure
from export import Export
import logging

class main_texture_ct(object):
    '''Main class to handle CT images, reads images and structures, calls radiomics calculation and export class to export results
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
    '''
    def __init__(self, sb, path_image, path_save, structure, pixNr, binSize, l_ImName, save_as, Dim, HUmin, HUmax, outlier_corr, wv, local, cropInput, exportList):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Start")
        final=[] # list with results
        image_modality = ['CT']
        dicomProblem = []

        final_file, wave_names, par_names = Export().Preset(exportList, wv,local, path_save, save_as, image_modality, path_image)

        for ImName in l_ImName:
            self.logger.info("Patient " + ImName)
            try:
                mypath_image = path_image+ImName+'\\'
                CT_UID = ['1.2.840.10008.5.1.4.1.1.2', '1.2.840.10008.5.1.4.1.1.2.1', 'CT Image Storage'] #CT and contarst-enhanced CT
                read = ReadImageStructure(CT_UID, mypath_image, structure, wv, local)

                dicomProblem.append([ImName, read.listDicomProblem])

                #parameters to recalculate intensities HU
                inter = float(dc.read_file(mypath_image+read.onlyfiles[1]).RescaleIntercept)
                slope = float(dc.read_file(mypath_image+read.onlyfiles[1]).RescaleSlope)

                bitsRead = str(dc.read_file(mypath_image+read.onlyfiles[1]).BitsAllocated)
                sign = int(dc.read_file(mypath_image+read.onlyfiles[1]).PixelRepresentation)

                if sign == 1:
                    bitsRead = 'int'+bitsRead
                elif sign ==0:
                    bitsRead = 'uint'+bitsRead

                IM_matrix = [] #list containing the images matrix
                for f in read.onlyfiles:
                    data = dc.read_file(mypath_image+f).PixelData
                    data16 = np.array(np.fromstring(data, dtype=bitsRead)) #converitng to decimal
                    data16 = data16*slope+inter
                    #recalculating for rows x columns
                    a=[]
                    for j in arange(0, read.rows):
                        a.append(data16[j*read.columns:(j+1)*read.columns])
                    a=np.array(a)
                    IM_matrix.append(np.array(a))
                IM_matrix = np.array(IM_matrix)

            except WindowsError: #error if there is not directory
                continue
            except IndexError: #empty folder
                continue
            #Texture(arguments).ret() -> function for texture calculation
            #arguments: status bar, image, stucture name, columns,rows, pixelSpacing, list of slice positions,  path to save the textutre maps, patient number, pixel discretization number or size, image modality, wavletes, contours, HU range, outlier_corr
            #function returns: number of removed points, minimum values, maximum values, structre used for calculations, mean,std, cov, skewness, kurtosis, enenrgy, entropy, contrast, corrrelation, homogenity, coarseness, neighContrast, busyness, complexity, intensity varation, size variation, fractal dimension, number of points used in the calculations, histogram (values bigger/smaller than median)
            stop_calc = '' #in case someting would be wrong with the image tags
            lista_results = Texture(sb,[IM_matrix], read.structure_f, read.columns, read.rows, read.xCTspace, read.slices, path_save, ImName, pixNr, binSize, image_modality, wv, local, cropInput, stop_calc, read.Xcontour, read.Xcontour_W, read.Ycontour, read.Ycontour_W, read.Xcontour_Rec, read.Ycontour_Rec, HUmin, HUmax, outlier_corr).ret()
            #v_min, v_max, organ, mean,std, cov, skew, kurt, var, median, percentile10, percentile90, iqr, Hrange, mad, rmad, H_energy, H_entropy, rms, H_uniformity,en, ent, con, cor, homo,homo_n, idiff, idiff_n, variance, average, sum_entropy, sum_variance, diff_entropy, diff_variance, IMC1, IMC2, MCC,  joint_max,  joint_average, diff_ave, dissim, inverse_var, autocorr, clust_t, clust_s, clust_p, coarse, neighCon, busy, comp,strength, len_inten, len_size, len_sse , len_lse, len_lgse, len_hgse, len_sslge, len_sshge, len_lslge, len_lshge, len_rpc, len_glv, len_lsv, len_size_entropy, inten, size, sse , lse, lgse, hgse, sslge, sshge, lslge, lshge, rpc, glv, lsv, size_entropy, frac, point = Texture(sb,[IM_matrix], structure, x_ct,y_ct, columns, rows, xCTspace, patientPos, rs, slices, path_save, ImName, pixNr, binSize, prefix, wv, None, None, None, None, HUmin, HUmax).ret()


            #final list contains of the sublist for each patient, sublist contains of [patient number, structure used for calcualtions, list of texture parameters, number of points used for calculcations]
            final = [[ImName, lista_results[2], lista_results[:2], lista_results[3:-1], lista_results[-1]]]

            final_file = Export().ExportResults(final, final_file, par_names, image_modality, wave_names, wv, local)

        final_file.close()
            #final.append([ImName, organ, [v_min, v_max], [mean,std, cov, skew, kurt, var, median, percentile10, percentile90, iqr, Hrange, mad, rmad, H_energy, H_entropy, rms, H_uniformity,en, ent, con, cor, homo,homo_n, idiff, idiff_n, variance, average, sum_entropy, sum_variance, diff_entropy, diff_variance, IMC1, IMC2, MCC,  joint_max,  joint_average, diff_ave, dissim, inverse_var, autocorr, clust_t, clust_s, clust_p, coarse, neighCon, busy, comp,strength, len_inten, len_size, len_sse , len_lse, len_lgse, len_hgse, len_sslge, len_sshge, len_lslge, len_lshge, len_rpc, len_glv, len_lsv, len_size_entropy, inten, size, sse , lse, lgse, hgse, sslge, sshge, lslge, lshge, rpc, glv, lsv, size_entropy, frac], point])
