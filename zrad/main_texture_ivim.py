# -*- coding: utf-8 -*-
'''read data and save texture parameters in txt file'''
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
import scipy.stats as st

#own classes
#import class to calculate texture parameters
from texture import Texture
from structure import Structures
from exception import MyException
from read import ReadImageStructure
from export import Export

class main_texture_ivim(object):
    '''
    To check - new data reading as I didn't have examplary data to test
    Main class to handle IVIM images, reads images and structures and calls radiomics calculation and export class to export results
    Type: object
    Attributes: 
    sb – Status bar in the frame 
    path_image – path to the patients subfolders
    path_save – path to save radiomics results
    structure – list of structures to be analysed
    pixNr number of analyzed bins, if not specified  = none
    binSize – bin size for the analysis, if not specified = none
    l_ImName – list of patients subfolders (here are data to be analysed)
    save_as – name of text files to save the radiomics results
    Dim – string variable of value 2D or 3D for dimensionality of calculation
    outlier – bool, correct for outliers
    wv – bool, calculate wavelet  
    exportList – list of matrices/features to be calculated and exported
    '''
    def __init__(self, sb, path_image, path_save, structure, pixNr, binSize, l_ImName, save_as, dim, outlier_corr,wv,local, cropStructure, exportList):
        final=[] # list with results
        image_modality = ['DSlow2', 'DFast2', 'F2']
        dicomProblem = []
        slope_list = [10**(-6),10**(-4),10**(-3)] #'dslow', 'dfast', 'F2' ocrresponding slopes to different maps
        for ImName in l_ImName:
            print('patient', ImName)
            try:
                sb.SetStatusText('Load '+ImName)

                mypath_image = path_image+ImName+'\\'+prefix[m_name]+"\\"
                UID = ['IVIM']
                
                read = ReadImageStructure(UID, mypath_image, structure, wv, None, image_modality)  # none for dimension

                dicomProblem.append([ImName, read.listDicomProblem])   
                
                l_IM_matrix = [] #list containing different perfusion maps
                for m_name in arange(0, len(image_modality)):
                    IM_matrix = [] #list containing the images matrix
                    for f in read.onlyfiles[m_name]:
                        data = dc.read_file(mypath_image+f).PixelData
                        data16 = np.array(np.fromstring(data, dtype=np.int16)) #converitng to decimal
                        data16 = data16*slope_list[m_name]
                        #recalculating for rows x columns
                        a=[]
                        for j in arange(0, read.rows):#(0, rows):
                            a.append(data16[j*read.columns:(j+1)*read.columns])
                        a=np.array(a)
                        IM_matrix.append(np.array(a))
                    IM_matrix = np.array(IM_matrix)

                    l_IM_matrix.append(IM_matrix)
                    
            except WindowsError: #error if there is not directory
                continue
            except IndexError: #file not in directory
                continue
            
                #for m_names in arange(0, len(prefix)):
                    #Texture(arguments).ret() -> function for texture calculation
                    #arguments: image, stucture name, image corner x, image corner x, columns, pixelSpacing, HFS or FFS, structure file, list of slice positions, patient number, path to save the textutre maps, map name (eg. AIF1), pixel discretization, site
                    #function returns: number of removed points, minimum values, maximum values, structre used for calculations, mean,std, cov, skewness, kurtosis, enenrgy, entropy, contrast, corrrelation, homogenity, coarseness, neighContrast, busyness, complexity, intensity varation, size variation, fractal dimension, number of points used in the calculations, histogram (values bigger/smaller than median)
            sb.SetStatusText('Calculate '+ImName)
            stop_calc = '' #in case someting would be wrong with the image tags
            lista_results = Texture(sb,l_IM_matrix, read.structure_f, read.columns, read.rows, read.xCTspace, read.slices, path_save, ImName, pixNr, binSize, image_modality, wv,local, cropStructure,stop_calc, read.Xcontour, read.Xcontour_W, read.Ycontour, read.Ycontour_W).ret()

            #final list contains of the sublist for each patient, sublist contains of [patient number, structure used for calcualtions, list of texture parameters, number of points used for calculcations]final.append([ImName, organ, lista_results[:2],
            final.append([ImName, lista_results[2], lista_results[:2], lista_results[3:-1], lista_results[-1]])

        #save the results
        Export(final, exportList, wv,path_save, save_as, image_modality, path_image)        

        