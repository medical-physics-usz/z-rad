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
from structure import Structures
from exception import MyException
from read import ReadImageStructure
from export import Export

class main_texture_mr(object):
    '''
    Main class to handle MR images, reads images and structures, normalize MR image according to two specified ROI, calls radiomics calculation and export class to export results
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
    struct_norm1 � ROI1 for the normalization coefficients for MR image based on two normal structures (for example muscle and white matter) and linear function
    struct_norm2 � ROI2 for the normalization
    wv � bool, calculate wavelet
    exportList � list of matrices/features to be calculated and exported
    '''
    def __init__(self,sb, path_image, path_save, structure, pixNr, binSize, l_ImName, save_as, Dim, struct_norm1, struct_norm2, wv, local, cropStructure, exportList):
        final=[] # list with results
        image_modality = ['MR']
        dicomProblem = []
        for ImName in l_ImName:
            print 'patient', ImName
            try:
                mypath_image = path_image+ImName+'\\'
                MR_UID = ['1.2.840.10008.5.1.4.1.1.4'] #MR

                read = ReadImageStructure(MR_UID, mypath_image, structure, wv)

                dicomProblem.append([ImName, read.listDicomProblem])

                #MR intiensities normalization
                norm_slope, norm_inter = self.normalization(struct_norm1, struct_norm2, read, mypath_image)

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
                    data16 = data16*norm_slope+norm_inter
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
                #arguments: image, stucture name, image corner x, image corner x, columns, pixelSpacing, HFS or FFS, structure file, list of slice positions, patient number, path to save the textutre maps, map name (eg. AIF1), pixel discretization, site
                #function returns: number of removed points, minimum values, maximum values, structre used for calculations, mean,std, cov, skewness, kurtosis, enenrgy, entropy, contrast, corrrelation, homogenity, coarseness, neighContrast, busyness, complexity, intensity varation, size variation, fractal dimension, number of points used in the calculations, histogram (values bigger/smaller than median)
                stop_calc = '' #in case someting would be wrong with the image tags                
                if Dim == '3D':
                    lista_results = Texture(sb,[IM_matrix], read.structure_f, read.columns, read.rows, read.xCTspace, read.slices, path_save, ImName, pixNr, binSize, image_modality, wv, local, cropStructure, stop_calc, read.Xcontour, read.Xcontour_W, read.Ycontour, read.Ycontour_W).ret()
#                elif Dim == '2D': #not working
#                    lista_results = Texture2D(sb,IM_matrix, structure, x_ct,y_ct, columns, rows, xCTspace, patientPos, rs, slices, path_save, ImName, pixNr, prefix).ret()


            final.append([ImName, lista_results[2], lista_results[:2], lista_results[3:-1], lista_results[-1]])

            #final.append([ImName, organ, [v_min, v_max], [mean,std, cov, skew, kurt, var, median, percentile10, percentile90, iqr, Hrange, mad, rmad, H_energy, H_entropy, rms, H_uniformity,en, ent, con, cor, homo,homo_n, idiff, idiff_n, variance, average, sum_entropy, sum_variance, diff_entropy, diff_variance, IMC1, IMC2, MCC,  joint_max,  joint_average, diff_ave, dissim, inverse_var, autocorr, clust_t, clust_s, clust_p, coarse, neighCon, busy, comp,strength, len_inten, len_size, len_sse , len_lse, len_lgse, len_hgse, len_sslge, len_sshge, len_lslge, len_lshge, len_rpc, len_glv, len_lsv, len_size_entropy, inten, size, sse , lse, lgse, hgse, sslge, sshge, lslge, lshge, rpc, glv, lsv, size_entropy, frac], point])

        #save the results
        Export(final, exportList, wv,path_save, save_as, image_modality, path_image)

    def normalization(self, struct_norm1, struct_norm2, read, mypath_image):
        '''get the normalization coefficients for MR image based on two normal sturctures (for example muscle and white matter) and linear function'''
        struct1 = Structures(read.rs, [struct_norm1], read.slices, read.x_ct, read.y_ct, read.xCTspace, len(read.slices), False) #False for the wavelet argument
        norm1_Xcontour = struct1.Xcontour
        norm1_Ycontour = struct1.Ycontour

        struct2 = Structures(read.rs, [struct_norm2], read.slices, read.x_ct, read.y_ct, read.xCTspace, len(read.slices), False) #False for the wavelet argument
        norm2_Xcontour = struct2.Xcontour
        norm2_Ycontour = struct2.Ycontour

        #to read only slices where there is a contour
        ind = []
        for f in arange(0, len(norm1_Xcontour)):
            if norm1_Xcontour[f] != []:
                ind.append(f)
        for f in arange(0, len(norm2_Xcontour)):
            if norm2_Xcontour[f] != []:
                ind.append(f)

        zmin = np.min(ind)
        zmax = np.max(ind)

        onlyfiles = read.onlyfiles[zmin:zmax+1]
        read.slices = read.slices[zmin:zmax+1]
        norm1_Xcontour = norm1_Xcontour[zmin:zmax+1]
        norm1_Ycontour = norm1_Ycontour[zmin:zmax+1]
        norm2_Xcontour = norm2_Xcontour[zmin:zmax+1]
        norm2_Ycontour = norm2_Ycontour[zmin:zmax+1]

        IM_matrix = [] #list containing the images matrix
        for i in onlyfiles:
            data = dc.read_file(mypath_image+i).PixelData
            data16 = np.array(np.fromstring(data, dtype=np.int16)) #converitng to decimal
            #recalculating for rows x columns
            a=[]
            for j in arange(0, read.rows):
                a.append(data16[j*read.columns:(j+1)*read.columns])
            a=np.array(a)
            IM_matrix.append(np.array(a))

        IM_matrix = np.array(IM_matrix)

        v1 = [] #values for structure 1
        v2 = [] #values for structure 2

        for i in arange(0, len(norm1_Xcontour)): #slices
            for j in arange(0, len(norm1_Xcontour[i])): #sub-structres in the slice
                for k in arange(0, len(norm1_Xcontour[i][j])):
                    v1.append(IM_matrix[i][norm1_Ycontour[i][j][k]][norm1_Xcontour[i][j][k]])

        for i in arange(0, len(norm2_Xcontour)): #slices
            for j in arange(0, len(norm2_Xcontour[i])): #sub-structres in the slice
                for k in arange(0, len(norm2_Xcontour[i][j])):
                    v2.append(IM_matrix[i][norm2_Ycontour[i][j][k]][norm2_Xcontour[i][j][k]])

        f1 = np.mean(v1)
        f2 = np.mean(v2)

        fa = (800-300)/(f1-f2) #find coefficients of a linear function
        fb = 800 - f1*fa

        return fa, fb
