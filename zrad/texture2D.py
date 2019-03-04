# -*- coding: utf-8 -*-
'''calculated the co matirxin 3D in all possible directions and average the results'''
import numpy as np
from numpy import arange,floor
import matplotlib
import pylab as py
import cv2 # interactive plots
import dicom as dc
import matplotlib
from scipy.fftpack import fft2, fftn, fftfreq
##from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as optimization
import scipy.stats as st
from os import makedirs
from os.path import isdir
from datetime import datetime

from texture_wavelet3D import Wavelet
##from texture_rs import Read_rs

##from margin4 import Margin
##from edge import Edge

class Texture(object):
    def __init__(self, IM,structure, x_ct,y_ct, columns, rows, xCTspace, patientPos, rs, slices, path, ImName, pixNr, prefix):
        self.IM = IM
        self.structure = structure
        self.x_ct = x_ct #image corner
        self.y_ct = y_ct
        self.xCTspace = xCTspace #image resolution
        self.columns = columns #columns
        self.rows = rows
        self.slices = slices #slice location
        self.mean = [] #list of texture parameters in the final verion contains [mean BF, mean MTT, mean BV]
        self.std = []
        self.cov = []
        self.skewness = []
        self.kurtosis = []
        self.energy = []
        self.entropy = []
        self.contrast = []
        self.correlation = []
        self.homogenity = []
        self.variance = []
        self.average = []
        self.sum_entropy = []
        self.sum_variance = []
        self.diff_entropy = []
        self.diff_variance = []
        self.IMC1 = []
        self.IMC2 = []
        self.MCC = [] 
        self.coarseness = []
        self.neighContrast = []
        self.busyness = []
        self.complexity = []
        self.intensity = []
        self.size = []
        self.sse = []
        self.lse = []
        self.lgse = []
        self.hgse = []
        self.sslge = []
        self.sshge = []
        self.lslge = []
        self.lshge = []
        self.rpc = []
        self.organs_n=[]
        self.points = []
        self.frac_dim = []
        self.bits = float(pixNr)
        self.position =patientPos
        perfusionMatrix=[]
        self.vmin = []
        self.vmax = []

        try:
            maps = [self.IM]
            names = [prefix[:-1]]
            #read a structure
            if structure != 'none' and names not in ['BV', 'BF', 'MTT']:
                self.structures(rs, structure) #read the structure points
                print 'RS done', datetime.now().strftime('%H:%M:%S')
                
            for i in arange(0, len(maps)):
                #wavelet transform
                wave_list = Wavelet(maps[i], path, names[i], ImName+'_'+pixNr).Return() #original, HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL
                rs_type = [1, 0, 0, 0, 0, 0, 0, 0, 0]
                for w in arange(0, len(wave_list)):
                    try:
                        #matrix - contains only the discretized pixels in the structre
                        #interval - one bin
                        #norm_points - how many points are used for the calculations
                        #matrix_v - contains only the pixels in the structre, original values
                        matrix, interval, norm_points, matrix_v = self.Matrix(wave_list[w], rs_type[w], structure)#, self.vmin[i], self.vmax[i])
                        print 'matrix done', datetime.now().strftime('%H:%M:%S')
                        #histogram
                        histogram = self.fun_histogram(matrix_v, names[i], ImName, pixNr, path, rs_type[w])
                        mean = self.fun_mean(histogram)
                        std = self.fun_std(histogram)
                        cov = self.fun_COV(histogram)
                        skewness = self.fun_skewness(histogram)
                        kurtosis = self.fun_kurtosis(histogram)
                        del histogram
                        print 'hist done', datetime.now().strftime('%H:%M:%S')
                        #coocurence matrix
                        norm_co = []
                        energy=[]
                        entropy=[]
                        contrast=[]
                        correlation =[]
                        homogenity = []
                        variance = []
                        average = []
                        sum_entropy = []
                        sum_variance = []
                        diff_entropy = []
                        diff_variance = []
                        IMC1 = []
                        IMC2 = []
                        MCC = []
                        #neighborhood matrix
                        norm_neigh = []
                        neighSum=0 #because in 2D it's more probable to get the 0 difference in one slice, which results in the end in to big coarseness
                        coarseness = []
                        neighContrast = []
                        busyness = []
                        complexity = []
                        # gray level size zone matrix
                        norm_M4 = []
                        intensity = []
                        size = []
                        sse = []
                        lse = []
                        lgse = []
                        hgse = []
                        sslge = []
                        sshge = []
                        lslge = []
                        lshge = []
                        rpc = []
                        norm_f=[]
                        frac = []
                        for m in arange(0, len(matrix)):
                            energy_t=[]
                            entropy_t=[]
                            contrast_t=[]
                            correlation_t =[]
                            homogenity_t = []
                            variance_t = []
                            average_t = []
                            sum_entropy_t = []
                            sum_variance_t = []
                            diff_entropy_t = []
                            diff_variance_t = []
                            IMC1_t = []
                            IMC2_t = []
                            MCC_t = []
                            lista_t = [[0,1],[0,-1],[1,0],[-1,0],[-1,1],[1,1],[1,-1],[-1,-1]]
                            for c in lista_t:
                                co_matrix, norm = self.coMatrix(matrix[m], c)
                                energy_t.append(self.fun_energy(co_matrix))
                                entropy_t.append(self.fun_entropy(co_matrix))
                                contrast_t.append(self.fun_contrast(co_matrix))
                                correlation_t.append(self.fun_correlation(co_matrix))
                                homogenity_t.append(self.fun_homogenity(co_matrix))
                                variance_t.append(self.fun_variance(co_matrix))
                                average_t.append(self.fun_average(co_matrix))
                                e, v = self.fun_sum_entropy_var(co_matrix)
                                sum_entropy_t.append(e)
                                sum_variance_t.append(v)
                                e, v = self.fun_diff_entropy_var(co_matrix)
                                diff_entropy_t.append(e)
                                diff_variance_t.append(v)
                                f1, f2 = self.fun_IMC(co_matrix)
                                IMC1_t.append(f1)
                                IMC2_t.append(f2)
                                MCC_t.append(self.fun_MCC(co_matrix))
                                del co_matrix
                            norm_co.append(norm)
                            energy.append(np.mean(energy_t))
                            entropy.append(np.mean(entropy_t))
                            contrast.append(np.mean(contrast_t))
                            correlation.append(np.mean(correlation_t))
                            homogenity.append(np.mean(homogenity_t))
                            variance.append(np.mean(variance_t))
                            average.append(np.mean(average_t))
                            sum_entropy.append(np.mean(sum_entropy_t))
                            sum_variance.append(np.mean(sum_variance_t))
                            diff_entropy.append(np.mean(diff_entropy_t))
                            diff_variance.append(np.mean(diff_variance_t))
                            IMC1.append(np.mean(IMC1_t))
                            IMC2.append(np.mean(IMC2_t))
                            MCC.append(np.mean(MCC_t))
                            #neighborhood matrix
                            neighMatrix, neighMatrixNorm, n  = self.M3(matrix[m])
                            neighSum+=n
                            norm_neigh.append(np.sum(neighMatrixNorm))
                            coarseness.append(self.fun_coarseness(neighMatrix, neighMatrixNorm, matrix))
                            neighContrast.append(self.fun_contrastM3(neighMatrix, neighMatrixNorm, matrix))
                            busyness.append(self.fun_busyness(neighMatrix, neighMatrixNorm, matrix))
                            complexity.append(self.fun_complexity(neighMatrix, neighMatrixNorm, matrix))
                            del neighMatrix
                            del neighMatrixNorm
                            # gray level size zone matrix
                            GLSZM, norm = self.M4(matrix[m])
                            norm_M4.append(norm)
                            intensity.append(self.intensityVariability(GLSZM))
                            size.append(self.sizeVariability(GLSZM))
                            sse.append(self.shortSize(GLSZM))
                            lse.append(self.longSize(GLSZM))
                            lgse.append(self.LGSE(GLSZM))
                            hgse.append(self.HGSE(GLSZM))
                            sslge.append(self.SSLGE(GLSZM))
                            sshge.append(self.SSHGE(GLSZM))
                            lslge.append(self.LSLGE(GLSZM))
                            lshge.append(self.LSHGE(GLSZM))
                            rpc.append(self.runPer(GLSZM))
                            fra, norm = self.fractal(matrix[m], path, pixNr, ImName)
                            norm_f.append(norm)                                                 
                            frac.append(fra)
                            del GLSZM
                            
                        norm_co=np.array(norm_co, dtype=np.float)/np.sum(norm_co)
                        energy = np.sum(np.array(energy)*np.array(norm_co))
                        entropy = np.sum(np.array(entropy)*np.array(norm_co))
                        contrast = np.sum(np.array(contrast)*np.array(norm_co))
                        correlation = np.sum(np.array(correlation)*np.array(norm_co))
                        homogenity = np.sum(np.array(homogenity)*np.array(norm_co))
                        variance = np.sum(np.array(variance)*np.array(norm_co))
                        average = np.sum(np.array(average)*np.array(norm_co))
                        sum_entropy = np.sum(np.array(sum_entropy)*np.array(norm_co))
                        sum_variance = np.sum(np.array(sum_variance)*np.array(norm_co))
                        diff_entropy = np.sum(np.array(diff_entropy)*np.array(norm_co))
                        diff_variance = np.sum(np.array(diff_variance)*np.array(norm_co))
                        IMC1 = np.sum(np.array(IMC1_t))
                        IMC2 = np.sum(np.array(IMC2_t))
                        MCC = np.sum(np.array(MCC_t))
                        #neighborhood matrix
                        norm_neigh = np.array(norm_neigh, dtype=np.float)/neighSum
                        coarseness = np.sum(np.array(coarseness)*np.array(norm_neigh))
                        neighContrast = np.sum(np.array(neighContrast)*np.array(norm_neigh))
                        busyness = np.sum(np.array(busyness)*np.array(norm_neigh))
                        complexity = np.sum(np.array(complexity)*np.array(norm_neigh))
                        # gray level size zone matrix
                        norm_M4 = np.array(norm_M4, dtype=np.float)/np.sum(norm_M4)
                        intensity = np.sum(np.array(intensity)*np.array(norm_M4))
                        size = np.sum(np.array(size)*np.array(norm_M4))
                        sse = np.sum(np.array(sse)*np.array(norm_M4))
                        lse = np.sum(np.array(lse)*np.array(norm_M4))
                        lgse = np.sum(np.array(lgse)*np.array(norm_M4))
                        hgse = np.sum(np.array(hgse)*np.array(norm_M4))
                        sslge = np.sum(np.array(sslge)*np.array(norm_M4))
                        sshge = np.sum(np.array(sshge)*np.array(norm_M4))
                        lslge = np.sum(np.array(lslge)*np.array(norm_M4))
                        lshge = np.sum(np.array(lshge)*np.array(norm_M4))
                        rpc = np.sum(np.array(rpc)*np.array(norm_M4))
                        norm_f = np.array(norm_f, dtype=np.float)/np.sum(norm_f)
                        frac = np.sum(np.array(frac)*np.array(norm_f))

                        print 'mean: ', round(mean, 3)
                        print 'std: ', round(std, 3)
                        print 'COV: ', round(cov, 3)
                        print 'skewness: ', round(skewness, 3)
                        print 'kurtosis: ', round(kurtosis, 3)
                        print 'energy: ', round(energy, 3)
                        print 'entropy: ', round(entropy, 3)
                        print 'contrast: ', round(contrast, 3)
                        print 'correlation: ', round(correlation,3)
                        print 'homogenity: ', round(homogenity,3)
                        print 'points:', np.sum(norm_points)
                        print 'neighborhood coarseness: ', round(coarseness, 3)
                        try:
                            print 'neighborhood contrast: ', round(neighContrast, 3)
                        except TypeError:
                            print 'neighborhood contrast: '
                        try:
                            print 'neighborhood busyness: ', round(busyness, 3)
                        except TypeError:
                            print 'neighborhood busyness: '
                        print 'neighborhood complexity: ', round(complexity, 3)
                        print 'intensity variability: ', round(intensity, 3)
                        print 'size variability: ', round(size, 3)
                        self.mean.append(round(mean, 3))
                        self.std.append(round(std, 3))
                        self.cov.append(round(cov, 3))
                        self.skewness.append(round(skewness, 3))
                        self.kurtosis.append(round(kurtosis, 3))
                        self.energy.append(round(energy, 3))
                        self.entropy.append(round(entropy, 3))
                        self.contrast.append(round(contrast, 3))
                        self.correlation.append(round(correlation, 3))
                        self.homogenity.append(round(homogenity,3))
                        self.variance.append(round(variance,3))
                        self.average.append(round(average,3))
                        self.sum_entropy.append(round(sum_entropy,3))
                        self.sum_variance.append(round(sum_variance,3))
                        self.diff_entropy.append(round(diff_entropy,3))
                        self.diff_variance.append(round(diff_variance,3))
                        try:
                            self.IMC1.append(round(IMC1,3))
                        except TypeError:
                            self.IMC1.append('')
                        self.IMC2.append(round(IMC2,3))
                        self.MCC.append(round(MCC,3))
                        self.coarseness.append(round(coarseness,3))
                        try:
                            self.neighContrast.append(neighContrast)
                        except TypeError:
                            self.neighContrast.append('')
                        try:
                            self.busyness.append(busyness)
                        except TypeError:
                            self.busyness.append('')
                        self.complexity.append(round(complexity,3))
                        self.intensity.append(round(intensity, 3))
                        self.size.append(round(size, 3))
                        self.sse.append(round(sse, 3))
                        self.lse.append(round(lse, 3))
                        self.lgse.append(round(lgse, 3))
                        self.hgse.append(round(hgse, 3))
                        self.sslge.append(round(sslge, 3))
                        self.sshge.append(round(sshge, 3))
                        self.lslge.append(round(lslge, 3))
                        self.lshge.append(round(lshge, 3))
                        self.rpc.append(round(rpc, 3))
                        try:
                            self.frac_dim.append(round(frac, 3))
                        except TypeError:
                            self.frac_dim.append('')
                        self.points.append(norm_points[0])
                        if rs_type[w]:
                            self.saveImage(path, names[i], matrix, i, ImName, pixNr)
                        del matrix
    ##                except NameError:
    ##                    print 'name'
    ##                except IndexError:
    ##                    print IndexError
    ##                    self.mean.append('')
    ##                    self.std.append('')
    ##                    self.cov.append('')
    ##                    self.skewness.append('')
    ##                    self.kurtosis.append('')
    ##                    self.energy.append('')
    ##                    self.entropy.append('')
    ##                    self.contrast.append('')
    ##                    self.correlation.append('')
    ##                    self.coarseness.append('')
    ##                    self.neighContrast.append('')
    ##                    self.busyness.append('')
    ##                    self.complexity.append('')
    ##                    self.points.append('')
    ##                    self.homogenity.append('')
    ##                    self.intensity.append('')
    ##                    self.size.append('')
    ##                    self.sse.append('')
    ##                    self.lse.append('')
    ##                    self.lgse.append('')
    ##                    self.hgse.append('')
    ##                    self.sslge.append('')
    ##                    self.sshge.append('')
    ##                    self.lslge.append('')
    ##                    self.lshge.append('')
    ##                    self.rpc.append('')
    ##                    perfusionMatrix.append('')
    ##                    self.frac_dim.append('')
                    except ValueError:#ValueError:
                        print ValueError
                        self.mean.append('')
                        self.std.append('')
                        self.cov.append('')
                        self.skewness.append('')
                        self.kurtosis.append('')
                        self.energy.append('')
                        self.entropy.append('')
                        self.contrast.append('')
                        self.correlation.append('')
                        self.coarseness.append('')
                        self.neighContrast.append('')
                        self.busyness.append('')
                        self.complexity.append('')
                        self.points.append('')
                        self.homogenity.append('')
                        self.variance.append('')
                        self.average.append('')
                        self.sum_entropy.append('')
                        self.sum_variance.append('')
                        self.diff_entropy.append('')
                        self.diff_variance.append('')
                        self.IMC1.append('')
                        self.IMC2.append('')
                        self.MCC.append('')
                        self.intensity.append('')
                        self.size.append('')
                        self.sse.append('')
                        self.lse.append('')
                        self.lgse.append('')
                        self.hgse.append('')
                        self.sslge.append('')
                        self.sshge.append('')
                        self.lslge.append('')
                        self.lshge.append('')
                        self.rpc.append('')
                        perfusionMatrix.append('')
                        self.frac_dim.append('')
        except ValueError:#IndexError:
            print IndexError
            for i in arange(0, len(maps)):
                self.mean.append('')
                self.std.append('')
                self.cov.append('')
                self.skewness.append('')
                self.kurtosis.append('')
                self.energy.append('')
                self.entropy.append('')
                self.contrast.append('')
                self.correlation.append('')
                self.points.append('')
                self.coarseness.append('')
                self.neighContrast.append('')
                self.busyness.append('')
                self.complexity.append('')
                self.homogenity.append('')
                self.variance.append('')
                self.average.append('')
                self.sum_entropy.append('')
                self.sum_variance.append('')
                self.diff_entropy.append('')
                self.diff_variance.append('')
                self.IMC1.append('')
                self.IMC2.append('')
                self.MCC.append('')
                self.intensity.append('')
                self.size.append('')
                self.sse.append('')
                self.lse.append('')
                self.lgse.append('')
                self.hgse.append('')
                self.sslge.append('')
                self.sshge.append('')
                self.lslge.append('')
                self.lshge.append('')
                self.rpc.append('')
                perfusionMatrix.append('')
                self.frac_dim.append('')
                                
##        self.Save()
##        self.Plot()
##        self.ReturnCon()
##        wynik = VolumesComparison(self.contours[0], self.segment, len(self.PET_im_small[0])).Return()
##        #print wynik
        #self.hist = self.voxelPerfusion(perfusionMatrix)
    def ret(self):
        print 'ret'                              
        return self.vmin, self.vmax, self.organs_n, self.mean, self.std, self.cov, self.skewness, self.kurtosis, self.energy, self.entropy, self.contrast, self.correlation, self.homogenity, self.variance, self.average, self.sum_entropy, self.sum_variance, self.diff_entropy, self.diff_variance, self.IMC1, self.IMC2, self.MCC, self.coarseness, self.neighContrast, self.busyness, self.complexity, self.intensity, self.size, self.sse , self.lse, self.lgse, self.hgse, self.sslge, self.sshge, self.lslge, self.lshge, self.rpc, self.frac_dim, self.points

    def structures(self, rs, structures):
        print 'rs', rs
        self.rs = dc.read_file(rs)
        list_organs =[]
        print 'structures:'
        for j in arange(0, len(self.rs.StructureSetROISequence)):
            list_organs.append([self.rs.StructureSetROISequence[j].ROIName, self.rs.StructureSetROISequence[j].ROINumber])
            print self.rs.StructureSetROISequence[j].ROIName

        organs = [structures] #define the structure you're intersed in

        self.contours=[] #list with structure contours
        #orientation of z axis
        if self.position == 'HFS':
            pos = 1
        elif self.position == 'FFS':
            pos = -1
        print self.position
        #search for organ I'm intersted in
        for i in arange(0, len(organs)):
            for j in arange(0, len(list_organs)):
                if list_organs[j][0] == organs[i]:
                    for k in arange(0, len(self.rs.ROIContourSequence)):
                        if self.rs.ROIContourSequence[k].ReferencedROINumber == list_organs[j][1]:
                            try:
                                lista = []
                                for l in arange(0, len(self.rs.ROIContourSequence[k].ContourSequence)):
                                    lista.append([float(pos*self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[2]), self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[::3], self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[1::3]])
                                lista.sort()
                                index = []
                                lista = self.multiContour(lista) #subcontrous in the slice
                                for m in arange(0, len(lista)):
                                    index.append(lista[m][0])
                                print index
                                print self.slices
                                diffI = round(index[1]-index[0], 1)
                                diffS = round(self.slices[1]-self.slices[0],1)
                                print diffI, diffS
                                if diffI != diffS:
                                    index.reverse()
                                    lista.reverse()
                                sliceB=index[-1]
                                sliceE=index[0]
                                indB = np.where(np.array(self.slices) == sliceB)[0][0]
                                indE = np.where(np.array(self.slices) == sliceE)[0][0]
                                if indE!=0:
                                    for m in arange(0, abs(indE-0)):
                                        lista.insert(0,[[],[[],[]]])
                                if indB!=(len(self.slices)-1):
                                    for m in arange(0, abs(indB-(len(self.slices)-1))):
                                        lista.append([[],[[],[]]])
                                for n in arange(0, len(lista)):
                                    lista[n] = lista[n][1:]
                                self.contours.append(lista)
                                self.organs_n.append(list_organs[j][0])
                                break
                            except AttributeError:
                                print 'no contours for: '+ organs[i]
        self.organs = organs
        #recalculating for pixels
        self.cnt=[]
        self.contours = np.array(self.contours)
        
        for i in arange(0, len(self.contours)): #controus
            for j in arange(0, len(self.contours[i])): #slice
                for n in arange(0, len(self.contours[i][j])): #number of contours per slice
                    if self.contours[i][j][n][0]!=[]:
                        self.contours[i][j][n][0]=np.array(abs(self.contours[i][j][n][0]-self.x_ct)/(self.xCTspace)) #from the interpolation in the CTP calculation
                        self.contours[i][j][n][1]=np.array(abs(self.contours[i][j][n][1]-self.y_ct)/(self.xCTspace))
                        for k in arange(0, len(self.contours[i][j][n][0])):
                            self.contours[i][j][n][0][k] = int(round(self.contours[i][j][n][0][k],0))
                            self.contours[i][j][n][1][k] = int(round(self.contours[i][j][n][1][k],0))
                        self.contours[i][j][n][0] = np.array(self.contours[i][j][n][0], dtype=np.int)
                        self.contours[i][j][n][1] = np.array(self.contours[i][j][n][1], dtype=np.int)

        x_c_min = [] #x position of contour points to define the region of interest where we look for the structure
        x_c_max = [] 
        y_c_min = []
        y_c_max = []
        for i in arange(0, len(self.contours)): #controus
            for j in arange(0, len(self.contours[i])): #slice
                for n in arange(0, len(self.contours[i][j])): #number of contours per slice
                    if self.contours[i][j][n][0]!=[]:
                        x_c_min.append(np.min(self.contours[i][j][n][0]))
                        x_c_max.append(np.max(self.contours[i][j][n][0]))
                        y_c_min.append(np.min(self.contours[i][j][n][1]))
                        y_c_max.append(np.max(self.contours[i][j][n][1]))

        x_min = np.min(x_c_min)
        x_max = np.max(x_c_max)
        y_min = np.min(y_c_min)
        y_max = np.max(y_c_max)

        print x_min
        print x_max
        print y_min 
        print y_max

        del x_c_min 
        del x_c_max 
        del y_c_min 
        del y_c_max 

        #finding points inside the contour           
        Xcontour=[]
        Ycontour=[]
        X, Y, cnt = self.getPoints(self.contours[0], x_min, x_max, y_min, y_max)
        Xcontour.append(X)
        Ycontour.append(Y)
        self.Xcontour=Xcontour[0]
        self.Ycontour=Ycontour[0]

        del self.contours
        self.contours=[]
        
        self.slices_w = list(np.array(self.slices).copy())
        if self.slices_w[0]-self.slices_w[1] < 0:
            self.slices_w.insert(0, self.slices[0]-2*abs(self.slices[0]-self.slices[1]))
            self.slices_w.insert(1, self.slices[0]-abs(self.slices[0]-self.slices[1]))
            self.slices_w.append(self.slices[-1]+abs(self.slices[0]-self.slices[1]))
            self.slices_w.append(self.slices[-1]+2*abs(self.slices[0]-self.slices[1]))
            self.slices_w.append(self.slices[-1]+3*abs(self.slices[0]-self.slices[1]))
            self.slices_w.append(self.slices[-1]+4*abs(self.slices[0]-self.slices[1]))
        else:
            self.slices_w.insert(0, self.slices[0]+2*abs(self.slices[0]-self.slices[1]))
            self.slices_w.insert(1, self.slices[0]+abs(self.slices[0]-self.slices[1]))
            self.slices_w.append(self.slices[-1]-abs(self.slices[0]-self.slices[1]))
            self.slices_w.append(self.slices[-1]-2*abs(self.slices[0]-self.slices[1]))
            self.slices_w.append(self.slices[-1]-3*abs(self.slices[0]-self.slices[1]))
            self.slices_w.append(self.slices[-1]-4*abs(self.slices[0]-self.slices[1]))
        for i in arange(0, len(self.slices_w)):
            self.slices_w[i] = round(self.slices_w[i],1)

        #get the points in the contour, as previously      
        for i in arange(0, len(organs)):
            for j in arange(0, len(list_organs)):
                if list_organs[j][0] == organs[i]:
                    for k in arange(0, len(self.rs.ROIContourSequence)):
                        if self.rs.ROIContourSequence[k].ReferencedROINumber == list_organs[j][1]:
                            try:
                                lista = []
                                for l in arange(0, len(self.rs.ROIContourSequence[k].ContourSequence)):
                                    lista.append([float(pos*self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[2]), self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[::3], self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[1::3]])
                                lista.sort()
                                index = []
                                lista = self.multiContour(lista) #subcontrous in the slice
                                print len(lista)
                                for m in arange(0, len(lista)):
                                    index.append(round(lista[m][0],1))
                                print 'index', index
                                print self.slices_w
                                diffI = round(index[1]-index[0], 1)
                                diffS = round(self.slices_w[1]-self.slices_w[0],1)
                                if diffI != diffS:
                                    index.reverse()
                                    lista.reverse()
                                sliceB=index[-1]
                                sliceE=index[0]
                                indB = np.where(np.array(self.slices_w) == sliceB)[0][0]
                                indE = np.where(np.array(self.slices_w) == sliceE)[0][0]
                                if indE!=0:
                                    for m in arange(0, abs(indE-0)):
                                        lista.insert(0,[[],[[],[]]])
                                if indB!=(len(self.slices_w)-1):
                                    for m in arange(0, abs(indB-(len(self.slices_w)-1))):
                                        lista.append([[],[[],[]]])
                                self.slices_w = self.slices_w[::2] #take only every second slice, a the resolution drops down after the transform
                                lista = lista[::2]
                                for n in arange(0, len(lista)):
                                    lista[n] = lista[n][1:]
                                self.contours.append(lista)
                                break
                            except AttributeError:
                                print 'no contours for: '+ organs[i]

        self.organs = organs
        
        #recalculating for pixels
        self.cnt=[]
        x_ct = self.x_ct - 2*self.xCTspace
        y_ct = self.y_ct - 2*self.xCTspace
        for i in arange(0, len(self.contours)): #controus
            for j in arange(0, len(self.contours[i])): #slice
                for n in arange(0, len(self.contours[i][j])): #number of contours per slice
                    if self.contours[i][j][n][0]!=[]:
                        self.contours[i][j][n][0]=np.array(abs(self.contours[i][j][n][0]-x_ct)/(2*self.xCTspace)) #from the interpolation in the CTP calculation
                        self.contours[i][j][n][1]=np.array(abs(self.contours[i][j][n][1]-y_ct)/(2*self.xCTspace))
                        for k in arange(0, len(self.contours[i][j][n][0])):
                            self.contours[i][j][n][0][k] = int(round(self.contours[i][j][n][0][k],0))
                            self.contours[i][j][n][1][k] = int(round(self.contours[i][j][n][1][k],0))
                        self.contours[i][j][n][0] = np.array(self.contours[i][j][n][0], dtype=np.int)
                        self.contours[i][j][n][1] = np.array(self.contours[i][j][n][1], dtype=np.int)

        x_c_min = [] #x position of contour points to define the region of interest where we look for the structure
        x_c_max = [] 
        y_c_min = []
        y_c_max = []
        for i in arange(0, len(self.contours)): #controus
            for j in arange(0, len(self.contours[i])): #slice
                for n in arange(0, len(self.contours[i][j])): #number of contours per slice
                    if self.contours[i][j][n][0]!=[]:
                        x_c_min.append(np.min(self.contours[i][j][n][0]))
                        x_c_max.append(np.max(self.contours[i][j][n][0]))
                        y_c_min.append(np.min(self.contours[i][j][n][1]))
                        y_c_max.append(np.max(self.contours[i][j][n][1]))

        x_min = np.min(x_c_min)
        x_max = np.max(x_c_max)
        y_min = np.min(y_c_min)
        y_max = np.max(y_c_max)

        del x_c_min 
        del x_c_max 
        del y_c_min 
        del y_c_max

        print x_min
        print x_max
        print y_min 
        print y_max
              
        Xcontour=[]
        Ycontour=[]
        X, Y, cnt = self.getPoints_w(self.contours[0], x_min, x_max, y_min, y_max)
        Xcontour.append(X)
        Ycontour.append(Y)
        self.Xcontour_W=Xcontour[0]
        self.Ycontour_W=Ycontour[0]

    def key_event(self, e):
        self.curr_pos

        if e.key == "6":
            self.curr_pos = self.curr_pos + 1 #moving forth
        elif e.key == "4":
            self.curr_pos = self.curr_pos - 1 #moving back
        else:
            return
        self.curr_pos = self.curr_pos % len(self.slices) #modulo opertaion to have always curr_pos smaller than the number of images

        self.ax.cla()
        self.ax.imshow(self.BV[self.curr_pos], cmap = py.cm.jet)
        for i in arange(0, len(self.contours[0][ self.curr_pos ])):
            self.ax.plot(self.contours[0][ self.curr_pos ][i][0], self.contours[0][ self.curr_pos ][i][1],'green', linewidth=2)
        self.ax.set_title(self.slices[self.curr_pos])
        self.fig.canvas.draw()

    def multiContour(self, lista):
        listap=[]
        lista_nr=[]
        for i in lista:
            listap.append(i[0])
            if i[0] not in lista_nr:
                lista_nr.append(i[0])
        counts = []
        for i in lista_nr:
            counts.append(listap.count(i))

        listka=[]
        nr=0
        kontur = []
        for i in arange(0, len(lista)):
            if lista[i][0] not in listka:
                m=[lista[i][0]]
                for j in arange(0, counts[nr]):
                    m.append([np.array(lista[i+j][1], dtype=np.float), np.array(lista[i+j][2], dtype=np.float)])
                    listka.append(lista[i][0])
                kontur.append(m)
                nr+=1
        return kontur

    def getPoints(self, segment, xmin, xmax, ymin, ymax):
        '''get points inside the contour'''
        cnt_all = []
        for k in arange(0, len(self.IM)):
            cnt = []
            for i in arange(0, len(segment[k])):
                c = []
                for j in arange(0, len(segment[k][i][0])):
                    c.append([segment[k][i][0][j], segment[k][i][1][j]])
                cnt.append(c)
            if cnt==[]:
                cnt=[[],[]]
            cnt_all.append(cnt)
            
        Xp=[]
        Yp=[]
        for k in arange(0, len(self.IM)):
            xp=[]
            yp=[]
            if cnt_all[k]!=[[]]:
                for n in arange(0, len(cnt_all[k])):
                    x=[]
                    y=[]
                    for i in arange(ymin, ymax+1):
                        for j in np.arange(xmin, xmax+1):
                            if cv2.pointPolygonTest(np.array(cnt_all[k][n]), (j,i),False) !=-1:
                                x.append(j)
                                y.append(i)
                    xp.append(x)
                    yp.append(y)
                Xp.append(xp)
                Yp.append(yp)
            else:
                Xp.append([])
                Yp.append([])
        return Xp, Yp, cnt_all

    def getPoints_w(self, segment):  
        cnt_all = []
        for k in arange(0, int(floor((len(self.BV)+5)/2.))):
            cnt = []
            for i in arange(0, len(segment[k])):
                c = []
                for j in arange(0, len(segment[k][i][0])):
                    c.append([segment[k][i][0][j], segment[k][i][1][j]])
                cnt.append(c)
            if cnt==[]:
                cnt=[[],[]]
            cnt_all.append(cnt)
            
        Xp=[]
        Yp=[]
        for k in arange(0, int(floor((len(self.BV)+5)/2.))):
            xp=[]
            yp=[]
            if cnt_all[k]!=[[]]:
                for n in arange(0, len(cnt_all[k])):
                    x=[]
                    y=[]
                    for i in arange(0, int(floor((len(self.BV[k])+5)/2.))):
                        for j in np.arange(0, int(floor((len(self.BV[k][0])+5)/2.))):
                            if cv2.pointPolygonTest(np.array(cnt_all[k][n]), (j,i),False) !=-1:
                                x.append(j)
                                y.append(i)
                    xp.append(x)
                    yp.append(y)
                Xp.append(xp)
                Yp.append(yp)
            else:
                Xp.append([])
                Yp.append([])
        return Xp, Yp, cnt_all
  
    def MatrixRemove(self, imap, nr, perf, ImName, pixNr, path, site):#, v1, v2):
        v = []
##        for i in arange(0, len(imap)):
##            ind = np.where(np.array(imap[i])<0)
##            for n in arange(0, len(ind[0])):
##                imap[i][ind[0][n]][ind[1][n]] = np.nan
        #searching for the matrix size in 3D
        for i in arange(0, len(self.Xcontour)): #slices
            for j in arange(0, len(self.Xcontour[i])): #sub-structres in the slice
                for k in arange(0, len(self.Xcontour[i][j])):
                    v.append(imap[i][self.Ycontour[i][j][k]][self.Xcontour[i][j][k]])

        #calculating the interval corresponding to one row in the co-occurence matrix
        ind = np.where(np.isnan(np.array(v)))[0]
        for j in arange(1, len(ind)+1):
            v.pop(ind[-j])
        #fitting gaussian function and deleting ouliers.
        remove = self.histfit(v, 15,nr, perf, ImName, pixNr, path, site)
        '''check for size influences'''
##        new_matrix = []
##        for j in arange(0, len(matrix)):
##            print len(matrix[j])/2,  len(matrix[j][0])/2
##            new_matrix.append(np.array(matrix[j])[:len(matrix[j])/4, :len(matrix[j][0])/4])
##            
##        norm_points=[]
##        for i in arange(0, len(new_matrix)):
##            norm_points.append(len(new_matrix[i])*len(new_matrix[i][0])-len(np.where(np.isnan(np.array(new_matrix[i])))[0]))
        #remove points outside gauss
        p_remove=[]
        for j in arange(0,len(remove)):
            p_remove.append(np.where(np.array(imap)==remove[j]))
        return p_remove, len(v)
    
    def Matrix(self, imap, rs_type, structure):
        '''fill the contour matrix with values from the image - including discretizaion'''
        '''matrix - matrix with discretized entries'''
        '''matrix_ture - matrix with original values for first-order statistics'''
        matrix = []
        norm_points = []
        min_value = []
        max_value = []
        lymin = []
        lymax = []
        lxmin = []
        lxmax = []
        v = []
        if structure != 'none':
        #searching for the matrix size in 3D
            if rs_type:
                Xcontour = self.Xcontour
                Ycontour = self.Ycontour
            else:
                Xcontour = self.Xcontour_W
                Ycontour = self.Ycontour_W
                
            for i in arange(0, len(Xcontour)): #slices
                ymins = []
                ymaxs = []
                xmins = []
                xmaxs = []
                for j in arange(0, len(Xcontour[i])): #sub-structres in the slice
                    ymins.append(np.min(Ycontour[i][j]))
                    ymaxs.append(np.max(Ycontour[i][j]))
                    xmins.append(np.min(Xcontour[i][j]))
                    xmaxs.append(np.max(Xcontour[i][j]))
                    for k in arange(0, len(Xcontour[i][j])):
                        v.append(imap[i][Ycontour[i][j][k]][Xcontour[i][j][k]])
                try:
                    lymin.append(np.min(ymins))
                except ValueError:
                    pass
                try:
                    lymax.append(np.max(ymaxs))
                except ValueError:
                    pass
                try:
                    lxmin.append(np.min(xmins))
                except ValueError:
                    pass
                try:
                    lxmax.append(np.max(xmaxs))
                except ValueError:
                    pass
            ymin = np.min(lymin) 
            ymax = np.max(lymax)
            xmin = np.min(lxmin)
            xmax = np.max(lxmax)
            vmin = np.min(v)
            vmax = np.max(v)
        else:
            xmin = 0
            xmax = self.columns-1
            ymin = 0
            ymax = self.rows-1
            vmin = np.min(imap)
            vmax = np.max(imap)
            
        #placing nan in all places in the matrix
        m = np.zeros([ymax-ymin+1, xmax-xmin+1]) #creating the matrix to fill it with points of the structure, y rows, x columns
        for im in arange(0, len(m)):
            for jm in arange(0, len(m[im])):
                m[im][jm]=np.nan
        matrix = []
        matrix_true = []
        
        if structure != 'none':
            for n in arange(0, len(Xcontour)):
                matrix.append(m.copy())
                matrix_true.append(m.copy())

            #calculating the interval corresponding to one row in the co-occurence matrix
            interval = (vmax-vmin)/(self.bits-1)
            del lymin 
            del lymax 
            del lxmin 
            del lxmax
            for i in arange(0, len(Xcontour)): #slices
                for j in arange(0, len(Xcontour[i])): #sub-structres in the slice
                    for k in arange(0, len(Xcontour[i][j])):
                        try:
                            matrix[i][Ycontour[i][j][k]-ymin][Xcontour[i][j][k]-xmin] = int(round((imap[i][Ycontour[i][j][k]][Xcontour[i][j][k]]-vmin)/interval, 0)) #first row, second column, changing to 256 channels as in co-ocurence matrix we have only the 256 channels
                            matrix_true[i][Ycontour[i][j][k]-ymin][Xcontour[i][j][k]-xmin] = imap[i][Ycontour[i][j][k]][Xcontour[i][j][k]] #first row, second column, changing to 256 channels as in co-ocurence matrix we have only the 256 channels
                        except ValueError:
                            pass
                    norm_points.append(len(v))

        else:
            for n in arange(0, len(imap)):
                matrix.append(m.copy())
                matrix_true.append(m.copy())

            interval = (vmax-vmin)/(self.bits-1) #matrix with self.bits channels
            del lymin 
            del lymax 
            del lxmin 
            del lxmax
            for i in arange(0, len(imap)): #slices
                for j in arange(0, len(imap[i])): #rows
                    for k in arange(0, len(imap[i][j])): #columns
                        try:
                            matrix[i][j-ymin][k-xmin] = int(round((imap[i][j][k]-vmin)/interval, 0)) #first row, second column, changing to self.bits channels as in co-ocurence matrix we have only the channels channels
                            matrix_true[i][j-ymin][k-xmin] = imap[i][j][k] #first row, second column, changing to self.bits channels as in co-ocurence matrix we have only the channels channels
                        except ValueError:
                            pass
                    norm_points.append(len(imap[0])*len(imap[0][0])) #how many points are used for calculation
                    

       #remove points outside gauss
        self.vmin.append(vmin)
        self.vmax.append(vmax)
        return matrix, interval, norm_points, matrix_true
    
    def histfit(self,x,N_bins, nr, name, ImName, pixNr, path, site):  
        ''' 
        x - dane
        N_bins -ilość binów w histogramie
         
        Funkcja rysuje histogram i na jego tle dorysowuje wykres 
        funkcji gęstości prawdopodobieństwa rozkładu normalnego 
        o średniej i wariancji estymowanych z x.
         
        Funkcja wymaga zaimportowania modułów pylab as py i scipy.stats as st'''
        x = np.array(x, dtype = np.float64)
        py.figure(figsize=(20,20))
        py.subplot(121)
        n, bins, patches = py.hist(x, N_bins, normed=True, facecolor='green', alpha=0.75)
            # Rysujemy histogram i w jawny sposób odbieramy zwracane przez p.hist obiekty
            #   - normujemy histogram do jedności
            #   - ustalamy kolor prostokątów na zielony
            #   - ustawiamy przezroczystość prostokątów na 0.75
     
     
        bincenters = 0.5*(bins[1:]+bins[:-1])
            # wytwarzamy tablicę z centrami binów korzystając z granic binów
            # zwróconych przez py.hist w macierzy bins
     
     
        y = st.norm.pdf( bincenters, loc = np.mean(x), scale = np.std(np.array(x)))
            # obliczamy momenty rozkładu x: średnią i wariancję (tak naprawdę to jej pierwiastek czyli standardowe odchylenie)
            # obliczamy wartości w normalnym rozkładzie gęstości prawdopodobieństwa
            # o średniej np.mean(x) i standardowym odchyleniu np.std(x) dla wartości bincenters
     
        l = py.plot(bincenters, y, 'r--', linewidth=1, label = 'std: '+str(round(np.std(x),2)))
        py.plot([np.mean(x)+3*np.std(x), np.mean(x)+3*np.std(x)], [0,0.1], 'b--')
        py.plot([np.mean(x)-3*np.std(x), np.mean(x)-3*np.std(x)], [0,0.1], 'b--')
        ind1 = np.where(np.array(x)>(np.mean(x)+3*np.std(x)))[0]
        ind2 = np.where(np.array(x)<(np.mean(x)-3*np.std(x)))[0]
        v = []
        for i in ind1:
            v.append(x[i])
        for i in ind2:
            v.append(x[i])
        x= list(x)
        leg=str(len(v)*100./len(x))
        for i in v:
            x.remove(i)
        py.legend()
        py.subplot(122)
        n, bins, patches = py.hist(x, N_bins, normed=True, facecolor='green', alpha=0.75, label = 'removed points: '+leg)
        py.legend()
        try:
            makedirs(path+'gauss\\')
        except OSError:
            if not isdir(path+'gauss\\'):
                raise
        py.savefig(path+'gauss\\'+'hist_'+site+'_'+name+'_'+str(nr)+ImName+'_'+pixNr+'.png')
        py.close()
        return v
            # do histogramu dorysowujemy linię
    def remove_points(self, M, p):
        for i in p:
            if i !=[]:
                for j in i:
                    for k in arange(0, len(j[0])):
                        M[j[0][k]][j[1][k]][j[2][k]] = np.nan
        return M

    def fun_histogram(self, M, name, ImName, pixNr, path, m_type):
        M1=[]
        for m in M:
            for i in arange(0, len(m)):
                for j in arange(0, len(m[i])):
                    if np.isnan(m[i][j]):
                        pass
                    else:
                        M1.append(m[i][j])

        matplotlib.rcParams.update({'font.size': 24})
        if m_type:
            fig = py.figure(300, figsize = (20,20))
            fig.text(0.5, 0.95, ImName+' '+name)
            py.hist(M1)
            try:
                makedirs(path+'histogram\\')
            except OSError:
                if not isdir(path+'histogram\\'):
                    raise
            fig.savefig(path+'histogram\\'+name+'_'+ImName+'_'+self.structure+'_'+pixNr+'.png')
            py.close()
        return M1

    def fun_mean(self, M1):
        m = np.mean(M1)
        return m

    def fun_std(self, M1):
        s = np.std(M1)
        return s

    def fun_COV(self, M1):
        miu = np.mean(M1)
        cov = 0
        for i in M1:
            cov+=(i-miu)**2
        cov=np.sqrt(cov/float(len(M1)))/miu
        return cov

    def fun_skewness(self, M1):
        miu = np.mean(M1)
        nom = 0
        denom =0
        for i in M1:
            nom+=(i-miu)**3
            denom+=(i-miu)**2
        nom=nom/float(len(M1))
        denom=denom/float(len(M1))
        denom = np.sqrt(denom**3)
        s=nom/denom
        return s

    def fun_kurtosis(self, M1):
        miu = np.mean(M1)
        nom = 0
        denom =0
        for i in M1:
            nom+=(i-miu)**4
            denom+=(i-miu)**2
        nom=nom/float(len(M1))
        denom=(denom/float(len(M1)))**2
        k=(nom/denom)-3
        return k

    def coMatrix(self, M,trans):
        #Calculate the 2D co-occurence matrix from 3D structure matrix 
        coMatrix = np.zeros((self.bits, self.bits))
        norm=0
        for y in arange(0, len(M)):
            for x in arange(0, len(M[y])):
                try:
                    value1 = M[y][x] #value in the structure matrix
                    value2 = M[y+trans[1]][x+trans[0]] #distance 1 angle 90 degree and 90 degree
                    y_cm = value1
                    x_cm = value2
                    try:
                        coMatrix[y_cm][x_cm]+=1
                        norm+=1
                    except IndexError:
                        pass
                except IndexError:
                    pass
##        print 'norm', norm
##        for i in coMatrix:
##            print i
        if norm ==0:
            return coMatrix, norm
        else:
            return coMatrix/norm, norm

    def fun_energy(self, coM):
        energy = 0
        ind = np.where(coM!=0)
        for j in arange(0, len(ind[0])):
            energy+=(coM[ind[0][j]][ind[1][j]])**2
        return energy

    def fun_entropy(self, coM):
        entropy = 0
        ind = np.where(coM!=0)
        for j in arange(0, len(ind[0])):
            s=(coM[ind[0][j]][ind[1][j]])*np.log10(coM[ind[0][j]][ind[1][j]])
            if np.isnan(s):
                pass
            else:
                entropy += -s
        if entropy ==0 and len(ind[0])==0:
            entropy=0
        elif entropy ==0:
            entropy=np.nan
        return entropy

    def fun_contrast(self, coM):
        contrast = 0
        ind = np.where(coM!=0)
        for j in arange(0, len(ind[0])):
            contrast+=((ind[0][j]-ind[1][j]+2)**2)*coM[ind[0][j]][ind[1][j]]
        if contrast ==0 and len(ind[0])==0:
            contrast = 0
        elif contrast ==0:
            contrast = np.nan
        return contrast

##    def fun_correlation(self, coM):
##        corr = 0
##        l = []
##        for j in arange(0, len(coM)):
##            for k in arange(0, len(coM[j])):
##                l.append(((j*k)*coM[j][k]-np.mean(coM[j])*np.mean(coM[:,k]))/(np.std(coM[j])*np.std(coM[:,k])))#l.append(((j-k)*coM[j][k]-np.mean(coM[j])*np.mean(coM[:,k]))/(np.std(coM[j])*np.std(coM[:,k])))
##        ind = np.where(np.isnan(np.array(l)))[0]
##        for j in arange(1, len(ind)+1):
##            l.pop(ind[-j])
##        corr += np.sum(l)
##        if corr==0:
##            corr=np.nan
##        return corr
    
##    def fun_correlation(self, coM):
##        corr = 0
##        l = []
##        for j in arange(0, len(coM)):
##            for k in arange(0, len(coM[j])):
##                l.append(((j+1)*(k+1))*coM[j][k])
##        X = []
##        Y = []
##        for i in arange(0, len(coM)):
##            X.append(np.sum(coM[i]))
##        for j in arange(0, len(coM[0])):
##            Y.append(np.sum(coM[:,j]))
##
##        meanX = np.mean(X)
##        stdX = np.std(X)
##        meanY = np.mean(Y)
##        stdY = np.std(Y)
##        
##        ind = np.where(np.isnan(np.array(l)))[0]
##        for j in arange(1, len(ind)+1):
##            l.pop(ind[-j])
##        corr += np.sum(l)
##        corr = (corr-meanX*meanY)/(stdX*stdY)
##        return corr

    def fun_correlation(self, coM):
        meanX=[]
        meanY=[]
        stdX=[]
        stdY=[]
        for i in arange(0, len(coM)):
            x=[]
            for j in arange(0, len(coM[i])):
                x.append((i+1)*coM[i][j])
            meanX.append(np.sum(x))
##        for i in arange(0, len(coM)):
##            x=[]
##            for j in arange(0, len(coM[i])):
##                x.append(((i+1-meanX[i])**2)*coM[i][j])
##            stdX.append(np.sum(x))

        for i in arange(0, len(coM[0])):
            y=[]
            for j in arange(0, len(coM)):
                y.append((i+1)*coM[i][j])
            meanY.append(np.sum(y))
##        for i in arange(0, len(coM[0])):
##            x=[]
##            for j in arange(0, len(coM)):
##                x.append(((i+1-meanY[i])**2)*coM[i][j])
##            stdY.append(np.sum(x))


##        corr = 0
##        l = []
##        for j in arange(0, len(coM)):
##            for k in arange(0, len(coM[j])):
##                l.append(((j-meanX[j])*(k-meanY[k]))/np.sqrt(stdX[j]*stdY[k]))
##                #l.append(((j*k)*coM[j][k]-np.mean(coM[j])*np.mean(coM[:,k]))/(np.std(coM[j])*np.std(coM[:,k])))#l.append(((j-k)*coM[j][k]-np.mean(coM[j])*np.mean(coM[:,k]))/(np.std(coM[j])*np.std(coM[:,k])))
##        ind = np.where(np.isnan(np.array(l)))[0]
##        for j in arange(1, len(ind)+1):
##            l.pop(ind[-j])
##
##        ind = np.where(np.isinf(np.array(l)))[0]
##        for j in arange(1, len(ind)+1):
##            l.pop(ind[-j])

        meanX=np.sum(meanX)
        meanY=np.sum(meanY)
        for i in arange(0, len(coM)):
            x=[]
            for j in arange(0, len(coM[i])):
                x.append(((i+1-meanX)**2)*coM[i][j])
            stdX.append(np.sum(x))
        for i in arange(0, len(coM[0])):
            x=[]
            for j in arange(0, len(coM)):
                x.append(((i+1-meanY)**2)*coM[i][j])
            stdY.append(np.sum(x))
        stdX=np.sum(stdX)
        stdY=np.sum(stdY)
        
        corr = 0
        l = []
        for j in arange(0, len(coM)):
            for k in arange(0, len(coM[j])):
                l.append(((j+1)*(k+1)*coM[j][k]-meanX*meanY)/(stdX*stdY))
                #l.append(((j*k)*coM[j][k]-np.mean(coM[j])*np.mean(coM[:,k]))/(np.std(coM[j])*np.std(coM[:,k])))#l.append(((j-k)*coM[j][k]-np.mean(coM[j])*np.mean(coM[:,k]))/(np.std(coM[j])*np.std(coM[:,k])))
        ind = np.where(np.isnan(np.array(l)))[0]
        for j in arange(1, len(ind)+1):
            l.pop(ind[-j])

        ind = np.where(np.isinf(np.array(l)))[0]
        for j in arange(1, len(ind)+1):
            l.pop(ind[-j])


        corr += np.sum(l)
        if corr==0:
            corr=np.nan
        return corr
    
    def fun_homogenity(self, coM):
        homo = 0
        ind = np.where(coM!=0)
        for j in arange(0, len(ind[0])):
           homo += coM[ind[0][j]][ind[1][j]]/(1+(ind[0][j]-ind[1][j])**2)#homo += coM[ind[0][j]][ind[1][j]]/(1+abs(ind[0][j]-ind[1][j]))
        if homo ==0:
            homo=np.nan
        return homo

    def fun_variance(self, coM):
        var=0
        miu = 0
        for i in arange(0, len(coM)):
            for j in arange(0, len(coM[i])):
                miu+=0.5*((i+1)*coM[i][j]+(j+1)*coM[i][j])
        ind = np.where(coM!=0)
        for j in arange(0, len(ind[0])):
            var += (ind[0][j]+1-miu)**2*coM[ind[0][j]][ind[1][j]]
        if var ==0:
            var = np.nan
        return var

    def fun_average(self, coM):
        a = 0
        pxy = np.zeros(2*len(coM))
        for i in arange(0, len(coM)):
            for j in arange(0, len(coM[i])):
                pxy[i+j]+=coM[i][j]

        for i in arange(0, len(pxy)):
            a += (i+2)*pxy[i]

        return a

    def fun_sum_entropy_var(self, coM):
        e = 0
        v = 0 
        pxy = np.zeros(2*len(coM))
        for i in arange(0, len(coM)):
            for j in arange(0, len(coM[i])):
                pxy[i+j]+=coM[i][j]
                
        for i in arange(0, len(pxy)):
            if pxy[i] !=0:
                e += -pxy[i]*np.log10(pxy[i])

        for i in arange(0, len(pxy)):
            v += (i+2-e)**2*pxy[i]
                  
        return e, v

    def fun_diff_entropy_var(self, coM):
        e = 0
        v = 0
        pxy = np.zeros(len(coM))
        for i in arange(0, len(coM)):
            for j in arange(0, len(coM[i])):
                pxy[abs(i-j)]+=coM[i][j]

        v = np.std(pxy)**2
        for i in arange(0, len(pxy)):
            if pxy[i] !=0:
                e += -pxy[i]*np.log10(pxy[i])

        return e, v

    def fun_IMC(self, coM):
        hxy = 0
        for i in arange(0, len(coM)):
            for j in arange(0, len(coM[i])):
                if coM[i][j] != 0:
                    hxy += -coM[i][j]*np.log10(coM[i][j])

        X = []
        Y = []
        for i in arange(0, len(coM)):
            X.append(np.sum(coM[i]))
        for j in arange(0, len(coM[0])):
            Y.append(np.sum(coM[:,j]))

        hxy1 = 0
        hxy2 = 0
        for i in arange(0, len(coM)):
            for j in arange(0, len(coM[i])):
                if X[i]*Y[j] != 0:
                    hxy1 += -coM[i][j]*np.log10(X[i]*Y[j])
                    hxy2 += -X[i]*Y[j]*np.log10(X[i]*Y[j])

        hx = 0
        hy = 0
        for i in arange(0, len(X)):
            if X[i] !=0:
                hx += -X[i]*np.log10(X[i])
        for i in arange(0, len(Y)):
            if Y[i] != 0:
                hy += -Y[i]*np.log10(Y[i])

        try:
            f12 = (hxy-hxy1)/max(hx,hy)
        except ZeroDivisionError:
            f12 = np.nan
        f13 = (1- np.exp(-2*(hxy2 - hxy)))*0.5

        return f12, f13

    def fun_MCC(self, coM):
        Q = np.zeros((len(coM),len(coM[0])))
        X = []
        Y = []
        for i in arange(0, len(coM)):
            X.append(np.sum(coM[i]))
        for j in arange(0, len(coM[0])):
            Y.append(np.sum(coM[:,j]))

        for i in arange(0, len(coM)):
            for j in arange(0, len(coM[i])):
                for k in arange(0, len(Y)):
                    if (X[i]*Y[k])!=0:
                        Q[i][j] += coM[i][k]*coM[j][k]/(X[i]*Y[k])
##                        if np.isnan(coM[i][k]*coM[j][k]/(X[i]*Y[k])):
##                            print coM[i][k], coM[j][k], X[i], Y[k]
                            
        l = np.linalg.eigvals(Q)

        l.sort()
        return l[-2]**0.5
        
    def M3(self, matrix): #Amadasun et al. Textural Features Corresponding to Textural Properties
        #neighborhood gray-tone difference matrix
        s = np.zeros(self.bits)
        Ni = np.zeros(self.bits)
##            print np.where(np.isnan(matrix[k]))
        for v in arange(0,self.bits):
            index = np.where(matrix==v)
##                print v
##                print index
            for ind in arange(0,len(index[0])):
                temp = []
                numerator = 0
##                    try:
                for y in [-1,1]:
                    for x in [-1,0,1]:
                        try:
                            temp.append(matrix[index[0][ind]+y][index[1][ind]+x])
                        except IndexError:
                            numerator+=1
                y=0
                for x in [-1,1]:
                    try:
                        temp.append(matrix[index[0][ind]+y][index[1][ind]+x])
                    except IndexError:
                        numerator+=1
                ind_nan = np.where(np.isnan(np.array(temp)))[0]
                for n in arange(1, len(ind_nan)+1):
                    temp.pop(ind_nan[-n])
                numerator+=len(ind_nan)
                if numerator!=8:
                    a = abs(v - (float(np.sum(temp))/(26-numerator)))
                    s[v]+=a
                    Ni[v] += 1
##                    except IndexError:
##                        print ind
##                        pass
        if np.sum(Ni)==0:
            return s, Ni, np.sum(Ni)
        else:
            if np.sum(s)==0:
                return s, s, np.sum(Ni)
            else:
                return s/np.sum(Ni), Ni, np.sum(Ni)
            
    def fun_coarseness(self, s, Ni, matrix):
        f = 0
        ind = np.where(np.array(Ni)!=0)[0]
        for i in ind:
            f+=s[i]*Ni[i]/np.sum(Ni)
        f = 1./(0.000000001+f)
        if f==0:
            f=np.nan
        return f
    def fun_busyness(self, s, Ni, matrix):
        try:
            nom = 0
            denom = 0
            ind = np.where(np.array(Ni)!=0)[0]
            for i in ind:
                nom += s[i]*Ni[i]/np.sum(Ni)
                for j in ind:
                    denom += abs(float(i*Ni[i])/(np.sum(Ni)) - float(j*Ni[j])/(np.sum(Ni)))
            if 2*nom/denom ==0:
                return np.nan
            else:
                return 2*nom/denom
        except ZeroDivisionError:
            if len(ind)==0:
                return 0
            else:
                return np.nan
    def fun_contrastM3(self, s, Ni, matrix):
        try:
            Ng = len(np.where(np.array(Ni)!=0)[0])
            ind = np.where(np.array(Ni)!=0)[0]
            s1 = 0
            for i in ind:
                for j in ind:
                    s1 += float(Ni[i])/(np.sum(Ni))*float(Ni[j])/(np.sum(Ni))*(i-j)**2
            s2 = 0
            ind = np.where(np.array(s)!=0)[0]
            for i in ind:
                s2+=s[i]

            f = (1./(Ng*(Ng-1)))*s1*(1./(np.sum(Ni)))*s2
            if f == 0:
                return np.nan
            else:
                return f
        except ZeroDivisionError:
            if len(ind)==0:
                return 0
            else:
                return np.nan
        
    def fun_complexity(self, s, Ni, matrix):
        ind = np.where(np.array(Ni)!=0)[0]
        s1 = 0
        for i in ind:
            for j in ind:
                s1 += (abs(i-j)/(float(Ni[i])+float(Ni[j])))*((s[i]*float(Ni[i])/(np.sum(Ni)))+(s[j]*float(Ni[j])/(np.sum(Ni))))
        if s1==0 and len(ind)==0:
            s1=0
        elif s1==0:
            s1=np.nan
        return s1
    def saveImage(self, path, name, matrix,i, ImName, pixNr):
        matplotlib.rcParams.update({'font.size': 24})
        fig = py.figure(10, figsize = (20,20))
        fig.text(0.5, 0.95, ImName)
        for j in arange(0, len(matrix)):
            axes = fig.add_subplot(5, 5, j+1)
            axes.set_title(j)
            im = axes.imshow(matrix[j], cmap=py.cm.jet, vmin = 0, vmax = self.bits)
        axes = fig.add_subplot(5, 5, 25)
        fig.colorbar(im)
        try:
            makedirs(path+ImName+'\\')
        except OSError:
            if not isdir(path+ImName+'\\'):
                raise
        fig.savefig(path+ImName+'\\'+name+'_'+self.structure+'_'+pixNr+'.png')
        py.close()
        del fig
        
    def M4(self, matrix): #Guillaume Thibault et al., ADVANCED STATISTICAL MATRICES FOR TEXTURE CHARACTERIZATION: APPLICATION TO DNA CHROMATIN AND MICROTUBULE NETWORK CLASSIFICATION
        GLSZM = []
        m = np.array(matrix).copy()
        m.dtype = np.float
        Smax = 1
        for i in arange(0, self.bits):
            GLSZM.append([0])
        for i in arange(0, len(m)):
            for j in arange(0, len(m[i])):
                if np.isnan(m[i][j]):
                    pass
                else:
                    v = int(m[i][j])
                    size = 1
                    m[i][j] = np.nan
                    points = self.neighbor(i,j, m,v)
                    size+=len(points)
                    zone = [[i,j]]
                    for ni in points:
                        zone.append(ni)
                        m[ni[0]][ni[1]] = np.nan
                    while len(points)!=0:
                        p = []
                        for n in arange(0, len(points)):
                            poin = self.neighbor(points[n][0],points[n][1], m,v)
                            for ni in poin:
                                zone.append(ni)
                                m[ni[0]][ni[1]] = np.nan
                                p.append(ni)
                                size+=1
                        points = p
                    if size>Smax:
                        for s in arange(0, len(GLSZM)):
                            for si in arange(0,size-Smax):
                                GLSZM[s].append(0)
                        Smax=size
                        GLSZM[v][size-1] +=1
                    else:
                        GLSZM[v][size-1] +=1
        for i in arange(0, len(GLSZM)):
            GLSZM[i] = np.array(GLSZM[i])
        if np.sum(GLSZM)!= 0:
            GLSZM = np.array(GLSZM)/float(np.sum(GLSZM))
        return GLSZM, np.sum(GLSZM)
                                
    def neighbor(self, y,x, matrix,v):
        points = []
        for i in arange(-1,2):
            for j in arange(-1,2):
                try:
                    if matrix[y+i][x+j] == v and y+i >= 0 and x+j >= 0:
                        points.append([y+i,x+j])
                except IndexError:
                    pass
        return points
    
    def sizeVariability(self, GLSZM):
        norm = 0
        for m in arange(0, len(GLSZM)):
            norm += np.sum(GLSZM[m])
        var = 0
        for n in arange(0, len(GLSZM[0])):
            s=0
            for m in arange(0, len(GLSZM)):
                s+=GLSZM[m][n]
            var+=s**2
        return float(var)
    def intensityVariability(self, GLSZM):
        norm = 0
        for m in arange(0, len(GLSZM)):
             norm += np.sum(GLSZM[m])
        var = 0
        for m in arange(0, len(GLSZM)):
            s=0
            for n in arange(0, len(GLSZM[m])):
                s+=GLSZM[m][n]
            var+=s**2
        return float(var)

    def shortSize(self, GLSZM):
        sse = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                sse += GLSZM[i][j]/(j+1)**2 #place 0 in the list corresponds to size 1
        return sse

    def longSize(self, GLSZM):
        lse = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                lse += GLSZM[i][j]*(j+1)**2 #place 0 in the list corresponds to size 1
        return lse

    def LGSE(self, GLSZM):
        lgse = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                lgse += GLSZM[i][j]/(i+1)**2 #otherwise level 0 is not included
        return lgse

    def HGSE(self, GLSZM):
        hgse = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                hgse += GLSZM[i][j]*(i+1)**2 #otherwise level 0 is not included
        return hgse

    def SSLGE(self, GLSZM):
        sslge = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                sslge += GLSZM[i][j]/((j+1)**2*(i+1)**2) #otherwise level 0 is not included
        return sslge
    
    def SSHGE(self, GLSZM):
        sshge = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                sshge += GLSZM[i][j]*(i+1)**2/(j+1)**2 #otherwise level 0 is not included
        return sshge

    def LSLGE(self, GLSZM):
        lslge = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                lslge += GLSZM[i][j]*(j+1)**2/(i+1)**2 #otherwise level 0 is not included
        return lslge
    
    def LSHGE(self, GLSZM):
        lshge = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                lshge += GLSZM[i][j]*(j+1)**2*(i+1)**2 #otherwise level 0 is not included
        return lshge

    def runPer(self, GLSZM):
        norm = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                norm+=(j+1)*GLSZM[i][j]
        rpc = np.sum(GLSZM)/norm
        return rpc

    def voxelPerfusion(self, perfMatrix):
        hist = []
        for i in arange(0, 8):
            hist.append(0)
        try:
            temp = []
            for i in arange(0,3):
                t = []
                for z in arange(0, len(perfMatrix[i])):
                    for y in arange(0, len(perfMatrix[i][z])):
                        for x in arange(0, len(perfMatrix[i][z][y])):
                            if not np.isnan(perfMatrix[i][z][y][x]):
                                t.append(perfMatrix[i][z][y][x])
                temp.append(t)
            median = [np.median(temp[0]), np.median(temp[1]), np.median(temp[2])]
            m = []
            for i in arange(0, len(perfMatrix)):
                m.append(perfMatrix[i]- median[i])
            for z in arange(0, len(m[0])):
                for y in arange(0, len(m[0][z])):
                    for x in arange(0, len(m[0][z][y])):
                        if m[0][z][y][x] > 0 and m[1][z][y][x] > 0 and m[2][z][y][x] > 0:
                            hist[0]+=1
                        elif m[0][z][y][x] > 0 and m[1][z][y][x] > 0 and m[2][z][y][x] < 0:
                            hist[1]+=1
                        elif m[0][z][y][x] > 0 and m[1][z][y][x] < 0 and m[2][z][y][x] < 0:
                            hist[2]+=1
                        elif m[0][z][y][x] < 0 and m[1][z][y][x] < 0 and m[2][z][y][x] < 0:
                            hist[3]+=1
                        elif m[0][z][y][x] < 0 and m[1][z][y][x] > 0 and m[2][z][y][x] > 0:
                            hist[4]+=1
                        elif m[0][z][y][x] < 0 and m[1][z][y][x] > 0 and m[2][z][y][x] < 0:
                            hist[5]+=1
                        elif m[0][z][y][x] < 0 and m[1][z][y][x] < 0 and m[2][z][y][x] > 0:
                            hist[6]+=1
                        elif m[0][z][y][x] > 0 and m[1][z][y][x] < 0 and m[2][z][y][x] > 0:
                            hist[7]+=1
        except TypeError:
            pass
        return hist

##    def Fourier(self, m):
##        
##        for i in arange(0, len(m)):
##            fig = py.figure(900+i)
##            mat = np.array(m[i]).copy()
##            ind = np.where(np.isnan(np.array(mat)))
##            for j in arange(0, len(ind[0])):
##                mat[ind[0][j]][ind[1][j]]=-1
##            F = fft2(np.array(mat))
##            Z=F**2
##            ax = fig.gca(projection='3d')
##            freqX = fftfreq(len(mat[0]))
##            freqY = fftfreq(len(mat))
##            X, Y = np.meshgrid(freqX, freqY)
##            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=py.cm.coolwarm, linewidth=0, antialiased=False)
##            fig.colorbar(surf, shrink=0.5, aspect=5)
##            py.show()
##            py.close()

    def fractal(self, m, path, pixNr, ImName):
        #https://en.wikipedia.org/wiki/Box_counting
        try:
            def func_lin(x,a,b):
                return x*a+b

            norm = len(m)*len(m[0])
            ind = np.where(np.isnan(np.array(m)))[0]
            norm = norm - len(ind)
            maxR = np.min([len(m),len(m[0])])
            print len(m),len(m[0])
            frac = []
            for r in arange(2, maxR+1): #because log(1) = 0
                N=0
                for y in arange(0, len(m),r):
                    for x in arange(0, len(m[0]),r):
                        m =np.array(m)
                        matrix=m[y:y+r,x:x+r] #doesn't produce indexerror
                        ind = len(np.where(np.isnan(matrix))[0])
                        if ind<(r**2):
                            N+=1
                frac.append(np.log(N))
            x = np.log(arange(2, maxR+1))
            xdata=np.transpose(x)
            x0 = np.array([0, 0]) #initial guess
            result = optimization.curve_fit(func_lin, xdata, frac, x0)
            fit = func_lin(x, result[0][0], result[0][1])
            py.figure(2000)
            ax = py.subplot(111)
            py.plot(x,frac, 'o')
            py.plot(x,fit, label = 'dim = '+str(-round(result[0][0],2)))
            py.xlabel('ln(r)')
            py.ylabel('ln(N(r))')
            py.legend()
            print path+'fractals\\'+ImName+'.png'
            try:
                makedirs(path+'fractals\\')
            except OSError:
                if not isdir(path+'fractals\\'):
                    raise
            py.savefig(path+'fractals\\'+ImName+'_'+self.structure+'_'+pixNr+'.png')
            py.close()
            return -result[0][0], norm
        except TypeError:
            return 0, 0
            pass
        except RuntimeError:
            return 0, 0
            pass

              
        
