# -*- coding: utf-8 -*-
'''calculated the co matirxin 3D in all possible directions and average the results'''
import numpy as np
from numpy import arange,floor
#import matplotlib
import pylab as py
import matplotlib
#from scipy.fftpack import fft2, fftn, fftfreq
#from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as optimization
#import scipy.stats as st
import cv2
from os import makedirs
from os.path import isdir
from datetime import datetime

from texture_wavelet3D import Wavelet
from texture_wavelet3D_ctp import WaveletCTP
from ROImatrix import Matrix
import matplotlib.pylab as plt
import logging
##from texture_rs import Read_rs

##from margin4 import Margin
##from edge import Edge

class Texture(object):
    '''Calculate texture, intenisty, fractal dim and center of the mass shift
    sb – status bar 
    maps – list of images to analyze, for CT it is one element list with 3D matrix, for perfusion it is a 3 elements list with 3D matrices for BF, MTT, BV
    structure – name of analyzed structure, used for naming the output files
    columns – number of columns in the image
    rows – number of rows in the image
    xCTspace – pixel spacing in xy 
    slices – z positions of slices
    path – path to save results 
    ImName – patient number
    pixNr – number of analyzed bins, if not specified  = none
    binSize – bin size for the analysis, if not specified = none
    modality – string, name of modality to customize things like HU range.
    wv – bool, calculate wavelet  
    cropStructure - {'hu_min': 'none', 'hu_max': 'none', 'ct_path': '', 'crop': False}, if no CT-based cropping on PET data
    stop_calc - string, stop calculation if image tag was incorrect, eg activty PET = 0; stop_calc=='' continue, stop_calc != '' break and save info in the excel file
    *cont – list of additional variables:
    Xc – structure points in X, list of slices, per slice list of substructures
    Yc - structure points in Y, list of slices, per slice list of substructures
    XcW - structure points in X in wavelet space, list of slices, per slice list of substructures
    YcW - structure points in Y in wavelet space, list of slices, per slice list of substructures
    HUmin – HU range min
    Humax – HU range max
    outlier – bool, correct for outliers
'''
    def __init__(self, sb, maps, structure, columns, rows, xCTspace, slices, path, ImName, pixNr, binSize, modality, wv, localRadiomics, cropStructure, stop_calc, *cont):#Xc, Yc, XcW, YcW, HUmin, HUmax, outlier,  ):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Start: Texture Calculation")
        self.structure = structure
        self.xCTspace = xCTspace #image resolution
        self.columns = columns #columns
        self.rows = rows
        self.slices = slices #slice location
        self.mean = [] #list of texture parameters in the final verion contains [mean BF, mean MTT, mean BV]
        self.std = []
        self.cov = []
        self.skewness = []
        self.kurtosis = []
        self.var = []
        self.median = []
        self.percentile10 = []
        self.percentile90 = []
        self.iqr = []
        self.range = []
        self.mad = []
        self.rmad = []
        self.H_energy = []
        self.H_entropy = []
        self.rms = []
        self.H_uniformity = []
        #co
        self.energy = []
        self.entropy = []
        self.contrast = []
        self.correlation = []
        self.homogenity = []
        self.homogenity_n = []
        self.idiff = []
        self.idiff_n = []
        self.variance = []
        self.sum_average = []
        self.sum_entropy = []
        self.sum_variance = []
        self.diff_entropy = []
        self.diff_variance = []
        self.IMC1 = []
        self.IMC2 = []
        self.MCC = []
        self.joint_max = []
        self.joint_average = []
        self.diff_ave = []
        self.dissim = []
        self.inverse_var = []
        self.autocorr = []
        self.clust_t = []
        self.clust_s = []
        self.clust_p = []
        #co merged
        self.COM_energy = []
        self.COM_entropy = []
        self.COM_contrast = []
        self.COM_correlation = []
        self.COM_homogenity = []
        self.COM_homogenity_n = []
        self.COM_idiff = []
        self.COM_idiff_n = []
        self.COM_variance = []
        self.COM_sum_average = []
        self.COM_sum_entropy = []
        self.COM_sum_variance = []
        self.COM_diff_entropy = []
        self.COM_diff_variance = []
        self.COM_IMC1 = []
        self.COM_IMC2 = []
        self.COM_MCC = []
        self.COM_joint_max = []
        self.COM_joint_average = []
        self.COM_diff_ave = []
        self.COM_dissim = []
        self.COM_inverse_var = []
        self.COM_autocorr = []
        self.COM_clust_t = []
        self.COM_clust_s = []
        self.COM_clust_p = []
        #ngtdm
        self.coarseness = []
        self.neighContrast = []
        self.busyness = []
        self.complexity = []
        self.strength = []
        #glrlm
        self.len_intensity = []
        self.len_intensity_n = []
        self.len_size = []
        self.len_size_n = []
        self.len_sse = []
        self.len_lse = []
        self.len_lgse = []
        self.len_hgse = []
        self.len_sslge = []
        self.len_sshge = []
        self.len_lslge = []
        self.len_lshge = []
        self.len_rpc = []
        self.len_glv = []
        self.len_lsv = []
        self.len_size_entropy = []
        #glrlm merged
        self.M_len_intensity = []
        self.M_len_intensity_n = []
        self.M_len_size = []
        self.M_len_size_n = []
        self.M_len_sse = []
        self.M_len_lse = []
        self.M_len_lgse = []
        self.M_len_hgse = []
        self.M_len_sslge = []
        self.M_len_sshge = []
        self.M_len_lslge = []
        self.M_len_lshge = []
        self.M_len_rpc = []
        self.M_len_glv = []
        self.M_len_lsv = []
        self.M_len_size_entropy = []
        #glszm
        self.intensity = []
        self.intensity_n = []
        self.size = []
        self.size_n = []
        self.sse = []
        self.lse = []
        self.lgse = []
        self.hgse = []
        self.sslge = []
        self.sshge = []
        self.lslge = []
        self.lshge = []
        self.rpc = []
        self.glv = []
        self.lsv = []
        self.size_entropy = []
        #GLDZM
        self.GLDZM_intensity = []
        self.GLDZM_intensity_n = []
        self.GLDZM_size = []
        self.GLDZM_size_n = []
        self.GLDZM_sse = []
        self.GLDZM_lse = []
        self.GLDZM_lgse = []
        self.GLDZM_hgse = []
        self.GLDZM_sslge = []
        self.GLDZM_sshge = []
        self.GLDZM_lslge = []
        self.GLDZM_lshge = []
        self.GLDZM_rpc = []
        self.GLDZM_glv = []
        self.GLDZM_lsv = []
        self.GLDZM_size_entropy = []
        #NGLDM
        self.NGLDM_intensity = []
        self.NGLDM_intensity_n = []
        self.NGLDM_size = []
        self.NGLDM_size_n = []
        self.NGLDM_sse = []
        self.NGLDM_lse = []
        self.NGLDM_lgse = []
        self.NGLDM_hgse = []
        self.NGLDM_sslge = []
        self.NGLDM_sshge = []
        self.NGLDM_lslge = []
        self.NGLDM_lshge = []
        self.NGLDM_glv = []
        self.NGLDM_lsv = []
        self.NGLDM_size_entropy = []
        self.NGLDM_energy = []
        # other
        self.points = []
        self.frac_dim = []
        self.cms = []
        self.mtv2 = []
        self.mtv3 = []
        self.mtv4 = []
        self.mtv5 = []
        self.mtv6 = []
        self.mtv7 = [] #metabolic tumor volume
        
        try:
            self.bits = int(pixNr)
        except ValueError: #must be an int
            self.bits= pixNr
        try:
            self.binSize = float(binSize)
        except ValueError: # must be a float
            self.binSize = binSize

        self.vmin = []
        self.vmax = []
        rs_type = [1, 0, 0, 0, 0, 0, 0, 0, 2] #structure type, structure resolution, transformed or non-transformed
        
        if structure != 'none': #take contour points
            self.Xcontour = cont[0] #Xc
            self.Xcontour_W = cont[1]#XcW
            self.Ycontour = cont[2]#Yc
            self.Ycontour_W = cont[3]#YcW
        else:
            self.Xcontour = ''
        
        self.Xcontour_Rec = cont[4] 
        self.Ycontour_Rec = cont[5]     
            
        #take modality specific parameters
        if 'CT' in modality:
            self.HUmin = cont[6]
            self.HUmax = cont[7]
            self.outlier_correction = cont[8]
        
        else:
            self.HUmin = 'none'
            self.HUmax = 'none'
            self.outlier_correction = False
            
        
        print(stop_calc)
        if self.Xcontour == 'one slice': #don't calculate, contour onl on one slice
            self.stop_calculation('one slice', rs_type)
        elif stop_calc != '' : #stop calculation if image file contains wrong tags, eg activity = 0 in PET file
            self.stop_calculation(stop_calc, rs_type)
        else: 

            for i in arange(0, len(maps)): #iterate through different map type for example for CTP: BV, BF, MTT
                #wavelet transform
                
                if wv:
                    if 'BV' not in modality:
                        wave_list = Wavelet(maps[i], path, modality[i], ImName+'_'+pixNr).Return() #order of trandformed images: original, LLL, HHH, HHL, HLH, HLL, LHH, LHL, LLH
                    else:
                        wave_list = WaveletCTP(maps[i], path, modality[i], ImName+'_'+pixNr).Return()
                    sb.SetStatusText(ImName +' wave done ' +str(datetime.now().strftime('%H:%M:%S')))
                    rs_type = [1, 2, 0, 0, 0, 0, 0, 0, 0] #structure type, structure resolution
                    iterations_n = len(wave_list)
                else:
                    rs_type = [1]
                    wave_list = [maps[i]]
                    iterations_n = len(wave_list)
                    

                #extract tumor only from the images, saves results as a list analog to wave_list
                #matrix - contains only the discretized pixels in the structre
                #interval - one bin
                #norm_points - how many points are used for the calculations
                #matrix_v - contains only the pixels in the structre, original values
                #matrix - contains only the discretized pixels in the structre region plus two voxels neighborhood (used in the local radiomics)

                matrix_list = [] 
                interval_list = [] 
                norm_points_list = [] 
                matrix_v_list = [] 
                matrix_full_list = []              
                n_bits_list = []   
                self.vmin = []
                self.vmax = []
                HUmask = [] #a mask for ROI postprocessing on wavelet maps
            
                if "PET" in modality and cropStructure["crop"] == True:
                    self.logger.info("PET and Crop = True")
                    self.logger.info("Start: create HU mask from CT structure")
                    wave_list_ct = []
#                   ### CT initialize ###
                    for i in arange(0, len(cropStructure["data"])): 
                        #wavelet transform
                        if wv:
                            wave_list_ct = Wavelet(cropStructure["data"][i], path, "CT", ImName+'_'+pixNr).Return() # order of trandformed images: original, LLL, HHH, HHL, HLH, HLL, LHH, LHL, LLH
                            sb.SetStatusText(ImName +' wave done ' + str(datetime.now().strftime('%H:%M:%S')))
                            rs_type = [1, 2, 0, 0, 0, 0, 0, 0, 0] #structure type, structure resolution
                            iterations_n = len(wave_list_ct)
                        else:
                            rs_type = [1]
                            wave_list_ct = [cropStructure["data"][i]]
                            iterations_n = len(wave_list_ct)

                    # create mask from CT data for PET data
                    self.logger.info("End: create HU mask from CT structure")
                    self.logger.info("Initialize Matrices used for Texture and Wavelet")
                    for w in arange(0, len(wave_list)):
                        self.logger.debug("RS-Type " + str(rs_type[w]))
                        self.logger.debug("Intialize CT Matrix")
                        CT_ROImatrix = Matrix(wave_list_ct[w], rs_type[w], structure, ["CT"], cropStructure["readCT"].Xcontour, cropStructure["readCT"].Ycontour, cropStructure["readCT"].Xcontour_W, cropStructure["readCT"].Ycontour_W, cropStructure["readCT"].Xcontour_Rec, cropStructure["readCT"].Ycontour_Rec, cropStructure["readCT"].columns, cropStructure["readCT"].rows, cropStructure["hu_min"],  cropStructure["hu_max"],  0,  0, False, HUmask, cropStructure)
                        self.logger.debug("Intialize PET Matrix")
                        ROImatrix = Matrix(wave_list[w], rs_type[w], structure, modality[i], cropStructure["readCT"].Xcontour, cropStructure["readCT"].Ycontour, cropStructure["readCT"].Xcontour_W, cropStructure["readCT"].Ycontour_W, cropStructure["readCT"].Xcontour_Rec, cropStructure["readCT"].Ycontour_Rec, self.columns, self.rows, self.HUmin, self.HUmax, self.binSize, self.bits, self.outlier_correction, CT_ROImatrix.HUmask, cropStructure)
                        matrix_list.append(ROImatrix.matrix) 
                        interval_list.append(ROImatrix.interval)
                        norm_points_list.append(ROImatrix.norm_points)
                        matrix_v_list.append(ROImatrix.matrix_true)
                        matrix_full_list.append(ROImatrix.matrix_full)
                        n_bits_list.append(ROImatrix.n_bits)
                        self.vmin.append(ROImatrix.vmin)
                        self.vmax.append(ROImatrix.vmax)
                        HUmask = CT_ROImatrix.HUmask

                    print("------------- end: created HU mask was used for PET structure --------------")
                    del wave_list_ct
                else:
                    self.logger.info("Normal Mode, no cropping")
                    for w in arange(0, len(wave_list)):
                        ROImatrix = Matrix(wave_list[w], rs_type[w], structure, modality[i], self.Xcontour, self.Ycontour, self.Xcontour_W, self.Ycontour_W, self.Xcontour_Rec, self.Ycontour_Rec, self.columns, self.rows, self.HUmin, self.HUmax, self.binSize, self.bits, self.outlier_correction, HUmask, cropStructure)
                        matrix_list.append(ROImatrix.matrix) #tumor matrix
                        interval_list.append(ROImatrix.interval)
                        norm_points_list.append(ROImatrix.norm_points)
                        matrix_v_list.append(ROImatrix.matrix_true)
                        matrix_full_list.append(ROImatrix.matrix_full)
                        n_bits_list.append(ROImatrix.n_bits)
                        self.vmin.append(ROImatrix.vmin)
                        self.vmax.append(ROImatrix.vmax)
                        HUmask = ROImatrix.HUmask # mask is returned in that for loop used for LLL...
                        
                

                del wave_list
                
                if localRadiomics:
                    self.logger.info("Start: Local Radiomics")
                    #call NaN optimizer   
                    
                    iterations_n = len(centers) #to define how many time the radiomics need to be calculated
                    self.structure = [] #list to collect info if the subvolume belongs to the tumor or to the recurrence
                    
                    #to have list with n_bits and intervals for each centers (it will be the same number for eachcenter, but it is done for compability with wavelet)
                    n_bits_temp = []
                    interval_list_temp = []
                    for c in len(centers):
                        n_bits_temp.append([n_bits_list[0]])
                        interval_list_temp.append([interval_list[0]])
                    n_bits_list = n_bits_temp
                    interval_list = interval_list_temp
                    del n_bits_temp
                    del interval_list_temp
                    
                    #make the matrix_list and matrix_v_list with all the subregions to be analyzed
                
                try: #ValueError
                    #calulate features for original and transformed images or local centers if provided 
                    #feed in the list of maps to calculate
                    for w in arange(0, iterations_n):
                        matrix = matrix_list[w]
                        matrix_v = matrix_v_list[w]    
                        self.n_bits = n_bits_list[w]
                        interval = interval_list[w]
                        
                        try:
                            sb.SetStatusText(ImName +' matrix done '+ str(datetime.now().strftime('%H:%M:%S')))
                        except AttributeError:
                            print('attributeerrror')
                            pass
                        if rs_type[w]==1: #save only for the original image
                            self.saveImage(path, modality[i], matrix, ImName, pixNr)
                        more_than_one_pix = True
                        
                        try:
                        #histogram
                            histogram = self.fun_histogram(matrix_v, modality[i], ImName, pixNr, path, w)
                        except IndexError:
                            self.stop_calculation('only one voxel', [0])
                            print('one voxel')
                            more_than_one_pix = False
                        if self.n_bits == 'values out of range':
                            more_than_one_pix = False
                            self.stop_calculation('values out of range', [0])
                        elif self.n_bits == 'too small contour':
                            more_than_one_pix = False
                            self.stop_calculation('too small contour', [0])
                            
                        if more_than_one_pix:
                            histogram = np.array(histogram)
                            mean = self.fun_mean(histogram)
                            std = self.fun_std(histogram)
                            cov = self.fun_COV(histogram)
                            skewness = self.fun_skewness(histogram)
                            kurtosis = self.fun_kurtosis(histogram)
                            var = self.fun_var(histogram)
                            median = self.fun_median(histogram)
                            percentile10 = self.fun_percentile(histogram, 10)
                            percentile90 = self.fun_percentile(histogram, 90)
                            iqr = self.fun_interqR(histogram)
                            Hrange = self.fun_range(histogram)
                            mad = self.fun_mad(histogram)
                            rmad = self.fun_rmad(histogram, percentile10, percentile90)
                            H_energy = self.fun_H_energy(histogram)
                            H_entropy = self.fun_H_entropy(histogram, interval)
                            rms = self.fun_rms(histogram)
                            H_uniformity = self.fun_H_uniformity(histogram, interval)
                            del histogram
                            try:
                                sb.SetStatusText(ImName +' hist done '+str(datetime.now().strftime('%H:%M:%S')))
                                self.logger.info( ImName +' hist done '+str(datetime.now().strftime('%H:%M:%S')))
                            except AttributeError: #if no GUI
                                pass
                            
                            self.cms.append(self.centerMassShift(matrix_v, self.xCTspace))
                            mtv = self.metabolicTumorVolume(matrix_v, self.xCTspace)
                            self.mtv2.append(mtv[0])
                            self.mtv3.append(mtv[1])
                            self.mtv4.append(mtv[2])
                            self.mtv5.append(mtv[3])
                            self.mtv6.append(mtv[4])
                            self.mtv7.append(mtv[5])
                            
                            #coocurence matrix
                            energy_t=[]
                            entropy_t=[]
                            contrast_t=[]
                            correlation_t =[]
                            homogenity_t = []
                            homogenity_n_t = []
                            idiff_t = []
                            idiff_n_t = []
                            variance_t = []
                            average_t = []
                            sum_entropy_t = []
                            sum_variance_t = []
                            diff_entropy_t = []
                            diff_variance_t = []
                            IMC1_t = []
                            IMC2_t = []
                            MCC_t = []
                            joint_max_t = []
                            joint_average_t = []
                            diff_ave_t = []
                            dissim_t = []
                            inverse_var_t = []
                            autocorr_t = []
                            clust_t_t = []
                            clust_s_t = []
                            clust_p_t = []
                            #directions in COM
                            lista_t = [[0,0,1], [0,1,0], [1,0,0], [0,1,1],[0,1,-1],[1,0,1],[1,0,-1],[1,1,0],[1,-1,0],[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1]]
                            list_CO_merged = []
                            for c in lista_t:
                                co_matrix, co_matrix_non_normalized, p_plus, p_minus = self.coMatrix(matrix, c) #p_plus, p_minus - marginale probabilities
                                list_CO_merged.append(co_matrix_non_normalized)
                                energy_t.append(self.fun_energy(co_matrix))
                                ent = self.fun_entropy(co_matrix)
                                entropy_t.append(ent)
                                contrast_t.append(self.fun_contrast(co_matrix))
                                correlation_t.append(self.fun_correlation(co_matrix))
                                homo, homo_n = self.fun_homogenity(co_matrix)
                                homogenity_t.append(homo)
                                homogenity_n_t.append(homo_n)
                                idiff, idiff_n = self.fun_inverse_diff(co_matrix)
                                idiff_t.append(idiff)
                                idiff_n_t.append(idiff_n)
                                variance_t.append(self.fun_variance(co_matrix))
                                a, v = self.fun_sum_average_var(p_plus)
                                average_t.append(a)
                                sum_variance_t.append(v)
                                sum_entropy_t.append(self.fun_sum_entropy(p_plus))
                                a, v = self.fun_diff_average_var(p_minus)
                                diff_ave_t.append(a)
                                diff_variance_t.append(v)
                                diff_entropy_t.append(self.fun_diff_entropy(p_minus))
                                f1, f2 = self.fun_IMC(co_matrix, ent)
                                IMC1_t.append(f1)
                                IMC2_t.append(f2)
                                MCC_t.append(self.fun_MCC(co_matrix))
                                joint_max_t.append(self.fun_joint_max(co_matrix))
                                joint_average_t.append(self.fun_joint_average(co_matrix))
                                dissim_t.append(self.fun_dissimilarity(co_matrix))
                                inverse_var_t.append(self.fun_inverse_var(co_matrix))
                                autocorr_t.append(self.fun_autocorr(co_matrix))
                                clust1, clust2, clust3 = self.fun_cluster(co_matrix)
                                clust_t_t.append(clust1)
                                clust_s_t.append(clust2)
                                clust_p_t.append(clust3)
                                del co_matrix
                            #take avarege over all directions
                            energy = np.mean(energy_t)
                            entropy = np.mean(entropy_t)
                            contrast = np.mean(contrast_t)
                            correlation = np.mean(correlation_t)
                            homogenity = np.mean(homogenity_t)
                            homogenity_n = np.mean(homogenity_n_t)
                            idiff = np.mean(idiff_t)
                            idiff_n = np.mean(idiff_n_t)
                            variance = np.mean(variance_t)
                            sum_average = np.mean(average_t)
                            sum_entropy = np.mean(sum_entropy_t)
                            sum_variance = np.mean(sum_variance_t)
                            diff_entropy = np.mean(diff_entropy_t)
                            diff_variance = np.mean(diff_variance_t)
                            IMC1 = np.mean(IMC1_t)
                            IMC2 = np.mean(IMC2_t)
                            try:
                                MCC = np.mean(MCC_t)
                            except TypeError:  # see MCC function
                                MCC = np.nan
                            joint_max = np.mean(joint_max_t)
                            joint_average = np.mean(joint_average_t)
                            diff_ave = np.mean(diff_ave_t)
                            dissim = np.mean(dissim_t)
                            inverse_var = np.mean(inverse_var_t)
                            autocorr = np.mean(autocorr_t)
                            clust_t = np.mean(clust_t_t)
                            clust_s = np.mean(clust_s_t)
                            clust_p = np.mean(clust_p_t)
                            
                            # merged COM
                            CO_merged, p_plus_merged, p_minus_merged = self.merge_COM(list_CO_merged)
                            del list_CO_merged
                            self.COM_energy.append(self.fun_energy(CO_merged))
                            ent = self.fun_entropy(CO_merged)
                            self.COM_entropy.append(ent)
                            self.COM_contrast.append(self.fun_contrast(CO_merged))
                            self.COM_correlation.append(self.fun_correlation(CO_merged))
                            homo, homo_n = self.fun_homogenity(CO_merged)
                            self.COM_homogenity.append(homo)
                            self.COM_homogenity_n.append(homo_n)
                            com_idiff, com_idiff_n = self.fun_inverse_diff(CO_merged)
                            self.COM_idiff.append(com_idiff)
                            self.COM_idiff_n.append(com_idiff_n)
                            self.COM_variance.append(self.fun_variance(CO_merged))
                            a, v = self.fun_sum_average_var(p_plus_merged)
                            self.COM_sum_average.append(a)
                            self.COM_sum_variance.append(v)
                            self.COM_sum_entropy.append(self.fun_sum_entropy(p_plus_merged))
                            a, v = self.fun_diff_average_var(p_minus_merged)
                            self.COM_diff_ave.append(a)
                            self.COM_diff_variance.append(v)
                            self.COM_diff_entropy.append(self.fun_diff_entropy(p_minus_merged))
                            f1, f2 = self.fun_IMC(CO_merged, ent)
                            self.COM_IMC1.append(f1)
                            self.COM_IMC2.append(f2)
                            self.COM_MCC.append(self.fun_MCC(CO_merged))
                            self.COM_joint_max.append(self.fun_joint_max(CO_merged))
                            self.COM_joint_average.append(self.fun_joint_average(CO_merged))
                            self.COM_dissim.append(self.fun_dissimilarity(CO_merged))
                            self.COM_inverse_var.append(self.fun_inverse_var(CO_merged))
                            self.COM_autocorr.append(self.fun_autocorr(CO_merged))
                            clust1, clust2, clust3 = self.fun_cluster(CO_merged)
                            self.COM_clust_t.append(clust1)
                            self.COM_clust_s.append(clust2)
                            self.COM_clust_p.append(clust3)
                            del CO_merged
                            try:
                                sb.SetStatusText(ImName +' COM done '+ str(datetime.now().strftime('%H:%M:%S')))
                                self.logger.info( ImName +' COM done '+ str(datetime.now().strftime('%H:%M:%S')))
                            except AttributeError: #if no GUI
                                pass
                            
                            #neighborhood matrix
                            neighMatrix, neighMatrixNorm  = self.M3(matrix)
                            coarseness = self.fun_coarseness(neighMatrix, neighMatrixNorm, matrix)
                            neighContrast = self.fun_contrastM3(neighMatrix, neighMatrixNorm, matrix)
                            busyness = self.fun_busyness(neighMatrix, neighMatrixNorm, matrix)
                            complexity = self.fun_complexity(neighMatrix, neighMatrixNorm, matrix)
                            strength = self.fun_strength(neighMatrix, neighMatrixNorm, matrix)
                            del neighMatrix
                            del neighMatrixNorm
                            try:
                                sb.SetStatusText(ImName +' NGTDM done ' + str(datetime.now().strftime('%H:%M:%S')))
                                self.logger.info( ImName +' NGTDM done ' + str(datetime.now().strftime('%H:%M:%S')))
                            except AttributeError: #if no GUI
                                pass
                            
                            # gray level run length matrix
                            len_intensity_t = []
                            len_intensity_n_t = []
                            len_size_t = []
                            len_size_n_t = []
                            len_sse_t = []
                            len_lse_t = []
                            len_lgse_t = []
                            len_hgse_t = []
                            len_sslge_t = []
                            len_sshge_t = []
                            len_lslge_t = []
                            len_lshge_t = []
                            len_rpc_t = []
                            len_glv_t = []
                            len_lsv_t = []
                            len_size_entropy_t = []
                            list_GLRLM_merged = []
                            for c in lista_t:
                                GLRLM, GLRLM_norm = self.M4L(matrix, c, lista_t) #norm - sum over all entries
                                list_GLRLM_merged.append(GLRLM)
                                if GLRLM_norm == 0: 
                                    len_intensity_t,len_intensity_n_t,len_size_t,len_size_n_t,len_sse_t,len_lse_t,len_lgse_t,len_hgse_t,len_sslge_t,len_sshge_t,len_lslge_t,len_lshge_t,len_rpc_t,  len_glv_t, len_lsv_t, len_size_entropy_t = self.returnNan([len_intensity_t,len_intensity_n_t,len_size_t,len_size_n_t,len_sse_t,len_lse_t,len_lgse_t,len_hgse_t,len_sslge_t,len_sshge_t,len_lslge_t,len_lshge_t,len_rpc_t,  len_glv_t, len_lsv_t, len_size_entropy_t])
                                else:
                                    int_var, int_var_n = self.intensityVariability(GLRLM, GLRLM_norm)
                                    len_intensity_t.append(int_var)
                                    len_intensity_n_t.append(int_var_n)
                                    size_var, size_var_n = self.sizeVariability(GLRLM, GLRLM_norm)
                                    len_size_t.append(size_var)
                                    len_size_n_t.append(size_var_n)
                                    len_sse_t.append(self.shortSize(GLRLM, GLRLM_norm))
                                    len_lse_t.append(self.longSize(GLRLM, GLRLM_norm))
                                    len_lgse_t.append(self.LGSE(GLRLM, GLRLM_norm))
                                    len_hgse_t.append(self.HGSE(GLRLM, GLRLM_norm))
                                    len_sslge_t.append(self.SSLGE(GLRLM, GLRLM_norm))
                                    len_sshge_t.append(self.SSHGE(GLRLM, GLRLM_norm))
                                    len_lslge_t.append(self.LSLGE(GLRLM, GLRLM_norm))
                                    len_lshge_t.append(self.LSHGE(GLRLM, GLRLM_norm))
                                    len_rpc_t.append(self.runPer(GLRLM, GLRLM_norm))
                                    len_glv_t.append(self.fun_GLvar(GLRLM, GLRLM_norm))
                                    len_lsv_t.append(self.fun_LSvar(GLRLM, GLRLM_norm))
                                    len_size_entropy_t.append(self.fun_size_ent(GLRLM, GLRLM_norm))
                            # average over all directions
                            self.len_intensity.append(round(np.mean(len_intensity_t), 3))
                            self.len_intensity_n.append(round(np.mean(len_intensity_n_t), 3))
                            self.len_size.append(round(np.mean(len_size_t), 3))
                            self.len_size_n.append(round(np.mean(len_size_n_t), 3))
                            self.len_sse.append(round(np.mean(len_sse_t), 3))
                            self.len_lse.append(round(np.mean(len_lse_t), 3))
                            self.len_lgse.append(round(np.mean(len_lgse_t), 3))
                            self.len_hgse.append(round(np.mean(len_hgse_t), 3))
                            self.len_sslge.append(round(np.mean(len_sslge_t), 3))
                            self.len_sshge.append(round(np.mean(len_sshge_t), 3))
                            self.len_lslge.append(round(np.mean(len_lslge_t), 3))
                            self.len_lshge.append(round(np.mean(len_lshge_t), 3))
                            self.len_rpc.append(round(np.mean(len_rpc_t), 3))
                            self.len_glv.append(round(np.mean(len_glv_t), 3))
                            self.len_lsv.append(round(np.mean(len_lsv_t), 3))
                            self.len_size_entropy.append(round(np.mean(len_size_entropy_t), 3))
                            # GLRLM merged
                            GLRLM_merged, norm_GLRLM_merged = self.merge_GLRLM(list_GLRLM_merged)
                            if norm_GLRLM_merged == 0:
                                self.returnNan([self.M_len_intensity, self.M_len_intensity_n, self.M_len_size,self.M_len_size_n, self.M_len_sse, self.M_len_lse, self.M_len_lgse,self.M_len_hgse, self.M_len_sslge,   self.M_len_sshge,self.M_len_lslge,self.M_len_lshge,self.M_len_rpc,self.M_len_glv,self.M_len_lsv, self.M_len_size_entropy])
                            else:
                                int_var, int_var_n = self.intensityVariability(GLRLM_merged, norm_GLRLM_merged)
                                self.M_len_intensity.append(int_var)
                                self.M_len_intensity_n.append(int_var_n)
                                size_var, size_var_n = self.sizeVariability(GLRLM_merged, norm_GLRLM_merged)
                                self.M_len_size.append(size_var)
                                self.M_len_size_n.append(size_var_n)
                                self.M_len_sse.append(self.shortSize(GLRLM_merged, norm_GLRLM_merged))
                                self.M_len_lse.append(self.longSize(GLRLM_merged, norm_GLRLM_merged))
                                self.M_len_lgse.append(self.LGSE(GLRLM_merged, norm_GLRLM_merged))
                                self.M_len_hgse.append(self.HGSE(GLRLM_merged, norm_GLRLM_merged))
                                self.M_len_sslge.append(self.SSLGE(GLRLM_merged, norm_GLRLM_merged))
                                self.M_len_sshge.append(self.SSHGE(GLRLM_merged, norm_GLRLM_merged))
                                self.M_len_lslge.append(self.LSLGE(GLRLM_merged, norm_GLRLM_merged))
                                self.M_len_lshge.append(self.LSHGE(GLRLM_merged, norm_GLRLM_merged))
                                self.M_len_rpc.append(self.runPer(GLRLM_merged, norm_GLRLM_merged))
                                self.M_len_glv.append(self.fun_GLvar(GLRLM_merged, norm_GLRLM_merged))
                                self.M_len_lsv.append(self.fun_LSvar(GLRLM_merged, norm_GLRLM_merged))
                                self.M_len_size_entropy.append(self.fun_size_ent(GLRLM_merged, norm_GLRLM_merged))
                            self.logger.info(ImName +' GLRLM ' + str(datetime.now().strftime('%H:%M:%S')))

                            
                            # gray level size zone matrix and gray level distane zone matrix
                            dist_matrix = self.distanceMatrix(matrix)
                            GLSZM, norm_GLSZM, GLDZM, norm_GLDZM = self.M4(matrix, dist_matrix) #norm - sum over all entries
                            #GLSZM
                            if norm_GLSZM == 0:
                                self.returnNan([self.intensity, self.intensity_n, self.size, self.size_n,self.sse,self.lse,  self.lgse, self.hgse,self.sslge, self.sshge, self.lslge, self.lshge, self.rpc, self.glv, self.lsv,self.size_entropy])
                            else:
                                intensity, intensity_n = self.intensityVariability(GLSZM, norm_GLSZM)
                                size, size_n = self.sizeVariability(GLSZM, norm_GLSZM)
                                sse = self.shortSize(GLSZM, norm_GLSZM)
                                lse = self.longSize(GLSZM, norm_GLSZM)
                                lgse = self.LGSE(GLSZM, norm_GLSZM)
                                hgse = self.HGSE(GLSZM, norm_GLSZM)
                                sslge = self.SSLGE(GLSZM, norm_GLSZM)
                                sshge = self.SSHGE(GLSZM, norm_GLSZM)
                                lslge = self.LSLGE(GLSZM, norm_GLSZM)
                                lshge = self.LSHGE(GLSZM, norm_GLSZM)
                                rpc = self.runPer(GLSZM, norm_GLSZM)
                                glv = self.fun_GLvar(GLSZM, norm_GLSZM)
                                lsv = self.fun_LSvar(GLSZM, norm_GLSZM)
                                size_entropy = self.fun_size_ent(GLSZM, norm_GLSZM)
                                # GLSZM
                                self.intensity.append(round(intensity, 3))
                                self.intensity_n.append(round(intensity_n, 3))
                                self.size.append(round(size, 3))
                                self.size_n.append(round(size_n, 3))
                                self.sse.append(round(sse, 3))
                                self.lse.append(round(lse, 3))
                                self.lgse.append(round(lgse, 3))
                                self.hgse.append(round(hgse, 3))
                                self.sslge.append(round(sslge, 3))
                                self.sshge.append(round(sshge, 3))
                                self.lslge.append(round(lslge, 3))
                                self.lshge.append(round(lshge, 3))
                                self.rpc.append(round(rpc, 3))
                                self.glv.append(round(glv, 3))
                                self.lsv.append(round(lsv, 3))
                                self.size_entropy.append(round(size_entropy, 3))
                            del GLSZM
                            self.logger.info(ImName +' GLSZM done '+ str(datetime.now().strftime('%H:%M:%S')))
                            
                            try:
                                sb.SetStatusText(ImName +' GLSZM done '+ str(datetime.now().strftime('%H:%M:%S')))
                                self.logger.info( ImName +' GLSZM done '+ str(datetime.now().strftime('%H:%M:%S')))
                            except AttributeError: #if no GUI
                                pass
                            # GLDZM
                            if norm_GLDZM == 0:
                                self.returnNan([self.GLDZM_intensity,self.GLDZM_intensity_n,self.GLDZM_size,self.GLDZM_size_n, self.GLDZM_sse,self.GLDZM_lse, self.GLDZM_lgse,self.GLDZM_hgse,self.GLDZM_sslge,self.GLDZM_sshge, self.GLDZM_lslge,self.GLDZM_lshge,self.GLDZM_rpc,self.GLDZM_glv,self.GLDZM_lsv,self.GLDZM_size_entropy])
                            else:
                                intensity, intensity_n = self.intensityVariability(GLDZM, norm_GLDZM)
                                self.GLDZM_intensity.append(intensity)
                                self.GLDZM_intensity_n.append(intensity_n)
                                size, size_n = self.sizeVariability(GLDZM, norm_GLDZM)
                                self.GLDZM_size.append(size)
                                self.GLDZM_size_n.append(size_n)
                                self.GLDZM_sse.append(self.shortSize(GLDZM, norm_GLDZM))
                                self.GLDZM_lse.append(self.longSize(GLDZM, norm_GLDZM))
                                self.GLDZM_lgse.append(self.LGSE(GLDZM, norm_GLDZM))
                                self.GLDZM_hgse.append(self.HGSE(GLDZM, norm_GLDZM))
                                self.GLDZM_sslge.append(self.SSLGE(GLDZM, norm_GLDZM))
                                self.GLDZM_sshge.append(self.SSHGE(GLDZM, norm_GLDZM))
                                self.GLDZM_lslge.append(self.LSLGE(GLDZM, norm_GLDZM))
                                self.GLDZM_lshge.append(self.LSHGE(GLDZM, norm_GLDZM))
                                self.GLDZM_rpc.append(rpc)  # redundant parameter
                                self.GLDZM_glv.append(self.fun_GLvar(GLDZM, norm_GLDZM))
                                self.GLDZM_lsv.append(self.fun_LSvar(GLDZM, norm_GLDZM))
                                self.GLDZM_size_entropy.append(self.fun_size_ent(GLDZM, norm_GLDZM))
                            del GLDZM
                            self.logger.info(ImName +' GLDZM done '+ str(datetime.now().strftime('%H:%M:%S')))
                            
                            # NGLDM
                            NGLDM, norm_NGLDM = self.NGLDM(matrix) #norm - sum over all entries
                            if norm_NGLDM == 0:
                                self.returnNan([self.NGLDM_intensity, self.NGLDM_intensity_n,self.NGLDM_size, self.NGLDM_size_n, self.NGLDM_sse,  self.NGLDM_lse, self.NGLDM_lgse, self.NGLDM_hgse,   self.NGLDM_sslge, self.NGLDM_sshge, self.NGLDM_lslge, self.NGLDM_lshge, self.NGLDM_glv,self.NGLDM_lsv, self.NGLDM_size_entropy,self.NGLDM_energy])
                            else:
                                intensity, intensity_n = self.intensityVariability(NGLDM, norm_NGLDM)
                                self.NGLDM_intensity.append(intensity)
                                self.logger.info('NGLDM ' + str(intensity))
                                self.NGLDM_intensity_n.append(intensity_n)
                                size, size_n = self.sizeVariability(NGLDM, norm_NGLDM)
                                self.NGLDM_size.append(size)
                                self.NGLDM_size_n.append(size_n)
                                self.NGLDM_sse.append(self.shortSize(NGLDM, norm_NGLDM))
                                self.NGLDM_lse.append(self.longSize(NGLDM, norm_NGLDM))
                                self.NGLDM_lgse.append(self.LGSE(NGLDM, norm_NGLDM))
                                self.NGLDM_hgse.append(self.HGSE(NGLDM, norm_NGLDM))
                                self.NGLDM_sslge.append(self.SSLGE(NGLDM, norm_NGLDM))
                                self.NGLDM_sshge.append(self.SSHGE(NGLDM, norm_NGLDM))
                                self.NGLDM_lslge.append(self.LSLGE(NGLDM, norm_NGLDM))
                                self.NGLDM_lshge.append(self.LSHGE(NGLDM, norm_NGLDM))
                                self.NGLDM_glv.append(self.fun_GLvar(NGLDM, norm_NGLDM))
                                self.NGLDM_lsv.append(self.fun_LSvar(NGLDM, norm_NGLDM))
                                self.NGLDM_size_entropy.append(self.fun_size_ent(NGLDM, norm_NGLDM))
                                self.NGLDM_energy.append(self.fun_NGLDM_energy(NGLDM, norm_NGLDM))
                            self.logger.info( ImName +' NGLDM done '+ str(datetime.now().strftime('%H:%M:%S')))
                            del NGLDM
                            
                            frac = self.fractal(matrix, path, pixNr, ImName)
                            try:
                                sb.SetStatusText(ImName +' fractal done ' + str(datetime.now().strftime('%H:%M:%S')))
                                self.logger.info(ImName +' fractal done ' + str(datetime.now().strftime('%H:%M:%S')))
                            except AttributeError:
                                pass
                            
                            #add to the results
                            self.mean.append(round(mean, 3))
                            self.std.append(round(std, 3))
                            self.cov.append(round(cov, 3))
                            self.skewness.append(round(skewness, 3))
                            self.kurtosis.append(round(kurtosis, 3))
                            self.var.append(round(var, 3))
                            self.median.append(round(median, 3))
                            self.percentile10.append(round(percentile10, 3))
                            self.percentile90.append(round(percentile90, 3))
                            self.iqr.append(round(iqr, 3))
                            self.range.append(round(Hrange, 3))
                            self.mad.append(round(mad, 3))
                            self.rmad.append(round(rmad, 3))
                            self.H_energy.append(round(H_energy, 3))
                            self.H_entropy.append(round(H_entropy, 3))
                            self.rms.append(round(rms, 3))
                            self.H_uniformity.append(round(H_uniformity, 3))
                            #co
                            self.energy.append(round(energy, 3))
                            self.entropy.append(round(entropy, 3))
                            self.contrast.append(round(contrast, 3))
                            self.correlation.append(round(correlation, 3))
                            self.homogenity.append(round(homogenity, 3))
                            self.homogenity_n.append(round(homogenity_n, 3))
                            self.idiff.append(round(idiff, 3))
                            self.idiff_n.append(round(idiff_n, 3))
                            self.variance.append(round(variance, 3))
                            self.sum_average.append(round(sum_average, 3))
                            self.sum_entropy.append(round(sum_entropy, 3))
                            self.sum_variance.append(round(sum_variance, 3))
                            self.diff_entropy.append(round(diff_entropy, 3))
                            self.diff_variance.append(round(diff_variance, 3))
                            try:
                                self.IMC1.append(round(IMC1, 3))
                            except TypeError:
                                self.IMC1.append('')
                            self.IMC2.append(round(IMC2, 3))
                            self.MCC.append(round(MCC, 3))
                            self.joint_max.append(round(joint_max, 3))
                            self.joint_average.append(round(joint_average, 3))
                            self.diff_ave.append(round(diff_ave, 3))
                            self.dissim.append(round(dissim, 3))
                            self.inverse_var.append(round(inverse_var, 3))
                            self.autocorr.append(round(autocorr, 3))
                            self.clust_t.append(round(clust_t, 3))
                            self.clust_s.append(round(clust_s, 3))
                            self.clust_p.append(round(clust_p, 3))
                            #NGTDM
                            self.coarseness.append(round(coarseness,4))
                            try:
                                self.neighContrast.append(round(neighContrast,4))
                            except TypeError:
                                self.neighContrast.append('nan')
                            try:
                                self.busyness.append(round(busyness,4))
                            except TypeError:
                                self.busyness.append('nan')
                            self.complexity.append(round(complexity,4))
                            self.strength.append(round(strength,4))
                            try:
                                self.frac_dim.append(round(frac, 3))
                            except TypeError:
                                self.frac_dim.append('nan')
                            self.points.append(norm_points_list[0])
                            del matrix
                        
                except ValueError:
                    print(ValueError)
                    self.stop_calculation('ValueError', [1])
                    matrix.append([])
                    interval.append([])
                    norm_points.append([])
                    matrix_v.append([])
                    matrix_full.append([])
                    
            #except IndexError:#IndexError:
            #    matrix = []
            #    print IndexError
            #    self.stop_calculation('IndexError', rs_type)

    def stop_calculation(self, info, rs_type):
        '''returns empty string and error type bt does not stop the calculation'''
        for i in arange(0, len(rs_type)):
            self.structure = info
            self.mean.append('')
            self.std.append('')
            self.cov.append('')
            self.skewness.append('')
            self.kurtosis.append('')
            self.var.append('')
            self.median.append('')
            self.percentile10.append('')
            self.percentile90.append('')
            self.iqr.append('')
            self.range.append('')
            self.mad.append('')
            self.rmad.append('')
            self.H_energy.append('')
            self.H_entropy.append('')
            self.rms.append('')
            self.H_uniformity.append('')
            self.energy.append('')
            self.entropy.append('')
            self.contrast.append('')
            self.correlation.append('')
            self.points.append('')
            self.coarseness.append('')
            self.neighContrast.append('')
            self.busyness.append('')
            self.complexity.append('')
            self.strength.append('')
            self.homogenity.append('')
            self.homogenity_n.append('')
            self.idiff.append('')
            self.idiff_n.append('')
            self.variance.append('')
            self.sum_average.append('')
            self.sum_entropy.append('')
            self.sum_variance.append('')
            self.diff_entropy.append('')
            self.diff_variance.append('')
            self.IMC1.append('')
            self.IMC2.append('')
            self.MCC.append('')
            self.joint_max.append('')
            self.joint_average.append('')
            self.diff_ave.append('')
            self.dissim.append('')
            self.inverse_var.append('')
            self.autocorr.append('')
            self.clust_t.append('')
            self.clust_s.append('')
            self.clust_p.append('')
            self.COM_energy.append('')
            self.COM_entropy.append('')
            self.COM_contrast.append('')
            self.COM_correlation.append('')
            self.COM_homogenity.append('')
            self.COM_homogenity_n.append('')
            self.COM_idiff.append('')
            self.COM_idiff_n.append('')
            self.COM_variance.append('')
            self.COM_sum_average.append('')
            self.COM_sum_entropy.append('')
            self.COM_sum_variance.append('')
            self.COM_diff_entropy.append('')
            self.COM_diff_variance.append('')
            self.COM_IMC1.append('')
            self.COM_IMC2.append('')
            self.COM_MCC.append('')
            self.COM_joint_max.append('')
            self.COM_joint_average.append('')
            self.COM_diff_ave.append('')
            self.COM_dissim.append('')
            self.COM_inverse_var.append('')
            self.COM_autocorr.append('')
            self.COM_clust_t.append('')
            self.COM_clust_s.append('')
            self.COM_clust_p.append('')
            self.len_intensity.append('')
            self.len_intensity_n.append('')
            self.len_size.append('')
            self.len_size_n.append('')
            self.len_sse.append('')
            self.len_lse.append('')
            self.len_lgse.append('')
            self.len_hgse.append('')
            self.len_sslge.append('')
            self.len_sshge.append('')
            self.len_lslge.append('')
            self.len_lshge.append('')
            self.len_rpc.append('')
            self.len_glv.append('')
            self.len_lsv.append('')
            self.len_size_entropy.append('')
            self.M_len_intensity.append('')
            self.M_len_intensity_n.append('')
            self.M_len_size.append('')
            self.M_len_size_n.append('')
            self.M_len_sse.append('')
            self.M_len_lse.append('')
            self.M_len_lgse.append('')
            self.M_len_hgse.append('')
            self.M_len_sslge.append('')
            self.M_len_sshge.append('')
            self.M_len_lslge.append('')
            self.M_len_lshge.append('')
            self.M_len_rpc.append('')
            self.M_len_glv.append('')
            self.M_len_lsv.append('')
            self.M_len_size_entropy.append('')
            self.intensity.append('')
            self.intensity_n.append('')
            self.size.append('')
            self.size_n.append('')
            self.sse.append('')
            self.lse.append('')
            self.lgse.append('')
            self.hgse.append('')
            self.sslge.append('')
            self.sshge.append('')
            self.lslge.append('')
            self.lshge.append('')
            self.rpc.append('')
            self.glv.append('')
            self.lsv.append('')
            self.size_entropy.append('')
            self.GLDZM_intensity.append('')
            self.GLDZM_intensity_n.append('')
            self.GLDZM_size.append('')
            self.GLDZM_size_n.append('')
            self.GLDZM_sse.append('')
            self.GLDZM_lse.append('')
            self.GLDZM_lgse.append('')
            self.GLDZM_hgse.append('')
            self.GLDZM_sslge.append('')
            self.GLDZM_sshge.append('')
            self.GLDZM_lslge.append('')
            self.GLDZM_lshge.append('')
            self.GLDZM_rpc.append('')
            self.GLDZM_glv.append('')
            self.GLDZM_lsv.append('')
            self.GLDZM_size_entropy.append('')
            self.NGLDM_intensity.append('')
            self.NGLDM_intensity_n.append('')
            self.NGLDM_size.append('')
            self.NGLDM_size_n.append('')
            self.NGLDM_sse.append('')
            self.NGLDM_lse.append('')
            self.NGLDM_lgse.append('')
            self.NGLDM_hgse.append('')
            self.NGLDM_sslge.append('')
            self.NGLDM_sshge.append('')
            self.NGLDM_lslge.append('')
            self.NGLDM_lshge.append('')
            self.NGLDM_glv.append('')
            self.NGLDM_lsv.append('')
            self.NGLDM_size_entropy.append('')
            self.NGLDM_energy.append('')
            self.frac_dim.append('')
            self.cms.append('')
            self.mtv2.append('')
            self.mtv3.append('')
            self.mtv4.append('')
            self.mtv5.append('')
            self.mtv6.append('')
            self.mtv7.append('')
    
    def returnNan(self, mylist):
        '''appends nan to list of variable
        used for example if the normalization factor equal 0 -> zerodivisionerror'''
        for i in mylist:
            i.append(np.nan)
        return mylist

    def ret(self):
        '''return function'''
        return (self.vmin, self.vmax, self.structure, self.mean, self.std, self.cov, self.skewness, self.kurtosis, self.var,
        self.median, self.percentile10, self.percentile90, self.iqr, self.range, self.mad, self.rmad, self.H_energy,
        self.H_entropy, self.rms, self.H_uniformity,
        self.energy, self.entropy, self.contrast, self.correlation, self.homogenity, self.homogenity_n, self.idiff,
        self.idiff_n, self.variance, self.sum_average, self.sum_entropy, self.sum_variance, self.diff_entropy,
        self.diff_variance, self.IMC1, self.IMC2, self.MCC, self.joint_max, self.joint_average, self.diff_ave,
        self.dissim, self.inverse_var, self.autocorr, self.clust_t, self.clust_s, self.clust_p,
        self.COM_energy, self.COM_entropy, self.COM_contrast, self.COM_correlation, self.COM_homogenity,
        self.COM_homogenity_n, self.COM_idiff, self.COM_idiff_n, self.COM_variance, self.COM_sum_average,
        self.COM_sum_entropy, self.COM_sum_variance, self.COM_diff_entropy, self.COM_diff_variance, self.COM_IMC1,
        self.COM_IMC2, self.COM_MCC, self.COM_joint_max, self.COM_joint_average, self.COM_diff_ave, self.COM_dissim,
        self.COM_inverse_var, self.COM_autocorr, self.COM_clust_t, self.COM_clust_s, self.COM_clust_p,
        self.coarseness, self.neighContrast, self.busyness, self.complexity, self.strength,
        self.len_intensity, self.len_intensity_n, self.len_size, self.len_size_n, self.len_sse, self.len_lse,
        self.len_lgse, self.len_hgse, self.len_sslge, self.len_sshge, self.len_lslge, self.len_lshge, self.len_rpc,
        self.len_glv, self.len_lsv, self.len_size_entropy,
        self.M_len_intensity, self.M_len_intensity_n, self.M_len_size, self.M_len_size_n, self.M_len_sse,
        self.M_len_lse, self.M_len_lgse, self.M_len_hgse, self.M_len_sslge, self.M_len_sshge, self.M_len_lslge,
        self.M_len_lshge, self.M_len_rpc, self.M_len_glv, self.M_len_lsv, self.M_len_size_entropy,
        self.intensity, self.intensity_n, self.size, self.size_n, self.sse, self.lse, self.lgse, self.hgse, self.sslge,
        self.sshge, self.lslge, self.lshge, self.rpc, self.glv, self.lsv, self.size_entropy,
        self.GLDZM_intensity, self.GLDZM_intensity_n, self.GLDZM_size, self.GLDZM_size_n, self.GLDZM_sse,
        self.GLDZM_lse, self.GLDZM_lgse, self.GLDZM_hgse, self.GLDZM_sslge, self.GLDZM_sshge, self.GLDZM_lslge,
        self.GLDZM_lshge, self.GLDZM_rpc, self.GLDZM_glv, self.GLDZM_lsv, self.GLDZM_size_entropy,
        self.NGLDM_intensity, self.NGLDM_intensity_n, self.NGLDM_size, self.NGLDM_size_n, self.NGLDM_sse,
        self.NGLDM_lse, self.NGLDM_lgse, self.NGLDM_hgse, self.NGLDM_sslge, self.NGLDM_sshge, self.NGLDM_lslge,
        self.NGLDM_lshge, self.NGLDM_glv, self.NGLDM_lsv, self.NGLDM_size_entropy, self.NGLDM_energy,
        self.frac_dim, self.cms, self.mtv2, self.mtv3,self.mtv4,self.mtv5,self.mtv6,self.mtv7, self.points)


    def fun_histogram(self, M, name, ImName, pixNr, path, w):
        ''' calcuate and plot the histogram'''
        M1=[] # take all values except of nan
        for m in M:
            for i in arange(0, len(m)):
                for j in arange(0, len(m[i])):
                    if np.isnan(m[i][j]):
                        pass
                    else:
                        M1.append(m[i][j])
                        
        matplotlib.rcParams.update({'font.size': 24})

        fig = py.figure(300, figsize = (20,20))
        try:
            fig.text(0.5, 0.95, ImName+' '+name)
            py.hist(M1)
            try:
                makedirs(path+'histogram\\')
            except OSError:
                if not isdir(path+'histogram\\'):
                    raise
        except ValueError: 
            pass
        
        fig.savefig(path+'histogram\\'+name+'_'+ImName+'_'+self.structure+'_'+pixNr+'_'+str(w)+'.png')
        py.close()
        return M1

    def fun_mean(self, M1): #3.1.1
        m = np.mean(M1)
        return m

    def fun_std(self, M1):
        s = np.std(M1)
        return s

    def fun_var(self, M1): #3.1.2
        v = np.std(M1)**2
        return v

    def fun_COV(self, M1):
        '''coefficient of variation'''
        miu = np.mean(M1)
        cov = 0
        for i in M1:
            cov+=(i-miu)**2
        cov=np.sqrt(cov/float(len(M1)))/miu
        return cov

    def fun_skewness(self, M1): #3.1.3
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

    def fun_median(self, M1): #3.1.5
        m = np.median(M1)
        return m

    def fun_percentile(self, M1, px): #3.1.7, 3.1.8
        '''px percentile for example 75'''
        p = np.percentile(M1, px)
        return p

    def fun_interqR(self, M1): #3.1.10
        '''interquartile range'''
        irq = self.fun_percentile(M1, 75) - self.fun_percentile(M1, 25)
        return irq

    def fun_range(self, M1): #3.1.11
        r = np.max(M1) - np.min(M1)
        return r

    def fun_mad(self, M1): #3.1.12
        '''mean absolute diviation'''
        mad = np.sum(abs((np.array(M1)-np.mean(M1))))/float(len(M1))
        return mad

    def fun_rmad(self, M1, p10, p90): #3.1.13
        '''robust meand absolute deivation'''
        temp = list(M1)
        ind1 = np.where(np.array(temp)<p10)[0]
        for i in arange(1, len(ind1)+1):
            temp.pop(ind1[-i])
        ind2 = np.where(np.array(temp)>p90)[0]
        for i in arange(1, len(ind2)+1):
            temp.pop(ind2[-i])
        mad = np.sum(abs((np.array(temp)-np.mean(temp))))/float(len(temp))
        return mad

    def fun_H_energy(self, M1): #3.1.14
        e = np.sum(M1**2)
        return e

    def fun_H_entropy(self, M1, interval): #3.1.16
        '''interval - bin size'''
        vmin = np.min(M1)
        dM1 = ((M1 - vmin) // interval) + 1

        s = set(dM1)
        sl = list(s)
        w = []
        for si in arange(0, len(sl)):
            i = np.where(dM1 == sl[si])[0]
            w.append(len(i))
        p = 1.0 * np.array(w) / np.sum(w)

        e = -np.sum(p * np.log2(p))
        return e

    def fun_rms(self, M1): #3.1.15
        '''root mea square'''
        rms = np.sqrt(np.sum(M1**2)/len(M1))
        return rms

    def fun_H_uniformity(self, M1, interval): #3.1.17
        '''interval - bin size'''
        vmin = np.min(M1)
        dM1 = ((M1 - vmin) // interval) + 1

        g = list(set(dM1))
        p = []
        for gi in arange(0, len(g)):
            ind = np.where(np.array(dM1) == g[gi])[0]
            p.append(len(ind) * 1.0 / len(dM1))
        u = np.sum(np.array(p) ** 2)
        return u

    def fun_kurtosis(self, M1): #3.1.4
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

    def coMatrix(self, M, trans):
        '''Calculate the 2D co-occurence matrix from 3D structure matrix '''
        '''trans correponds to translation vector'''
        coMatrix = np.zeros((self.n_bits, self.n_bits))
        norm = 0
        for i in arange(0, len(M)):
            for y in arange(0, len(M[i])):
                for x in arange(0, len(M[i][y])):
                    if i + trans[2] >= 0 and y + trans[1] >= 0 and x + trans[0] >= 0 and i + trans[2] < len(M) and y + trans[1] < len(M[0]) and x + trans[0] < len(M[0][0]): #to check for indexerror
                        value1 = M[i][y][x]  # value in the structure matrix
                        value2 = M[i + trans[2]][y + trans[1]][x + trans[0]]
                        if not np.isnan(value1) and not np.isnan(value2):
                            y_cm = int(value1)
                            x_cm = int(value2)
                            coMatrix[y_cm][x_cm] += 1.
                            coMatrix[x_cm][y_cm] += 1.
                            norm += 2  # symmetrical matrix
                            # values at the digonal should be added only once, but not in IBSI definition
                            #        for i in arange(0, len(coMatrix)):
                            #            coMatrix[i][i] = coMatrix[i][i]/2.
        norm = np.sum(coMatrix)
        if norm != 0:
            coMatrixN = coMatrix / norm
        else:
            coMatrixN = np.zeros(shape = coMatrix.shape)
            coMatrixN[:] = np.nan
        #marignal probabilites
        p_minus = np.zeros(self.n_bits)
        p_plus = np.zeros(2 * self.n_bits + 3)
        for i in arange(0, len(coMatrixN)):
            for j in arange(0, len(coMatrixN[i])):
                p_minus[abs(i - j)] += coMatrixN[i][j]
                p_plus[abs(i + j) + 2] += coMatrixN[i][j]

        return coMatrixN, coMatrix, p_plus, p_minus

    def merge_COM(self, list_CO_merged):
        '''Merge GLCM from all directions and calculate average'''
        CO_merged = list_CO_merged[0]
        for i in arange(1, len(list_CO_merged)):
            CO_merged += list_CO_merged[i]
        CO_merged = CO_merged / float(np.sum(CO_merged))

        p_minus_merged = np.zeros(self.n_bits)
        p_plus_merged = np.zeros(2 * self.n_bits + 3)
        for i in arange(0, len(CO_merged)):
            for j in arange(0, len(CO_merged[i])):
                p_minus_merged[abs(i - j)] += CO_merged[i][j]
                p_plus_merged[abs(i + j) + 2] += CO_merged[i][j]

        return CO_merged, p_plus_merged, p_minus_merged

    def fun_energy(self, coM):  # 3.3.11
        energy = 0
        ind = np.where(coM != 0) #non-zero entries only to speed up calculation
        for j in arange(0, len(ind[0])):
            energy += (coM[ind[0][j]][ind[1][j]]) ** 2
        return energy

    def fun_entropy(self, coM):  # 3.3.4
        entropy = 0
        ind = np.where(coM != 0) #non-zero entries only to speed up calculation
        for j in arange(0, len(ind[0])):
            s = (coM[ind[0][j]][ind[1][j]]) * np.log2(coM[ind[0][j]][ind[1][j]])
            if np.isnan(s):
                pass
            else:
                entropy += -s
        if entropy == 0: #if empty matrix
            entropy = np.nan
        return entropy

    def fun_contrast(self, coM):  # 3.3.12
        contrast = 0
        ind = np.where(coM != 0) #non-zero entries only to speed up calculation
        for j in arange(0, len(ind[0])):
            contrast += ((ind[0][j] - ind[1][j]) ** 2) * coM[ind[0][j]][ind[1][j]]
        if contrast == 0:
            contrast = np.nan
        return contrast

    def fun_correlation(self, coM):  # 3.3.19
        '''symmetrical matrix'''
        mean = 0
        for i in arange(0, len(coM)):
            mean += (i + 1) * np.sum(coM[i]) #i+1 gray values starting from 1 not from 0

        std = 0
        for i in arange(0, len(coM)):
            std += ((i + 1 - mean) ** 2) * np.sum(coM[i]) #i+1 gray values starting from 1 not from 0
        std = np.sqrt(std)

        ind = np.where(np.array(coM) != 0) #non-zero entries only to speed up calculation#non-zero entries only to speed up calculation
        corr = 0
        for i in arange(0, len(ind[0])):
            corr += (ind[0][i] + 1) * (ind[1][i] + 1) * coM[ind[0][i]][ind[1][i]] #i+1 gray values starting from 1 not from 0
        corr = (corr - mean ** 2) / std ** 2
        return corr

    def fun_homogenity(self, coM):  # 3.3.16 and 3.3.17
        homo = 0
        nhomo = 0
        ind = np.where(coM != 0) #non-zero entries only to speed up calculation
        for j in arange(0, len(ind[0])):
            homo += coM[ind[0][j]][ind[1][j]] / (1 + (ind[0][j] - ind[1][j]) ** 2)
            nhomo += coM[ind[0][j]][ind[1][j]] / (1 + ((ind[0][j] - ind[1][j]) / float(len(coM))) ** 2)
        if homo == 0:
            homo = np.nan
        return homo, nhomo

    def fun_inverse_diff(self, coM):  # 3.3.14 and 3.3.15
        homo = 0
        nhomo = 0
        ind = np.where(coM != 0) #non-zero entries only to speed up calculation
        for j in arange(0, len(ind[0])):
            homo += coM[ind[0][j]][ind[1][j]] / (1 + abs(ind[0][j] - ind[1][j]))
            nhomo += coM[ind[0][j]][ind[1][j]] / (1 + abs(ind[0][j] - ind[1][j]) / float(len(coM)))
        if homo == 0:
            homo = np.nan
        return homo, nhomo

    def fun_variance(self, coM): #3.3.3
        var = 0
        miu = 0
        for i in arange(0, len(coM)):
            miu += (i + 1) * np.sum(coM[i]) #i+1 gray values starting from 1 not from 0
        ind = np.where(coM != 0) #non-zero entries only to speed up calculation
        for j in arange(0, len(ind[0])):
            var += (ind[0][j] + 1 - miu) ** 2 * coM[ind[0][j]][ind[1][j]] #i+1 gray values starting from 1 not from 0
        if var == 0:
            var = np.nan
        return var

    def fun_sum_average_var(self, p_plus):  # 3.3.8 and 3.3.9
        a = 0
        v = 0
        for k in arange(2, len(p_plus)):
            a += k * p_plus[k]

        for k in arange(2, len(p_plus)):
            v += (k - a) ** 2 * p_plus[k]

        return a, v

    def fun_sum_entropy(self, p_plus):  # 3.3.10
        e = 0
        for i in arange(2, len(p_plus)):
            if p_plus[i] != 0:
                e += -p_plus[i] * np.log2(p_plus[i])
        return e

    def fun_diff_entropy(self, p_minus):  # 3.3.7
        e = 0

        for i in arange(0, len(p_minus)):
            if p_minus[i] != 0:
                e += -p_minus[i] * np.log2(p_minus[i])

        return e

    def fun_IMC(self, coM, entropy):  # 3.3.24 and 3.3.25
        hxy = entropy

        X = []
        for i in arange(0, len(coM)):
            X.append(np.sum(coM[i]))

        hxy1 = 0
        hxy2 = 0
        for i in arange(0, len(coM)):
            for j in arange(0, len(coM[i])):
                if X[i] * X[j] != 0:
                    hxy1 += -coM[i][j] * np.log2(X[i] * X[j])
                    hxy2 += -X[i] * X[j] * np.log2(X[i] * X[j])

        hx = 0
        for i in arange(0, len(X)):
            if X[i] != 0:
                hx += -X[i] * np.log2(X[i])

        try:
            f12 = (hxy - hxy1) / hx
        except ZeroDivisionError:
            f12 = np.nan

        if hxy > hxy2:
            f13 = 0
        else:
            f13 = np.sqrt(1 - np.exp(-2 * (hxy2 - hxy)))

        return f12, f13

    def fun_MCC(self, coM):
        try:
            Q = np.zeros((len(coM), len(coM[0])))
            X = []
            for i in arange(0, len(coM)):
                X.append(np.sum(coM[i]))

            for i in arange(0, len(coM)):
                for j in arange(0, len(coM[i])):
                    for k in arange(0, len(X)):
                        if (X[i] * X[k]) != 0:
                            Q[i][j] += coM[i][k] * coM[j][k] / (X[i] * X[k])

            l = np.linalg.eigvals(Q)

            l.sort()
            try:
                return l[-2] ** 0.5
            except IndexError:  # due to not sufficient number of bits in wavelet transform
                return ''
        except np.linalg.linalg.LinAlgError:
            return np.nan

    def fun_joint_max(self, coM):  # 3.3.1
        return np.max(coM)

    def fun_joint_average(self, coM):  # 3.3.2
        s = 0
        for i in arange(0, len(coM)):
            s += (i + 1) * np.sum(coM[i]) #i+1 gray values starting from 1 not from 0
        return s

    def fun_diff_average_var(self, p_minus):  # 3.3.5 and 3.3.6
        a = 0
        v = 0
        for k in arange(0, len(p_minus)):
            a += k * p_minus[k]

        for k in arange(0, len(p_minus)):
            v += (k - a) ** 2 * p_minus[k]

        return a, v

    def fun_dissimilarity(self, coM):  # 3.3.13
        ds = 0
        ind = np.where(np.array(coM) != 0) #non-zero entries only to speed up calculation
        for i in arange(0, len(ind[0])):
            ds += abs(ind[0][i] - ind[1][i]) * coM[ind[0][i]][ind[1][i]]
        return ds

    def fun_inverse_var(self, coM):  # 3.3.18
        f = 0
        for i in arange(0, len(coM)):
            for j in arange(i + 1, len(coM[0])):
                f += coM[i][j] / (i - j) ** 2 
        return 2 * f

    def fun_autocorr(self, coM):  # 3.3.20
        c = 0
        ind = np.where(np.array(coM) != 0) #non-zero entries only to speed up calculation
        for i in arange(0, len(ind[0])):
            c += (ind[0][i] + 1) * (ind[1][i] + 1) * coM[ind[0][i]][ind[1][i]] #i+1 gray values starting from 1 not from 0
        return c

    def fun_cluster(self, coM):  # 3.3.21, 3.3.22 and 3.3.23
        mean = 0
        for i in arange(0, len(coM)):
            mean += (i + 1) * np.sum(coM[i])

        clust_t = 0
        clust_s = 0
        clust_p = 0
        ind = np.where(np.array(coM) != 0) #non-zero entries only to speed up calculation
        for i in arange(0, len(ind[0])):
            clust_t += ((ind[0][i] + ind[1][i] + 2 - 2 * mean) ** 2) * coM[ind[0][i]][ind[1][i]] #i+1 gray values starting from 1 not from 0
            clust_s += ((ind[0][i] + ind[1][i] + 2 - 2 * mean) ** 3) * coM[ind[0][i]][ind[1][i]]
            clust_p += ((ind[0][i] + ind[1][i] + 2 - 2 * mean) ** 4) * coM[ind[0][i]][ind[1][i]]
        return clust_t, clust_s, clust_p

    def M3(self, matrix):  # Amadasun et al. Textural Features Corresponding to Textural Properties
        '''neighborhood gray-tone difference matrix'''
        s = np.zeros(self.n_bits) #average gray level of voxels adjjacent to voxels with given gray value
        Ni = np.zeros(self.n_bits) #number of voxels of a given gray level
        for k in arange(0, len(matrix)):
            for v in arange(0, self.n_bits):
                index = np.where(matrix[k] == v)  # serach for a value level
                for ind in arange(0, len(index[0])):
                    temp = []
                    numerator = 0
                    for z in [-1, 1]:
                        for y in [-1, 0, 1]:
                            for x in [-1, 0, 1]:
                                if k + z >= 0 and index[0][ind] + y >= 0 and index[1][ind] + x >= 0 and k + z < len(
                                        matrix) and index[0][ind] + y < len(matrix[0]) and index[1][ind] + x < len(
                                        matrix[0][0]): #check for the index error
                                    temp.append(matrix[k + z][index[0][ind] + y][index[1][ind] + x])
                                else:
                                    numerator += 1
                    for y in [-1, 1]:
                        for x in [-1, 0, 1]:
                            if index[0][ind] + y >= 0 and index[1][ind] + x >= 0 and index[0][ind] + y < len(
                                    matrix[0]) and index[1][ind] + x < len(matrix[0][0]):
                                temp.append(matrix[k][index[0][ind] + y][index[1][ind] + x])
                            else:
                                numerator += 1
                    y = 0
                    for x in [-1, 1]:
                        if index[1][ind] + x >= 0 and index[1][ind] + x < len(matrix[0][0]):
                            temp.append(matrix[k][index[0][ind] + y][index[1][ind] + x])
                        else:
                            numerator += 1
                    ind_nan = np.where(np.isnan(np.array(temp)))[0]
                    for n in arange(1, len(ind_nan) + 1):
                        temp.pop(ind_nan[-n])
                    numerator += len(ind_nan)
                    if numerator != 26:
                        a = abs(v - (float(np.sum(temp)) / (26 - numerator)))
                        s[v] += a
                        Ni[v] += 1
        return s, Ni

    def fun_coarseness(self, s, Ni, matrix): #3.6.1
        f = 0
        ind = np.where(np.array(Ni) != 0)[0]
        for i in ind:
            f += s[i] * Ni[i] / np.sum(Ni)
        if f == 0:
            f = np.nan
        else:
            f = 1. / (0.000000001 + f)
        return f

    def fun_busyness(self, s, Ni, matrix): #3.6.3
        try:
            nom = 0
            denom = 0
            ind = np.where(np.array(Ni) != 0)[0]
            for i in ind:
                nom += s[i] * Ni[i] / np.sum(Ni)
                for j in ind:
                    denom += abs(float((i + 1) * Ni[i]) / (np.sum(Ni)) - float((j + 1) * Ni[j]) / (
                    np.sum(Ni)))  # to adapt i = [1:Ng]
            if nom / denom == 0:
                return np.nan
            else:
                return nom / denom  # don't divide by 2 to adapt for the oncoray
        except ZeroDivisionError:
            return ''

    def fun_contrastM3(self, s, Ni, matrix): #3.6.2.
        try:
            Ng = len(np.where(np.array(Ni) != 0)[0])
            ind = np.where(np.array(Ni) != 0)[0]
            s1 = 0
            for i in ind:
                for j in ind:
                    s1 += float(Ni[i]) / (np.sum(Ni)) * float(Ni[j]) / (np.sum(Ni)) * (i - j) ** 2
            s2 = 0
            ind = np.where(np.array(s) != 0)[0]
            for i in ind:
                s2 += s[i]

            f = (1. / (Ng * (Ng - 1))) * s1 * (1. / (np.sum(Ni))) * s2
            if f == 0:
                return np.nan
            else:
                return f
        except ZeroDivisionError:
            return ''

    def fun_complexity(self, s, Ni, matrix): #3.6.4
        ind = np.where(np.array(Ni) != 0)[0]
        s1 = 0
        for i in ind:
            for j in ind:
                s1 += (abs(i - j) / (float(Ni[i]) + float(Ni[j]))) * (
                (s[i] * float(Ni[i]) / (np.sum(Ni))) + (s[j] * float(Ni[j]) / (np.sum(Ni))))
        if s1 == 0:
            s1 = np.nan
        return s1

    def fun_strength(self, s, Ni, matrix): #3.6.5
        ind = np.where(np.array(Ni) != 0)[0]
        s1 = 0
        for i in ind:
            for j in ind:
                s1 += ((float(Ni[i]) + float(Ni[j])) / np.sum(Ni)) * (i - j) ** 2
        s2 = np.sum(s)

        s = s1 / s2

        if s2 == 0:
            s = 0
        return s

    def M4L(self, matrix, di, direction):
        '''gray-level run length matrix'''
        GLRLM = []
        Smax = 1
        for i in arange(0, self.n_bits):
            GLRLM.append([0])

        # library of planes
        planes = {}
        planes['[0, 0, 1]'] = self.M4L_choose_plane([0, 0, 1], matrix)
        planes['[0, 1, 0]'] = self.M4L_choose_plane([0, 1, 0], matrix)
        planes['[1, 0, 0]'] = self.M4L_choose_plane([1, 0, 0], matrix)
        planes['[0, 1, 1]'] = self.M4L_choose_plane([0, 1, 1], matrix)
        planes['[0, 1, -1]'] = self.M4L_choose_plane([0, 1, -1], matrix)
        planes['[1, 0, 1]'] = self.M4L_choose_plane([1, 0, 1], matrix)
        planes['[1, 0, -1]'] = self.M4L_choose_plane([1, 0, -1], matrix)
        planes['[1, 1, 0]'] = self.M4L_choose_plane([1, 1, 0], matrix)
        planes['[1, -1, 0]'] = self.M4L_choose_plane([1, -1, 0], matrix)
        planes['[1, 1, 1]'] = self.M4L_choose_plane([1, 1, 1], matrix)
        planes['[1, 1, -1]'] = self.M4L_choose_plane([1, 1, -1], matrix)
        planes['[1, -1, 1]'] = self.M4L_choose_plane([1, -1, 1], matrix)
        planes['[1, -1, -1]'] = self.M4L_choose_plane([1, -1, -1], matrix)

        seeds = planes[str(di)]  # plane used as the begining of the rays

        vector_len = max([len(matrix), len(matrix[0]), len(matrix[0][0]),
                          int(np.sqrt(len(matrix) ** 2 + len(matrix[0]) ** 2)) + 1,
                          int(np.sqrt(len(matrix) ** 2 + len(matrix[0][0]) ** 2)) + 1,
                          int(np.sqrt(len(matrix[0][0]) ** 2 + len(matrix[0]) ** 2)) + 1])
        v = arange(0, vector_len)

        Vx = []  # coordinates of the vectors to extract
        Vy = []
        Vz = []

        vm = []  # vecotr with gray values to calculate the matrix

        # rays from the plane
        vm = self.M4L_ray(matrix, seeds, di, v, shiftX=0, shiftY=0, shiftZ=0)
        if np.ndim(vm) != 1: # #to account for different dimensions
            nan_list = [np.nan for inan in arange(0,np.ndim(vm)+1)]
            vm.append(nan_list)
        
        # move the plane to account for diagonal crossings
        if np.sum(di) != 1 or di[0] * di[1] * di[2] != 0:
            v = arange(0, vector_len + 1)
            if di[2] == -1:
                new_vm = self.M4L_ray(matrix, seeds, di, v, shiftX=1, shiftY=0, shiftZ=0) #to account for different dimensions
                if np.ndim(new_vm) != 1:
                    nan_list = [np.nan for inan in arange(0,np.ndim(new_vm)+1)]
                    new_vm.append(nan_list)
                vm = np.concatenate((vm, new_vm))
            else:
                new_vm = self.M4L_ray(matrix, seeds, di, v, shiftX=-1, shiftY=0, shiftZ=0)
                if np.ndim(new_vm) != 1:
                    nan_list = [np.nan for inan in arange(0,np.ndim(new_vm)+1)]
                    new_vm.append(nan_list)
                vm = np.concatenate((vm, new_vm))
            if di in direction[5:]:
                new_vm = self.M4L_ray(matrix, seeds, di, v, shiftX=0, shiftY=0, shiftZ=-1)
                if np.ndim(new_vm) != 1:
                    nan_list = [np.nan for inan in arange(0,np.ndim(new_vm)+1)]
                    new_vm.append(nan_list)
                vm = np.concatenate((vm, new_vm))
            if di in direction[9:]:
                if di[2] == -1:
                    new_vm = self.M4L_ray(matrix, seeds, di, v, shiftX=1, shiftY=0, shiftZ=-1)
                    if np.ndim(new_vm) != 1:
                        nan_list = [np.nan for inan in arange(0,np.ndim(new_vm)+1)]
                        new_vm.append(nan_list)
                    vm = np.concatenate((vm, new_vm))
                else:
                    new_vm = self.M4L_ray(matrix, seeds, di, v, shiftX=-1, shiftY=0, shiftZ=-1)
                    if np.ndim(new_vm) != 1:
                        nan_list = [np.nan for inan in arange(0,np.ndim(new_vm)+1)]
                        new_vm.append(nan_list)
                    vm = np.concatenate((vm, new_vm))

        GLRLM, Smax = self.M4L_fill(vm, GLRLM, Smax, int(np.nanmin(matrix)), int(np.nanmax(matrix)))

        GLRLM = np.array(GLRLM)
        GLRLM.astype(np.float)
        return GLRLM, float(np.sum(GLRLM))

    def M4L_fill(self, v, M, Smax, vmin, vmax):
        v = np.array(v)  # matrix of the vector with gray values
        for g in arange(vmin, vmax + 1):  # g - gray values in the vector
            for vi in arange(0, len(v)):  # vi - single vector
                ind = np.where(np.array(v[vi]) == g)[0]  # where in this vector is g
                s = 1  # length of the vector
                S = 0
                for i in arange(0, len(ind) - 1):
                    if ind[i + 1] - ind[i] == 1:  # if the neighboor has the same gray value, increase the size
                        s += 1
                    else:
                        if s != 1:  # if not the length needs to be added to main GLRLM matrix
                            if s > Smax:  # check is if Smax needs to be increased
                                for Ms in arange(0, len(M)):
                                    for si in arange(0, s - Smax):
                                        M[Ms].append(0)
                                Smax = s
                            M[g][s - 1] += 1
                            S += s
                            s = 1
                if s != 1:  # check if in the matrix left something to be added
                    if s > Smax:
                        for Ms in arange(0, len(M)):
                            for si in arange(0, s - Smax):
                                M[Ms].append(0)
                        Smax = s
                    M[g][s - 1] += 1
                    S += s
                M[g][0] += len(ind) - S  # fill all the lenght equal to 1

        return M, Smax

    def M4L_choose_plane(self, d, matrix):
        '''choose plane where rays for a given angl should start'''
        max_dim = max([len(matrix), len(matrix[0]), len(matrix[0][0])]) + 1
        seeds = []
        if d[0] >= 0 and d[1] >= 0 and d[2] >= 0:
            corner = [0, 0, 0]
        elif np.sum(d[1:]) == -2:
            corner = [0, len(matrix[0]) - 1, len(matrix[0][0]) - 1]
        elif d[1] == -1:
            corner = [0, len(matrix[0]), 0]
        elif d[2] == -1:
            corner = [0, 0, len(matrix[0][0])]
        if d[0] == 0:
            if np.sum(d) == 1:
                for z in arange(0, len(matrix) + 1):
                    for cor in arange(-max_dim, max_dim):
                        seeds.append(
                            [z + corner[0], (cor + corner[1]) * abs(d[1] - 1), (cor + corner[2]) * abs(d[2] - 1)])
            else:
                for z in arange(0, len(matrix) + 1):
                    for cor in arange(-max_dim, max_dim):
                        seeds.append([z + corner[0], (cor + corner[1]) * d[1], (cor + corner[2]) * (-d[2])])
        else:
            if d[1] == 0 and d[2] == 0:
                for x in arange(0, max_dim + 1):
                    for y in arange(0, max_dim + 1):
                        seeds.append([0, (y + corner[1]), (x + corner[2])])
            elif d[0] >= 0 and d[1] >= 0 and d[2] >= 0:
                for z in arange(-max_dim, max_dim + 1):
                    for cor in arange(-max_dim, max_dim + 1):
                        seeds.append([z + corner[0], (cor + corner[1]) - z, -(cor + corner[2]) - z])
            elif np.sum(d[1:]) == -2:
                for cor in arange(-max_dim - 1, max_dim + 1):
                    for z in arange(-max_dim, max_dim + 1):
                        seeds.append([z + corner[0], (cor + corner[1]) + z, (-cor + corner[2]) + z])
            elif d[2] == -1:
                for cor in arange(-max_dim, max_dim + 1):
                    for z in arange(-max_dim, max_dim + 1):
                        seeds.append([z + corner[0], (cor + corner[1]) - z, (cor + corner[2]) + z])
            elif d[1] == -1:
                for cor in arange(-max_dim, max_dim + 1):
                    for z in arange(-max_dim, max_dim + 1):
                        seeds.append([z + corner[0], (cor + corner[1]) + z, (cor + corner[2]) - z])

        return seeds

    def M4L_remove(self, vx, vy, vz, dx, dy, dz):
        indx = np.where(np.array(vx) >= dx)[0]
        indx = np.concatenate((indx, np.where(np.array(vx) < 0)[0]))
        indy = np.where(np.array(vy) >= dy)[0]
        indy = np.concatenate((indy, np.where(np.array(vy) < 0)[0]))
        indz = np.where(np.array(vz) >= dz)[0]
        indz = np.concatenate((indz, np.where(np.array(vz) < 0)[0]))
        ind = np.concatenate((indx, indy, indz))
        ind = list(set(ind))
        ind.sort()
        ind.reverse()
        vx = list(vx)
        vy = list(vy)
        vz = list(vz)
        for i in ind:
            vx.pop(i)
            vy.pop(i)
            vz.pop(i)
        return vx, vy, vz

    def M4L_ray(self, matrix, seeds, di, v, shiftX, shiftY, shiftZ):
        vm = []
        for s in seeds:
            vx = s[2] + shiftX + di[2] * v
            vy = s[1] + shiftY + di[1] * v
            vz = s[0] + shiftZ + di[0] * v
            vx, vy, vz = self.M4L_remove(vx, vy, vz, len(matrix[0][0]), len(matrix[0]), len(matrix))
            if vx != []:
                vm.append(matrix[(np.array(vz), np.array(vy), np.array(vx))])
        return vm

    def merge_GLRLM(self, list_GLRLM_merged):
        # list_GLRLM_merged - contains list of GLRLM for each direction
        # find maximum size
        size = [len(m[0]) for m in list_GLRLM_merged]
        size_max = np.max(size)
        # extend matricies
        for i in arange(0, len(list_GLRLM_merged)):
            temp = list_GLRLM_merged[i].tolist()
            for j in arange(0, len(temp)):
                for n in arange(len(temp[j]), size_max):
                    temp[j].append(0)
            list_GLRLM_merged[i] = np.array(temp)

        GLRLM_merged = list_GLRLM_merged[0]
        for i in arange(1, len(list_GLRLM_merged)):
            GLRLM_merged += list_GLRLM_merged[i]
        norm_GLRLM_merged = float(np.sum(GLRLM_merged))

        GLRLM_merged = np.array(GLRLM_merged)
        GLRLM_merged.astype(np.float)

        return GLRLM_merged, norm_GLRLM_merged

    def M4(self, matrix, dist_matrix): #Guillaume Thibault et al., ADVANCED STATISTICAL MATRICES FOR TEXTURE CHARACTERIZATION: APPLICATION TO DNA CHROMATIN AND MICROTUBULE NETWORK CLASSIFICATION
        '''gray-level size zone matrix'''
        '''gray-level distance zone matrix'''
        GLSZM = []
        GLDZM = []
        m = np.array(matrix).copy()
        m.dtype = np.float
        Smax = 1 #maximal size
        Dmax = 1 #maximal distance
        for i in arange(0, self.n_bits):
            GLSZM.append([0])
            GLDZM.append([0])
        for k in arange(0, len(m)):
            for i in arange(0, len(m[k])):
                for j in arange(0, len(m[k][i])):
                    if np.isnan(m[k][i][j]):
                        pass
                    else:
                        v = int(m[k][i][j])
                        size = 1
                        m[k][i][j] = np.nan
                        points = self.neighbor(k,i,j, m,v) #searching for neighbors with the same value
                        size+=len(points)
                        zone = [[k,i,j]] #contains coordinates of points in the zone
                        for ni in points:
                            zone.append(ni)
                            m[ni[0]][ni[1]][ni[2]] = np.nan
                        while len(points)!=0:
                            p = []
                            for n in arange(0, len(points)):
                                poin = self.neighbor(points[n][0],points[n][1],points[n][2], m,v)
                                for ni in poin:
                                    zone.append(ni)
                                    m[ni[0]][ni[1]][ni[2]] = np.nan
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

                        #define minimum distance
                        distance = []
                        for zi in zone:
                            distance.append(dist_matrix[zi[0], zi[1],zi[2]])
                        min_distance = int(np.min(distance))
                        if min_distance>Dmax:
                            for s in arange(0, len(GLDZM)):
                                for si in arange(0,min_distance-Dmax):
                                    GLDZM[s].append(0)
                            Dmax=min_distance
                            GLDZM[v][min_distance-1] +=1
                        else:
                            GLDZM[v][min_distance-1] +=1

        for i in arange(0, len(GLSZM)):
            GLSZM[i] = np.array(GLSZM[i])
        GLSZM = np.array(GLSZM)#no normalization according to IBSI /float(np.sum(GLSZM))
        norm_GLSZM = np.sum(GLSZM)
        GLSZM.astype(np.float)

        GLDZM = np.array(GLDZM)#no normalization according to IBSI /float(np.sum(GLSZM))
        norm_GLDZM = np.sum(GLDZM)
        GLDZM.astype(np.float)

        return GLSZM, norm_GLSZM, GLDZM, norm_GLDZM
                                
    def neighbor(self, z,y,x, matrix,v):
        '''search for neighbours with the same gray level'''
        points = []
        for k in arange(-1,2):
            for i in arange(-1,2):
                for j in arange(-1,2):
                    try:
                        if matrix[z+k][y+i][x+j] == v and z+k >= 0 and y+i >= 0 and x+j >= 0:
                            points.append([z+k,y+i,x+j])
                    except IndexError:
                        pass
        return points

    def distanceMatrix(self, matrix):
        # distance to the ROI border, this distance is not exactly the same as in the IBSI, it just seacrhine for the closest nan voxel not a clostest nan voxel from the original ROI
        dm = np.array(matrix).copy()
        (indz, indy, indx) = np.where(~np.isnan(matrix))
        for i in arange(0, len(indz)):
            dist = []  # vector of distances for one voxel
            z = matrix[:, indy[i], indx[i]]
            nanz = np.where(np.isnan(z))[0]
            d = []
            if len(nanz) != 0:
                d = list(abs(nanz - indz[i]))
            d.append(indz[i] + 1 - 0)
            d.append(len(matrix) - indz[i])
            dist.append(np.min(d))

            y = matrix[indz[i], :, indx[i]]
            nany = np.where(np.isnan(y))[0]
            d = []
            if len(nany) != 0:
                d = list(abs(nany - indy[i]))
            d.append(indy[i] + 1 - 0)
            d.append(len(matrix[0]) - indy[i])
            dist.append(np.min(d))

            x = matrix[indz[i], indy[i], :]
            nanx = np.where(np.isnan(x))[0]
            d = []
            if len(nanx) != 0:
                d = list(abs(nanx - indx[i]))
            d.append(indx[i] + 1 - 0)
            d.append(len(matrix[0][0]) - indx[i])
            dist.append(np.min(d))

            dm[indz[i], indy[i], indx[i]] = np.min(dist)

        dm = np.array(dm)
        return dm

    def sizeVariability(self, GLSZM, norm):  # 3.4.11 and #3.4.12
        var = 0
        GLSZM = GLSZM / np.sqrt(norm)
        for n in arange(0, len(GLSZM[0])):
            s = 0
            for m in arange(0, len(GLSZM)):
                s += GLSZM[m][n]
            var += s ** 2 #to avoid overflow
        return var, float(var) / norm 

    def intensityVariability(self, GLSZM, norm):  # 3.4.9 and #3.4.10
        var = 0
        GLSZM = GLSZM / np.sqrt(norm)
        for m in arange(0, len(GLSZM)):
            s = 0
            for n in arange(0, len(GLSZM[m])):
                s += GLSZM[m][n]
            var += s ** 2 #to avoid overflow error
        return var, float(var) / norm 

    def shortSize(self, GLSZM, norm):  # 3.4.1
        sse = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                sse += GLSZM[i][j] / float(np.uint(j + 1) ** 2)  # place 0 in the list corresponds to size 1
        return sse / norm

    def longSize(self, GLSZM, norm):  # 3.4.2
        lse = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                lse += GLSZM[i][j] * float(np.uint(j + 1) ** 2)  # place 0 in the list corresponds to size 1
        return lse / norm

    def LGSE(self, GLSZM, norm):  # 3.4.3
        lgse = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                lgse += GLSZM[i][j] / float(np.uint(i + 1) ** 2)  # otherwise level 0 is not included
        return lgse / norm

    def HGSE(self, GLSZM, norm):  # 3.4.4
        hgse = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                hgse += GLSZM[i][j] * float(np.uint(i + 1) ** 2)  # otherwise level 0 is not included
        return hgse / norm

    def SSLGE(self, GLSZM, norm):  # 3.4.5
        sslge = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                sslge += GLSZM[i][j] / float((np.uint(j + 1) ** 2 * (i + 1) ** 2))  # otherwise level 0 is not included
        return sslge / norm

    def SSHGE(self, GLSZM, norm):  # 3.4.6
        sshge = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                sshge += GLSZM[i][j] * float((i + 1) ** 2) / float(np.uint(j + 1) ** 2)  # otherwise level 0 is not included
        return sshge / norm

    def LSLGE(self, GLSZM, norm):  # 3.4.7
        lslge = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                lslge += GLSZM[i][j] * float(np.uint(j + 1) ** 2) / float((i + 1) ** 2)  # otherwise level 0 is not included
        return lslge / norm

    def LSHGE(self, GLSZM, norm):  # 3.4.8
        lshge = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                lshge += GLSZM[i][j] * float(np.uint(j + 1) ** 2 * (i + 1) ** 2)  # otherwise level 0 is not included
        return lshge / norm

    def runPer(self, GLSZM, norm):  # 3.4.13
        Nv = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                Nv += (j + 1) * GLSZM[i][j]
        rpc = norm / float(Nv)
        return rpc

    def fun_GLvar(self, GLSZM, norm):  # 3.4.14
        pGLSZM = GLSZM / float(norm)
        miu = 0
        for i in arange(0, len(GLSZM)):
            miu += (i + 1) * np.sum(pGLSZM[i])

        glv = 0
        for i in arange(0, len(GLSZM)):
            glv += ((i + 1 - miu) ** 2) * np.sum(pGLSZM[i])

        return glv

    def fun_LSvar(self, GLSZM, norm):  # 3.4.15
        pGLSZM = GLSZM / float(norm)
        miu = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                miu += (j + 1) * np.sum(pGLSZM[i][j])

        lsv = 0
        for i in arange(0, len(GLSZM)):
            for j in arange(0, len(GLSZM[i])):
                lsv += ((j + 1 - miu) ** 2) * pGLSZM[i][j]

        return lsv

    def fun_size_ent(self, GLSZM, norm):  # 3.4.16
        pGLSZM = GLSZM / float(norm)
        ind = np.where(pGLSZM != 0)
        e = 0
        for i in arange(0, len(ind[0])):
            e += -pGLSZM[ind[0][i]][ind[1][i]] * np.log2(pGLSZM[ind[0][i]][ind[1][i]])
        return e

    def NGLDM(self, matrix):  # Oncoray 4.2
        """Calculate neighborhood gray-level dependence matrix"""
        s = []
        for i in arange(0, self.n_bits):
            s.append([0])
        maxSize = 0

        for k in arange(0, len(matrix)):
            for v in arange(0, self.n_bits):
                index = np.where(matrix[k] == v)  # serach for a value level
                for ind in arange(0, len(index[0])):
                    temp = []
                    numerator = 0
                    for z in [-1, 1]:
                        for y in [-1, 0, 1]:
                            for x in [-1, 0, 1]:
                                if k + z >= 0 and index[0][ind] + y >= 0 and index[1][ind] + x >= 0 and k + z < len(
                                        matrix) and index[0][ind] + y < len(matrix[0]) and index[1][ind] + x < len(
                                        matrix[0][0]):
                                    temp.append(matrix[k + z][index[0][ind] + y][index[1][ind] + x])
                                else:
                                    numerator += 1
                    for y in [-1, 1]:
                        for x in [-1, 0, 1]:
                            if index[0][ind] + y >= 0 and index[1][ind] + x >= 0 and index[0][ind] + y < len(
                                    matrix[0]) and index[1][ind] + x < len(matrix[0][0]):
                                temp.append(matrix[k][index[0][ind] + y][index[1][ind] + x])
                            else:
                                numerator += 1
                    y = 0
                    for x in [-1, 1]:
                        # try:
                        if index[1][ind] + x >= 0 and index[1][ind] + x < len(matrix[0][0]):
                            temp.append(matrix[k][index[0][ind] + y][index[1][ind] + x])
                        # except IndexError:
                        else:
                            numerator += 1
                    ind_nan = np.where(np.isnan(np.array(temp)))[0]
                    for n in arange(1, len(ind_nan) + 1):
                        temp.pop(ind_nan[-n])
                    numerator += len(ind_nan)
                    if numerator != 26:  # if it has neigbourhood
                        size = len(np.where(np.array(temp) == v)[0])
                        if size > maxSize:
                            for gray in arange(0, len(s)):
                                for app in arange(maxSize, size):
                                    s[gray].append(0)
                            maxSize = size  # update maxSize
                        s[int(v)][size] += 1
                       

        s = np.array(s)
        s.astype(np.float)
        norm = float(np.sum(s))
        return s, norm

    def fun_NGLDM_energy(self, NGLDM, norm): #4.2.17
        pNGLDM = NGLDM / float(norm)
        e = np.sum(pNGLDM ** 2)
        return e

    def fractal(self, m, path, pixNr, ImName):
        #https://en.wikipedia.org/wiki/Box_counting
        '''fractal dimension'''
        try:
            def func_lin(x,a,b):
                return x*a+b
            
            maxR = np.min([len(m),len(m[0]),len(m[0][0])])
            frac = []
            for r in arange(2, maxR+1): #because log(1) = 0
                N=0
                for z in arange(0, len(m), r):
                    for y in arange(0, len(m[0]),r):
                        for x in arange(0, len(m[0][0]),r):
                            m =np.array(m)
                            matrix=m[z:z+r,y:y+r,x:x+r] #doesn't produce indexerror
                            ind = len(np.where(np.isnan(matrix))[0])
                            if ind<(r**3):
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
            #print path+'fractals\\'+ImName+'.png'
            try:
                makedirs(path+'fractals\\')
            except OSError:
                if not isdir(path+'fractals\\'):
                    raise
            py.savefig(path+'fractals\\'+ImName+'_'+self.structure+'_'+pixNr+'.png')
            py.close()
            return -result[0][0]
        except TypeError:
            return ''
            pass

    def centerMassShift(self, matrix, voxel_size):
        # calculated on original gray values
        ind = np.where(~np.isnan(matrix))
        ind_r = list(ind)

        ind_r[0] = ind_r[0] * voxel_size
        ind_r[1] = ind_r[1] * voxel_size
        ind_r[2] = ind_r[2] * voxel_size
        geo = np.array([np.sum(ind_r[0]), np.sum(ind_r[1]), np.sum(ind_r[2])])
        geo = geo / float(len(ind[0]))
        gl = np.array([np.sum(ind_r[0] * matrix[ind]), np.sum(ind_r[1] * matrix[ind]), np.sum(ind_r[2] * matrix[ind])])
        gl = gl / np.sum(matrix[ind])
        cms = geo - gl
        cms = np.sqrt(np.sum(cms ** 2))
        return cms

    def metabolicTumorVolume(self, matrix, voxel_size):
        # calculated on original gray values
        # find vmax
        vmax = np.nanmax(matrix)

        percent = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        mtv = []
        for p in percent:        
            ind = np.where(matrix > p*vmax)
            if len(ind) !=3:
                mtv.append['']
            else:
                vol = len(ind[0])*voxel_size**3
                mtv.append(vol)
        
        return mtv

    def saveImage(self, path, name, matrix, ImName, pixNr):
        '''save image of anylsed roi'''
        matplotlib.rcParams.update({'font.size': 24})

        pixNr = str(pixNr)
        
        # print matrix
        
        for n in arange(0, len(matrix)//24+1):
            fig = py.figure(10, figsize = (20,20))
            fig.text(0.5, 0.95, ImName+' '+name)
            for j in arange(0, 24):
                axes = fig.add_subplot(5, 5, j+1)
                axes.set_title(24*n+j)
                try:
                    im = axes.imshow(matrix[24*n+j], cmap=py.cm.jet, vmin = 0, vmax = self.n_bits)
                except IndexError:
                    break
                    pass
            axes = fig.add_subplot(5, 5, 25)
            fig.colorbar(im)
            try:
                makedirs(path+ImName+'\\')
            except OSError:
                if not isdir(path+ImName+'\\'):
                    raise
            fig.savefig(path+ImName+'\\'+name+'_'+self.structure+'_'+pixNr+'_'+str(n+1)+'.png')
            py.close()
            
            del fig
        for n in arange(0, len(matrix)//24+1):
            fig = py.figure(20, figsize = (20,20))
            fig.text(0.5, 0.95, ImName+' '+name)
            for j in arange(0, 24):
                axes = fig.add_subplot(5, 5, j+1, facecolor='#FFFF99')
                axes.set_title(24*n+j)
                try:
                    im = axes.imshow(matrix[24*n+j], cmap=py.cm.Greys_r, vmin = 0, vmax = self.n_bits)
                except IndexError:
                    break
                    pass
            axes = fig.add_subplot(5, 5, 25)
            fig.colorbar(im)
            fig.savefig(path+ImName+'\\black_'+name+'_'+self.structure+'_'+pixNr+'_'+str(n+1)+'.png')
            py.close()
            del fig
        
