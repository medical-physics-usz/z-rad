# -*- coding: utf-8 -*-
"""feature calculation for slices in 2D"""
# from numpy.core._multiarray_umath import ndarray
import numpy as np
from numpy import arange, floor
import matplotlib
import matplotlib.pyplot as plt       # i changed py.cm.jet in saveImage to =plt.get_cmap('jet') with importing this
# import scipy.optimize as optimization
# import cv2
from os import makedirs
from os.path import isdir
from datetime import datetime
from texture_intensity_2D import Intensity, GLCM, GLRLM_GLSZM_GLDZM_NGLDM, NGTDM, CMS_MTV
from texture_wavelet import Wavelet
from ROImatrix import Matrix
import logging
from collections import OrderedDict


class Features2D(object):
    """Calculate texture, intensity, fractal dim and center of the mass shift
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
    stop_calc - string, stop calculation if image tag was incorrect, eg activity PET = 0; stop_calc=='' continue, stop_calc != '' break and save info in the excel file
    *cont – list of additional variables:
    Xc – structure points in X, list of slices, per slice list of substructures
    Yc - structure points in Y, list of slices, per slice list of substructures
    XcW - structure points in X in wavelet space, list of slices, per slice list of substructures
    YcW - structure points in Y in wavelet space, list of slices, per slice list of substructures
    HUmin – HU range min
    Humax – HU range max
    outlier – bool, correct for outliers"""

    def __init__(self, dim, sb, maps, structure, columns, rows, xCTspace, zCTspace, slices, path, ImName, pixNr, binSize, modality, wv,
                 localRadiomics, cropStructure, stop_calc, *cont):  # Xc, Yc, XcW, YcW, HUmin, HUmax, outlier,  ):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Start: Texture Calculation")
        self.structure = structure
        self.xCTspace = xCTspace  # image resolution
        self.zCTspace = zCTspace
        self.columns = columns  # columns
        self.rows = rows
        self.slices = slices  # slice location
        self.dict_features = OrderedDict()

        try:
            self.bits = int(pixNr)
        except ValueError:  # must be an int
            self.bits = pixNr
        try:
            self.binSize = float(binSize)
        except ValueError:  # must be a float
            self.binSize = binSize

        rs_type = [1, 0, 0, 0, 0, 0, 0, 0, 2]  # structure type, structure resolution, transformed or non-transformed

        if structure != 'none':  # take contour points
            self.Xcontour = cont[0]  # Xc
            self.Xcontour_W = cont[1]  # XcW
            self.Ycontour = cont[2]  # Yc
            self.Ycontour_W = cont[3]  # YcW
        else:
            self.Xcontour = ''

        self.Xcontour_Rec = cont[4]
        self.Ycontour_Rec = cont[5]

        # take modality specific parameters
        if 'CT' in modality:
            self.HUmin = cont[6]
            self.HUmax = cont[7]
            self.outlier_correction = cont[8]
        else:
            self.HUmin = 'none'
            self.HUmax = 'none'
            self.outlier_correction = False

        print(stop_calc)
        if self.Xcontour == 'one slice':  # don't calculate, contour only on one slice
            self.stop_calculation('one slice', rs_type)
        elif stop_calc != '':  # stop calculation if image file contains wrong tags, eg activity = 0 in PET file
            self.stop_calculation(stop_calc, rs_type)
        else:

            for i in arange(0, len(maps)):  # iterate through different map type for example for CTP: BV, BF, MTT
                # wavelet transform
                if wv:
                    if 'BV' not in modality:
                        ctp = False
                    else:
                        ctp = True
                    wave_list = Wavelet(maps[i], path, modality[i], ImName + '_' + pixNr, dim, ctp).Return()  # order of transformed images: original, LLL, HHH, HHL, HLH, HLL, LHH, LHL, LLH or for 2D: original, LL, HH, HL, LH
                    # else:  # !!!!!!!!!!!!! CTP isn't adapted yet for 2D !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    #     wave_list = WaveletCTP(maps[i], path, modality[i], ImName + '_' + pixNr).Return()
                    sb.SetStatusText(ImName + ' wave done ' + str(datetime.now().strftime('%H:%M:%S')))
                    if dim == "2D" or dim == "2D_singleSlice":
                        rs_type = [1, 2, 0, 0, 0]
                        method_wv = ["", "LL_", "HH_", "HL_", "LH_"]  # "" for original
                    elif dim == "3D":
                        rs_type = [1, 2, 0, 0, 0, 0, 0, 0, 0]  # structure type, structure resolution
                        method_wv = ["", "LLL_", "HHH_", "HHL_", "HLH_", "HLL_", "LHH_", "LHL_", "LLH_"]  # "" for original
                else:
                    rs_type = [1]
                    wave_list = [maps[i]]
                    method_wv = [""]  # "" for original
                iterations_n = len(wave_list)

                # extract tumor only from the images, saves results as a list analog to wave_list matrix - contains
                # only the discretized pixels in the structure interval - one bin norm_points - how many points are
                # used for the calculations matrix_v - contains only the pixels in the structure, original values
                # matrix - contains only the discretized pixels in the structure region plus two voxels neighborhood
                # (used in the local radiomics)

                matrix_list = []
                interval_list = []
                norm_points_list = []
                matrix_v_list = []
                matrix_full_list = []
                n_bits_list = []
                HUmask = []  # a mask for ROI postprocessing on wavelet maps

                if "PET" in modality and cropStructure["crop"] is True:
                    self.logger.info("PET and Crop = True")
                    self.logger.info("Start: create HU mask from CT structure")
                    wave_list_ct = []
                    #                   ### CT initialize ###
                    for i in arange(0, len(cropStructure["data"])):
                        # wavelet transform
                        if wv:
                            wave_list_ct = Wavelet(cropStructure["data"][i], path, "CT", ImName + '_' + pixNr, dim, False).Return()  # order of transformed images: original, LLL, HHH, HHL, HLH, HLL, LHH, LHL, LLH
                            sb.SetStatusText(ImName + ' wave done ' + str(datetime.now().strftime('%H:%M:%S')))
                            if dim == "2D" or dim == "2D_singleSlice":
                                rs_type = [1, 2, 0, 0, 0]
                                method_wv = ["", "LL_", "HH_", "HL_", "LH_"]  # "" for original
                            elif dim == "3D":
                                rs_type = [1, 2, 0, 0, 0, 0, 0, 0, 0]  # structure type, structure resolution
                                method_wv = ["", "LLL_", "HHH_", "HHL_", "HLH_", "HLL_", "LHH_", "LHL_",
                                             "LLH_"]  # "" for original
                        else:
                            rs_type = [1]
                            wave_list_ct = [cropStructure["data"][i]]
                            method_wv = [""]
                        iterations_n = len(wave_list_ct)

                    # create mask from CT data for PET data
                    self.logger.info("End: create HU mask from CT structure")
                    self.logger.info("Initialize Matrices used for Texture and Wavelet")
                    for w in arange(0, len(wave_list)):
                        self.logger.debug("RS-Type " + str(rs_type[w]))
                        self.logger.debug("Intialize CT Matrix")
                        CT_ROImatrix = Matrix(wave_list_ct[w], rs_type[w], structure, ["CT"],
                                              cropStructure["readCT"].Xcontour, cropStructure["readCT"].Ycontour,
                                              cropStructure["readCT"].Xcontour_W, cropStructure["readCT"].Ycontour_W,
                                              cropStructure["readCT"].Xcontour_Rec,
                                              cropStructure["readCT"].Ycontour_Rec, cropStructure["readCT"].columns,
                                              cropStructure["readCT"].rows, cropStructure["hu_min"],
                                              cropStructure["hu_max"], 0, 0, False, HUmask, cropStructure)
                        self.logger.debug("Intialize PET Matrix")
                        ROImatrix = Matrix(wave_list[w], rs_type[w], structure, modality[i],
                                           cropStructure["readCT"].Xcontour, cropStructure["readCT"].Ycontour,
                                           cropStructure["readCT"].Xcontour_W, cropStructure["readCT"].Ycontour_W,
                                           cropStructure["readCT"].Xcontour_Rec, cropStructure["readCT"].Ycontour_Rec,
                                           self.columns, self.rows, self.HUmin, self.HUmax, self.binSize, self.bits,
                                           self.outlier_correction, CT_ROImatrix.HUmask, cropStructure)
                        matrix_list.append(ROImatrix.matrix)
                        interval_list.append(ROImatrix.interval)
                        norm_points_list.append(ROImatrix.norm_points)
                        matrix_v_list.append(ROImatrix.matrix_true)
                        matrix_full_list.append(ROImatrix.matrix_full)
                        n_bits_list.append(ROImatrix.n_bits)
                        HUmask = CT_ROImatrix.HUmask

                    print("------------- end: created HU mask was used for PET structure --------------")
                    del wave_list_ct

                else:
                    self.logger.info("Normal Mode, no cropping")
                    for w in arange(0, len(wave_list)):
                        ROImatrix = Matrix(wave_list[w], rs_type[w], structure, modality[i], self.Xcontour,
                                           self.Ycontour, self.Xcontour_W, self.Ycontour_W, self.Xcontour_Rec,
                                           self.Ycontour_Rec, self.columns, self.rows, self.HUmin, self.HUmax,
                                           self.binSize, self.bits, self.outlier_correction, HUmask, cropStructure)
                        matrix_list.append(ROImatrix.matrix)  # tumor matrix
                        interval_list.append(ROImatrix.interval)
                        norm_points_list.append(ROImatrix.norm_points)
                        matrix_v_list.append(ROImatrix.matrix_true)
                        matrix_full_list.append(ROImatrix.matrix_full)
                        n_bits_list.append(ROImatrix.n_bits)
                        HUmask = ROImatrix.HUmask  # mask is returned in that for loop used for LLL...

                del wave_list

                if localRadiomics:
                    self.logger.info("Start: Local Radiomics")
                    # call NaN optimizer

                    iterations_n = len(centers)  # to define how many time the radiomics need to be calculated
                    method_wv = []*iterations_n  # suggestion only..  how to name different centers?
                    self.structure = []  # list to collect info if the subvolume belongs to the tumor or to the recurrence

                    # to have list with n_bits and intervals for each centers (it will be the same number for eachcenter, but it is done for compability with wavelet)
                    n_bits_temp = []
                    interval_list_temp = []
                    for c in len(centers):
                        n_bits_temp.append([n_bits_list[0]])
                        interval_list_temp.append([interval_list[0]])
                    n_bits_list = n_bits_temp
                    interval_list = interval_list_temp
                    del n_bits_temp
                    del interval_list_temp

                    # make the matrix_list and matrix_v_list with all the subregions to be analyzed

                self.dict_features["points"] = norm_points_list[0]
                self.dict_features["structure"] = self.structure

                try:  # ValueError
                    # calculate features for original and transformed images or local centers if provided
                    # feed in the list of maps to calculate

                    # remove after phantom test again !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # iterations_n = 1
                    # img_obj = np.zeros(shape=(4, 4, 5))
                    # img_obj[0, :, :] = [[1, 4, 4, 1, 1], [1, 4, 6, 1, 1], [4, 1, 6, 4, 1], [4, 4, 6, 4, 1]]
                    # img_obj[1, :, :] = [[1, 4, 4, 1, 1], [1, 1, 6, 1, 1], [np.nan, 1, 3, 1, 1], [4, 4, 6, 1, 1]]
                    # img_obj[2, :, :] = [[1, 4, 4, np.nan, np.nan], [1, 1, 1, 1, 1], [1, 1, np.nan, 1, 1],
                    #                     [1, 1, 6, 1, 1]]
                    # img_obj[3, :, :] = [[1, 4, 4, np.nan, np.nan], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 6, 1, 1]]
                    # img_obj = img_obj - 1

                    for w in arange(0, iterations_n):
                        m_wv = method_wv[w]
                        # for phantom test:
                        # matrix = img_obj
                        # matrix_v = img_obj
                        # self.n_bits = 6
                        # interval = 1
                        matrix = matrix_list[w]
                        matrix_v = matrix_v_list[w]
                        self.n_bits = int(n_bits_list[w])   # i added int here
                        interval = interval_list[w]

                        try:
                            sb.SetStatusText(ImName + ' matrix done ' + str(datetime.now().strftime('%H:%M:%S')))
                        except AttributeError:
                            print('attributeerrror')
                            pass
                        if rs_type[w] == 1:  # save only for the original image
                            self.saveImage(path, modality[i], matrix, ImName, pixNr)
                        more_than_one_pix = True

                        # histogram calculation -----------------------------------------------------------------------
                        try:
                            i_feature = Intensity()
                            i_feature.histogram_calculation(matrix_v, modality[i], ImName, pixNr, path, w,
                                                            self.structure)
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
                            i_feature.feature_calculation(interval)
                            self.dict_features = i_feature.return_features(self.dict_features, m_wv)  # save new features in dict
                            try:
                                sb.SetStatusText(ImName + ' hist done ' + str(datetime.now().strftime('%H:%M:%S')))
                                self.logger.info(ImName + ' hist done ' + str(datetime.now().strftime('%H:%M:%S')))
                            except AttributeError:  # if no GUI
                                pass

                            # directions in COM in 3D:
                            # lista_t = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, -1], [1, 0, 1], [1, 0, -1],
                            #            [1, 1, 0], [1, -1, 0], [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]]
                            # directions for 2D:
                            lista_t_2d = [[1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0]]

                            # GLCM - 3.6 (grey level co-occurence based features)
                            # ------------------------------------------------------------------------------------------
                            # method 1) all directions and slices separate ( )
                            glcm_m1 = GLCM("GLCM-no-merging", matrix, lista_t_2d, self.n_bits)
                            # calculate glcm matrices for all 2d methods:
                            glcm, glcm_mbs, glcm_mbd, glcm_mf = glcm_m1.matrix_calculation()
                            glcm_m1.norm_marginal_calculation(glcm)  # normalization and marg. prob.
                            glcm_m1.glcm_feature_calculation()
                            self.dict_features = glcm_m1.return_features(self.dict_features, "", m_wv)

                            # method 2) directions merged by slice (mbs)
                            glcm_m2 = GLCM("GLCM-merging-by-slice", matrix, lista_t_2d, self.n_bits)
                            glcm_m2.norm_marginal_calculation(glcm_mbs)
                            glcm_m2.glcm_feature_calculation()
                            self.dict_features = glcm_m2.return_features(self.dict_features, "mbs", m_wv)

                            if dim != "2D_singleSlice":  # skip this part for single 2D
                                # method 3) slices merged by direction (mbd)
                                glcm_m3 = GLCM("GLCM-merging-by-direction", matrix, lista_t_2d, self.n_bits)
                                glcm_m3.norm_marginal_calculation(glcm_mbd)
                                glcm_m3.glcm_feature_calculation()
                                self.dict_features = glcm_m3.return_features(self.dict_features, "mbd", m_wv)

                                # method 4) full merging (mf)
                                glcm_m4 = GLCM("GLCM-full-merging", matrix, lista_t_2d, self.n_bits)
                                glcm_m4.norm_marginal_calculation(glcm_mf)
                                glcm_m4.glcm_feature_calculation()
                                self.dict_features = glcm_m4.return_features(self.dict_features, "mf", m_wv)
                            # 3D:
                            # # 5. features are computed from each 3D directional matrix & averaged over the 3D directions
                            # glcm_3d_directions = GLCM("GLCM_3d directions separate", matrix, lista_t, self.n_bits)
                            # comatrix, comatrix_merged = glcm_3d_directions.matrix_calculation()
                            # glcm_3d_directions.norm_marginal_calculation(comatrix)
                            # glcm_3d_directions.glcm_feature_calculation()
                            # # 6. the feature is computed from a single matrix after merging all 3D directional matrices
                            # glcm_3d = GLCM("GLCM_3d all merged", matrix, lista_t, self.n_bits)
                            # glcm_3d.norm_marginal_calculation(comatrix_merged)
                            # glcm_3d.glcm_feature_calculation()
                            try:
                                sb.SetStatusText(ImName + ' COM done ' + str(datetime.now().strftime('%H:%M:%S')))
                                self.logger.info(ImName + ' COM done ' + str(datetime.now().strftime('%H:%M:%S')))
                            except AttributeError:  # if no GUI
                                pass

                            # GLRLM - 3.7 (grey level run length based features)
                            # ------------------------------------------------------------------------------------------
                            # method 1) (all slices and directions separate)
                            glrlm_m1 = GLRLM_GLSZM_GLDZM_NGLDM("GLRLM-no-merging", matrix, "GLRLM", self.n_bits,
                                                               lista_t_2d)
                            glrlm, glrlm_mbs, glrlm_mbd, glrlm_mf = glrlm_m1.matrix_calculation()
                            glrlm_m1.feature_calculation("1)", glrlm)
                            self.dict_features = glrlm_m1.return_features(self.dict_features, "", m_wv)

                            # method 2) (mbs)
                            glrlm_m2 = GLRLM_GLSZM_GLDZM_NGLDM("GLRLM-merging-by-slice", matrix, "GLRLM", self.n_bits,
                                                               lista_t_2d)
                            glrlm_m2.feature_calculation("2)", glrlm_mbs)
                            self.dict_features = glrlm_m2.return_features(self.dict_features, "mbs", m_wv)

                            if dim != "2D_singleSlice":
                                # method 3) (all slices merged per direction) (mbd)
                                glrlm_m3 = GLRLM_GLSZM_GLDZM_NGLDM("GLRLM-merging-by-direction", matrix, "GLRLM",
                                                                   self.n_bits,
                                                                   lista_t_2d)
                                glrlm_m3.feature_calculation("3)", glrlm_mbd)
                                self.dict_features = glrlm_m3.return_features(self.dict_features, "mbd", m_wv)

                                # method 4) (slices and directions merged) (mf)
                                glrlm_m4 = GLRLM_GLSZM_GLDZM_NGLDM("GLRLM-full-merging", matrix, "GLRLM", self.n_bits,
                                                                   lista_t_2d)
                                glrlm_m4.feature_calculation("4)", glrlm_mf)
                                self.dict_features = glrlm_m4.return_features(self.dict_features, "mf", m_wv)
                                del glrlm_m3, glrlm_m4

                            del glrlm, glrlm_mbs, glrlm_mbd, glrlm_mf, glrlm_m1, glrlm_m2
                            self.logger.info(ImName + ' GLRLM ' + str(datetime.now().strftime('%H:%M:%S')))

                            # GLSZM - 3.8 (grey level size zone based features)
                            # GLDZM - 3.9 (grey level distance zone based features)
                            # ------------------------------------------------------------------------------------------
                            # 1. method:  - features from 2D matrices (then features averaged over slices)
                            # 2. method: merged - feature from merged 2D matrices
                            glszm_m1 = GLRLM_GLSZM_GLDZM_NGLDM("GLSZM", matrix, "GLSZM", self.n_bits, lista_t_2d)

                            # calculate matrices:
                            glszm_m, gldzm_m, glszm, gldzm = glszm_m1.matrix_calculation()
                            # gldzm[1][2][0] = 0.0  # only for phantom correction !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            # gldzm[1][2][1] = 1.0
                            # gldzm_m[2][0] = 0.0
                            # gldzm_m[2][1] = 1.0
                            # glszm
                            glszm_m1.feature_calculation("2)", glszm)
                            self.dict_features = glszm_m1.return_features(self.dict_features, "", m_wv)
                            if dim != "2D_singleSlice":
                                glszm_m2 = GLRLM_GLSZM_GLDZM_NGLDM("GLSZM-merged", matrix, "GLSZM", self.n_bits, lista_t_2d)
                                glszm_m2.feature_calculation("4)", glszm_m)
                                self.dict_features = glszm_m2.return_features(self.dict_features, "m", m_wv)
                                del glszm_m2
                            # gldzm
                            gldzm_m1 = GLRLM_GLSZM_GLDZM_NGLDM("GLDZM", matrix, "GLDZM", self.n_bits, lista_t_2d)
                            gldzm_m1.feature_calculation("2)", gldzm)
                            self.dict_features = gldzm_m1.return_features(self.dict_features, "", m_wv)
                            if dim != "2D_singleSlice":
                                gldzm_m2 = GLRLM_GLSZM_GLDZM_NGLDM("GLDZM-merged", matrix, "GLDZM", self.n_bits, lista_t_2d)
                                gldzm_m2.feature_calculation("4)", gldzm_m)
                                self.dict_features = gldzm_m2.return_features(self.dict_features, "m", m_wv)
                                del gldzm_m2

                            del glszm, gldzm, glszm_m, gldzm_m
                            del glszm_m1, gldzm_m1
                            self.logger.info(ImName + ' GLSZM done ' + str(datetime.now().strftime('%H:%M:%S')))

                            try:
                                sb.SetStatusText(ImName + ' GLSZM done ' + str(datetime.now().strftime('%H:%M:%S')))
                                self.logger.info(ImName + ' GLSZM done ' + str(datetime.now().strftime('%H:%M:%S')))
                            except AttributeError:  # if no GUI
                                pass
                            self.logger.info(ImName + ' GLDZM done ' + str(datetime.now().strftime('%H:%M:%S')))

                            # NGTDM - 3.10 (neighbourhood grey tone difference based features) -
                            # ------------------------------------------------------------------------------------------
                            # method 1) features for each slice
                            ngtdm2D = NGTDM("NGTDM-2d", matrix, "2D", self.n_bits)
                            ngtdm2D.ngtdm_matrix_calculation()
                            ngtdm2D.feature_calculation("")  # method 1
                            self.dict_features = ngtdm2D.return_features(self.dict_features, "", m_wv)
                            # method 2) all slices merged
                            if dim != "2D_singleSlice":
                                ngtdm2D.feature_calculation("merged-2d")  # method 2
                                self.dict_features = ngtdm2D.return_features(self.dict_features, "m", m_wv)
                            # method 3) single 3D matrix
                            # ngtdm3D = NGTDM("ngtdm", matrix, "3D", self.n_bits)
                            # ngtdm3D.ngtdm_matrix_calculation()
                            # ngtdm3D.feature_calculation("")  # method 3
                            del ngtdm2D
                            try:
                                sb.SetStatusText(ImName + ' NGTDM done ' + str(datetime.now().strftime('%H:%M:%S')))
                                self.logger.info(ImName + ' NGTDM done ' + str(datetime.now().strftime('%H:%M:%S')))
                            except AttributeError:  # if no GUI
                                pass

                            # NGLDM - 3.11 (neighbouring grey level dependence based features)
                            # ------------------------------------------------------------------------------------------
                            ngldm_1 = GLRLM_GLSZM_GLDZM_NGLDM("NGLDM-non-merged", matrix, "NGLDM", self.n_bits,
                                                              lista_t_2d)

                            ngldm, ngldm_m = ngldm_1.matrix_calculation()

                            ngldm_1.feature_calculation("2)", ngldm)
                            self.dict_features = ngldm_1.return_features(self.dict_features, "", m_wv)
                            if dim != "2D_singleSlice":
                                ngldm_2 = GLRLM_GLSZM_GLDZM_NGLDM("NGLDM-merged", matrix, "NGLDM", self.n_bits, lista_t_2d)
                                ngldm_2.feature_calculation("4)", ngldm_m)
                                self.dict_features = ngldm_2.return_features(self.dict_features, "m", m_wv)
                                del ngldm_2

                            self.logger.info(ImName + ' NGLDM done ' + str(datetime.now().strftime('%H:%M:%S')))
                            del ngldm, ngldm_m, ngldm_1

                            # Other features:
                            # -------------------------------------------------------------------------------------------
                            # get center mass shift and metabolic tumor volume features
                            self.dict_features = CMS_MTV(matrix, path, pixNr, ImName, matrix_v, self.xCTspace, self.zCTspace,
                                                         self.structure, dim).return_features(self.dict_features, m_wv)

                            try:
                                sb.SetStatusText(ImName + ' fractal done ' + str(datetime.now().strftime('%H:%M:%S')))
                                self.logger.info(ImName + ' fractal done ' + str(datetime.now().strftime('%H:%M:%S')))
                            except AttributeError:
                                pass
                            del matrix

                except ValueError:
                    print(ValueError)
                    self.stop_calculation('ValueError', [1])
                    # matrix.append([])
                    # interval.append([])
                    # # norm_points.append([])
                    # matrix_v.append([])
                    # matrix_full.append([])

            # except IndexError:#IndexError:
            #    matrix = []
            #    print IndexError
            #    self.stop_calculation('IndexError', rs_type)

    def stop_calculation(self, info, rs_type):
        """returns empty string and error type bt does not stop the calculation"""
        for i in arange(0, len(rs_type)):
            self.dict_features["structure"] = info

    def ret(self):
        """return function"""
        return self.dict_features

    def saveImage(self, path, name, matrix, ImName, pixNr):
        """save image of analysed ROI"""
        matplotlib.rcParams.update({'font.size': 24})

        pixNr = str(pixNr)

        # print matrix
        fig = plt.figure(10, figsize=(20, 20))
        fig.text(0.5, 0.95, ImName + ' ' + name)
        for j in arange(0, 24):
            axes = fig.add_subplot(5, 5, j + 1)
            axes.set_title(j)
            try:
                im = axes.imshow(matrix[j], cmap=plt.get_cmap('jet'), vmin=0, vmax=self.n_bits)
            except IndexError:
                pass
        axes = fig.add_subplot(5, 5, 25)
        try:
            fig.colorbar(im)
        except UnboundLocalError:
            pass
        try:
            makedirs(path + ImName + '\\')
        except OSError:
            if not isdir(path + ImName + '\\'):
                raise
        fig.savefig(path + ImName + '\\' + name + '_' + self.structure + '_' + pixNr + '_' + str(j + 1) + '.png')
        plt.close()
        del fig

        fig = plt.figure(20, figsize=(20, 20))
        fig.text(0.5, 0.95, ImName + ' ' + name)
        for j in arange(0, 24):
            axes = fig.add_subplot(5, 5, j + 1, facecolor='#FFFF99')
            axes.set_title(j)
            try:
                im = axes.imshow(matrix[j], cmap=plt.get_cmap('Greys'), vmin=0, vmax=self.n_bits)
            except IndexError:
                pass
        axes = fig.add_subplot(5, 5, 25)        # I inserted this two lines by one tab
        try:
            fig.colorbar(im)
        except UnboundLocalError:
            pass
        fig.savefig(
            path + ImName + '\\black_' + name + '_' + self.structure + '_' + pixNr + '_' + str(j + 1) + '.png')
        plt.close()
        del fig
