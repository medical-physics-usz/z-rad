# -*- coding: cp1252 -*-

# import libraries
try:
    import pydicom as dc # dicom library
except ImportError:
    import dicom as dc # dicom library
import numpy as np  # numerical computation
from numpy import arange, floor
import pylab as py  # drawing plots
from os import listdir, makedirs  # managing files
from os.path import isfile, join, isdir

from scipy.stats import norm  # statistical analysis
import scipy.optimize as optimization
import matplotlib
from scipy import ndimage
from time import gmtime

# own classes
# import class to calculate texture parameters
from texture import Texture
from exception import MyException
from read import ReadImageStructure
from export import Export
from LocRadSlice import Slice


class Texture():
    '''Main class to handle CT images, reads images and structures, calls radiomics calculation and export class to export results
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
    HUmin - HU range min
    HUmax – HU range max
    outlier – bool, correct for outliers
    wv – bool, calculate wavelet
    exportList – list of matrices/features to be calculated and exported
    '''

    matrix = 0
    interval = 0
    norm_points = 0
    matrix_v = 0
    slices = []
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0


    def __init__(self, path_image, path_save, structure, pixNr, binSize, l_ImName, HUmin, HUmax,
                 outlier_corr, wv):

        # Declare needed variables for matrix calc
        self.structure = structure
        self.HUmin = HUmin
        self.HUmax = HUmax
        self.bin_size = binSize
        self.outlier_correction = outlier_corr
        self.vmin = []
        self.vmax = []
        self.slice_data = []
        self.matrix, self.interval, self.norm_points, self.matrix_v = 0,0,0,0

        image_modality = ['CT']
        rs_type = [1, 0, 0, 0, 0, 0, 0, 0, 2]  # structure type, structure resolution, transformed or non-transformed

        try:
            self.bits = int(pixNr)
        except ValueError:  # must be an int
            self.bits = pixNr
        try:
            self.binSize = float(binSize)
        except ValueError:  # must be a float
            self.binSize = binSize

        dicomProblem = []
        for ImName in l_ImName:
            print 'patient', ImName
            try:
                mypath_image = path_image + ImName + '\\'
                CT_UID = ['1.2.840.10008.5.1.4.1.1.2', '1.2.840.10008.5.1.4.1.1.2.1']  # CT and contarst-enhanced CT

                read = ReadImageStructure(CT_UID, mypath_image, structure, wv)

                dicomProblem.append([ImName, read.listDicomProblem])

                # parameters to recalculate intensities HU
                inter = float(dc.read_file(mypath_image + read.onlyfiles[1]).RescaleIntercept)
                slope = float(dc.read_file(mypath_image + read.onlyfiles[1]).RescaleSlope)

                IM_matrix = []  # list containing the images matrix
                for f in read.onlyfiles:
                    data = dc.read_file(mypath_image + f).PixelData
                    data16 = np.array(np.fromstring(data, dtype=np.int16))  # converitng to decimal
                    data16 = data16 * slope + inter
                    # recalculating for rows x columns
                    a = []
                    for j in arange(0, read.rows):
                        a.append(data16[j * read.columns:(j + 1) * read.columns])
                    a = np.array(a)
                    IM_matrix.append(np.array(a))
                    # Texture.slices.append(Slice(dc.read_file(mypath_image + f).SliceLocation, np.array(a)))
                    self.slice_data.append(Slice(dc.read_file(mypath_image + f).SliceLocation, np.array(a)))
                    print f
                IM_matrix = np.array(IM_matrix)

            except WindowsError:  # error if there is not directory
                continue
            except IndexError:  # empty folder
                continue

            # Get needed values from read dicom files
            self.Xcontour = read.Xcontour
            self.Xcontour_W = read.Xcontour_W
            self.Ycontour = read.Ycontour
            self.Ycontour_W = read.Ycontour_W
            self.xCTspace = read.xCTspace
            self.columns = read.columns
            self.rows = read.rows
            self.slices = read.slices

            IM_matrix = [IM_matrix]

            for i in arange(0, len(IM_matrix)):
                rs_type = [1]
                wave_list = [IM_matrix[i]]
                print i
                for w in arange(0, len(wave_list)):
                    self.matrix, self.interval, self.norm_points, self.matrix_v = \
                          self.Matrix(wave_list[w], rs_type[w], structure, image_modality[0])
                    # Texture.matrix, Texture.interval, Texture.norm_points, Texture.matrix_v = \
                    #     self.Matrix(wave_list[w], rs_type[w], structure, image_modality[0])


    def Matrix(self, imap, rs_type, structure, map_name):
        '''fill the contour matrix with values from the image - including discretizaion'''
        '''matrix - matrix with discretized entries'''
        '''matrix_ture - matrix with original values for first-order statistics'''
        matrix = []
        norm_points = []
        lymin = []
        lymax = []
        lxmin = []
        lxmax = []
        v = []
        if structure != 'none':  # if the structure is defined
            # searching for the matrix size in 3D
            if rs_type == 1:  # original vs wavelet
                Xcontour = self.Xcontour
                Ycontour = self.Ycontour
            else:
                Xcontour = self.Xcontour_W
                Ycontour = self.Ycontour_W

            # set minx, miny, minz, maxx, maxy, maxz in the matrix
            for i in arange(0, len(Xcontour)):  # slices
                ymins = []
                ymaxs = []
                xmins = []
                xmaxs = []
                for j in arange(0, len(Xcontour[i])):  # sub-structres in the slice
                    ymins.append(np.min(Ycontour[i][j]))
                    ymaxs.append(np.max(Ycontour[i][j]))
                    xmins.append(np.min(Xcontour[i][j]))
                    xmaxs.append(np.max(Xcontour[i][j]))
                    for k in arange(0, len(Xcontour[i][j])):
                        v.append(imap[i][Ycontour[i][j][k]][Xcontour[i][j][k]])
                try:
                    lymin.append(np.min(ymins))
                except ValueError:  # in case of an empty slice
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
            if Texture.xmax == 0:
                ymin = np.min(lymin)
                ymax = np.max(lymax)
                xmin = np.min(lxmin)
                xmax = np.max(lxmax)
                Texture.ymin = ymin
                Texture.ymax = ymax
                Texture.xmin = xmin
                Texture.xmax = xmax
            else:
                ymin = Texture.ymin
                ymax = Texture.ymax
                xmin = Texture.xmin
                xmax = Texture.xmax

            # ymin = np.min(lymin)
            # ymax = np.max(lymax)
            # xmin = np.min(lxmin)
            # xmax = np.max(lxmax)
            print "YMIN = " + str(ymin)
            print "YMAX = " + str(ymax)
            print "XMIN = " + str(xmin)
            print "XMAX = " + str(xmax)
            del lymin
            del lymax
            del lxmin
            del lxmax

            # for the outlier correction, first remove values outside of HU range (for CT only) and then calccuate standard deviation
            ind = np.where(np.isnan(np.array(v)))[0]
            for j in arange(1, len(ind) + 1):
                v.pop(ind[-j])
            if self.HUmin != 'none' and rs_type != 0:
                ind = np.where(np.array(v) < self.HUmin)[0]
                for j in arange(1, len(ind) + 1):
                    v.pop(ind[-j])
                ind = np.where(np.array(v) > self.HUmax)[0]
                for j in arange(1, len(ind) + 1):
                    v.pop(ind[-j])
            vmin = np.min(v)
            vmax = np.max(v)
            vmean = np.mean(v)
            vsd = np.std(v)
            v_out_low = vmean - 3 * vsd
            v_out_high = vmean + 3 * vsd
        else:  # no structure, analyse full image
            xmin = 0
            xmax = self.columns - 1
            ymin = 0
            ymax = self.rows - 1
            vmin = np.nanmin(imap)
            vmax = np.nanmax(imap)

        # placing nan in all places in the matrix
        m = np.zeros([ymax - ymin + 1,
                      xmax - xmin + 1])  # creating the matrix to fill it with points of the structure, y rows, x columns
        for im in arange(0, len(m)):
            for jm in arange(0, len(m[im])):
                m[im][jm] = np.nan

        matrix = []  # matrix with discretizd values
        matrix_true = []  # matrix with real image values

        if structure != 'none':  # if the structure to be anylsed was defined
            print 'Vmin, vmax'
            print vmin
            print vmax
            for n in arange(0, len(Xcontour)):
                matrix.append(m.copy())
                matrix_true.append(m.copy())

            if self.bits == '':  # fixed bin defined
                if map_name == 'MTT':
                    interval = self.binSize * 10  # as the range of relative MTT is normally much grater than range of BV and BF
                else:
                    interval = self.binSize
                self.n_bits = int((vmax - vmin) / interval) + 1  # calcuate corresponding number of bins
            else:  # fixed number of bin defined
                interval = (vmax - vmin) / (self.bits - 1)
                self.n_bits = self.bits
            print 'n bins, interval', (self.n_bits, interval)

            for i in arange(0, len(Xcontour)):  # slices
                for j in arange(0, len(Xcontour[i])):  # sub-structres in the slice
                    for k in arange(0, len(Xcontour[i][j])):  # each point
                        try:
                            matrix[i][Ycontour[i][j][k] - ymin][Xcontour[i][j][k] - xmin] = int((imap[i][
                                                                                                     Ycontour[i][j][k]][
                                                                                                     Xcontour[i][j][
                                                                                                         k]] - vmin) / interval)  # first row, second column, changing to self.bits channels as in co-ocurence matrix we have only the channels channels
                            matrix_true[i][Ycontour[i][j][k] - ymin][Xcontour[i][j][k] - xmin] = \
                            imap[i][Ycontour[i][j][k]][Xcontour[i][j][
                                k]]  # first row, second column, changing to self.bits channels as in co-ocurence matrix we have only the channels channels
                        except ValueError:  # in case of nan
                            pass
                    norm_points.append(len(v))  # how many points are used for calculation
        else:  # no sturcture defined
            print 'Vmin, vmax'
            print vmin, vmax
            for n in arange(0, len(imap)):
                matrix.append(m.copy())
                matrix_true.append(m.copy())

            if self.bits == '':  # fixed bin defined
                if map_name == 'MTT':
                    interval = self.binSize * 10  # as the range of relative MTT is normally much grater than range of BV and BF
                else:
                    interval = self.binSize
                self.n_bits = int(((vmax - vmin) / interval)) + 1  # calcuate corresponding number of bins
            else:  # fixed number of bin defined
                interval = (vmax - vmin) / (self.bits - 1)
                self.n_bits = self.bits
            print 'n bins, interval', (self.n_bits, interval)

            for i in arange(0, len(imap)):  # slices
                for j in arange(0, len(imap[i])):  # rows
                    for k in arange(0, len(imap[i][j])):  # columns
                        try:  # each point
                            matrix[i][j - ymin][k - xmin] = int((imap[i][j][
                                                                     k] - vmin) / interval)  # first row, second column, changing to self.bits channels as in co-ocurence matrix we have only the channels channels
                            matrix_true[i][j - ymin][k - xmin] = imap[i][j][
                                k]  # first row, second column, changing to self.bits channels as in co-ocurence matrix we have only the channels channels
                        except ValueError:  # in case of nan
                            pass
                    norm_points.append(np.where(~np.isnan(imap))[0])  # how many points are used for calculation$

        matrix = np.array(matrix)
        matrix_true = np.array(matrix_true)

        # if there are constrains for HU in CT
        if self.HUmin != 'none' and rs_type != 0:  # used for original image and LLL filter
            ind_min = np.where(matrix_true < self.HUmin)
            ind_max = np.where(matrix_true > self.HUmax)
            matrix_true[ind_min] = np.nan
            matrix_true[ind_max] = np.nan
            matrix[ind_min] = np.nan
            matrix[ind_max] = np.nan

        # cut the matrix in z direction to remove empty slices
        zind = np.where(~np.isnan(matrix))[0]
        zmin = np.min(zind) - 1
        zmax = np.max(zind) + 1
        if zmin < 0:
            zmin = 0
        if zmax >= len(matrix):
            zmax = len(matrix) - 1

        matrix = matrix[zmin:zmax + 1]

        print ("ZMIN = " + str(zmin))
        print ("ZMAX = " + str(zmax))
        matrix_true = matrix_true[zmin:zmax + 1]

        # correct for outliers
        if self.outlier_correction:
            ind_min = np.where(matrix_true < v_out_low)
            ind_max = np.where(matrix_true > v_out_high)
            matrix_true[ind_min] = np.nan
            matrix_true[ind_max] = np.nan
            matrix[ind_min] = np.nan
            matrix[ind_max] = np.nan
            vmin = np.min(matrix_true[np.where(~np.isnan(matrix_true))])
            vmax = np.max(matrix_true[np.where(~np.isnan(matrix_true))])

        self.vmin.append(vmin)
        self.vmax.append(vmax)

        for counter in range(len(matrix)):
            np.array(matrix[counter])

            self.slice_data[zmin + counter].pixel_data_roi.append(matrix[counter])
            # Texture.slices[zmin + counter].pixel_data_roi.append(matrix[counter])

        return matrix, interval, norm_points, matrix_true

