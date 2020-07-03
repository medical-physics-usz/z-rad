# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import logging
from exception import MyException

class Matrix(object):
    def __init__(self, imap, rs_type, structure, map_name, XcontourPoints, YcontourPoints, Xcontour_WPoints,
                 Ycontour_WPoints, Xcontour_Rec, Ycontour_Rec, columns, rows, HUmin, HUmax, binSize, bits,
                 outlier_correction, HUmask, cropStructure={"crop": False}):
        """ fill the contour matrix with values from the image - including discretization
            matrix - matrix with discretized entries
            matrix_true - matrix with original values for first-order statistics"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Read in Data into ROIMatrix")
        matrix = []
        lymin = []
        lymax = []
        lxmin = []
        lxmax = []
        v = []
        try:
            if structure != 'none':  # if the structure is defined

                # searching for the matrix size in 3D
                if rs_type == 1:  # original vs wavelet
                    Xcontour = XcontourPoints
                    Ycontour = YcontourPoints
                else:
                    Xcontour = Xcontour_WPoints
                    Ycontour = Ycontour_WPoints

                # set minx, miny, minz, maxx, maxy, maxz in the matrix
                for i in range(len(Xcontour)):  # slices
                    ymins = []
                    ymaxs = []
                    xmins = []
                    xmaxs = []
                    for j in range(len(Xcontour[i])):  # sub-structures in the slice
                        if len(Ycontour[i][j]) != 0:
                            ymins.append(np.min(Ycontour[i][j]))
                            ymaxs.append(np.max(Ycontour[i][j]))
                            xmins.append(np.min(Xcontour[i][j]))
                            xmaxs.append(np.max(Xcontour[i][j]))
                            for k in range(len(Xcontour[i][j])):
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

                # add margin if local radiomics is computed including healthy tissue (2 voxels)
                ymin = np.min(lymin) - 2
                ymax = np.max(lymax) + 2
                xmin = np.min(lxmin) - 2
                xmax = np.max(lxmax) + 2
                self.logger.info("ymin {}, ymax {}, xmax {}, xmin {}".format(ymin, ymax, xmin, xmax))
                if xmin < 0:
                    xmin = 0
                if xmax >= columns:
                    xmax = columns
                if ymin < 0:
                    ymin = 0
                if ymax >= rows:
                    ymax = rows
                del lymin
                del lymax
                del lxmin
                del lxmax

                # for the outlier correction, first remove values outside of HU range (for CT only) and then calccuate standard deviation

                ind = np.where(np.isnan(np.array(v)))[0]
                for j in range(1, len(ind) + 1):
                    v.pop(ind[-j])

                if (HUmin != 'none' and rs_type != 0) or (cropStructure["crop"] and rs_type != 0):  # no parameter for outlier correction!
                    print("***** outlier correction for CT ********")
                    ind = np.where(np.array(v) < HUmin)[0]
                    for j in range(1, len(ind) + 1):
                        v.pop(ind[-j])
                    ind = np.where(np.array(v) > HUmax)[0]
                    for j in range(1, len(ind) + 1):
                        v.pop(ind[-j])
                vmin = np.min(v)
                vmax = np.max(v)
                vmean = np.mean(v)
                vsd = np.std(v)
                v_out_low = vmean - 3 * vsd
                v_out_high = vmean + 3 * vsd
            else:  # no structure, analyse full image
                xmin = 0
                xmax = columns - 1
                ymin = 0
                ymax = rows - 1
                vmin = np.nanmin(imap)
                vmax = np.nanmax(imap)

            # placing nan in all places in the matrix
            m = np.zeros([ymax - ymin + 1, xmax - xmin + 1])  # creating the matrix to fill it with points of the structure, y rows, x columns
            for im in range(len(m)):
                for jm in range(len(m[im])):
                    m[im][jm] = np.nan

            matrix = []  # matrix with discretizd values
            matrix_full = []  # matrix with discretizd values but not cut to the ROI
            matrix_true = []  # matrix with real image values
            matrix_rec = []

            if structure != 'none':  # if the structure to be anaylsed was defined
                self.logger.info("Vmin, Vmax " + ", ".join(map(str, (round(vmin,3), round(vmax,3)))))

                for n in range(len(Xcontour)):
                    matrix.append(m.copy())
                    matrix_true.append(m.copy())
                    matrix_full.append(m.copy())
                    matrix_rec.append(m.copy())

                if bits == '':  # fixed bin defined
                    if map_name == 'MTT':
                        interval = binSize * 10  # as the range of relative MTT is normally much grater than range of BV and BF
                    else:
                        interval = binSize
                    n_bits = int((vmax-vmin)//interval+1)  # calcuate corresponding number of bins
                else: #fixed number of bin defined
                    interval = round((vmax-vmin)/(bits-1), 2)
                    if vmin + (bits-1)*interval < vmax:
                        interval = interval+0.01 # problems with rounding, for example the number 0.1249 would be rounderd to 0.12
                        if vmin + (bits-1)*interval < vmax:
                            MyException('Problem with rounding precision, increase rounding precision of the interval in ROImatrix')
                            raise StopIteration
                    if interval == 0: #in case the image is homogenous after the filtering
                        interval = 0.01 
                    n_bits = bits
                self.logger.info('n bins, interval ' + ", ".join(map(str, (n_bits, interval))))

                for i in range(len(Xcontour)):  # slices
                    for j in range(len(Xcontour[i])):  # sub-structures in the slice
                        for k in range(len(Xcontour[i][j])):  # each point
                            try:
                                matrix[i][Ycontour[i][j][k] - ymin][Xcontour[i][j][k] - xmin] = int((imap[i][Ycontour[i][j][k]][Xcontour[i][j][k]] - vmin) / interval)  # first row, second column, changing to bits channels as in co-ocurence matrix we have only the channels channels
                                matrix_true[i][Ycontour[i][j][k] - ymin][Xcontour[i][j][k] - xmin] = imap[i][Ycontour[i][j][k]][Xcontour[i][j][k]]  # first row, second column, changing to bits channels as in co-ocurence matrix we have only the channels channels
                            except ValueError:  # in case of nan
                                pass

                for i in range(len(imap)):  # slices
                    for j in range(ymin, ymax + 1):  # rows
                        for k in range(xmin, xmax + 1):  # columns
                            try:  # each point
                                matrix_full[i][j - ymin][k - xmin] = int((imap[i][j][k] - vmin) / interval)  # first row, second column, changing to bits channels as in co-ocurence matrix we have only the channels channels
                            except ValueError:  # in case of nan
                                pass

                # define the recurrence place
                if Xcontour_Rec != []:
                    for i in range(len(Xcontour_Rec)):  # slices
                        for j in range(len(Xcontour_Rec[i])):  # sub-structures in the slice
                            for k in range(len(Xcontour_Rec[i][j])):  # each point
                                try:
                                    matrix_rec[i][Ycontour_Rec[i][j][k] - ymin][Xcontour_Rec[i][j][k] - xmin] = int((imap[i][Ycontour_Rec[i][j][k]][Xcontour_Rec[i][j][k]] - vmin) / interval)  # first row, second column, changing to bits channels as in co-ocurence matrix we have only the channels channels
                                except ValueError:  # in case of nan
                                    pass

            else:  # no structure defined
                self.logger.info("Vmin, Vmax " + ", ".join(map(str, (vmin, vmax))))
                for n in range(len(imap)):
                    matrix.append(m.copy())
                    matrix_true.append(m.copy())

                if bits == '':  # fixed bin defined
                    if map_name == 'MTT':
                        interval = binSize * 10  # as the range of relative MTT is normally much grater than range of BV and BF
                    else:
                        interval = binSize
                    n_bits = int((vmax-vmin)//interval + 1)  # calcuate corresponding number of bins
                else: #fixed number of bin defined
                    interval = round((vmax-vmin)/(bits-1), 2)
                    if vmin + (bits-1)*interval < vmax:
                        interval = interval+0.01 # problems with rounding, for example the number 0.1249 would be rounderd to 0.12
                        if vmin + (bits-1)*interval < vmax:
                            MyException('Problem with rounding precision, increase rounding precision of the interval in ROImatrix')
                            raise StopIteration
                    if interval == 0: #in case the image is homogenous after the filtering
                        interval = 0.01 
                    n_bits = bits
                self.logger.info('n bins, interval ' + ", ".join(map(str, (n_bits, interval))))

                for i in range(len(imap)):  # slices
                    for j in range(len(imap[i])):  # rows
                        for k in range(len(imap[i][j])):  # columns
                            try:  # each point
                                matrix[i][j - ymin][k - xmin] = int((imap[i][j][k] - vmin) / interval)  # first row, second column, changing to bits channels as in co-ocurence matrix we have only the channels channels
                                matrix_true[i][j - ymin][k - xmin] = imap[i][j][k]  # first row, second column, changing to bits channels as in co-ocurence matrix we have only the channels channels
                            except ValueError:  # in case of nan
                                pass

            matrix = np.array(matrix)
            matrix_true = np.array(matrix_true)
            matrix_full = np.array(matrix_full)
            matrix_rec = np.array(matrix_rec)

            # if there are constrains for HU in CT
            # crop structures CT has HU range but PET not!
            if HUmin != 'none':
                if rs_type != 0:  # used for original image and LLL filter

                    ind_min = np.where(matrix_true < HUmin)
                    ind_max = np.where(matrix_true > HUmax)
                    matrix_true[ind_min] = np.nan
                    matrix_true[ind_max] = np.nan
                    matrix[ind_min] = np.nan
                    matrix[ind_max] = np.nan
                    matrix_full[ind_min] = np.nan
                    matrix_full[ind_max] = np.nan
                    matrix_rec[ind_min] = np.nan
                    matrix_rec[ind_max] = np.nan
                    print("rs_type", rs_type, " matrix_true", matrix_true.shape)
                    if rs_type == 2:
                        self.HUmask = [ind_min, ind_max]
                    elif rs_type == 1 and cropStructure["crop"] == True:
                        self.HUmask = [ind_min, ind_max]
                    else:
                        self.HUmask = []
                        # if cropStructure["crop"]:
                        # print self.HUmask
                #                    print "Hu mask min bound: min index in x,y,z", np.amin(self.HUmask[0][0]), np.amin(self.HUmask[0][1]), np.amin(self.HUmask[0][2])
                #                    print "Hu mask min bound: max index in x,y,z", np.amax(self.HUmask[0][0]), np.amax(self.HUmask[0][1]), np.amax(self.HUmask[0][2])
                #                    print "Hu mask max bound: min index in x,y,z", np.amin(self.HUmask[1][0]), np.amin(self.HUmask[1][1]), np.amin(self.HUmask[1][2])
                #                    print "Hu mask max bound: max index in x,y,z", np.amax(self.HUmask[1][0]), np.amax(self.HUmask[1][1]), np.amax(self.HUmask[1][2])
                else:
                    matrix_true[HUmask[0]] = np.nan
                    matrix_true[HUmask[1]] = np.nan
                    matrix[HUmask[0]] = np.nan
                    matrix[HUmask[1]] = np.nan
                    matrix_full[HUmask[0]] = np.nan
                    matrix_full[HUmask[1]] = np.nan
                    matrix_rec[HUmask[0]] = np.nan
                    matrix_rec[HUmask[1]] = np.nan
                    self.HUmask = HUmask
            else:
                if cropStructure["crop"] and HUmin != 'none':
                    matrix_true[HUmask[0]] = np.nan
                    matrix_true[HUmask[1]] = np.nan
                    matrix[HUmask[0]] = np.nan
                    matrix[HUmask[1]] = np.nan
                    matrix_full[HUmask[0]] = np.nan
                    matrix_full[HUmask[1]] = np.nan
                    matrix_rec[HUmask[0]] = np.nan
                    matrix_rec[HUmask[1]] = np.nan
                self.HUmask = HUmask

            # cut the matrix in z direction to remove empty slices
            zind = np.where(~np.isnan(matrix))[0]
            zmin = np.min(zind) - 2
            zmax = np.max(zind) + 2
            if zmin < 0:
                zmin = 0
            if zmax >= len(matrix):
                zmax = len(matrix) - 1

            matrix = matrix[zmin:zmax + 1]
            matrix_true = matrix_true[zmin:zmax + 1]
            matrix_full = matrix_full[zmin:zmax + 1]
            matrix_rec = matrix_rec[zmin:zmax + 1]

            # correct for outliers
            if outlier_correction:
                ind_min = np.where(matrix_true < v_out_low)
                ind_max = np.where(matrix_true > v_out_high)
                matrix_true[ind_min] = np.nan
                matrix_true[ind_max] = np.nan
                matrix[ind_min] = np.nan
                matrix[ind_max] = np.nan
                vmin = np.min(matrix_true[np.where(~np.isnan(matrix_true))])
                vmax = np.max(matrix_true[np.where(~np.isnan(matrix_true))])

            self.matrix = matrix
            self.interval = interval
            self.norm_points = len(np.where(~np.isnan(matrix))[0])
            self.matrix_true = matrix_true
            self.matrix_full = matrix_full
            self.matrix_rec = matrix_rec
            self.n_bits = n_bits
            self.vmin = vmin
            self.vmax = vmax

        except ValueError:
            print('roi value')
            if structure != 'none' and Xcontour == '':
                self.n_bits = 'too small contour'
            else:
                self.n_bits = 'values out of range'
            self.matrix = []
            self.interval = ''
            self.norm_points = 0
            self.matrix_true = []
            self.matrix_full = []
            self.matrix_rec = []
            self.HUmask = []
            self.vmin = 0
            self.vmax = 0
        except IndexError:
            print('roi index')
            if HUmask == []:
                self.n_bits = 'values out of range'
            self.matrix = []
            self.interval = ''
            self.norm_points = 0
            self.matrix_true = []
            self.matrix_full = []
            self.matrix_rec = []
            self.HUmask = []
            self.vmin = 0
            self.vmax = 0
