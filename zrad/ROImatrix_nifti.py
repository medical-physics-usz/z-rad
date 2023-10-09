import logging

import numpy as np
from scipy.interpolate import interpn

from exception import MyException
from tqdm import tqdm


class MatrixNifti(object):
    def __init__(self, imap, rs_type, structure, map_name, contour, columns, rows, HUmin, HUmax, binSize, bits,
                 outlier_correction, HUmask):
        """ fill the contour matrix with values from the image - including discretization
            matrix - matrix with discretized entries
            matrix_true - matrix with original values for first-order statistics"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Read in Data into ROIMatrix")
        matrix = []
        v = []
        try:
            if structure != 'none':  # if the structure is defined
                
                # searching for the matrix size in 3D
                if rs_type == 1:  # original vs wavelet
                    matrix_true = imap * contour
                    ind = np.where(contour == 0)
                    matrix_true[ind] = np.nan  # matrix with real image values
                    
                # for the outlier correction, first remove values outside of HU range (for CT only) and then calccuate
                # standard deviation

                    ind = np.where(~np.isnan(np.array(matrix_true)))
                    v = list(matrix_true[ind])
    
                    # no parameter for outlier correction!
                    if HUmin != 'none':
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

                else:
                    contour_wavelet = np.zeros((int(round(len(contour)/2+0.01,0)), int(round(len(contour[0])/2+0.01, 0)), int(round(len(contour[0][0])/2+0.01,0))))
                    imap = imap[1:-1, 1:-1, 1:-1]
                    
                    x = np.arange(0, len(contour[0][0]))
                    y = np.arange(0, len(contour[0]))
                    z = np.arange(0, len(contour))
                    
                    new_points = []
                    for xi in np.arange(0, len(contour[0][0]),2):
                        for yi in np.arange(0, len(contour[0]),2):
                            for zi in np.arange(0, len(contour),2):
                                new_points.append([zi, yi, xi])
                    new_points = np.array(new_points)
                    
                    new_values = interpn((z,y,x), contour, new_points, method='linear')
                    
                    for pni, pn in enumerate(new_points):
                        contour_wavelet[pn[0]//2, pn[1]//2, pn[2]//2] = new_values[pni]
                    ind = np.where(contour_wavelet >= 0.5)
                    contour_wavelet[ind] = 1.
                    ind = np.where(contour_wavelet < 0.5)
                    contour_wavelet[ind] = 0
                    
                    # print(contour_wavelet.shape)
                    
                    matrix_true = imap * contour_wavelet # matrix with real image values
                    ind = np.where(contour_wavelet == 0)
                    matrix_true[ind] = np.nan
                    
                    ind = np.where(~np.isnan(np.array(matrix_true)))
                    v = list(matrix_true[ind])
    
                    # no parameter for outlier correction!
                    if HUmin != 'none' and rs_type == 2:
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
                vmin = np.nanmin(imap)
                vmax = np.nanmax(imap)

            if structure != 'none':  # if the structure to be anaylsed was defined
                self.logger.info("Vmin, Vmax " + ", ".join(map(str, (round(vmin, 3), round(vmax, 3)))))

                matrix = np.zeros(matrix_true.shape)  # matrix with discretizd values
                matrix[:, :, :] = np.nan
                matrix_full = np.zeros(matrix_true.shape)  # matrix with discretizd values but not cut to the ROI
                matrix_full[:, :, :] = np.nan
                matrix_rec = np.zeros(matrix_true.shape)  
                matrix_rec[:, :, :] = np.nan

                if bits == '':  # fixed bin defined
                    if map_name == 'MTT':
                        # as the range of relative MTT is normally much grater than range of BV and BF
                        interval = binSize * 10
                    else:
                        interval = binSize
                    n_bits = int((vmax - vmin) // interval + 1)  # calculate corresponding number of bins
                else:  # fixed number of bin defined
                    interval = round((vmax-vmin)/(bits-1), 2)
                    if vmin + (bits-1)*interval < vmax:
                        interval = interval+0.01  # problems with rounding, for example the number 0.1249 would be rounderd to 0.12
                        if vmin + (bits-1)*interval < vmax:
                            MyException('Problem with rounding precision, increase rounding precision of the interval in ROImatrix')
                            raise StopIteration
                    if interval == 0:  # in case the image is homogenous after the filtering
                        interval = 0.01 
                    n_bits = bits
                self.logger.info('n bins, interval ' + ", ".join(map(str, (n_bits, interval))))
                
                Xcontour = np.where(~np.isnan(matrix_true))

                for i in range(len(Xcontour[0])):  # slices
                    try:
                        # first row, second column, changing to bits channels as in co-ocurence matrix we have
                        # only the channels
                        matrix[Xcontour[0][i]][Xcontour[1][i]][Xcontour[2][i]] = int((imap[Xcontour[0][i]][Xcontour[1][i]][Xcontour[2][i]] - vmin) / interval)
                    except ValueError:  # in case of nan
                        pass

                for i in range(len(matrix_full)):  # slices
                    for j in range(len(matrix_full[0])):  # rows
                        for k in range(len(matrix_full[0][0])):  # columns
                            try:  # each point
                                # first row, second column, changing to bits channels as in co-ocurence matrix we have
                                # only the channels
                                matrix_full[i][j][k] = int((imap[i][j][k] - vmin) / interval)
                            except ValueError:  # in case of nan
                                pass


            else:  # no structure defined
                self.logger.info("Vmin, Vmax " + ", ".join(map(str, (vmin, vmax))))
                matrix = np.zeros(matrix_true.shape) #matrix with discretizd values
                matrix[:,:,:] = np.nan 
                matrix_true = imap


                if bits == '':  # fixed bin defined
                    if map_name == 'MTT':
                        # as the range of relative MTT is normally much grater than range of BV and BF
                        interval = binSize * 10
                    else:
                        interval = binSize

                    n_bits = int((vmax-vmin) // interval + 1)  # calculate corresponding number of bins
                else:  # fixed number of bin defined
                    interval = round((vmax - vmin) / (bits - 1), 2)
                    if vmin + (bits - 1) * interval < vmax:
                        # problems with rounding, for example the number 0.1249 would be rounded to 0.12
                        interval = interval + 0.01
                        if vmin + (bits-1) * interval < vmax:
                            MyException('Problem with rounding precision, increase rounding precision of the interval in ROImatrix')
                            raise StopIteration
                    if interval == 0:  # in case the image is homogenous after the filtering
                        interval = 0.01 

                    n_bits = bits
                self.logger.info('n bins, interval ' + ", ".join(map(str, (n_bits, interval))))

                for i in range(len(imap)):  # slices
                    for j in range(len(imap[i])):  # rows
                        for k in range(len(imap[i][j])):  # columns
                            try:  # each point
                                # first row, second column, changing to bits channels as in co-occurrence matrix we have
                                # only the channels channels
                                matrix[i][j][k] = int((imap[i][j][k] - vmin) / interval)
                            except ValueError:  # in case of nan
                                pass               
                
                matrix_rec = imap
                matrix_full = imap

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
                    # print("rs_type", rs_type, " matrix_true", matrix_true.shape)
                    if rs_type == 2:
                        self.HUmask = [ind_min, ind_max]
                    elif rs_type == 1:
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
                self.HUmask = []

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
#
        except ValueError:
            # print('roi value')
            if rs_type == 1:
                ind = np.where(contour==1)
            else:
                ind = np.where(contour_wavelet==1)
            if structure != 'none' and len(ind[0]) < 2:
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
            # print('roi index')
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
