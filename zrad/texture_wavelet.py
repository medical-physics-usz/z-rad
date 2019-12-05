import pywt
try:
    import pydicom as dc  # dicom library
except ImportError:
    import dicom as dc  # dicom library
from numpy import sqrt, arange, array, fromstring, int16, zeros, float16, isnan, where, nan, concatenate, floor
from pylab import imshow, figure, show, close, cm, subplot, plot, savefig
from numpy import min as nmin
from numpy import max as nmax
from numpy import float as floatn
from numpy import int as intn
from os import makedirs
from os.path import isdir
import logging


class Wavelet(object):
    """wavelet transform"""
    def __init__(self, m, path, im_name, name, dim):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Start: Wavelet Calculation")
        self.map = m
        self.dim = dim

        # if len(self.map) != 1:
        if self.dim == "3D":
            HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL = self.wavelet_calculation(self.map)  # xyz

            self.HHH = HHH/(2.*sqrt(2))
            self.HHL = HHL/(2.*sqrt(2))
            self.HLH = HLH/(2.*sqrt(2))
            self.HLL = HLL/(2.*sqrt(2))
            self.LHH = LHH/(2.*sqrt(2))
            self.LHL = LHL/(2.*sqrt(2))
            self.LLH = LLH/(2.*sqrt(2))
            self.LLL = LLL/(2.*sqrt(2))
            k = 1
            try:
                makedirs(path+'\\wavelet\\'+name+'\\')
            except OSError:
                if not isdir(path+'\\wavelet\\'+name+'\\'):
                    raise
#            for i in arange(0, len(self.map), 2):
#                figure()
#                subplot(331)
#                imshow(self.map[i], cm.Greys_r)
#                subplot(332)
#                imshow(self.HHH[k], cm.Greys_r)
#                subplot(333)
#                imshow(self.HHL[k], cm.Greys_r)
#                subplot(334)
#                imshow(self.HLH[k], cm.Greys_r)
#                subplot(335)
#                imshow(self.HLL[k], cm.Greys_r)
#                subplot(336)
#                imshow(self.LHH[k], cm.Greys_r)
#                subplot(337)
#                imshow(self.LHL[k], cm.Greys_r)
#                subplot(338)
#                imshow(self.LLH[k], cm.Greys_r)
#                subplot(339)
#                imshow(self.LLL[k], cm.Greys_r)
#                savefig(path+'\\wavelet\\'+name+'\\'+im_name+'_'+str(k)+'.png')
#                close()
#                k+=1
        elif self.dim == "2D" or dim == "2D_singleSlice":
            HH, HL, LH, LL = self.wavelet_calculation(self.map)

            self.HH = HH/2
            self.HL = HL/2
            self.LH = LH/2
            self.LL = LL/2
            k = 0
            try:
                makedirs(path+'\\wavelet\\2D_'+name+'\\')
            except OSError:
                if not isdir(path+'\\wavelet\\2D_'+name+'\\'):
                    raise
#            for i in arange(0, len(self.map), 2):
#                figure()
#                subplot(321)
#                imshow(self.map[i], cm.Greys_r)
#                subplot(322)
#                imshow(self.HH[k], cm.Greys_r)
#                subplot(323)
#                imshow(self.HL[k], cm.Greys_r)
#                subplot(324)
#                imshow(self.LH[k], cm.Greys_r)
#                subplot(325)
#                imshow(self.LL[k], cm.Greys_r)
#                savefig(path+'\\wavelet\\2D_'+name+'\\'+im_name+'_'+str(k)+'.png')
#                close()
#                k+=1

    def Return(self):
        # if len(self.map) != 1:
        if self.dim == "3D":
            return self.map, self.LLL, self.HHH, self.HHL, self.HLH, self.HLL, self.LHH, self.LHL, self.LLH
        else:
            return self.map, self.LL, self.HH, self.HL, self.LH

    def wavelet_calculation(self, a):
        # x direction
        H = []
        L = []
        for z in arange(0, len(a)):
            h = []
            l = []
            for i in a[z]:  # calculate wavelet for each row
                cA, cD = pywt.dwt(i, 'coif1')  # dwt: discrete wavelet transform
                h.append(cD)  # detail coefficient (from the high-pass filter)
                l.append(cA)  # approximation coefficient (from the low-pass filter)
            H.append(h)
            L.append(l)

        H = array(H)
        L = array(L)

        # y direction
        hh = zeros((len(a), int(floor((len(a[0])+5)/2.)), int(floor((len(a[0][0])+5)/2.))))
        hl = zeros((len(a), int(floor((len(a[0])+5)/2.)), int(floor((len(a[0][0])+5)/2.))))
        lh = zeros((len(a), int(floor((len(a[0])+5)/2.)), int(floor((len(a[0][0])+5)/2.))))
        ll = zeros((len(a), int(floor((len(a[0])+5)/2.)), int(floor((len(a[0][0])+5)/2.))))

        for z in arange(0, len(a)):
            for i in arange(0, len(H[z][0])):  # calculate wavelets over all columns
                cA, cD = pywt.dwt(H[z][:, i], 'coif1')  # for detailed coeff. from x direction
                hh[z][:, i] = cD  # xy
                hl[z][:, i] = cA
                
        for z in arange(0, len(a)):
            for i in arange(0, len(L[z][0])):
                cA, cD = pywt.dwt(L[z][:, i], 'coif1')
                lh[z][:, i] = cD
                ll[z][:, i] = cA

        # if 9 slice-CT: no need to calculate wavelet in z direction, exit here
        if self.dim == "2D" or self.dim == "2D_singleSlice":
            return hh, hl, lh, ll

        # z direction
        hhh = zeros((int(floor((len(a)+5)/2.)), int(floor((len(a[0])+5)/2.)), int(floor((len(a[0][0])+5)/2.))))
        hhl = zeros((int(floor((len(a)+5)/2.)), int(floor((len(a[0])+5)/2.)), int(floor((len(a[0][0])+5)/2.))))
        hlh = zeros((int(floor((len(a)+5)/2.)), int(floor((len(a[0])+5)/2.)), int(floor((len(a[0][0])+5)/2.))))
        hll = zeros((int(floor((len(a)+5)/2.)), int(floor((len(a[0])+5)/2.)), int(floor((len(a[0][0])+5)/2.))))
        lhh = zeros((int(floor((len(a)+5)/2.)), int(floor((len(a[0])+5)/2.)), int(floor((len(a[0][0])+5)/2.))))
        lhl = zeros((int(floor((len(a)+5)/2.)), int(floor((len(a[0])+5)/2.)), int(floor((len(a[0][0])+5)/2.))))
        llh = zeros((int(floor((len(a)+5)/2.)), int(floor((len(a[0])+5)/2.)), int(floor((len(a[0][0])+5)/2.))))
        lll = zeros((int(floor((len(a)+5)/2.)), int(floor((len(a[0])+5)/2.)), int(floor((len(a[0][0])+5)/2.))))

        for y in arange(0, len(hh[0])):
            for x in arange(0, len(hh[0][y])):
                cA, cD = pywt.dwt(hh[:, y, x], 'coif1')
                hhh[:, y, x] = cD
                hhl[:, y, x] = cA
                
        for y in arange(0, len(hl[0])):
            for x in arange(0, len(hl[0][y])):
                cA, cD = pywt.dwt(hl[:, y, x], 'coif1')
                hlh[:, y, x] = cD  # xyz
                hll[:, y, x] = cA

        for y in arange(0, len(lh[0])):
            for x in arange(0, len(lh[0][y])):
                cA, cD = pywt.dwt(lh[:, y, x], 'coif1')
                lhh[:, y, x] = cD
                lhl[:, y, x] = cA

        for y in arange(0, len(ll[0])):
            for x in arange(0, len(ll[0][y])):
                cA, cD = pywt.dwt(ll[:, y, x], 'coif1')
                llh[:, y, x] = cD
                lll[:, y, x] = cA

        return hhh, hhl, hlh, hll, lhh, lhl, llh, lll