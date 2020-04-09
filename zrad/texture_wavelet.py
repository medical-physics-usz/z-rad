# -*- coding: utf-8 -*-s

# import libraries
import pywt
from numpy import sqrt, arange, array, zeros, where, floor, isnan, nan
import matplotlib.pyplot as plt
from os import makedirs
from os.path import isdir
import logging

class Wavelet(object):
    """wavelet transform"""
    def __init__(self, m, path, im_name, name, dim, ctp):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Start: Wavelet Calculation")
        self.map = m
        self.dim = dim

        if ctp:
            perf = im_name
            ind = where(isnan(m))
            mean = 0
            licz = 0
            cnt = []
            for i in range(len(ind[0])):
                cnt.append([ind[0][i], ind[1][i], ind[2][i]])
            for z in range(len(m)):
                for y in range(len(m[z])):
                    for x in range(len(m[z][y])):
                        if [z, y, x] not in cnt:
                            mean += m[z][y][x]
                            licz += 1.
            mean = mean/licz
            for i in range(len(ind[0])):
                self.map[ind[0][i]][ind[1][i]][ind[2][i]] = mean

        if self.dim == "3D":
            HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL = self.wavelet_calculation(self.map)  # xyz

            if ctp:
                for i in range(len(ind[0])):
                    self.map[ind[0][i]][ind[1][i]][ind[2][i]] = nan
                for i in range(len(cnt)):
                    cnt[i] = [int(round(cnt[i][0] / 2. + 1, 0)), int(round(cnt[i][1] / 2. + 1, 0)),
                              int(round(cnt[i][2] / 2. + 1, 0))]

                for i in cnt:
                    HHH[i[0]][i[1]][i[2]] = nan
                    HHL[i[0]][i[1]][i[2]] = nan
                    HLH[i[0]][i[1]][i[2]] = nan
                    HLL[i[0]][i[1]][i[2]] = nan
                    LHH[i[0]][i[1]][i[2]] = nan
                    LHL[i[0]][i[1]][i[2]] = nan
                    LLH[i[0]][i[1]][i[2]] = nan
                    LLL[i[0]][i[1]][i[2]] = nan

                self.HHH = HHH
                self.HHL = HHL
                self.HLH = HLH
                self.HLL = HLL
                self.LHH = LHH
                self.LHL = LHL
                self.LLH = LLH
                self.LLL = LLL
            else:
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
            # if not ctp:
               # for i in range(len(self.map), 2):
               #     plt.figure()
               #     plt.subplot(331)
               #     plt.imshow(self.map[i], cmap=plt.get_cmap('Greys_r'))
               #     plt.title('original')
               #     plt.subplot(332)
               #     plt.imshow(self.HHH[k], cmap=plt.get_cmap('Greys_r'))
               #     plt.title('HHH')
               #     plt.subplot(333)
               #     plt.imshow(self.HHL[k], cmap=plt.get_cmap('Greys_r'))
               #     plt.title('HHL')
               #     plt.subplot(334)
               #     plt.imshow(self.HLH[k], cmap=plt.get_cmap('Greys_r'))
               #     plt.title('HLH')
               #     plt.subplot(335)
               #     plt.imshow(self.HLL[k], cmap=plt.get_cmap('Greys_r'))
               #     plt.title('HLL')
               #     plt.subplot(336)
               #     plt.imshow(self.LHH[k], cmap=plt.get_cmap('Greys_r'))
               #     plt.title('LHH')
               #     plt.subplot(337)
               #     plt.imshow(self.LHL[k], cmap=plt.get_cmap('Greys_r'))
               #     plt.title('LHL')
               #     plt.subplot(338)
               #     plt.imshow(self.LLH[k], cmap=plt.get_cmap('Greys_r'))
               #     plt.title('LLH')
               #     plt.subplot(339)
               #     plt.imshow(self.LLL[k], cmap=plt.get_cmap('Greys_r'))
               #     plt.title('LLL')
               #     plt.savefig(path+'\\wavelet\\'+name+'\\'+im_name+'_'+str(k)+'.png')
               #     plt.close()
               #     k+=1
            if ctp:  # if ctp
                for i in range(len(self.map), 2):
                    plt.figure()
                    plt.subplot(331)
                    plt.imshow(self.map[i], vmin=0, vmax=10, cmap=plt.get_cmap('Jet'))
                    plt.subplot(332)
                    plt.imshow(self.HHH[k], vmin=-1, vmax=5, cmap=plt.get_cmap('Greys_r'))
                    plt.subplot(333)
                    plt.imshow(self.HHL[k], vmin=-1, vmax=5, cmap=plt.get_cmap('Greys_r'))
                    plt.subplot(334)
                    plt.imshow(self.HLH[k], vmin=-1, vmax=5, cmap=plt.get_cmap('Greys_r'))
                    plt.subplot(335)
                    plt.imshow(self.HLL[k], vmin=-1, vmax=5, cmap=plt.get_cmap('Greys_r'))
                    plt.subplot(336)
                    plt.imshow(self.LHH[k], vmin=-1, vmax=5, cmap=plt.get_cmap('Greys_r'))
                    plt.subplot(337)
                    plt.imshow(self.LHL[k], vmin=-1, vmax=5, cmap=plt.get_cmap('Greys_r'))
                    plt.subplot(338)
                    plt.imshow(self.LLH[k], vmin=-1, vmax=5, cmap=plt.get_cmap('Greys_r'))
                    plt.subplot(339)
                    plt.imshow(self.LLL[k], vmin=0, vmax=10, cmap=plt.get_cmap('Greys_r'))
                    plt.savefig(path + '\\wavelet\\' + name + '\\' + perf + '_' + str(k) + '.png')
                    plt.close()
                    k += 1

        elif self.dim == "2D" or dim == "2D_singleSlice":
            HH, HL, LH, LL = self.wavelet_calculation(self.map)

            if ctp:
                print("ctp - wavelets for 2D is not adapted yet!!!!!!")
                import sys
                sys.exit("Error message: check implementation in texture_wavelet.py line 141")

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

            # for i in range(len(self.map), 2):
            #     plt.figure()
            #     plt.subplot(321)
            #     plt.imshow(self.map[i], cmap=plt.get_cmap('Greys_r'))
            #     plt.title('original')
            #     plt.subplot(322)
            #     plt.imshow(self.HH[k], cmap=plt.get_cmap('Greys_r'))
            #     plt.title('HH')
            #     plt.subplot(323)
            #     plt.imshow(self.HL[k], cmap=plt.get_cmap('Greys_r'))
            #     plt.title('HL')
            #     plt.subplot(324)
            #     plt.imshow(self.LH[k], cmap=plt.get_cmap('Greys_r'))
            #     plt.title('LH')
            #     plt.subplot(325)
            #     plt.imshow(self.LL[k], cmap=plt.get_cmap('Greys_r'))
            #     plt.title('LL')
            #     plt.savefig(path + '\\wavelet\\2D_' + name + '\\' + im_name + '_' + str(k) + '.png')
            #     plt.close()
            #     k += 1

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
        for z in range(len(a)):
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

        for z in range(len(a)):
            for i in range(len(H[z][0])):  # calculate wavelets over all columns
                cA, cD = pywt.dwt(H[z][:, i], 'coif1')  # for detailed coeff. from x direction
                hh[z][:, i] = cD  # xy
                hl[z][:, i] = cA
                
        for z in range(len(a)):
            for i in range(len(L[z][0])):
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

        for y in range(len(hh[0])):
            for x in range(len(hh[0][y])):
                cA, cD = pywt.dwt(hh[:, y, x], 'coif1')
                hhh[:, y, x] = cD
                hhl[:, y, x] = cA
                
        for y in range(len(hl[0])):
            for x in range(len(hl[0][y])):
                cA, cD = pywt.dwt(hl[:, y, x], 'coif1')
                hlh[:, y, x] = cD  # xyz
                hll[:, y, x] = cA

        for y in range(len(lh[0])):
            for x in range(len(lh[0][y])):
                cA, cD = pywt.dwt(lh[:, y, x], 'coif1')
                lhh[:, y, x] = cD
                lhl[:, y, x] = cA

        for y in range(len(ll[0])):
            for x in range(len(ll[0][y])):
                cA, cD = pywt.dwt(ll[:, y, x], 'coif1')
                llh[:, y, x] = cD
                lll[:, y, x] = cA

        return hhh, hhl, hlh, hll, lhh, lhl, llh, lll
