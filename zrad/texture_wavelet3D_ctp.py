import pywt
try:
    import pydicom as dc # dicom library
except ImportError:
    import dicom as dc # dicom library
from numpy import arange, array, fromstring, int16, zeros, float16, isnan, where, nan, concatenate, floor
from pylab import imshow, figure, show, close, cm, subplot, plot, savefig
from numpy import min as nmin
from numpy import max as nmax
from numpy import float as floatn
from numpy import int as intn
from os import makedirs
from os.path import isdir

class WaveletCTP(object):
    def __init__(self, m, path, perf, name):
        self.map = m
        ind = where(isnan(m))
        mean = 0
        licz=0
        cnt = []
        for i in arange(0, len(ind[0])):
            cnt.append([ind[0][i], ind[1][i], ind[2][i]])
        for z in arange(0, len(m)):
            for y in arange(0, len(m[z])):
                for x in arange(0, len(m[z][y])):
                    if [z,y,x] not in cnt:
                        mean += m[z][y][x]
                        licz += 1.
        mean = mean/licz
        for i in arange(0, len(ind[0])):
            self.map[ind[0][i]][ind[1][i]][ind[2][i]] = mean
            
        HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL = self.threeD(self.map)

        for i in arange(0, len(ind[0])):
            self.map[ind[0][i]][ind[1][i]][ind[2][i]] = nan
        for i in arange(0, len(cnt)):
            cnt[i] = [int(round(cnt[i][0]/2.+1, 0)),int(round(cnt[i][1]/2.+1, 0)),int(round(cnt[i][2]/2.+1, 0))]

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
        k=1
        try:
            makedirs(path+'\\wavelet\\'+name+'\\')
        except OSError:
            if not isdir(path+'\\wavelet\\'+name+'\\'):
                raise
        for i in arange(0, len(self.map), 2):
            figure()
            subplot(331)
##            nani = where(isnan(self.map[i]))
##            for j in arange(0, len(nani[0])):
##                self.map[i][nani[0][j]][nani[1][j]] = 2
##            nani = where(isnan(self.HHH[k]))
##            for j in arange(0, len(nani[0])):
##                self.HHH[k][nani[0][j]][nani[1][j]] = 0
##                self.HHL[k][nani[0][j]][nani[1][j]] = 0
##                self.HLH[k][nani[0][j]][nani[1][j]] = 0
##                self.HLL[k][nani[0][j]][nani[1][j]] = 0
##                self.LHH[k][nani[0][j]][nani[1][j]] = 0
##                self.LHL[k][nani[0][j]][nani[1][j]] = 0
##                self.LLH[k][nani[0][j]][nani[1][j]] = 0
##                self.LLL[k][nani[0][j]][nani[1][j]] = 2
                
            imshow(self.map[i], vmin = 0, vmax = 10, cmap = cm.jet)
            subplot(332)
            imshow(self.HHH[k], vmin = -1, vmax = 5, cmap = cm.jet)
            subplot(333)
            imshow(self.HHL[k], vmin = -1, vmax = 5, cmap = cm.jet)
            subplot(334)
            imshow(self.HLH[k], vmin = -1, vmax = 5,cmap = cm.jet)
            subplot(335)
            imshow(self.HLL[k], vmin = -1, vmax = 5,cmap = cm.jet)
            subplot(336)
            imshow(self.LHH[k], vmin = -1, vmax = 5,cmap = cm.jet)
            subplot(337)
            imshow(self.LHL[k], vmin = -1, vmax = 5,cmap = cm.jet)
            subplot(338)
            imshow(self.LLH[k], vmin = -1, vmax = 5,cmap = cm.jet)
            subplot(339)
            imshow(self.LLL[k], vmin = 0, vmax = 10,cmap = cm.jet)
            savefig(path+'\\wavelet\\'+name+'\\'+perf+'_'+str(k)+'.png')
            close()
            k+=1
    def Return(self):
        return self.map, self.LLL, self.HHH, self.HHL, self.HLH, self.HLL, self.LHH, self.LHL, self.LLH, 

    def threeD(self, a):
        H = []
        L = []
        for z in arange(0, len(a)):
            h = []
            l = []
            for i in a[z]:
                cA, cD = pywt.dwt(i, 'coif1')
                h.append(cD)
                l.append(cA)
            H.append(h)
            L.append(l)

        H = array(H)
        L = array(L)

        hh = zeros((len(a), int(floor((len(a[0])+5)/2.)), int(floor((len(a[0][0])+5)/2.))))
        hl = zeros((len(a), int(floor((len(a[0])+5)/2.)), int(floor((len(a[0][0])+5)/2.))))
        lh = zeros((len(a), int(floor((len(a[0])+5)/2.)), int(floor((len(a[0][0])+5)/2.))))
        ll = zeros((len(a), int(floor((len(a[0])+5)/2.)), int(floor((len(a[0][0])+5)/2.))))

        for z in arange(0,len(a)):
            for i in arange(0, len(H[z][0])):
                cA, cD = pywt.dwt(H[z][:,i], 'coif1')
                hh[z][:,i] = cD
                hl[z][:,i] = cA
                
        for z in arange(0,len(a)):
            for i in arange(0, len(L[z][0])):
                cA, cD = pywt.dwt(L[z][:,i], 'coif1')
                lh[z][:,i] = cD
                ll[z][:,i] = cA

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
                cA, cD = pywt.dwt(hh[:,y,x], 'coif1')
                hhh[:,y,x] = cD
                hhl[:,y,x] = cA
                
        for y in arange(0, len(hl[0])):
            for x in arange(0, len(hl[0][y])):
                cA, cD = pywt.dwt(hl[:,y,x], 'coif1')
                hlh[:,y,x] = cD
                hll[:,y,x] = cA

        for y in arange(0, len(lh[0])):
            for x in arange(0, len(lh[0][y])):
                cA, cD = pywt.dwt(lh[:,y,x], 'coif1')
                lhh[:,y,x] = cD
                lhl[:,y,x] = cA

        for y in arange(0, len(ll[0])):
            for x in arange(0, len(ll[0][y])):
                cA, cD = pywt.dwt(ll[:,y,x], 'coif1')
                llh[:,y,x] = cD
                lll[:,y,x] = cA

        
        return hhh, hhl, hlh, hll, lhh, lhl, llh, lll


        
##        
##    ct.Rows = len(hh)
##    ct.Columns = len(hh[0])
##    ct.PixelSpacing[0] = ct.PixelSpacing[0]*2#**n
##    ct.PixelSpacing[1] = ct.PixelSpacing[1]*2#**n
##    a=array(hh, dtype = float16)
##    ct.PixelData = a.tostring()
##    ct.save_as(path+'proba_hh'+str(n)+'.dcm')
##    
##    a=array(hl, dtype = float16)
##    ct.PixelData = a.tostring()
##    ct.save_as(path+'proba_hl'+str(n)+'.dcm')
##    
##    a=array(lh, dtype = float16)
##    ct.PixelData = a.tostring()
##    ct.save_as(path+'proba_lh'+str(n)+'.dcm')
##    
##    a=array(ll, dtype = float16)
##    ct.PixelData = a.tostring()
##    ct.save_as(path+'proba_ll'+str(n)+'.dcm')
##    
##    return hh, hl, lh, ll
##
##figure()
##subplot(221)
##imshow(a, cmap = cm.Greys_r,vmin=0,vmax=10)#, vmin = -50 , vmax=100)
##for i in arange(0, len(contours[0])):
##    plot(contours[0][i][0], contours[0][i][1],'green', linewidth=2)
##subplot(222)
##imshow(ll1, cmap = cm.Greys_r,vmin=0,vmax=10)#, vmin = -50 , vmax=100)
##for i in arange(0, len(contours_w[0])):
##    plot(contours_w[0][i][0], contours_w[0][i][1],'green', linewidth=2)
##    plot(contours_n[0][i][0], contours_n[0][i][1],'blue', linewidth=2)
##subplot(223)
##imshow(hl1, cmap = cm.Greys_r,vmin=0,vmax=10)#, vmin = -50 , vmax=100)
##for i in arange(0, len(contours_w1[0])):
##    plot(contours_w[0][i][0], contours_w[0][i][1],'green', linewidth=2)
##    plot(contours_n[0][i][0], contours_n[0][i][1],'blue', linewidth=2)
##subplot(224)
##imshow(lh1, cmap = cm.Greys_r,vmin=0,vmax=10)#, vmin = -50 , vmax=100)
##for i in arange(0, len(contours_w1[0])):
##    plot(contours_w[0][i][0], contours_w[0][i][1],'green', linewidth=2)
##    plot(contours_n[0][i][0], contours_n[0][i][1],'blue', linewidth=2)
##show()
##close()





