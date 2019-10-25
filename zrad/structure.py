# -*- coding: utf-8 -*-s

from numpy import arange,floor
import cv2
import numpy as np
try:
    import pydicom as dc # dicom library
except ImportError:
    import dicom as dc # dicom library

from exception import MyException
import logging

class Structures(object):
    '''Class to extract each point in the image grid which belongs to the defined ROI
    rs - to the structure set including structure set name
    structure – list of organs to be analyzed, always choose first existing organ for this list, for example [GTV_art, GTV], if GTV_art  is not defined in the rs then choose GTV, if GTV not defined then /AttributeError with a message: missing structure
    slices – list of positions of slice in z 
    x_ct - x coordinate of image top left corner
    y_ct – y coordinate of image top left corner
    xCTspace – pixel spacing in xy
    position – patient position on the table for example HFS (head first supine)
    len_IM – number of slices
    wv - bool, calculate wavelet, to see if we need contours in wavelet space
        '''
    def __init__(self, rs, structure, slices, x_ct, y_ct, xCTspace, len_IM, wv, local):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Start Reading in StructureSet")
        self.slices = slices
        self.x_ct = x_ct
        self.y_ct = y_ct
        self.xCTspace = xCTspace
        self.len_IM = len_IM
        self.wv = wv

        self.Xcontour_W = []
        self.Ycontour_W = []
        self.slices_w = []
        self.Xcontour = []
        self.Ycontour = []
        self.Xcontour_Rec = []
        self.Ycontour_Rec = []

        self.find(rs, structure, local)

    def find(self, rs, structure, local):
        self.logger.debug("Structure set filename " + rs)
        self.rs = dc.read_file(rs) #read RS file
        list_organs =[] #list of organs defined in the RS file
        self.logger.info('structures in RS file: ')
        for j in arange(0, len(self.rs.StructureSetROISequence)):
            list_organs.append([self.rs.StructureSetROISequence[j].ROIName, self.rs.StructureSetROISequence[j].ROINumber])
            self.logger.info("Structures in Structure Set File: " + self.rs.StructureSetROISequence[j].ROIName)

        organs = structure #define the structure you're intersed in

        self.contours=[] #list with structure contours

        #search for organ I'm intersted in
        for i in arange(0, len(organs)): #organs definde by user
            for j in arange(0, len(list_organs)): #organ in RS
                if list_organs[j][0] == organs[i]: #if the same name
                    for k in arange(0, len(self.rs.ROIContourSequence)): #searach by ROI number
                        if self.rs.ROIContourSequence[k].ReferencedROINumber == list_organs[j][1]: # double check the ROI number
                            try:
                                lista = [] #z position of the slice
                                #controus in dicom are save as a list with sequence x1, y1, zi, x2, y2, zi, ... xn, yn, zi
                                #where zi is the slice position
                                #if there are subcontours in the slice then these re two different sequences with the same zi
                                for l in arange(0, len(self.rs.ROIContourSequence[k].ContourSequence)):
                                    lista.append([round(float(self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[2]),3), self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[::3], self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[1::3]])
                                lista.sort()
                                index = []
                                lista = self.multiContour(lista) #subcontrous in the slice
                                for m in arange(0, len(lista)):
                                    index.append(lista[m][0])
                                self.logger.info('z positions contour \n' + ", ".join(map(str,index)))
                                self.logger.info('z positions image \n' + ", ".join(map(str,self.slices)))
                                slice_count = True #True is more than one slice
                                try:
                                    diffI = round(index[1]-index[0], 2) #double check if the orientation is ok
                                except IndexError:
                                    info = 'only one slice'
                                    slice_count = False
                                if slice_count: #if more than one slice
                                    diffS = round(self.slices[1]-self.slices[0],2)
                                    self.logger.info("resolution image, ROI " + ", ".join(map(str, (diffI, diffS))))
                                    if np.sign(diffI) != np.sign(diffS): #if different orientation then reverse the contour points
                                        index.reverse()
                                        lista.reverse()
                                    #check for slices withut contour in between other contour slices
                                    diff = abs(np.array(index[1:])-np.array(index[:-1]))/diffS
                                    self.logger.info("difference in t position between slices normalized to lice spacing  " + ", ".join(map(str, diff)))
                                    dk = 0
                                    for d in arange(0, len(diff)):
                                        for di in arange(1, abs(int(round(diff[d],0)))): #if no empt slice in between then abs(int(round(diff[d],0))) = 1
                                            index.insert(d+dk+1, index[d+dk]+diffS) #if not add empt slices to index and lista
                                            lista.insert(d+dk+1, [[],[[],[]]])
                                            dk+=1
                                    #include empty list to slices where structure was not contour, so in the end lista and index has the same length as image
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
                                    self.contours.append(lista) #list of contours for all user definded strctures
                                    break
                                else: #if only one slice of contour
                                    ind = np.where(np.array(self.slices) == index[0])[0][0]
                                    self.logger.info("contour only in slice")
                                    if ind!=0:
                                        for m in arange(0, abs(ind-0)):
                                            lista.insert(0,[[],[[],[]]])
                                    if ind!=(len(self.slices)-1):
                                        for m in arange(0, abs(ind-(len(self.slices)-1))):
                                            lista.append([[],[[],[]]])
                                    for n in arange(0, len(lista)):
                                        lista[n] = lista[n][1:]
                                    self.contours.append(lista)
                            except AttributeError:
                                self.logger.info("no contours for: " + organs[i])

        #recalculating for pixels the points into pixels
        self.cnt=[]
        if local:
            self.contours = np.array(self.contours)
            self.organs = organs[0]
        else:
            try:
                if len(self.contours)!=1:
                    self.contours = np.array([self.contours[0]])
                    self.organs = organs[0]
                else:
                    self.contours = np.array(self.contours)
                    self.organs = organs[-1]
            except IndexError:
    #            info = "Check structure names" #for Lucas
    #            MyException(info)
                raise IndexError
            
        if list(self.contours[0]) == ['one slice']: #stop the calculation if it's only one slice
            self.Xcontour = 'one slice'
            self.Ycontour = 'one slice'
            self.Xcontour_W = 'one slice'
            self.Ycontour_W = 'one slice'
        else: #continue, recalculate contour points from mm to pixels
            print self.x_ct, self.xCTspace
            for i in arange(0, len(self.contours)): #controus
                for j in arange(0, len(self.contours[i])): #slice
                    for n in arange(0, len(self.contours[i][j])): #number of contours per slice
                        if list(self.contours[i][j][n][0])!=[]:
                            self.contours[i][j][n][0] = np.array(abs(self.contours[i][j][n][0]-self.x_ct)/(self.xCTspace))
                            self.contours[i][j][n][1] = np.array(abs(self.contours[i][j][n][1]-self.y_ct)/(self.xCTspace))
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

            try:
                x_min = np.min(x_c_min)
                x_max = np.max(x_c_max)+1
                y_min = np.min(y_c_min)
                y_max = np.max(y_c_max)+1
                
                self.logger.info("xmin, xmax, ymin, ymax " + ", ".join(map(str, (x_min, x_max, y_min, y_max))))

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
                
                if local:
                    Xcontour_Rec=[]
                    Ycontour_Rec=[]
                    X, Y, cnt = self.getPoints(self.contours[-1], x_min, x_max, y_min, y_max)
                    Xcontour_Rec.append(X)
                    Ycontour_Rec.append(Y)
                    self.Xcontour_Rec=Xcontour_Rec[0]
                    self.Ycontour_Rec=Ycontour_Rec[0]
                
                del self.contours

                #finding the points for transformed images
                if self.wv:
                    self.contours=[]

                    #slices position in the transformed image
                    self.slices_w = list(np.array(self.slices).copy())
                    #boundary conditions
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
                        self.slices_w[i] = round(self.slices_w[i],3)

                    #get the points in the contour, as previously
                    self.logger.info("Calculate Wavelets")
                    #same as above for a original image ROI
                    for i in arange(0, len(organs)):
                        for j in arange(0, len(list_organs)):
                            if list_organs[j][0] == organs[i]:
                                for k in arange(0, len(self.rs.ROIContourSequence)):
                                    if self.rs.ROIContourSequence[k].ReferencedROINumber == list_organs[j][1]:
                                        try:
                                            lista = []
                                            for l in arange(0, len(self.rs.ROIContourSequence[k].ContourSequence)):
                                                lista.append([round(float(self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[2]),3), self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[::3], self.rs.ROIContourSequence[k].ContourSequence[l].ContourData[1::3]])
                                            lista.sort()
                                            index = []
                                            lista = self.multiContour(lista) #subcontrous in the slice
                                            for m in arange(0, len(lista)):
                                                index.append(round(lista[m][0],3))
                                            slice_count = True
                                            try:
                                                diffI = round(index[1]-index[0], 3) #double check if the orientation is ok
                                            except IndexError:
                                                info = 'only one slice'
                                                slice_count = False
                                            if slice_count:
                                                diffS = round(self.slices_w[1]-self.slices_w[0],3)
                                                if np.sign(diffI) != np.sign(diffS):
                                                    index.reverse()
                                                    lista.reverse()
                                                #empty slices
                                                diff = abs(np.array(index[1:])-np.array(index[:-1]))/diffS
                                                dk = 0
                                                for d in arange(0, len(diff)):
                                                    for di in arange(1, abs(int(diff[d]))):
                                                        index.insert(d+dk+1, index[d+dk]+diffS)
                                                        lista.insert(d+dk+1, [[],[[],[]]])
                                                        dk+=1
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
                                                lista = lista[::2] #adjust resoltuion drops down by 2
                                                for n in arange(0, len(lista)):
                                                    lista[n] = lista[n][1:]
                                                self.contours.append(lista)
                                            else:
                                                ind = np.where(np.array(self.slices_w) == index[0])[0][0]
                                                if ind!=0:
                                                    for m in arange(0, abs(ind-0)):
                                                        lista.insert(0,[[],[[],[]]])
                                                if ind!=(len(self.slices)-1):
                                                    for m in arange(0, abs(ind-(len(self.slices_w)-1))):
                                                        lista.append([[],[[],[]]])
                                                lista = lista[::2] #adjust resoltuion drops down by 2
                                                for n in arange(0, len(lista)):
                                                    lista[n] = lista[n][1:]
                                                self.contours.append(lista)
                                            break
                                        except AttributeError:
                                            print 'no contours for: '+ organs[i]
                    self.slices_w = self.slices_w[::2] #adjust resoltuion drops down by 2
                    #recalculating for pixels
                    self.cnt=[]
                    x_ct = self.x_ct - 2*self.xCTspace #adjust resoltuion drops down by 2
                    y_ct = self.y_ct - 2*self.xCTspace
                    for i in arange(0, len(self.contours)): #controus
                        for j in arange(0, len(self.contours[i])): #slice
                            for n in arange(0, len(self.contours[i][j])): #number of contours per slice
                                if list(self.contours[i][j][n][0])!=[]:
                                    self.contours[i][j][n][0]=np.array(abs(self.contours[i][j][n][0]-x_ct)/(2*self.xCTspace)) 
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
                    self.logger.info("Wavelet xmin, xmax, ymin, ymax " + ", ".join(map(str, (x_min, x_max, y_min, y_max))))
                    
                    #get all point inside the contour
                    Xcontour=[]
                    Ycontour=[]
                    X, Y, cnt = self.getPoints_w(self.contours[0], x_min, x_max, y_min, y_max)
                    Xcontour.append(X)
                    Ycontour.append(Y)
                    self.Xcontour_W=Xcontour[0]
                    self.Ycontour_W=Ycontour[0]
        
            except ValueError: #ValueError
                self.Xcontour_W=''
                self.Ycontour_W=''
                self.logger.warn( 'too small structure')
                pass
               # raise ValueError
                
    def multiContour(self, lista):
        '''accaount for mutlicontrous in one slice,
        checks z positions in each sublist of the list and if the have the same z then creats a new sublist
        for example input l = [[z1, [x1, x2],[y1,y2]], [z1, [x3, x4, x5],[y3, y4, y5]], [z2, [x1, x2],[y1,y2]]] - 3 contorus on 2 slices
        output l = [[z1, [[x1, x2],[y1,y2]], [[x3, x4, x5],[y3, y4, y5]]], [z2, [[x1, x2],[y1,y2]]]]'''
        
        listap=[]
        lista_nr=[]
        for i in lista:
            listap.append(i[0])
            if i[0] not in lista_nr:
                lista_nr.append(i[0])
        counts = []
        for i in lista_nr:
            counts.append(listap.count(i)) #how many times a ceratin z position occurs on the list

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
        '''get points inside the contour
        segment - contour points'''
        cnt_all = []
        #print 'slices in image: ', self.len_IM
        #print 'slices in structure: ', len(segment)
        for k in arange(0, self.len_IM):
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
        for k in arange(0, self.len_IM):
            if cnt_all[k]!=[[]]:
                M = []
                for n in arange(0, len(cnt_all[k])):
                    m = np.zeros((ymax+1-ymin, xmax+1-xmin))
                    for i in arange(ymin, ymax+1):
                        for j in np.arange(xmin, xmax+1): #check if the point in inside the polygon definded by contour points, 0 - on contour, 1 - inside, -1 -outside
                            m[i-ymin][j-xmin] = cv2.pointPolygonTest(np.array(cnt_all[k][n]), (j,i),False)
                    M.append(m)
                for n in arange(1, len(M)): #to account for multiple subcontours ina slice, includng holes in a contour
                    M[0] = M[0]*M[n]
                M[0] = M[0]*(-1)**(len(M)+1)
                ind = np.where(M[0]>=0)
                xp = ind[1]+xmin
                yp = ind[0]+ymin
                Xp.append([xp])
                Yp.append([yp])
            else:
                Xp.append([])
                Yp.append([])
        return Xp, Yp, cnt_all

    def getPoints_w(self, segment, xmin, xmax, ymin, ymax):
        '''get points inside the contour for wavelet transform
        same as getPoints but for wavelet resolution'''
        cnt_all = []
        for k in arange(0, int(floor((self.len_IM+5)/2.))):
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
        for k in arange(0, int(floor((self.len_IM+5)/2.))):
            if cnt_all[k]!=[[]]:
                M=[]
                for n in arange(0, len(cnt_all[k])):
                    m = np.zeros((ymax+1-ymin, xmax+1-xmin))
                    for i in arange(ymin, ymax+1):
                        for j in np.arange(xmin, xmax+1):
                            m[i-ymin][j-xmin] = cv2.pointPolygonTest(np.array(cnt_all[k][n]), (j,i),False)
                    M.append(m)
                for n in arange(1, len(M)):
                    M[0] = M[0]*M[n]
                M[0] = M[0]*(-1)**(len(M)+1)
                ind = np.where(M[0]>=0)
                xp = ind[1]+xmin
                yp = ind[0]+ymin
                Xp.append([xp])
                Yp.append([yp])
            else:
                Xp.append([])
                Yp.append([])
        return Xp, Yp, cnt_all
