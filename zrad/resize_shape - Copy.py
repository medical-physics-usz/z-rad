import os
import dicom as dc
from dicom.filereader import InvalidDicomError
import numpy as np
from numpy import arange
from os import path, makedirs, listdir
from scipy.ndimage.morphology import distance_transform_edt
from scipy.interpolate import interp1d
from os.path import isfile, join
import pylab as py
import dicom as dc
import cv2


class ResizeShape(object):
    '''Class to resize images and listed structures to a resolution defined by user and saved the results as dicom file
    inp_resolution – resolution defined by user for texture calcualtion
    inp_struct – string containing the structure name
    inp_mypath_load – path with the data to be resized
    inp_mypath_save – path to save resized data
    image_type – image modality
    begin – start number
    stop – stop number
    '''      
    
    def __init__(self, inp_struct, inp_mypath_load, inp_mypath_save, image_type, low, high, inp_resolution):
        inp_resolution = float(inp_resolution)
        if inp_resolution < 1.: #set a round factor for slice position and resolution for shape calculation 1mm if texture resolution > 1mm and 0.1 if texture resolution < 1mm
            self.round_factor = 3
            self.resolution = 0.1 #finer resolution for the shape calcaultion if texture is below 1 mm 
        else:
            self.round_factor = 2
            self.resolution = 1.0            
       
        #save a text file which specifies which resolution was used for shape resizing
        try:
            makedirs(inp_mypath_save)
        except OSError:
            if not path.isdir(inp_mypath_save):
                raise           
        f = open(inp_mypath_save+'shape_resolution.txt','w')
        f.write(str(self.resolution))
        f.close()
            
        self.list_structure = [inp_struct] #sturucture to be resized, placed in a list due to similarities with texture resizing

        self.mypath_load = inp_mypath_load+"\\"
        self.mypath_s = inp_mypath_save+"\\"
        self.image_type = image_type
        self.listDicomProblem = [] #cannot open as dicom

        self.lista_dir =  [str(i) for i in np.arange(low, high+1)] #list of directories to be analyzed

        self.resize()

    def resize(self):
        '''resize structure to text file'''
        
        if self.image_type == 'CT':
            UID_t = ['1.2.840.10008.5.1.4.1.1.2', '1.2.840.10008.5.1.4.1.1.2.1'] #CT and contarst-enhanced CT
        elif self.image_type == 'PET':
            UID_t = ['1.2.840.10008.5.1.4.1.1.20','Positron Emission Tomography Image Storage']
        elif self.image_type == 'MR':
            UID_t = ['1.2.840.10008.5.1.4.1.1.4']

        for name in self.lista_dir: #iterate through the patients
            try:
                print 'patient ', name
                mypath_file =self.mypath_load +name + '\\' #go to subfolder for given patient
                mypath_save = self.mypath_s 

                onlyfiles = []
                for f in listdir(mypath_file):
                    try:
                        if dc.read_file(mypath_file+f).SOPClassUID in UID_t and isfile(join(mypath_file,f)): #read only dicoms of certain modality
                            onlyfiles.append((round(float(dc.read_file(mypath_file+'\\'+f).ImagePositionPatient[2]),2), f)) #sort files by slice position
                    except InvalidDicomError: #not a dicom file   
                        self.listDicomProblem.append(f)
                        pass
                        
                onlyfiles.sort() #sort and take only file names
                slices = []
                for i in arange(0, len(onlyfiles)):
                    slices.append(onlyfiles[i][0])
                    onlyfiles[i] = onlyfiles[i][1]

                CT = dc.read_file(mypath_file+onlyfiles[0]) #example image
                xCTspace=float(CT.PixelSpacing[0]) #XY resolution
                xct = float(CT.ImagePositionPatient[0]) #x position of top left corner
                yct = float(CT.ImagePositionPatient[1]) #y position of top left corner

                #define z interpolation grid
                sliceThick = round(abs(slices[0]-slices[1]),self.round_factor)
                #check slice sorting,for the interpolation funtcion one need increaing slice position
                if slices[1]-slices[0] < 0:
                    new_gridZ = np.arange(slices[-1], slices[0]+sliceThick, self.resolution)
                    old_gridZ = np.arange(slices[-1], slices[0]+sliceThick, sliceThick)
                else:
                    new_gridZ = np.arange(slices[0], slices[-1]+sliceThick, self.resolution)
                    old_gridZ = np.arange(slices[0], slices[-1]+sliceThick, sliceThick)
                
                #check the length in case of rounding problems
                if len(old_gridZ) != len(slices):
                    if slices[1]-slices[0] < 0:
                        slices_r = np.array(slices).copy()
                        slices_r = list(slices_r)
                        slices_r.reverse()
                        old_gridZ = np.array(slices_r)
                    else:
                        old_gridZ = np.array(slices)

                del CT
                del onlyfiles

                RS_UID = ['1.2.840.10008.5.1.4.1.1.481.3', 'RT Structure Set Storage'] #structure set
                rs = []
                for f in listdir(mypath_file):
                    try:
                        if dc.read_file(mypath_file+f).SOPClassUID in RS_UID and isfile(join(mypath_file,f)): #read only dicoms of certain modality
                            rs.append(f)
                    except InvalidDicomError: #not a dicom file   
                        pass

                resize_rs = True #if there is no RS or too many RS files change folder name
                if len(rs)!=1:
                    resize_rs = False
                else:                          
                    rs_name = mypath_file+rs[0]
                    
                if resize_rs: #only if only one RS found in the directory
                    rs = dc.read_file(rs_name) #read rs

                    list_organs = [] #ROI (name, number)
                    list_organs_names = [] #ROI names
                    for j in arange(0, len(rs.StructureSetROISequence)):
                        list_organs.append([rs.StructureSetROISequence[j].ROIName, rs.StructureSetROISequence[j].ROINumber])
                        list_organs_names.append(rs.StructureSetROISequence[j].ROIName)

                    change_struct = [] #check if structure avaiable in RS and add to list (the list if only due to similarity with texture), 
                    for j in self.list_structure:
                        if j in list_organs_names:
                            change_struct.append(j)
                            try:
                                makedirs(mypath_save+'\\'+j+'\\'+name+'\\')
                            except OSError:
                                if not path.isdir(mypath_save+'\\'+j+'\\'+name+'\\'):
                                    raise

                    for s in arange(0, len(change_struct)):
                        print 'structure: ', s
                        #read a contour points for given structure
                        #M - 3D matrix filled with 1 insdie contour and 0 outside
                        #xmin - minimum value of x in the contour
                        #ymin - minimum value of y in the contour
                        #st_nr - number of the ROI of the defined name
                        M, xmin, ymin, st_nr= self.structures(rs_name, change_struct[s], slices, xct, yct, xCTspace, len(slices))
                        
                        insertedZ=[] #list of contour slices alread inserted for the given ROI

                        # roudning new patient position to the defined precision                        
                        for gz in range(0, len(new_gridZ)):
                            new_gridZ[gz] = round(new_gridZ[gz],self.round_factor)
                        for n_s in arange(0, len(M)-1): # n_s slice number
                            if M[n_s] != [] and M[n_s+1] != []: #if two consecutive slices not empty - interpolate
                                if self.round_factor == 2:
                                    zi = np.linspace(old_gridZ[n_s], old_gridZ[n_s+1], sliceThick/0.01+1) #create an interpolation grid between those slicse
                                    #round interpolation grid accroding to specified precision 
                                    for gz in arange(0, len(zi)):
                                        zi[gz] = round(zi[gz],self.round_factor)
                                    #interpolate, X list of x positions of the interpolated contour, Y list of y positions of the interpoated contour 
                                    X, Y = self.interpolate(M[n_s], M[n_s+1], np.linspace(0,1,sliceThick/0.01+1))
                                elif self.round_factor == 3 :
                                    zi = np.linspace(old_gridZ[n_s], old_gridZ[n_s+1], sliceThick/0.001+1) #create an interpolation grid between those slicse
                                    #round interpolation grid accroding to specified precision
                                    for gz in arange(0, len(zi)):
                                        zi[gz] = round(zi[gz],self.round_factor)
                                    #interpolate, X list of x positions of the interpolated contour, Y list of y positions of the interpoated contour 
                                    X, Y = self.interpolate(M[n_s], M[n_s+1], np.linspace(0,1,sliceThick/0.001+1))
                                #check which position in the interpolation grid correcpods to the new slice position
                                for i in arange(0, len(zi)):
                                    if zi[i] in new_gridZ and zi[i] not in insertedZ: #insertedZ gathers all slice positions which are alreay filled in case that slice position is on the ovelap of two slices from orignal
                                        ind = str(np.where(new_gridZ == zi[i])[0][0]) #slice position to save correct file name, also importat for LN dist
                                        file_n = open(mypath_save+'\\'+change_struct[s]+'\\'+name+'\\'+'slice_'+ind, 'w')
                                        insertedZ.append(zi[i])
                                        for j in arange(0, len(X[i])): #save positions of all points inside structure in to columns for X and Y
                                            file_n.write(str(X[i][j]+xmin)+'\t' +str(Y[i][j]+ymin))
                                            file_n.write('\n')
                                        file_n.close()
 
            except WindowsError: #no path with data for the patient X
                pass
            except KeyError:
                self.wrongROI.append(name) #problem with image
                pass
        
        if len(self.wrongROI)!=0:
            config = open(self.mypath_s+'\\'+self.list_structure[0]+'\\'+'key_error.txt', 'w')
            for i in self.wrongROI:
                config.write(i+'\n')
            config.close()
            
        if len(self.listDicomProblem)!=0:
            config = open(self.mypath_s+'\\'+self.list_structure[0]+'\\'+'dicom_file_error.txt', 'w')
            for i in self.listDicomProblem:
                config.write(i+'\n')
            config.close()
            

    def structures(self, rs, structure, slices, x_ct, y_ct, xCTspace, l_IM):
        print rs
        rs = dc.read_file(rs) #read RS file
        list_organs =[] #list of organs defined in the RS file
        print 'structures in RS file:'
        for j in arange(0, len(rs.StructureSetROISequence)):
            list_organs.append([rs.StructureSetROISequence[j].ROIName, rs.StructureSetROISequence[j].ROINumber])
            print rs.StructureSetROISequence[j].ROIName

        organs = [structure] #define the structure you're intersed in

        contours=[] #list with structure contours

        #search for organ I'm intersted in
        for i in arange(0, len(organs)): #organs definde by user
            for j in arange(0, len(list_organs)): #organ in RS
                if list_organs[j][0] == organs[i]: #if the same name
                    for k in arange(0, len(rs.ROIContourSequence)): #searach by ROI number
                        if rs.ROIContourSequence[k].ReferencedROINumber == list_organs[j][1]:  # double check the ROI number
                            st_nr = k #ROI number
                            try:
                                lista = [] #z position of the slice
                                for l in arange(0, len(rs.ROIContourSequence[k].ContourSequence)):
                                    lista.append([round(float(rs.ROIContourSequence[k].ContourSequence[l].ContourData[2]),self.round_factor), rs.ROIContourSequence[k].ContourSequence[l].ContourData[::3], rs.ROIContourSequence[k].ContourSequence[l].ContourData[1::3]])
                                lista.sort()
                                index = []
                                lista = self.multiContour(lista) #subcontrous in the slice
                                for m in arange(0, len(lista)):
                                    index.append(lista[m][0])
                                print index
                                print slices
                                diffI = round(index[1]-index[0], 1) #double check if the orientation is ok
                                diffS = round(slices[1]-slices[0],1)
                                print diffI, diffS
                                if np.sign(diffI) != np.sign(diffS):
                                    index.reverse()
                                    lista.reverse()
                                #empty slices
                                diff = abs(np.array(index[1:])-np.array(index[:-1]))/diffS
                                print diff
                                dk = 0
                                for d in arange(0, len(diff)):
                                    for di in arange(1, abs(round(diff[d],0))):
                                        index.insert(d+dk+1, index[d+dk]+diffS)
                                        lista.insert(d+dk+1, [[],[[],[]]])
                                        dk+=1
                                print dk
                                print ''
                                print index
            
                                #include empty list to slices where structure was not contour
                                sliceB=index[-1]
                                sliceE=index[0]
                                indB = np.where(np.array(slices) == sliceB)[0][0]
                                indE = np.where(np.array(slices) == sliceE)[0][0]
                                print indE
                                print indB
                                if indE!=0:
                                    for m in arange(0, abs(indE-0)):
                                        lista.insert(0,[[],[[],[]]])
                                if indB!=(len(slices)-1):
                                    for m in arange(0, abs(indB-(len(slices)-1))):
                                        lista.append([[],[[],[]]])
                                for n in arange(0, len(lista)):
                                    lista[n] = lista[n][1:]
                                contours.append(lista)
                                break
                            except AttributeError:
                                print 'no contours for: '+ organs[i]
        print len(slices)
        print len(contours[0])

         #recalculating for pixels the points into pixels
        cnt=[]
        contours = np.array(contours)

        for i in arange(0, len(contours)): #controus
            for j in arange(0, len(contours[i])): #slice
                for n in arange(0, len(contours[i][j])): #number of contours per slice
                    if contours[i][j][n][0]!=[]:
                        contours[i][j][n][0]=np.array(abs(contours[i][j][n][0]-x_ct)/(xCTspace))
                        contours[i][j][n][1]=np.array(abs(contours[i][j][n][1]-y_ct)/(xCTspace))
                        for k in arange(0, len(contours[i][j][n][0])):
                            contours[i][j][n][0][k] = int(round(contours[i][j][n][0][k],0))
                            contours[i][j][n][1][k] = int(round(contours[i][j][n][1][k],0))
                        contours[i][j][n][0] = np.array(contours[i][j][n][0], dtype=np.int)
                        contours[i][j][n][1] = np.array(contours[i][j][n][1], dtype=np.int)


        x_c_min = [] #x position of contour points to define the region of interest where we look for the structure
        x_c_max = []
        y_c_min = []
        y_c_max = []
        for i in arange(0, len(contours)): #controus
            for j in arange(0, len(contours[i])): #slice
                for n in arange(0, len(contours[i][j])): #number of contours per slice
                    if contours[i][j][n][0]!=[]:
                        x_c_min.append(np.min(contours[i][j][n][0]))
                        x_c_max.append(np.max(contours[i][j][n][0]))
                        y_c_min.append(np.min(contours[i][j][n][1]))
                        y_c_max.append(np.max(contours[i][j][n][1]))

        x_min = np.min(x_c_min)
        x_max = np.max(x_c_max)+1
        y_min = np.min(y_c_min)
        y_max = np.max(y_c_max)+1

        #finding points inside the contour
        M = self.getPoints(contours[0], x_min, x_max, y_min, y_max, l_IM) #x_min and y_min global for whole contour
        return M, x_min, y_min, st_nr

    def multiContour(self, lista):
        '''accaount for mutlicontrous in one slice'''
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

    def getPoints(self, segment, xmin, xmax, ymin, ymax, l_IM):
        '''get points inside the contour'''
        cnt_all = []
        for k in arange(0, l_IM):
            cnt = []
            for i in arange(0, len(segment[k])):
                c = []
                for j in arange(0, len(segment[k][i][0])):
                    c.append([segment[k][i][0][j], segment[k][i][1][j]])
                cnt.append(c)
            if cnt==[]:
                cnt=[[],[]]
            cnt_all.append(cnt)

        M = []
        for k in arange(0, l_IM):
            if cnt_all[k]!=[[]]:
                m = np.ones((ymax+1-ymin, xmax+1-xmin))*(-1)
                for n in arange(0, len(cnt_all[k])): #subcontours
                    for i in arange(ymin, ymax+1):
                        for j in np.arange(xmin, xmax+1):
                            if cv2.pointPolygonTest(np.array(cnt_all[k][n]), (j,i),False) !=-1:
                                m[i-ymin][j-xmin]=cv2.pointPolygonTest(np.array(cnt_all[k][n]), (j,i),False)
                M.append(m)
            else:
                M.append([])
        M = np.array(M)
        return M

    def interpolate(self, s1, s2, znew):
        s1 = np.array(s1)
        s2 = np.array(s2)
        s1a = s1.copy()
        s2a = s2.copy()
        s1a[np.where(s1==-1)] = 1
        s2a[np.where(s2==-1)] = 1
        im1 = distance_transform_edt(s1a, return_distances=True)
        im2 = distance_transform_edt(s2a, return_distances=True)
        del s1a
        del s2a


        out1 = np.ones((len(s1), len(s1[0])), dtype = np.int)
        out2 = np.ones((len(s2), len(s2[0])), dtype = np.int)
        out1[np.where(s1!=-1)]=0 #outside
        out2[np.where(s2!=-1)]=0
        del s1
        del s2

        im1 = -im1*out1 + im1*(~out1+2)
        im2 = -im2*out2 + im2*(~out2+2)

        con = np.array([im1, im2])
        '''py.figure()
        py.subplot(211)
        py.imshow(im1)
        py.subplot(212)
        py.imshow(im2)
        py.show()'''

        z = [0,1]
        con_m = np.zeros((len(znew), len(im1), len(im1[0])))
        for i in arange(0, len(im1)):
            for j in arange(0, len(im1[0])):
                f = interp1d(z, con[:,i, j])
                con_m[:,i,j]=f(znew)
    #    for i in arange(0, len(con_m)):
    #        py.subplot(3, len(con_m)/3 +1, i+1)
    #    py.imshow(con_m[0])
    #    py.colorbar()
    #    py.show()
        del con
        del im1
        del im2

        Xfin = []
        Yfin = []
        for n in arange(0, len(con_m)):
            indx = np.where(con_m[n]>=0)[0]
            indy = np.where(con_m[n]>=0)[1]

            Xfin.append(indx)
            Yfin.append(indy)
        del con_m
        return Xfin, Yfin









        
    
