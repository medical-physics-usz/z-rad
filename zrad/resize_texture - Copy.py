# -*- coding: utf-8 -*-s
import os
import dicom as dc
from dicom.filereader import InvalidDicomError
import numpy as np
from numpy import arange
from os import path, makedirs, listdir, rmdir
from os.path import isfile, join
from scipy.ndimage.morphology import distance_transform_edt
from scipy.interpolate import interp1d
import cv2

from resize_interpolate_roi import InterpolateROI


class ResizeTexture(object):
    '''Class to resize images and listed structures to a resolution defined by user and saved the results as dicom file
    inp_resolution – resolution defined by user
    inp_struct – string containing the structures to be resized separated by ‘,’
    inp_mypath_load – path with the data to be resized
    inp_mypath_save – path to save resized data
    image_type – image modality
    begin – start number
    stop – stop number
    '''  
    
    def __init__(self, inp_resolution,inp_struct,inp_mypath_load,inp_mypath_save,image_type, begin, stop):
        self.resolution = float(inp_resolution)
        if self.resolution < 1.: #set a round factor for slice position 
            self.round_factor = 3
        else:
            self.round_factor = 2        
        
        #divide s atring with structures names to a list of names
        if ',' not in inp_struct: 
            self.list_structure = [inp_struct]
        else:
            self.list_structure =inp_struct.split(',')
            
        for i in range(0,len(self.list_structure)):
            if self.list_structure[i][0]==' ':
                self.list_structure[i] = self.list_structure[i][1:]
            if self.list_structure[i][-1]==' ':
                self.list_structure[i] = self.list_structure[i][:-1]

        self.mypath_load = inp_mypath_load+"\\" 
        self.mypath_s = inp_mypath_save+"\\"
        self.UID = '2030' #UID for new images
        self.image_type = image_type
        
        self.list_dir = [str(f) for f in arange(begin, stop+1)] #list of directories to be analyzed

        self.resize()
        
    def resize(self):
        '''resize image and structure'''

        wrongROI = []
        lista_voi = [] #problems with VOI
        self.listDicomProblem = [] #cannot open as dicom
        error = []
        for name in self.list_dir:
            try:
                print 'patient ', name
                mypath_file = self.mypath_load +name + '\\' #go to subfolder for given patient
                mypath_save = self.mypath_s + name+'\\' #create subfolder for given patient
                try:
                    makedirs(mypath_save)
                except OSError:
                    if not path.isdir(mypath_save):
                        raise

                if self.image_type == 'CT':
                    UID_t = ['1.2.840.10008.5.1.4.1.1.2', '1.2.840.10008.5.1.4.1.1.2.1'] #CT and contarst-enhanced CT
                elif self.image_type == 'PET':
                    UID_t = ['1.2.840.10008.5.1.4.1.1.20','Positron Emission Tomography Image Storage']
                elif self.image_type == 'MR':
                    UID_t = ['1.2.840.10008.5.1.4.1.1.4']

                
                #resize standard modalities: CT, PET and MR
                if self.image_type not in ['CTP', 'IVIM']:
                    onlyfiles = []
                    for f in listdir(mypath_file):
                        try:
                            if dc.read_file(mypath_file+f).SOPClassUID in UID_t and isfile(join(mypath_file,f)): #read only dicoms of certain modality
                                onlyfiles.append((round(float(dc.read_file(mypath_file+'\\'+f).ImagePositionPatient[2]),2), f)) #sort files by slice position
                        except InvalidDicomError: #not a dicom file   
                            self.listDicomProblem.append(f)
                            pass
                        
                    onlyfiles.sort() #sort and take only file names
                    for i in arange(0, len(onlyfiles)):
                        onlyfiles[i] = onlyfiles[i][1]    
                        
                    #data needed to decode images
                    CT = dc.read_file(mypath_file+onlyfiles[0]) #example image
                    position = CT.PatientPosition # HFS or FFS
                    xCTspace=float(CT.PixelSpacing[0]) #XY resolution
                    xct = float(CT.ImagePositionPatient[0]) #x position of top left corner
                    yct = float(CT.ImagePositionPatient[1]) #y position of top left corner
                    columns = CT.Columns # number of columns
                    rows = CT.Rows #number of rows
                    new_gridX = np.arange(round(xct, 10), round(xct+xCTspace*columns, 10), self.resolution) #new grid of X for interpolation
                    old_gridX = np.arange(round(xct, 10), round(xct+xCTspace*columns, 10), xCTspace) #original grid of X 
                    new_gridY = np.arange(round(yct, 10), round(yct+xCTspace*rows, 10), self.resolution) #new grid of Y for interpolation
                    old_gridY = np.arange(round(yct, 10), round(yct+xCTspace*rows, 10), xCTspace) #original grid of Y
                    
                    if len(old_gridX) > columns: # due to rounding
                        old_gridX = old_gridX[:-1]
                        old_gridY = old_gridY[:-1]
                        
                    new_rows = len(new_gridY) # number of new rows
                    new_columns = len(new_gridX) # number of new columns
                    
                    IM = [] #list of images
                    slices = [] # list of slices
                    for k in onlyfiles:
                        CT = dc.read_file(mypath_file+k)
                        slices.append(round(float(CT.ImagePositionPatient[2]), self.round_factor))
                        
                        #read image data   
                        data = CT.PixelData #imaging data                        
                        if self.image_type == 'PET':
                            data16 = np.array(np.fromstring(data, dtype=np.int16)) #converitng to decimal
                            data16 = data16*float(CT.RescaleSlope)+float(CT.RescaleIntercept)
                        else:
                            data16 = np.array(np.fromstring(data, dtype=np.int16)) #converitng to decimal, for CT no intercept as it is the same per image
                        #recalculating for rows x columns
                        a=[]
                        for j in np.arange(0, rows):
                            a.append(data16[j*columns:(j+1)*columns])
                        del data
                        del data16
                        
                        #interpolate XY 
                        a=np.array(a)
                        b_new = np.zeros((len(old_gridY), len(new_gridX)))
                        for j in np.arange(0, len(a)):
                            b_new[j] = np.interp(new_gridX, old_gridX, a[j])
                        a_new = np.zeros((len(new_gridY), len(new_gridX)))
                        for j in np.arange(0, len(b_new[0])):
                            a_new[:,j] = np.interp(new_gridY, old_gridY, b_new[:,j])
                        del b_new
                        del a
                        IM.append(a_new)
                        del a_new

                    #define z interpolation grid
                    IM = np.array(IM)
                    sliceThick = round(abs(slices[0]-slices[1]),self.round_factor)
                    #check slice sorting,for the interpolation funtcion one need increaing slice position
                    if slices[1]-slices[0] < 0:
                        new_gridZ = np.arange(slices[-1], slices[0]+sliceThick, self.resolution)
                        old_gridZ = np.arange(slices[-1], slices[0]+sliceThick, sliceThick)
                        Image = IM.copy()
                        for j in arange(0, len(IM)):
                            IM[j] = Image[-j-1]
                        del Image
                    else:
                        new_gridZ = np.arange(slices[0], slices[-1]+sliceThick, self.resolution)
                        old_gridZ = np.arange(slices[0], slices[-1]+sliceThick, sliceThick)
                    print 'new grid Z '
                    print new_gridZ
                    print 'old grid Z '
                    print old_gridZ

                    new_image = np.zeros((len(new_gridZ), new_rows, new_columns)) #matrix with zeros for the new image
                    #interpolate in z direction
                    try:
                        for x in arange(0, new_columns):
                            for y in arange(0, new_rows):
                                new_image[:,y,x] = np.interp(new_gridZ, old_gridZ, IM[:,y,x])
                    except ValueError:
                        if slices[1]-slices[0] < 0:
                            slices_r = np.array(slices).copy()
                            slices_r = list(slices_r)
                            slices_r.reverse()
                            old_gridZ = np.array(slices_r)
                        else:
                            print 'tu'
                            old_gridZ = np.array(slices)
                        for x in arange(0, new_columns):
                            for y in arange(0, new_rows):
                                new_image[:,y,x] = np.interp(new_gridZ, old_gridZ, IM[:,y,x])

                    #save interpolated images
                    for im in arange(0, len(new_image)):
                        im_nr = int(im*float(len(onlyfiles))/len(new_image)) #choose an original dicom file to be modify
                        CT = dc.read_file(mypath_file+onlyfiles[im_nr]) #read file to be modified
                        CT.FrameOfReferenceUID = CT.FrameOfReferenceUID[:-2]+self.UID #change UID so it is treated as new image
                        CT.SeriesInstanceUID = CT.SeriesInstanceUID[:-2]+self.UID
                        CT.SOPInstanceUID = CT.SOPInstanceUID[:-1]+self.UID+str(im)
                        CT.Columns = new_columns #adapt columns
                        CT.Rows = new_rows #adapt rows
                        CT.sliceThick = str(self.resolution) #adapt slice thickness
                        CT.PixelSpacing[0] = str(self.resolution) #adapt XY resolution
                        CT.PixelSpacing[1] = str(self.resolution)
                        #adapt slice location tag if exists
                        try:
                            if position == 'FFS':
                                CT.SliceLocation = str(-new_gridZ[im]) #minus comes from the standard
                            else:
                                CT.SliceLocation = str(new_gridZ[im])
                        except AttributeError:
                            pass
                        CT.ImagePositionPatient[2] = str(new_gridZ[im]) #adapt patient position tag for new z position of the image
                        #for PET images, calculate new slope and intercept
                        if self.image_type == 'PET':
                            vmax = int(np.max(new_image[im])+1)
                            rslope = float(vmax)/(2**15) #16 bits
                            new_image[im] = new_image[im]/rslope
                            CT.RescaleSlope = round(rslope,5)
                            CT.RescaleIntercept = 0
                            array=np.array(new_image[im], dtype = np.int16) 
                            CT.LargestImagePixelValue = np.max(array)
                        else:
                            array=np.array(new_image[im], dtype = np.int16)
                        data = array.tostring() #convert to string
                        CT.PixelData = data #data pixel data to new image
                    
                        CT.save_as(mypath_save+'image'+str(im+1)+'.dcm') #save image
                        
                    del new_image
                    del CT
                    del onlyfiles
                
                #disabled as it does not handle the well the interpolation for NaN, recommended calcaulte perfusion on cubic voxels.
#                elif self.image_type == 'CTP': 
#                    UID_t = ['1.2.840.10008.5.1.4.1.1.2', '1.2.840.10008.5.1.4.1.1.2.1'] #CT and contarst-enhanced CT
#                    onlyfiles =[f for f in listdir(mypath_file) if dc.read_file(mypath_file+f).SOPClassUID in UID_t and isfile(join(mypath_file,f))] #read only PET files in the folder
#                    
#                    slices_tot =[]
#                    for i in arange(0, len(onlyfiles)):
#                        onlyfiles[i]= (float(dc.read_file(mypath_file+onlyfiles[i]).InstanceNumber) ,onlyfiles[i],float(dc.read_file(mypath_file+onlyfiles[i]).ImagePositionPatient[2]))
#                        if round(float(dc.read_file(mypath_file+onlyfiles[i][1]).ImagePositionPatient[2]),self.round_factor) not in slices_tot:
#                            slices_tot.append(round(float(dc.read_file(mypath_file+onlyfiles[i][1]).ImagePositionPatient[2]),self.round_factor))
#                    #sorting the files according to theirs number
#                    #saving to the list onlyfiles_perf just the file names
#                    onlyfiles.sort()
#                    for i in arange(0, len(onlyfiles)):
#                        onlyfiles[i]= onlyfiles[i][1]
#                    print onlyfiles
#                    for s in arange(0, len(onlyfiles)/len(slices_tot)):
#                        IM = []
#                        slices = []
#                        for k in onlyfiles[s*len(slices_tot): (s+1)*len(slices_tot)]:
#                            CT = dc.read_file(mypath_file+k)
#                            slices.append(round(float(CT.ImagePositionPatient[2]), self.round_factor))
#                            position = CT.PatientPosition
#                            xCTspace=float(CT.PixelSpacing[0])
#                            xct = float(CT.ImagePositionPatient[0])
#                            yct = float(CT.ImagePositionPatient[1])
#                            columns = CT.Columns
#                            rows = CT.Rows
#                            new_gridX = np.arange(xct, xct+xCTspace*columns, self.resolution)
#                            old_gridX = np.arange(xct, xct+xCTspace*columns, xCTspace)
#                            new_gridY = np.arange(yct, yct+xCTspace*rows, self.resolution)
#                            old_gridY = np.arange(yct, yct+xCTspace*rows, xCTspace)
#                            new_rows = len(new_gridY)
#                            new_columns = len(new_gridX)
#                            data = CT.PixelData
#                            data16 = np.array(np.fromstring(data, dtype=np.int16))
#                            #recalculating for rows x columns
#                            a=[]
#                            for j in np.arange(0, rows):
#                                a.append(data16[j*columns:(j+1)*columns])
#        ##                    del data
#        ##                    del data16
#                            a=np.array(a)
#                            b_new = np.zeros((len(old_gridY), len(new_gridX)))
#                            for j in np.arange(0, len(a)):
#                                b_new[j] = np.interp(new_gridX, old_gridX, a[j])
#                            a_new = np.zeros((len(new_gridY), len(new_gridX)))
#                            for j in np.arange(0, len(b_new[0])):
#                                a_new[:,j] = np.interp(new_gridY, old_gridY, b_new[:,j])
#                            del b_new
#                            del a
#                            IM.append(a_new)
#                            del a_new
#
#                        #interpolate z positions
#                        IM = np.array(IM)
#                        sliceThick = round(abs(slices[0]-slices[1]),self.round_factor)
#                        print sliceThick
#                        print slices
#                        print slices[1]-slices[0]
#                        if slices[1]-slices[0] < 0:
#                            new_gridZ = np.arange(slices[-1], slices[0]+sliceThick, self.resolution)
#                            old_gridZ = np.arange(slices[-1], slices[0]+sliceThick, sliceThick)
#                        else:
#                            print 'tu'
#                            new_gridZ = np.arange(slices[0], slices[-1]+sliceThick, self.resolution)
#                            old_gridZ = np.arange(slices[0], slices[-1]+sliceThick, sliceThick)
#                            instance_nr = np.arange(s*len(new_gridZ)+1,(s+1)*len(new_gridZ)+1)
#                            instance_nr = list(instance_nr)
#                        print new_gridZ
#                        print old_gridZ
#                        new_image = np.zeros((len(new_gridZ), new_rows, new_columns))
#                    ##    py.plot(slices, old_gridZ, 'o')
#                    ##    py.show()
#                        #reverse image        
#                        if slices[1]-slices[0] < 0:
#                            print 'r'
#                            Image = IM.copy()
#                            for j in arange(0, len(IM)):
#                                IM[j] = Image[-j-1]
#                            del Image
#                        
#                        try:
#                            for x in arange(0, new_columns):
#                                for y in arange(0, new_rows):
#                                    new_image[:,y,x] = np.interp(new_gridZ, old_gridZ, IM[:,y,x])
#                        except ValueError:
#                            if slices[1]-slices[0] < 0:
#                                slices_r = np.array(slices).copy()
#                                slices_r = list(slices_r)
#                                slices_r.reverse()
#                                old_gridZ = np.array(slices_r)
#                            else:
#                                print 'tu'
#                                old_gridZ = np.array(slices)
#                            for x in arange(0, new_columns):
#                                for y in arange(0, new_rows):
#                                    new_image[:,y,x] = np.interp(new_gridZ, old_gridZ, IM[:,y,x])
#
#                        instance_nr = np.arange(s*len(new_gridZ)+1,(s+1)*len(new_gridZ)+1)
#
#                        #save image
#                        for im in arange(0, len(new_image)):
#                            im_nr = int(s*len(old_gridZ)+im*float(len(old_gridZ))/len(new_image))
#                            CT = dc.read_file(mypath_file+onlyfiles[im_nr])
#                            CT.InstanceNumber = str(instance_nr[im])
#                            CT.FrameOfReferenceUID = CT.FrameOfReferenceUID[:-2]+self.UID
#                            CT.SeriesInstanceUID = CT.SeriesInstanceUID[:-2]+self.UID
#                            CT.SOPInstanceUID = CT.SOPInstanceUID[:-1]+self.UID+str(im)
#                            CT.Columns = new_columns
#                            CT.Rows = new_rows
#                            CT.sliceThick = str(self.resolution)
#                            CT.PixelSpacing[0] = str(self.resolution)
#                            CT.PixelSpacing[1] = str(self.resolution)
#                            try:
#                                if position == 'FFS':
#                                    CT.SliceLocation = str(-new_gridZ[im])
#                                else:
#                                    CT.SliceLocation = str(new_gridZ[im])
#                            except AttributeError:
#                                pass
#                            CT.ImagePositionPatient[2] = str(new_gridZ[im])
#                            vmax = int(np.max(new_image[im])+1)
#                ##            py.imshow(new_image[im])
#                ##            py.colorbar()
#                ##            py.show()
#                            array=np.array(new_image[im], dtype = np.int16)
#                            data = array.tostring()
#                            CT.PixelData = data
#                        
#                            CT.save_as(mypath_save+'CT.'+str(instance_nr[im])+'.dcm')
#                        del new_image
#                        del CT
#                    del onlyfiles

                elif self.image_type == 'IVIM':
                    maps_list = ['DSlow2', 'DFast2', 'F2']
                    
                    for ivim_map in maps_list:
                        try:
                            makedirs(mypath_save+ivim_map)
                        except OSError:
                            if not path.isdir(mypath_save):
                                raise
                    
                        mypath_ivim = mypath_file+ivim_map+'\\' #IVIM subtype
                        onlyfiles =[f for f in listdir(mypath_ivim) if isfile(join(mypath_ivim,f))] #read all files in

                        for i in arange(0, len(onlyfiles)):
                            onlyfiles[i]= (float(dc.read_file(mypath_ivim+onlyfiles[i]).ImagePositionPatient[2]) ,onlyfiles[i])

                        #sorting the files according to theirs slice position
                        onlyfiles.sort()
                        for i in arange(0, len(onlyfiles)):
                            onlyfiles[i]= onlyfiles[i][1]
                            
                        CT = dc.read_file(mypath_ivim+onlyfiles[0])    #example image
                        position = CT.PatientPosition # HFS or FFS
                        xCTspace=float(CT.PixelSpacing[0]) #XY resolution
                        xct = float(CT.ImagePositionPatient[0]) #x position of top left corner
                        yct = float(CT.ImagePositionPatient[1]) #y position of top left corner
                        columns = CT.Columns # number of columns
                        rows = CT.Rows #number of rows
                        new_gridX = np.arange(xct, xct+xCTspace*columns, self.resolution) #new grid of X for interpolation
                        old_gridX = np.arange(xct, xct+xCTspace*columns, xCTspace) #original grid of X 
                        new_gridY = np.arange(yct, yct+xCTspace*rows, self.resolution) #new grid of Y for interpolation
                        old_gridY = np.arange(yct, yct+xCTspace*rows, xCTspace) #original grid of Y
                        
                        if len(old_gridX) > columns: # due to rounding
                            old_gridX = old_gridX[:-1]
                            old_gridY = old_gridY[:-1]                        
                        
                        new_rows = len(new_gridY) # number of new rows
                        new_columns = len(new_gridX) # number of new columns

                        IM = [] #list of images
                        slices = [] # list of slices
                        for k in onlyfiles: # list of slices
                            CT = dc.read_file(mypath_ivim+k)
                            slices.append(round(float(CT.ImagePositionPatient[2]), self.round_factor))
                            
                            #read image data
                            data = CT.PixelData
                            data16 = np.array(np.fromstring(data, dtype=np.int16)) #converitng to decimal
                            #recalculating for rows x columns
                            a=[]
                            for j in np.arange(0, rows):
                                a.append(data16[j*columns:(j+1)*columns])
                            del data
                            del data16
                            
                            #interpolate XY 
                            a=np.array(a)
                            b_new = np.zeros((len(old_gridY), len(new_gridX)))
                            for j in np.arange(0, len(a)):
                                b_new[j] = np.interp(new_gridX, old_gridX, a[j])
                            a_new = np.zeros((len(new_gridY), len(new_gridX)))
                            for j in np.arange(0, len(b_new[0])):
                                a_new[:,j] = np.interp(new_gridY, old_gridY, b_new[:,j])
                            del b_new
                            del a
                            IM.append(a_new)
                            del a_new

                        #define z interpolation grid
                        IM = np.array(IM)
                        sliceThick = round(abs(slices[0]-slices[1]),self.round_factor)
                        #check slice sorting,for the interpolation funtcion one need increaing slice position
                        if slices[1]-slices[0] < 0:
                            new_gridZ = np.arange(slices[-1], slices[0]+sliceThick, self.resolution)
                            old_gridZ = np.arange(slices[-1], slices[0]+sliceThick, sliceThick)
                            Image = IM.copy()
                            for j in arange(0, len(IM)):
                                IM[j] = Image[-j-1]
                            del Image
                        else:
                            new_gridZ = np.arange(slices[0], slices[-1]+sliceThick, self.resolution)
                            old_gridZ = np.arange(slices[0], slices[-1]+sliceThick, sliceThick)
                        print 'new grid Z '
                        print new_gridZ
                        print 'old grid Z '
                        print old_gridZ
                        
                        new_image = np.zeros((len(new_gridZ), new_rows, new_columns))  #matrix with zeros for the new image
                        #interpolate in z direction
                        try:
                            for x in arange(0, new_columns):
                                for y in arange(0, new_rows):
                                    new_image[:,y,x] = np.interp(new_gridZ, old_gridZ, IM[:,y,x])
                        except ValueError:
                            if slices[1]-slices[0] < 0:
                                slices_r = np.array(slices).copy()
                                slices_r = list(slices_r)
                                slices_r.reverse()
                                old_gridZ = np.array(slices_r)
                            else:
                                print 'tu'
                                old_gridZ = np.array(slices)
                            for x in arange(0, new_columns):
                                for y in arange(0, new_rows):
                                    new_image[:,y,x] = np.interp(new_gridZ, old_gridZ, IM[:,y,x])

                        #save interpolated images
                        for im in arange(0, len(new_image)):
                            im_nr = int(im*float(len(onlyfiles))/len(new_image)) #choose an original dicom file to be modify
                            CT = dc.read_file(mypath_ivim+onlyfiles[im_nr]) #read file to be modified
                            CT.FrameOfReferenceUID = CT.FrameOfReferenceUID[:-2]+self.UID #change UID so it is treated as new image
                            CT.SeriesInstanceUID = CT.SeriesInstanceUID[:-2]+self.UID
                            CT.SOPInstanceUID = CT.SOPInstanceUID[:-1]+self.UID+str(im)
                            CT.Columns = new_columns #adapt columns
                            CT.Rows = new_rows #adapt rows
                            CT.sliceThick = str(self.resolution) #adapt slice thickness
                            CT.PixelSpacing[0] = str(self.resolution) #adapt XY resolution
                            CT.PixelSpacing[1] = str(self.resolution)
                            #adapt slice location tag if exists
                            try:
                                if position == 'FFS':
                                    CT.SliceLocation = str(-new_gridZ[im]) #minus comes from the standard
                                else:
                                    CT.SliceLocation = str(new_gridZ[im])
                            except AttributeError:
                                pass
                            CT.ImagePositionPatient[2] = str(new_gridZ[im]) # adapt patient position tag for new z position of the image
                            array=np.array(new_image[im], dtype = np.int16)
                            data = array.tostring() #convert to string
                            CT.PixelData = data #data pixel data to new image
                        
                            CT.save_as(mypath_save+ivim_map+'\\image'+str(im+1)+'.dcm') #save image
                        del new_image
                        del CT
                        del onlyfiles

                #resize structure
                RS_UID = ['1.2.840.10008.5.1.4.1.1.481.3', 'RT Structure Set Storage'] #structure set
                
                rs=[]
                for f in listdir(mypath_file):
                    try:
                        if dc.read_file(mypath_file+f).SOPClassUID in RS_UID and isfile(join(mypath_file,f)): #read only dicoms of certain modality
                            rs.append(f)
                    except InvalidDicomError: #not a dicom file   
                        pass
                
                resize_rs = True
                if len(rs)!=1:
                    resize_rs = False
                else:                          
                    rs_name = mypath_file+rs[0]
                
                if not resize_rs: #if there is no RS or too many RS files change folder name
                    os.rename(mypath_save[:-1],mypath_save[:-1]+'_noRS')
                else:
                    rs = dc.read_file(rs_name) #read rs

                    list_organs = [] #ROI (name, number)
                    list_organs_names = [] #ROI names
                    for j in arange(0, len(rs.StructureSetROISequence)):
                        list_organs.append([rs.StructureSetROISequence[j].ROIName, rs.StructureSetROISequence[j].ROINumber])
                        list_organs_names.append(rs.StructureSetROISequence[j].ROIName)

                    change_struct = [] #structure to be resized and which are avaible in RS
                    for j in self.list_structure:
                        if j in list_organs_names:
                            change_struct.append(j)
                    
                    structure_nr_to_save = []
                    for s in arange(0, len(change_struct)):
                        print 'structure: ', s
                        #read a contour points for given structure
                        #M - 3D matrix filled with 1 insdie contour and 0 outside
                        #xmin - minimum value of x in the contour
                        #ymin - minimum value of y in the contour
                        #st_nr - number of the ROI of the defined name
                        M, xmin, ymin, st_nr = self.structures(rs_name, change_struct[s], slices, xct, yct, xCTspace, len(slices))
                        
                        contour=[] #list of contour points
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
                                elif self.round_factor == 3:
                                    zi = np.linspace(old_gridZ[n_s], old_gridZ[n_s+1], sliceThick/0.001+1) #create an interpolation grid between those slicse
                                    #round interpolation grid accroding to specified precision  
                                    for gz in arange(0, len(zi)):
                                        zi[gz] = round(zi[gz],self.round_factor)
                                    #interpolate, X list of x positions of the interpolated contour, Y list of y positions of the interpoated contour 
                                    X, Y = self.interpolate(M[n_s], M[n_s+1], np.linspace(0,1,sliceThick/0.001+1))   
                                #check which position in the interpolation grid correcpods to the new slice position
                                for i in arange(0, len(zi)):
                                    if zi[i] in new_gridZ and zi[i] not in insertedZ: #insertedZ gathers all slice positions which are alreay filled in case that slice position is on the ovelap of two slices from orignal
                                        insertedZ.append(zi[i])
                                        for j in arange(0, len(X[i])): #substructres in the slice
                                            l = np.zeros((3*len(X[i][j])))
                                            l[::3] = (X[i][j]+xmin)*xCTspace + xct #convert to the original coordinates in mm
                                            l[1::3]=(Y[i][j]+ymin)*xCTspace + yct #convert to the original coordinates in mm
                                            l[2::3] = round(zi[i],self.round_factor) #convert to the original coordinates in mm
                                            l.round(self.round_factor)
                                            li = [str(round(ci,self.round_factor)) for ci in l] #convert to string
                                            contour.append(li) # add to contour list
                                            del l
                                            del li

                        #search for ROI number I'm intersted in
                        st_nr = 1000000
                        for j in arange(0, len(list_organs)):
                            if list_organs[j][0] == change_struct[s]:
                                for k in arange(0, len(rs.ROIContourSequence)):
                                    if rs.ROIContourSequence[k].ReferencedROINumber == list_organs[j][1]:
                                        st_nr = k
                                        break
                        structure_nr_to_save.append(rs.ROIContourSequence[st_nr].ReferencedROINumber)

                        rs.StructureSetROISequence[st_nr].ROIName = change_struct[s]

                        #modify the rs file, replace old contour sequence with the new one
                        for j in arange(0, len(contour)):
                            try:
                                rs.ROIContourSequence[st_nr].ContourSequence[j].ContourData = contour[j]
                                nr = len(contour[j])/3  #number of points
                                rs.ROIContourSequence[st_nr].ContourSequence[j].NumberOfContourPoints = str(nr)
                            except IndexError: #if the new contour is a longer sequence
                                a = dc.dataset.Dataset()
                                a.add_new((0x3006,0x42), 'CS', 'CLOSED_PLANAR')
                                a.add_new((0x3006,0x46), 'IS', str(len(contour[j])/3))
                                a.add_new((0x3006,0x48), 'IS', str(j)) #sequence element number
                                a.add_new((0x3006,0x50), 'DS', contour[j])
                                rs.ROIContourSequence[0].ContourSequence.append(a)
                        #delete the sequence elements if the original seuquence was onger than interpolated
                        for j in arange(len(contour), len(rs.ROIContourSequence[st_nr].ContourSequence)):
                            del rs.ROIContourSequence[st_nr].ContourSequence[-1]
                            
                        print 'length of new contour: ', len(contour)
                        print 'length of new contour sequence: ', len(rs.ROIContourSequence[st_nr].ContourSequence)
                        print 'the numbers above should be the same'
                    
                    #delete structures which were not resized
                    #modify sepatratelty just to be sure that sorting is correct ROIContourSequence, RTROIObservationsSequence, StructureSetROISequence
                    #ROIContourSequence                    
                    nr_del = []
                    for i in range(0,len(rs.ROIContourSequence)):
                        if rs.ROIContourSequence[i].RefdROINumber not in structure_nr_to_save:
                            nr_del.append(i)
                    nr_del.reverse()
                    for i in nr_del:
                        del rs.ROIContourSequence[i]
                    #RTROIObservationsSequence
                    nr_del = []
                    for i in range(0,len(rs.RTROIObservationsSequence)):
                        if rs.RTROIObservationsSequence[i].RefdROINumber not in structure_nr_to_save:
                            nr_del.append(i)
                    nr_del.reverse()
                    for i in nr_del:
                        del rs.RTROIObservationsSequence[i]
                    #StructureSetROISequence
                    nr_del = []
                    for i in range(0,len(rs.StructureSetROISequence)):
                        if rs.StructureSetROISequence[i].ROINumber not in structure_nr_to_save:
                            nr_del.append(i)
                    nr_del.reverse()
                    for i in nr_del:
                        del rs.StructureSetROISequence[i]
                    
                    
                    rs.save_as(mypath_save+'RS.00001.dcm') #save modified RS file

            except WindowsError: #no path with data for the patient X
                rmdir(mypath_save)
                pass
            except KeyError: #problem with image
                wrongROI.append(name) 
                pass
            except IndexError:
                lista_voi.append(name)
                pass
            

        if len(wrongROI)!=0:      
            config = open(self.mypath_s+'\\'+'key_error.txt', 'w')
            for i in wrongROI:
                config.write(i+'\n')
            config.close()

        if len(lista_voi)!=0:      
            config = open(self.mypath_s+'\\'+'voi_problem.txt', 'w')
            for i in lista_voi:
                config.write(i+'\n')
            config.close()

        if len(error)!=0:      
            config = open(self.mypath_s+'\\'+'unrecognized error.txt', 'w')
            for i in error:
                config.write(i+'\n')
            config.close()

    def structures(self,rs, structure, slices, x_ct, y_ct, xCTspace, l_IM):
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
                        if rs.ROIContourSequence[k].ReferencedROINumber == list_organs[j][1]: # double check the ROI number
                            st_nr = k #ROI number
                            try:
                                lista = [] #z position of the slice
                                #controus in dicom are save as a list with sequence x1, y1, zi, x2, y2, zi, ... xn, yn, zi
                                #where zi is the slice position
                                #if there are subcontours in the slice then these re two different sequences with the same zi
                                for l in arange(0, len(rs.ROIContourSequence[k].ContourSequence)):
                                    lista.append([round(float(rs.ROIContourSequence[k].ContourSequence[l].ContourData[2]),self.round_factor), rs.ROIContourSequence[k].ContourSequence[l].ContourData[::3], rs.ROIContourSequence[k].ContourSequence[l].ContourData[1::3]])
                                lista.sort()
                                index = []
                                lista = self.multiContour(lista) #subcontrous in the slice
                                for m in arange(0, len(lista)):
                                    index.append(lista[m][0])
                                print 'z positions contour'
                                print index
                                print 'z positions image'
                                print slices
                                if len(index) != 1: #if more than one slice
                                    diffI = round(index[1]-index[0], 1) #double check if the orientation is ok
                                    diffS = round(slices[1]-slices[0],1)
                                    print 'resolution image, ROI'
                                    print diffI, diffS
                                    if np.sign(diffI) != np.sign(diffS): #if different orientation then reverse the contour points
                                        index.reverse()
                                        lista.reverse()
                                    #check for slices withut contour in between other contour slices
                                    diff = abs(np.array(index[1:])-np.array(index[:-1]))/diffS
                                    print 'difference in z position between slices normalized to slice spacing'''
                                    print diff
                                    dk = 0
                                    for d in arange(0, len(diff)):
                                        for di in arange(1, int(round(abs(diff[d]),0))): #if no empt slice in between then abs(int(round(diff[d],0))) = 1
                                            index.insert(d+dk+1, index[d+dk]+diffS) #if not add empt slices to index and lista
                                            lista.insert(d+dk+1, [[],[[],[]]])
                                            dk+=1
                                    #include empty list to slices where structure was not contour, so in the end lista and index has the same length as image
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
                                    contours.append(lista) #list of contours for all user definded strctures
                                else: #if only one slice of contour
                                    ind = np.where(np.array(slices) == index[0])[0][0]
                                    print 'contour only in slice'
                                    print ind
                                    if ind!=0:
                                        for m in arange(0, abs(ind-0)):
                                            lista.insert(0,[[],[[],[]]])
                                    if ind!=(len(slices)-1):
                                        for m in arange(0, abs(ind-(len(slices)-1))):
                                            lista.append([[],[[],[]]])
                                    for n in arange(0, len(lista)):
                                        lista[n] = lista[n][1:]
                                    contours.append(lista)
                                break
                            except AttributeError:
                                print 'no contours for: '+ organs[i]

        #recalculating for pixels the points into pixels
        contours = np.array(contours)
        
        #recalculate contour points from mm to pixels
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
        
        print 'xmin, xmax, ymin, ymax'
        print x_min
        print x_max
        print y_min
        print y_max

        #finding points inside the contour, M - 3D matrix with -1 outside, 0 on the border and 1 inside
        M = self.getPoints(contours[0], x_min, x_max, y_min, y_max, l_IM)

        return M, x_min, y_min, st_nr

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
            counts.append(listap.count(i)) #how many timesa ceratin z position occurs on the list

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
        '''get points inside the contour
        segment - contour points'''
        cnt_all = []
        print 'slices in image: ', len(l_IM)
        print 'slices in structure: ', len(segment)
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
                m = np.ones((ymax+1-ymin, xmax+1-xmin))*(-1) #initilize  the 2D matrix with -1, meaning empty slice 
                for n in arange(0, len(cnt_all[k])): #subcontours
                    for i in arange(ymin, ymax+1):
                        for j in np.arange(xmin, xmax+1):
                            if cv2.pointPolygonTest(np.array(cnt_all[k][n]), (j,i),False) !=-1: #not to interfere with other subcontours in this slice
                                m[i-ymin][j-xmin]=cv2.pointPolygonTest(np.array(cnt_all[k][n]), (j,i),False) #check if the point in inside the polygon definded by contour points, 0 - on contour, 1 - inside, -1 -outside
                M.append(m)
            else:
                M.append([])
        M = np.array(M)

        #adjust if there is a contur only in one slice, add slice filled with -1 before and after
        ind = []
        for k in arange(0, len(M)):
            if M[k] != []:
                ind.append(k)

        if len(ind) == 1:
            if ind[0] !=0:
                M[ind[0]-1] = np.ones((ymax-ymin+1,xmax-xmin+1))*-1
            if ind[0]!=len(M):
                M[ind[0]+1] = np.ones((ymax-ymin+1,xmax-xmin+1))*-1
                 
        return M #M - 3D matrix with -1 outside, 0 on the border and 1 inside

    def interpolate(self, s1, s2, znew):
        '''interpolate structure between slices'''
        s1 = np.array(s1) #ROI points in 2D matrix, -1 out, 0 border, 1 in
        s2 = np.array(s2) #ROI points in 2D matrix, -1 out, 0 border, 1 in
        s1a = s1.copy()
        s2a = s2.copy()
        s1a[np.where(s1==-1)] = 1 #copy of ROI matrix with 0 on the border and 1 otherwise
        s2a[np.where(s2==-1)] = 1 #copy of ROI matrix with 0 on the border and 1 otherwise
        im1 = distance_transform_edt(s1a, return_distances=True) #calculate distance to border abs value
        im2 = distance_transform_edt(s2a, return_distances=True)
        del s1a
        del s2a


        out1 = np.ones((len(s1), len(s1[0])), dtype = np.int)
        out2 = np.ones((len(s2), len(s2[0])), dtype = np.int)
        out1[np.where(s1!=-1)]=0 #a matrix with 1 inside ROI and 0 outside
        out2[np.where(s2!=-1)]=0
        del s1
        del s2

        im1 = -im1*out1 + im1*(~out1+2) #to transform distance as negative inside the ROI and positive outside
        im2 = -im2*out2 + im2*(~out2+2)

        con = np.array([im1, im2]) # stack slcies to make a 3D matrix for interpolation

        z = [0,1]
        con_m = np.zeros((len(znew), len(im1), len(im1[0]))) #interpolated 3D matirx
        for i in arange(0, len(im1)): #interpolate each voxel in z direction
            for j in arange(0, len(im1[0])):
                f = interp1d(z, con[:,i, j])
                con_m[:,i,j]=f(znew)
        del con
        del im1
        del im2

        #find a polygon for the interpolated ROI
        Xfin = []
        Yfin = []
        for n in arange(0, len(con_m)): #slice by  slice
            a = np.zeros((len(con_m[n]), len(con_m[n][0])), dtype=np.uint8)
            a[np.where(con_m[n]>=0)] = 1 #everything inside countour equal 1

            contour , hier = cv2.findContours(a, mode =1, method =2) #find a contour for the structure 

            Xf = []
            Yf = []
            for i in arange(0, len(contour)): #for substructures, like holes
                Xf.append(contour[i][:,0,0])
                Yf.append(contour[i][:,0,1])
            del contour
            Xfin.append(Xf) #append given slice contour
            Yfin.append(Yf)
            del Xf, Yf
        del con_m
        return Xfin, Yfin #returns list of contour points, if a slice contrain a substractures then the element for slice is list of lists
