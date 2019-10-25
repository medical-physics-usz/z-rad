# -*- coding: utf-8 -*-s
import os
try:
    import dicom as dc
    from dicom.filereader import InvalidDicomError
except:
    import pydicom as dc
    from pydicom.filereader import InvalidDicomError
import numpy as np
from numpy import arange
from os import path, makedirs, listdir, rmdir
from os.path import isfile, join
from scipy.ndimage.morphology import distance_transform_edt
from scipy.interpolate import interp1d
import cv2
from glob import glob

from resize_interpolate_roi import InterpolateROI
import logging

class ResizeTexture(object):
    '''Class to resize images and listed structures to a resolution defined by user and saved the results as dicom file
    inp_resolution – resolution defined by user
    inp_struct – list of structure names to be resized
    inp_mypath_load – path with the data to be resized
    inp_mypath_save – path to save resized data
    image_type – image modality
    begin – start number
    stop – stop number
    '''

    def __init__(self, inp_resolution,inp_struct,inp_mypath_load,inp_mypath_save,image_type, begin, stop, cropInput):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Start: Resize Texture")

        self.resolution = float(inp_resolution)
        if self.resolution < 1.: #set a round factor for slice position
            self.round_factor = 3
        else:
            self.round_factor = 3
        self.cropStructure = cropInput

        self.list_structure = inp_struct

        self.mypath_load = inp_mypath_load+"\\"
        self.mypath_s = inp_mypath_save+"\\"
        self.UID = '2030' #UID for new images
        self.image_type = image_type

        self.list_dir = [str(f) for f in arange(begin, stop+1)] #list of directories to be analyzed
        self.resize()

        #if there were non dicom files save file
        if len(self.listDicomProblem)!=0:
            config = open(self.mypath_s+'\\'+'dicom_problem.txt', 'w')
            for i in self.listDicomProblem:
                config.write(i+'\n')
            config.close()

    def resize(self):
        '''resize image and structure'''

        wrongROI = []
        lista_voi = [] #problems with VOI
        emptyROI = [] #empty contour for one of the structures
        self.listDicomProblem = [] #cannot open as dicom
        error = []
        for name in self.list_dir:
            try:
                print('patient ', name)
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
                    UID_t = ['1.2.840.10008.5.1.4.1.1.20','Positron Emission Tomography Image Storage', '1.2.840.10008.5.1.4.1.1.128']
                elif self.image_type == 'MR':
                    UID_t = ['1.2.840.10008.5.1.4.1.1.4']


                #resize standard modalities: CT, PET and MR
                if self.image_type not in ['CTP', 'IVIM']:
                    onlyfiles = []

                    for f in listdir(mypath_file):
                        try:
                            if isfile(join(mypath_file,f)) and dc.read_file(mypath_file+f).SOPClassUID in UID_t: #read only dicoms of certain modality
                                onlyfiles.append((round(float(dc.read_file(mypath_file+'\\'+f).ImagePositionPatient[2]), self.round_factor), f)) #sort files by slice position
                        except InvalidDicomError: #not a dicom file
                            self.listDicomProblem.append(name+' '+f)
                            pass

                    onlyfiles.sort() #sort and take only file names
                    for i in arange(0, len(onlyfiles)):
                        onlyfiles[i] = onlyfiles[i][1]

                    #data needed to decode images
                    print(onlyfiles[0])
                    CT = dc.read_file(mypath_file+onlyfiles[0]) #example image
                    position = CT.PatientPosition # HFS or FFS
                    xCTspace=float(CT.PixelSpacing[1]) #XY resolution
                    yCTspace=float(CT.PixelSpacing[0]) #XY resolution
                    xct = float(CT.ImagePositionPatient[0]) #x position of top left corner
                    yct = float(CT.ImagePositionPatient[1]) #y position of top left corner

                    columns = CT.Columns # number of columns
                    rows = CT.Rows #number of rows

                    if self.cropStructure["ct_path"] == "":
                        new_gridX = np.arange(round(xct, 10), round(xct + xCTspace*columns, 10), self.resolution) #new grid of X for interpolation
                        old_gridX = np.arange(round(xct, 10), round(xct + xCTspace*columns, 10), xCTspace) #original grid of X
                        new_gridY = np.arange(round(yct, 10), round(yct + yCTspace*rows, 10), self.resolution) #new grid of Y for interpolation
                        old_gridY = np.arange(round(yct, 10), round(yct + yCTspace*rows, 10), yCTspace) #original grid of Y

                    elif self.cropStructure["ct_path"] != "":

                        self.logger.info("Start Shifting Matrix to adjust for Image Corner differences")
                        CTfiles = glob(self.cropStructure["ct_path"] + "\\" + name + "\\*dcm")
                        cropCT = dc.read_file(CTfiles[0]) # take one ct slices to extract image origin
                        if ("RTST" in cropCT.Modality):
                            cropCT = dc.read_file(CTfiles[1])
                        crop_corner_x = float(cropCT.ImagePositionPatient[0])# x position of top left corner
                        crop_corner_y = float(cropCT.ImagePositionPatient[1])# y position of top left corner
                        self.logger.info("x_ct_old" + str(xct))
                        self.logger.info("y_ct_old" + str(yct))
                        self.logger.info("x_ct_new" + str(crop_corner_x))
                        self.logger.info("y_ct_new" + str(crop_corner_y))
                        new_gridX = np.arange(round(crop_corner_x, 10), round(crop_corner_x + xCTspace*columns, 10), self.resolution) #new grid of X for interpolation
                        old_gridX = np.arange(round(xct, 10), round(xct + xCTspace*columns, 10), xCTspace) #original grid of X
                        new_gridY = np.arange(round(crop_corner_y, 10), round(crop_corner_y + yCTspace*rows, 10), self.resolution) #new grid of Y for interpolation
                        old_gridY = np.arange(round(yct, 10), round(yct + yCTspace*rows, 10), yCTspace) #original grid of Y


                    print("length of rows and columns in dicom", rows, columns)
                    print("shape of old gridX", old_gridX.shape)
                    print("shape of old gridY", old_gridY.shape)
                    if len(old_gridX) > columns: # due to rounding
                        old_gridX = old_gridX[:columns]
                    if len(old_gridY) > rows: # due to rounding
                        old_gridY = old_gridY[:rows]

                    new_rows = len(new_gridY) # number of new rows
                    new_columns = len(new_gridX) # number of new columns

                    IM = [] #list of images
                    slices = [] # list of slices
                    for k in onlyfiles:
                        CT = dc.read_file(mypath_file+k)
                        slices.append(round(float(CT.ImagePositionPatient[2]), self.round_factor))

                        #read image data
                        data = CT.PixelData #imaging data
                        bits = str(CT.BitsAllocated)
                        sign = int(CT.PixelRepresentation)

                        if sign == 1:
                            bits = 'int'+bits
                        elif sign ==0:
                            bits = 'uint'+bits

                        if self.image_type == 'PET':
                            data16 = np.array(np.fromstring(data, dtype=bits)) #converitng to decimal
                            data16 = data16*float(CT.RescaleSlope)+float(CT.RescaleIntercept)
                        else:
                            data16 = np.array(np.fromstring(data, dtype=bits)) #converitng to decimal, for CT no intercept as it is the same per image
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
                    print('new grid Z ', len(new_gridZ))
                    print(new_gridZ)
                    print('old grid Z ')
                    print(old_gridZ)
                    print("new rows and cols", new_rows, new_columns)
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
                            print('tu')
                            old_gridZ = np.array(slices)
                        for x in arange(0, new_columns):
                            for y in arange(0, new_rows):
                                new_image[:,y,x] = np.interp(new_gridZ, old_gridZ, IM[:,y,x])

                    #check if all dicom tags are valid
                    CT = dc.read_file(mypath_file+onlyfiles[0])
                    invalid_tags = []
                    key = list(CT.keys())
                    for ki in key:
                        try:
                            CT[ki].VR
                        except KeyError:
                            invalid_tags.append(ki)

                    for ki in invalid_tags:
                        wrongROI.append(name+' invalid tag'+str(ki))

                    #save interpolated images
                    for im in arange(0, len(new_image)):
                        im_nr = int(im*float(len(onlyfiles))/len(new_image)) #choose an original dicom file to be modify
                        CT = dc.read_file(mypath_file+onlyfiles[im_nr]) #read file to be modified
                        for ki in invalid_tags:
                            del CT[ki]
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
                            if bits[0] =='u':
                                rslope = float(vmax)/(2**int(CT.BitsAllocated)) #16 bits
                            else:
                                rslope = float(vmax)/(2**(int(CT.BitsAllocated)-1)) #16 bits
                            new_image[im] = new_image[im]/rslope
                            CT.RescaleSlope = round(rslope,5)
                            CT.RescaleIntercept = 0
                            array=np.array(new_image[im], dtype = bits)
                            CT.LargestImagePixelValue = np.max(array)
                            CT[0x28,0x107].VR = 'US'
                        else:
                            array=np.array(new_image[im], dtype = bits)
                        data = array.tostring() #convert to string
                        CT.PixelData = data #data pixel data to new image
                        if self.cropStructure["ct_path"] != "":
                            CT.ImagePositionPatient[0] = crop_corner_x
                            CT.ImagePositionPatient[1] = crop_corner_y
                            #CT.ImagePositionPatient[2] = zct
                        #try:
                        CT.save_as(mypath_save+'image'+str(im+1)+'.dcm') #save image
                        #except KeyError:
                        #    print "images cannot be saved, irregular Dicom Tag found"
                        #    wrongROI.append(name)

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
    #                            CT.sldataseticeThick = str(self.resolution)
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
                        xCTspace=float(CT.PixelSpacing[1]) #XY resolution
                        yCTspace=float(CT.PixelSpacing[0]) #XY resolution
                        xct = float(CT.ImagePositionPatient[0]) #x position of top left corner
                        yct = float(CT.ImagePositionPatient[1]) #y position of top left corner
                        columns = CT.Columns # number of columns
                        rows = CT.Rows #number of rows
                        new_gridX = np.arange(xct, xct+xCTspace*columns, self.resolution) #new grid of X for interpolation
                        old_gridX = np.arange(xct, xct+xCTspace*columns, xCTspace) #original grid of X
                        new_gridY = np.arange(yct, yct+yCTspace*rows, self.resolution) #new grid of Y for interpolation
                        old_gridY = np.arange(yct, yct+yCTspace*rows, yCTspace) #original grid of Y

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
                        print('new grid Z ')
                        print(new_gridZ)
                        print('old grid Z ')
                        print(old_gridZ)

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
                                print('tu')
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
                self.logger.info("Resize structure Set for texture")
                RS_UID = ['1.2.840.10008.5.1.4.1.1.481.3', 'RT Structure Set Storage'] #structure set

                rs=[]
                for f in listdir(mypath_file):
                    try:
                        if  isfile(join(mypath_file,f)) and dc.read_file(mypath_file+f).SOPClassUID in RS_UID: #read only dicoms of certain modality
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
                        print('structure: ', change_struct[s])
                        try:
                            #read a contour points for given structure
                            #M - 3D matrix filled with 1 insdie contour and 0 outside
                            #xmin - minimum value of x in the contour
                            #ymin - minimum value of y in the contour
                            #st_nr - number of the ROI of the defined name
                            M, xmin, ymin, st_nr = InterpolateROI().structures(rs_name, change_struct[s], slices, xct, yct, xCTspace, yCTspace, len(slices), self.round_factor)

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
                                        #interpolate, X list of x positions of the interpolated contour, Y list of y positions of the interpoated contour , interpolation type  texture find polygon encompassing the sturcture
                                        X, Y = InterpolateROI().interpolate(M[n_s], M[n_s+1], np.linspace(0,1,sliceThick/0.01+1), 'texture')
                                    elif self.round_factor == 3:
                                        zi = np.linspace(old_gridZ[n_s], old_gridZ[n_s+1], sliceThick/0.001+1) #create an interpolation grid between those slicse
                                        #round interpolation grid accroding to specified precision
                                        for gz in arange(0, len(zi)):
                                            zi[gz] = round(zi[gz],self.round_factor)
                                        #interpolate, X list of x positions of the interpolated contour, Y list of y positions of the interpoated contour, interpolation type  texture find polygon encompassing the sturcture
                                        X, Y = InterpolateROI().interpolate(M[n_s], M[n_s+1], np.linspace(0,1,sliceThick/0.001+1), 'texture')
                                    #check which position in the interpolation grid correcpods to the new slice position
                                    for i in arange(0, len(zi)):
                                        if zi[i] in new_gridZ and zi[i] not in insertedZ: #insertedZ gathers all slice positions which are alreay filled in case that slice position is on the ovelap of two slices from orignal
                                            insertedZ.append(zi[i])
                                            for j in arange(0, len(X[i])): #substructres in the slice
                                                l = np.zeros((3*len(X[i][j])))
                                                # this needs to be new position for structure!
                                                if self.cropStructure["ct_path"] != "":
                                                    xct = crop_corner_x
                                                    yct = crop_corner_y
                                                l[::3] = (X[i][j]+xmin)*xCTspace + xct #convert to the original coordinates in mm
                                                l[1::3]=(Y[i][j]+ymin)*yCTspace + yct #convert to the original coordinates in mm
                                                l[2::3] = round(zi[i],self.round_factor) #convert to the original coordinates in mm
                                                l.round(self.round_factor)
                                                li = [str(round(ci,self.round_factor)) for ci in l] #convert to string
                                                contour.append(li) # add to contour list
                                                del l
                                                del li

                            #search for ROI number I'm interested in
                            st_nr = 1000000
                            for j in arange(0, len(list_organs)):
                                if list_organs[j][0] == change_struct[s]:
                                    for k in arange(0, len(rs.ROIContourSequence)):
                                        if rs.ROIContourSequence[k].ReferencedROINumber == list_organs[j][1]:
                                            st_nr = k
                                            break
                            print('number: ', st_nr)
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
                                    rs.ROIContourSequence[st_nr].ContourSequence.append(a)
                            #delete the sequence elements if the original seuquence was longer than interpolated
                            for j in arange(len(contour), len(rs.ROIContourSequence[st_nr].ContourSequence)):
                                del rs.ROIContourSequence[st_nr].ContourSequence[-1]

                            print('length of new contour: ', len(contour))
                            print('length of new contour sequence: ', len(rs.ROIContourSequence[st_nr].ContourSequence))
                            print('the numbers above should be the same')
                        except IndexError:
                            emptyROI.append(name + '    '+ change_struct[s])
                            pass


                    #delete structures which were not resized
                    #modify sepatratelty just to be sure that sorting is correct ROIContourSequence, RTROIObservationsSequence, StructureSetROISequence
                    #ROIContourSequence
                    nr_del = []
                    for i in range(0,len(rs.ROIContourSequence)):

                        if rs.ROIContourSequence[i].ReferencedROINumber not in structure_nr_to_save:
                            nr_del.append(i)
                    nr_del.reverse()
                    for i in nr_del:
                        del rs.ROIContourSequence[i]
                    #RTROIObservationsSequence
                    nr_del = []
                    for i in range(0,len(rs.RTROIObservationsSequence)):
                        if rs.RTROIObservationsSequence[i].ReferencedROINumber not in structure_nr_to_save:
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

        if len(emptyROI)!=0:
            config = open(self.mypath_s+'\\'+'empty_roi.txt', 'w')
            for i in emptyROI:
                config.write(i+'\n')
            config.close()

        if len(error)!=0:
            config = open(self.mypath_s+'\\'+'unrecognized error.txt', 'w')
            for i in error:
                config.write(i+'\n')
            config.close()
