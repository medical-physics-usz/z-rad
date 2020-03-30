# -*- coding: utf-8 -*-

# import libraries
import pydicom as dc
from pydicom.filereader import InvalidDicomError
import numpy as np
from os import path, makedirs, listdir
from os.path import isfile, join
from glob import glob
import logging

# own classes
from resize_interpolate_roi import InterpolateROI

class ResizeShape(object):
    '''Class to resize listed structures to a resolution (1mm for texture resultion > 1mm and 0.1 mm for texture resultion <1mm) and saved the results as text files in subfolder resize_1mm
    inp_resolution – resolution defined by user for texture calcualtion
    inp_struct – list of structure names to be resized
    inp_mypath_load – path with the data to be resized
    inp_mypath_save – path to save resized data
    image_type – image modality
    begin – start number
    stop – stop number
    '''

    def __init__(self, inp_struct, inp_mypath_load, inp_mypath_save, image_type, low, high, inp_resolution, interpolation_type, cropStructure):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Resize Shape")
        inp_resolution = float(inp_resolution)
        if inp_resolution < 1.: #set a round factor for slice position and resolution for shape calculation 1mm if texture resolution > 1mm and 0.1 if texture resolution < 1mm
            self.round_factor = 3
            self.resolution = 0.1 #finer resolution for the shape calcaultion if texture is below 1 mm
        else:
            self.round_factor = 3
            self.resolution = 1.0

        self.cropStructure = cropStructure
        self.interpolation_algorithm=interpolation_type
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
        self.wrongROI = [] #in case of key errors

        self.lista_dir =  [str(i) for i in np.arange(low, high+1)] #list of directories to be analyzed

        self.resize()

    def resize(self):
        '''resize structure to text file'''

        if self.image_type == 'CT':
            UID_t = ['1.2.840.10008.5.1.4.1.1.2', '1.2.840.10008.5.1.4.1.1.2.1'] #CT and contarst-enhanced CT
        elif self.image_type == 'PET':
            UID_t = ['1.2.840.10008.5.1.4.1.1.20','Positron Emission Tomography Image Storage', '1.2.840.10008.5.1.4.1.1.128']
        elif self.image_type == 'MR':
            UID_t = ['1.2.840.10008.5.1.4.1.1.4']


        for name in self.lista_dir: #iterate through the patients
            try:
                print('patient ', name)
                mypath_file =self.mypath_load +name + '\\' #go to subfolder for given patient
                mypath_save = self.mypath_s

                onlyfiles = []
                for f in listdir(mypath_file):
                    try:
                        if isfile(join(mypath_file,f)) and dc.read_file(mypath_file+f).SOPClassUID in UID_t: #read only dicoms of certain modality
                            onlyfiles.append((round(float(dc.read_file(mypath_file+'\\'+f).ImagePositionPatient[2]), self.round_factor), f)) #sort files by slice position
                    except InvalidDicomError: #not a dicom file
                        self.listDicomProblem.append(name+'   '+f)
                        pass

                onlyfiles.sort() #sort and take only file names
                slices = []
                for i in np.arange(0, len(onlyfiles)):
                    slices.append(onlyfiles[i][0])
                    onlyfiles[i] = onlyfiles[i][1]

                CT = dc.read_file(mypath_file+onlyfiles[0]) #example image
                # position in DICOM: pixel spacing = y,x,z in image position patient x,y,z
                xCTspace=float(CT.PixelSpacing[1]) # XY resolution
                yCTspace=float(CT.PixelSpacing[0]) # XY resolution
                xct = float(CT.ImagePositionPatient[0]) # x position of top left corner
                yct = float(CT.ImagePositionPatient[1]) # y position of top left corner
                if self.cropStructure["ct_path"] != "":
                    CTfiles = glob(self.cropStructure["ct_path"] + "\\" + name + "\\*dcm")
                    cropCT = dc.read_file(CTfiles[0]) # take one ct slices to extract image origin
                    if ("RTST" in cropCT.Modality):
                        cropCT = dc.read_file(CTfiles[1])
                    xct = float(cropCT.ImagePositionPatient[0]) #x position of top left corner
                    yct =  float(cropCT.ImagePositionPatient[1]) #y position of top left corner
                #define z interpolation grid
                sliceThick = round(abs(slices[0]-slices[1]),self.round_factor)
                #check slice sorting,for the interpolation funtcion one need increasing slice position
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
                        if isfile(join(mypath_file,f)) and dc.read_file(mypath_file+f).SOPClassUID in RS_UID: #read only dicoms of certain modality
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
                    for j in np.arange(0, len(rs.StructureSetROISequence)):
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

                    for s in np.arange(0, len(change_struct)):
                        print('structure: ', change_struct[s])
                        #read a contour points for given structure
                        #M - 3D matrix filled with 1 insdie contour and 0 outside
                        #xmin - minimum value of x in the contour
                        #ymin - minimum value of y in the contour
                        #st_nr - number of the ROI of the defined name
                        M, xmin, ymin, st_nr = InterpolateROI().structures(rs_name, change_struct[s], slices, xct, yct, self.resolution, self.resolution, len(slices), self.round_factor)

                        insertedZ=[] #list of contour slices alread inserted for the given ROI

                        # roudning new patient position to the defined precision
                        for gz in range(0, len(new_gridZ)):
                            new_gridZ[gz] = round(new_gridZ[gz],self.round_factor)
                        for n_s in np.arange(0, len(M)-1): # n_s slice number
                            if M[n_s] != [] and M[n_s+1] != []: #if two consecutive slices not empty - interpolate
                                if self.round_factor == 2:
                                    zi = np.linspace(old_gridZ[n_s], old_gridZ[n_s+1], sliceThick/0.01+1) #create an interpolation grid between those slicse
                                    #round interpolation grid accroding to specified precision
                                    for gz in np.arange(0, len(zi)):
                                        zi[gz] = round(zi[gz],self.round_factor)
                                    #interpolate, X list of x positions of the interpolated contour, Y list of y positions of the interpoated contour , interpolation type shape returns all the points in the structure
                                    X, Y = InterpolateROI().interpolate(self.interpolation_algorithm, M[n_s], M[n_s+1], np.linspace(0,1,sliceThick/0.01+1), 'shape')
                                elif self.round_factor == 3 :
                                    zi = np.linspace(old_gridZ[n_s], old_gridZ[n_s+1], sliceThick/0.001+1) #create an interpolation grid between those slicse
                                    #round interpolation grid accroding to specified precision
                                    for gz in np.arange(0, len(zi)):
                                        zi[gz] = round(zi[gz],self.round_factor)
                                    #interpolate, X list of x positions of the interpolated contour, Y list of y positions of the interpoated contour , interpolation type shape returns all the points in the structure
                                    X, Y = InterpolateROI().interpolate(self.interpolation_algorithm, M[n_s], M[n_s+1], np.linspace(0,1,sliceThick/0.001+1), 'shape')
                                #check which position in the interpolation grid correcpods to the new slice position
                                for i in np.arange(0, len(zi)):
                                    if zi[i] in new_gridZ and zi[i] not in insertedZ: #insertedZ gathers all slice positions which are alreay filled in case that slice position is on the ovelap of two slices from orignal
                                        ind = str(np.where(new_gridZ == zi[i])[0][0]) #slice position to save correct file name, also importat for LN dist
                                        file_n = open(mypath_save+'\\'+change_struct[s]+'\\'+name+'\\'+'slice_'+ind, 'w')
                                        insertedZ.append(zi[i])
                                        for j in np.arange(0, len(X[i])): #save positions of all points inside structure in to columns for X and Y
                                            file_n.write(str(X[i][j]+xmin)+'\t' +str(Y[i][j]+ymin))
                                            file_n.write('\n')
                                        file_n.close()

            except WindowsError: #no path with data for the patient X
                pass
            except KeyError:
                self.wrongROI.append(name) #problem with image
                pass
            except IndexError:
                pass
#
        if len(self.wrongROI)!=0:
            config = open(self.mypath_s+'\\'+self.list_structure[0]+'_key_error.txt', 'w')
            for i in self.wrongROI:
                config.write(i+'\n')
            config.close()

        if len(self.listDicomProblem)!=0:
            config = open(self.mypath_s+'\\'+'dicom_file_error.txt', 'w')
            for i in self.listDicomProblem:
                config.write(i+'\n')
            config.close()
