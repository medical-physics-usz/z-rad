import logging
import os
from os import listdir
from os.path import isfile, join

import pydicom as dc
from numpy import arange
import nibabel as nib

from structure import Structures


class ReadImageStructure(object):
    """Reads certain modality dicom files in a given folder, provides list of files, number of rows, number of columns,
    pixel spacing, position of the left top corner (x,y), patient position;
    contour points (x,y) also for wavelets
    Type: object
    Attributes:
    UID – list of UIDs characteristic for this image modality
    mypath_image – path where images are saved
    structure – list of structures to be analysed
    wv – bool, calculate wavelet
    *modality - optional argument, list of prefixes in the functional maps for example for CTP modality = ['BV', 'MTT',
    'BF']
    """

    def __init__(self, file_type, UID, mypath_image, structure, wv, dim, local, *modality):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Start")
        if 'CTP' in UID or 'IVIM' in UID:
            self.modality = modality
        self.dim = dim
        self.stop_calc = ''
        if file_type == 'dicom':
            self.ReadImages(UID, mypath_image)
            self.ReadStucture(mypath_image, structure, wv, local)
        elif file_type == 'nifti':
            self.ReadNiftiImageStructure(mypath_image, structure) 

    def ReadImages(self, UID, mypath_image):
        self.logger.info("Reading Images")
        onlyfiles = []
        self.listDicomProblem = []  # cannot open as dicom
        if 'CTP' in UID or 'IVIM' in UID:  # CTP and IVIM
            onlyfiles = []
            for u in UID:  # iterate through different maps types in te functional imaging
                of = []
                for f in listdir(mypath_image):
                    if isfile(join(mypath_image, f)) and f[:len(self.modality[u])] == self.modality[u]:
                        of.append((round(float(dc.read_file(mypath_image + os.sep + f).ImagePositionPatient[2]), 3), f))
                        # hier müsste eine Fehlermeldung kommen, falls nicht!
                of.sort()
                onlyfiles.append(of)
            for u in arange(len(UID)):
                s = []
                for i in arange(len(onlyfiles[u])):
                    s.append(onlyfiles[u][i][0])
                    onlyfiles[u][i] = onlyfiles[u][i][1]
            self.slices = s
        else:  # CT or PET
            for f in listdir(mypath_image):
                try:
                    # read only dicoms of certain modality
                    if isfile(join(mypath_image, f)) and dc.read_file(mypath_image + f).SOPClassUID in UID:
                        # sort files by slice position
                        onlyfiles.append((round(float(dc.read_file(mypath_image + os.sep + f).ImagePositionPatient[2]), 3), f))
                except 'InvalidDicomError':  # not a dicom file
                    self.listDicomProblem.append(f)
                    pass

            onlyfiles.sort()  # sort and take only file names
            self.slices = []
            for i in arange(len(onlyfiles)):
                self.slices.append(onlyfiles[i][0])
                onlyfiles[i] = onlyfiles[i][1]

        IM = dc.read_file(mypath_image + os.sep + onlyfiles[0])

        self.rows = IM.Rows
        self.columns = IM.Columns
        self.xCTspace = float(IM.PixelSpacing[0])
        self.x_ct = float(IM.ImagePositionPatient[0])  # x image corner
        self.y_ct = float(IM.ImagePositionPatient[1])  # y image corner
        self.zCTspace = float(IM.SliceThickness)
        self.onlyfiles = onlyfiles

        del IM

    def ReadStucture(self, mypath_image, structure, wv, local):
        """reading RS file"""
        self.logger.info("Reading StructureSet")
        self.logger.info("Selected Structure " + " ".join(structure))

        RS_UID = ['1.2.840.10008.5.1.4.1.1.481.3', 'RT Structure Set Storage']  # structure set
        # take only the first RS file you found
        rs = [f for f in listdir(mypath_image) if isfile(join(mypath_image, f)) and dc.read_file(mypath_image + f).SOPClassUID in RS_UID][0]

        self.rs = mypath_image + rs
        self.logger.info("StructureSet File" + self.rs)
        struct = Structures(self.rs, structure, self.slices, self.x_ct, self.y_ct, self.xCTspace, len(self.slices), wv,
                            self.dim, local)
        self.Xcontour = struct.Xcontour
        self.Xcontour_W = struct.Xcontour_W
        self.Ycontour = struct.Ycontour
        self.Ycontour_W = struct.Ycontour_W
        self.slices_w = struct.slices_w
        self.Xcontour_Rec = struct.Xcontour_Rec
        self.Ycontour_Rec = struct.Ycontour_Rec
        self.structure_f = struct.organs

    def ReadNiftiImageStructure(self, mypath_image, structure):
        self.onlyfiles = listdir(mypath_image)
        if len(self.onlyfiles ) == 2:
            matrix1 = nib.load(mypath_image + self.onlyfiles[0])
            matrix2 = nib.load(mypath_image + self.onlyfiles[1])     
            img_matrix1 = matrix1.get_fdata() 
            img_matrix2 = matrix2.get_fdata()
            xCTspace = matrix1.header['pixdim'][2]
            yCTspace = matrix1.header['pixdim'][3]
            zCTspace = matrix1.header['pixdim'][1] 
            xCTspaceCont = matrix2.header['pixdim'][2]
            yCTspaceCont = matrix2.header['pixdim'][3]
            zCTspaceCont = matrix2.header['pixdim'][1]
            if not (xCTspace == yCTspace == zCTspace and xCTspaceCont == yCTspaceCont == zCTspaceCont and xCTspace == xCTspaceCont):
                self.stop_calc = 'image and contour voxels are not cubic or voxel size in image and contour differ'
            if img_matrix1.shape != img_matrix2.shape:
                self.stop_calc = 'image and contour shape differ'
            self.xCTspace = xCTspace
            self.yCTspace = yCTspace
            self.zCTspace = zCTspace
            self.columns = img_matrix1.shape[2] 
            self.rows = img_matrix1.shape[1] 
            self.slices = img_matrix1.shape[0]                         
        else:
            self.stop_calc = 'expecting 2 files per directory'
            self.xCTspace = ''
            self.columns = ''
            self.rows = ''
            self.slices = ''
        self.Xcontour = ''
        self.Xcontour_W = ''
        self.Ycontour = '' 
        self.Ycontour_W = ''
        self.Xcontour_Rec = ''
        self.Ycontour_Rec = ''
        self.listDicomProblem = []
        self.structure_f = structure[0]
