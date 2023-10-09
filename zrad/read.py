import logging
import os
from os import listdir
from os.path import isfile, join

import pydicom as dc
from numpy import arange
import nibabel as nib

from structure import Structures
import numpy as np


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

    def __init__(self, file_type, UID, mypath_image, structure, wv, dim, local):
        self.x_ct = None
        self.y_ct = None
        self.onlyfiles = None
        self.rs = None
        self.slices_w = None
        self.yCTspace = None
        self.zCTspace = None
        self.xCTspace = None
        self.columns = None
        self.rows = None
        self.slices = None
        self.Xcontour = None
        self.Xcontour_W = None
        self.Ycontour = None
        self.Ycontour_W = None
        self.Xcontour_Rec = None
        self.Ycontour_Rec = None
        self.listDicomProblem = None
        self.structure_f = None
        self.logger = logging.getLogger(__name__)
        self.logger.info("Start")
        self.dim = dim
        self.stop_calc = ''
        self.UID = UID
        self.mypath_image = mypath_image
        self.structure = structure
        self.wv = wv
        self.local = local
        if file_type == 'dicom':
            self.ReadImages()
            self.ReadStucture()
        elif file_type == 'nifti':
            self.ReadNiftiImageStructure()

    def ReadImages(self):
        self.logger.info("Reading Images")
        onlyfiles = []
        self.listDicomProblem = []  # cannot open as dicom
        for f in listdir(self.mypath_image):
            try:
                # read only dicoms of certain modality
                if isfile(join(self.mypath_image, f)) and dc.read_file(self.mypath_image + f).SOPClassUID in self.UID:
                    # sort files by slice position
                    onlyfiles.append((round(float(dc.read_file(self.mypath_image + os.sep + f).ImagePositionPatient[2]), 3), f))
            except 'InvalidDicomError':  # not a dicom file
                self.listDicomProblem.append(f)
                pass

        onlyfiles.sort()  # sort and take only file names
        self.slices = []
        for i in arange(len(onlyfiles)):
            self.slices.append(onlyfiles[i][0])
            onlyfiles[i] = onlyfiles[i][1]

        IM = dc.read_file(self.mypath_image + os.sep + onlyfiles[0])

        self.rows = IM.Rows
        self.columns = IM.Columns
        self.xCTspace = float(IM.PixelSpacing[0])
        self.x_ct = float(IM.ImagePositionPatient[0])  # x image corner
        self.y_ct = float(IM.ImagePositionPatient[1])  # y image corner
        self.zCTspace = float(IM.SliceThickness)
        self.onlyfiles = onlyfiles

        del IM

    def ReadStucture(self):
        """reading RS file"""
        self.logger.info("Reading StructureSet")
        self.logger.info("Selected Structure " + " ".join(self.structure))

        RS_UID = ['1.2.840.10008.5.1.4.1.1.481.3', 'RT Structure Set Storage']  # structure set

        rtstruct_files = []
        for filename in listdir(self.mypath_image):
            filepath = join(self.mypath_image, filename)
            if isfile(filepath) and dc.read_file(filepath).SOPClassUID in RS_UID:
                rtstruct_files.append(filename)

        # take only the first RS file you found
        rs = rtstruct_files[0]

        self.rs = self.mypath_image + rs
        self.logger.info("StructureSet File" + self.rs)
        struct = Structures(self.rs, self.structure, self.slices, self.x_ct, self.y_ct, self.xCTspace, len(self.slices), self.wv,
                            self.dim, self.local)
        self.Xcontour = struct.Xcontour
        self.Xcontour_W = struct.Xcontour_W
        self.Ycontour = struct.Ycontour
        self.Ycontour_W = struct.Ycontour_W
        self.slices_w = struct.slices_w
        self.Xcontour_Rec = struct.Xcontour_Rec
        self.Ycontour_Rec = struct.Ycontour_Rec
        self.structure_f = struct.organs

    def ReadNiftiImageStructure(self):
        self.onlyfiles = [e for e in listdir(self.mypath_image) if e[0] != '.']
        if len(self.onlyfiles) == 2:
            matrix1 = nib.load(self.mypath_image + self.onlyfiles[0])
            matrix2 = nib.load(self.mypath_image + self.onlyfiles[1])
            img_matrix1 = matrix1.get_fdata().transpose(2, 1, 0)
            img_matrix2 = matrix2.get_fdata().transpose(2, 1, 0)
            xCTspace = matrix1.header['pixdim'][1]
            yCTspace = matrix1.header['pixdim'][2]
            zCTspace = matrix1.header['pixdim'][3]
            if img_matrix1.shape != img_matrix2.shape:
                self.stop_calc = 'image and contour shape differ'
            self.xCTspace = xCTspace
            self.yCTspace = yCTspace
            self.zCTspace = zCTspace
            self.columns = img_matrix1.shape[2] 
            self.rows = img_matrix1.shape[1] 
            self.slices = img_matrix1.shape[0]
            for i, matrix in enumerate([matrix1, matrix2]):
                print(f"File {i} pixel dimensions {matrix.header['pixdim'][1:4]}")
                print(f"Min {np.min(matrix.get_fdata())}")
                # print(f"p2 {np.percentile(matrix.get_fdata(), 2)}")
                print(f"Median {np.min(matrix.get_fdata())}")
                print(f"Mean {np.mean(matrix.get_fdata())}")
                # print(f"p98 {np.percentile(matrix.get_fdata(), 98)}")
                print(f"Max {np.max(matrix.get_fdata())}")
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
        self.structure_f = self.structure[0]
