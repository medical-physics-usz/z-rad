# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:59:51 2017

@author: Marta Bogowicz
"""

try:
    import pydicom as dc  # dicom library
except ImportError:
    import dicom as dc  # dicom library
from os import listdir, makedirs  # managing files
from os.path import isfile, join, isdir
from numpy import arange
import logging

from structure import Structures


class ReadImageStructure(object):
    """reads certain modality dicom files in a given folder, provides list of files, number of rows, number of columns, pixel spacing, position of the left top corner (x,y), patient position;
    contour points (x,y) also for wavelets
    Type: object
    Attributes:
    UID – list of UIDs characteristic for this image modality
    mypath_image – path where images are saved
    structure – list of structures to be analysed
    wv – bool, calculate wavelet
    *modality - optional argument, list of prefixes in the functional maps for example for CTP modality = ['BV', 'MTT', 'BF']
    """

    def __init__(self, UID, mypath_image, stucture, wv, dim, local, *modality):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Start")
        if 'CTP' in UID or 'IVIM' in UID:
            self.modality = modality
        self.dim = dim
        self.ReadImages(UID, mypath_image)
        self.ReadStucture(mypath_image, stucture, wv, local)

    def ReadImages(self, UID, mypath_image):
        print('''reads and sorts images''')
        # self.logger.info("Reading Images")
        onlyfiles = []
        self.listDicomProblem = []  # cannot open as dicom
        if 'CTP' in UID or 'IVIM' in UID:  # CTP and IVIM
            onlyfiles = []
            for u in UID:  # iterate through different maps types in te functional imaging
                of = []
                for f in listdir(mypath_image):
                    if isfile(join(mypath_image, f)) and f[:len(self.modality[u])] == self.modality[u]:
                        of.append((round(float(dc.read_file(mypath_image + '\\' + f).ImagePositionPatient[2]), 3), f))
                        # hier müsste eine Fehlermeldung kommen, falls nicht!
                of.sort()
                onlyfiles.append(of)
            for u in arange(0, len(UID)):
                s = []
                for i in arange(0, len(onlyfiles[u])):
                    s.append(onlyfiles[u][i][0])
                    onlyfiles[u][i] = onlyfiles[u][i][1]
            self.slices = s
        else:  # CT or PET
            for f in listdir(mypath_image):
                try:
                    if isfile(join(mypath_image, f)) and dc.read_file(mypath_image + f).SOPClassUID in UID:  # read only dicoms of certain modality
                        onlyfiles.append((round(float(dc.read_file(mypath_image + '\\' + f).ImagePositionPatient[2]), 3), f))  # sort files by slice position
                except 'InvalidDicomError':  # not a dicom file
                    self.listDicomProblem.append(f)
                    pass

            onlyfiles.sort()  # sort and take only file names
            self.slices = []
            for i in arange(0, len(onlyfiles)):
                self.slices.append(onlyfiles[i][0])
                onlyfiles[i] = onlyfiles[i][1]

        IM = dc.read_file(mypath_image + '\\' + onlyfiles[0])

        self.rows = IM.Rows
        self.columns = IM.Columns
        self.xCTspace = float(IM.PixelSpacing[0])
        self.x_ct = float(IM.ImagePositionPatient[0])  # x image corner
        self.y_ct = float(IM.ImagePositionPatient[1])  # y image corner
        self.zCTspace = float(IM.SliceThickness)
        self.onlyfiles = onlyfiles

        del IM

    def ReadStucture(self, mypath_image, structure, wv, local):
        self.logger.info("Reading StructureSet")
        self.logger.info("Selected Structure " + " ".join(structure))
        '''reading RS file'''
        RS_UID = ['1.2.840.10008.5.1.4.1.1.481.3', 'RT Structure Set Storage']  # structure set
        rs = [f for f in listdir(mypath_image) if isfile(join(mypath_image, f)) and dc.read_file(mypath_image + f).SOPClassUID in RS_UID][0]  # take only the first RS file you found

        self.rs = mypath_image + rs
        self.logger.info("StructureSet File" + self.rs)
        struct = Structures(self.rs, structure, self.slices, self.x_ct, self.y_ct, self.xCTspace, len(self.slices), wv,self.dim, local)
        self.Xcontour = struct.Xcontour
        self.Xcontour_W = struct.Xcontour_W
        self.Ycontour = struct.Ycontour
        self.Ycontour_W = struct.Ycontour_W
        self.slices_w = struct.slices_w

        self.Xcontour_Rec = struct.Xcontour_Rec
        self.Ycontour_Rec = struct.Ycontour_Rec

        self.structure_f = struct.organs
