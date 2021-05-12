import logging
import os
from functools import reduce
from glob import glob
from os import path, makedirs, listdir, rmdir
from os.path import isfile, join
from shutil import copyfile

import numpy as np
import pydicom as dc
from joblib import Parallel, delayed
from pydicom.filereader import InvalidDicomError
from scipy.interpolate import interp1d
from tqdm import tqdm

from resize_interpolate_roi import InterpolateROI
from utils import tqdm_joblib


class ResizeTexture(object):
    """Class to resize images and listed structures to a resolution defined by user and saved the results as dicom file
    inp_resolution – resolution defined by user
    inp_struct – list of structure names to be resized
    inp_mypath_load – path with the data to be resized
    inp_mypath_save – path to save resized data
    image_type – image modality
    begin – start number
    stop – stop number
    dim - dimension for resizing
    """

    def __init__(self, inp_resolution, interpolation_type, inp_struct, inp_mypath_load, inp_mypath_save, image_type,
                 begin, stop, cropInput, dim, n_jobs):
        self.logger = logging.getLogger("Resize_Texture")
        self.interpolation_alg = interpolation_type
        self.resolution = float(inp_resolution)
        if self.resolution < 1.:  # set a round factor for slice position
            self.round_factor = 3
        else:
            self.round_factor = 3
        self.cropStructure = cropInput
        self.dim = dim
        self.list_structure = inp_struct
        if inp_mypath_load[-1] != os.sep:
            inp_mypath_load += os.sep
        self.mypath_load = inp_mypath_load
        if inp_mypath_save[-1] != os.sep:
            inp_mypath_save += os.sep
        self.mypath_s = inp_mypath_save
        self.UID = '2030'  # UID for new images
        self.image_type = image_type

        pat_range = [str(f) for f in range(begin, stop + 1)]
        pat_dirs = glob(self.mypath_load + os.sep + "*[0-9]*")
        list_dir_candidates = [e.split(os.sep)[-1] for e in pat_dirs if e.split(os.sep)[-1].split("_")[0] in pat_range]
        self.list_dir = sorted(list_dir_candidates)

        self.listDicomProblem = []  # cannot open as dicom
        self.n_jobs = n_jobs
        self.resize()

        # if there were non dicom files save file
        if len(self.listDicomProblem) != 0:
            config = open(self.mypath_s + os.sep + 'dicom_problem.txt', 'w')
            for i in self.listDicomProblem:
                config.write(i + '\n')
            config.close()

    def resize(self):
        """resize image and structure"""

        def parfor(name):
            wrong_roi_this = []
            list_voi_this = []  # problems with VOI
            empty_roi_this = []  # empty contour for one of the structures

            try:
                self.logger.info('Patient ' + str(name))
                mypath_file = self.mypath_load + name + os.sep  # go to subfolder for given patient
                mypath_save = self.mypath_s + name + os.sep  # create subfolder for given patient
                try:
                    makedirs(mypath_save)
                except OSError:
                    if not path.isdir(mypath_save):
                        raise

                if self.image_type == 'CT':
                    UID_t = ['1.2.840.10008.5.1.4.1.1.2', '1.2.840.10008.5.1.4.1.1.2.1']  # CT and contrast-enhanced CT
                elif self.image_type == 'PET':
                    UID_t = ['1.2.840.10008.5.1.4.1.1.20', 'Positron Emission Tomography Image Storage',
                             '1.2.840.10008.5.1.4.1.1.128']
                elif self.image_type == 'MR':
                    UID_t = ['1.2.840.10008.5.1.4.1.1.4']

                # resize standard modalities: CT, PET and MR
                if self.image_type not in ['CTP', 'IVIM']:
                    onlyfiles = []

                    for f in listdir(mypath_file):
                        try:
                            # read only dicoms of certain modality
                            if isfile(join(mypath_file, f)) and dc.read_file(mypath_file + f).SOPClassUID in UID_t:
                                # sort files by slice position
                                onlyfiles.append((round(
                                    float(dc.read_file(mypath_file + os.sep + f).ImagePositionPatient[2]),
                                    self.round_factor), f))
                        except InvalidDicomError:  # not a dicom file
                            self.listDicomProblem.append(name + ' ' + f)
                            pass

                    onlyfiles.sort()  # sort and take only file names
                    for i in range(len(onlyfiles)):
                        onlyfiles[i] = onlyfiles[i][1]

                    # data needed to decode images
                    self.logger.info("First found image filename: " + onlyfiles[0])
                    CT = dc.read_file(mypath_file + onlyfiles[0])  # example image
                    position = CT.PatientPosition  # HFS or FFS
                    xCTspace = float(CT.PixelSpacing[1])  # XY resolution
                    yCTspace = float(CT.PixelSpacing[0])  # XY resolution
                    xct = float(CT.ImagePositionPatient[0])  # x position of top left corner
                    yct = float(CT.ImagePositionPatient[1])  # y position of top left corner

                    columns = CT.Columns  # number of columns
                    rows = CT.Rows  # number of rows

                    if self.cropStructure["ct_path"] == "":
                        # new grid of X for interpolation
                        new_gridX = np.arange(round(xct, 10), round(xct + xCTspace * columns, 10), self.resolution)
                        # original grid of X
                        old_gridX = np.arange(round(xct, 10), round(xct + xCTspace * columns, 10), xCTspace)
                        # new grid of Y for interpolation
                        new_gridY = np.arange(round(yct, 10), round(yct + yCTspace * rows, 10), self.resolution)
                        # original grid of Y
                        old_gridY = np.arange(round(yct, 10), round(yct + yCTspace * rows, 10), yCTspace)

                    elif self.cropStructure["ct_path"] != "":

                        self.logger.info("Start Shifting Matrix to adjust for Image Corner differences")
                        CTfiles = glob(self.cropStructure["ct_path"] + os.sep + name + os.sep + "*dcm")
                        cropCT = dc.read_file(CTfiles[0])  # take one ct slices to extract image origin
                        if "RTST" in cropCT.Modality:
                            cropCT = dc.read_file(CTfiles[1])
                        crop_corner_x = float(cropCT.ImagePositionPatient[0])  # x position of top left corner
                        crop_corner_y = float(cropCT.ImagePositionPatient[1])  # y position of top left corner
                        self.logger.info("x_ct_old" + str(xct))
                        self.logger.info("y_ct_old" + str(yct))
                        self.logger.info("x_ct_new" + str(crop_corner_x))
                        self.logger.info("y_ct_new" + str(crop_corner_y))
                        # new grid of X for interpolation
                        new_gridX = np.arange(round(crop_corner_x, 10), round(crop_corner_x + xCTspace * columns, 10),
                                              self.resolution)
                        # original grid of X
                        old_gridX = np.arange(round(xct, 10), round(xct + xCTspace * columns, 10), xCTspace)
                        # new grid of Y for interpolation
                        new_gridY = np.arange(round(crop_corner_y, 10), round(crop_corner_y + yCTspace * rows, 10),
                                              self.resolution)
                        # original grid of Y
                        old_gridY = np.arange(round(yct, 10), round(yct + yCTspace * rows, 10), yCTspace)

                    self.logger.info("length of rows {} and columns {} in dicom".format(rows, columns))
                    self.logger.info("shape of old gridX " + str(old_gridX.shape))
                    self.logger.info("shape of old gridY " + str(old_gridY.shape))
                    if len(old_gridX) > columns:  # due to rounding
                        old_gridX = old_gridX[:columns]
                    if len(old_gridY) > rows:  # due to rounding
                        old_gridY = old_gridY[:rows]

                    new_rows = len(new_gridY)  # number of new rows
                    new_columns = len(new_gridX)  # number of new columns

                    IM = []  # list of images
                    slices = []  # list of slices
                    for k in onlyfiles:  # go over DICOM files
                        CT = dc.read_file(mypath_file + k)
                        slices.append(round(float(CT.ImagePositionPatient[2]), self.round_factor))

                        # read image data
                        data = CT.PixelData  # imaging data
                        bits = str(CT.BitsAllocated)
                        sign = int(CT.PixelRepresentation)

                        if sign == 1:
                            bits = 'int' + bits
                        elif sign == 0:
                            bits = 'uint' + bits

                        if self.image_type == 'PET':
                            data16 = np.array(np.fromstring(data, dtype=bits))  # converting to decimal
                            data16 = data16 * float(CT.RescaleSlope) + float(CT.RescaleIntercept)
                        else:
                            # converting to decimal, for CT no intercept as it is the same per image
                            data16 = np.array(np.fromstring(data, dtype=bits))
                        # recalculating for rows x columns
                        a = np.reshape(data16, (rows, columns))
                        del data
                        del data16
                        # interpolate XY
                        b_new = np.zeros((len(old_gridY), len(new_gridX)))
                        for j in range(len(a)):  # interpolate in x direction (each row)

                            f = interp1d(old_gridX, a[j], kind=self.interpolation_alg, fill_value="extrapolate")
                            b_new[j] = f(new_gridX)
                        a_new = np.zeros((len(new_gridY), len(new_gridX)))
                        for j in range(len(b_new[0])):
                            f = interp1d(old_gridY, b_new[:, j], kind=self.interpolation_alg, fill_value="extrapolate")
                            a_new[:, j] = f(new_gridY)

                        del b_new
                        del a
                        IM.append(a_new)
                        del a_new

                    if self.dim == "2D" or self.dim == "2D_singleSlice":
                        # skip z interpolation if dim = 2D
                        new_image = np.array(IM)
                    else:
                        # define z interpolation grid
                        IM = np.array(IM)
                        sliceThick = round(abs(slices[0] - slices[1]), self.round_factor)
                        # check slice sorting,for the interpolation function one need increasing slice position
                        if slices[1] - slices[0] < 0:
                            new_gridZ = range(slices[-1], slices[0] + sliceThick, self.resolution)
                            old_gridZ = range(slices[-1], slices[0] + sliceThick, sliceThick)
                            Image = IM.copy()
                            for j in range(len(IM)):
                                IM[j] = Image[-j - 1]
                            del Image
                        else:
                            new_gridZ = np.arange(slices[0], slices[-1] + sliceThick, self.resolution)
                            old_gridZ = np.arange(slices[0], slices[-1] + sliceThick, sliceThick)
                        self.logger.info('new grid Z ' + str(len(new_gridZ)))
                        self.logger.info(" " + ", ".join(map(str, new_gridZ)))
                        self.logger.info('old grid Z ' + str(len(old_gridZ)))
                        self.logger.info(" " + ", ".join(map(str, old_gridZ)))
                        self.logger.info("new rows and cols {}, {}".format(new_rows, new_columns))
                        new_image = np.zeros(
                            (len(new_gridZ), new_rows, new_columns))  # matrix with zeros for the new image
                        # interpolate in z direction
                        try:
                            for x in range(new_columns):
                                for y in range(new_rows):
                                    f = interp1d(old_gridZ, IM[:, y, x], kind=self.interpolation_alg,
                                                 fill_value="extrapolate")
                                    new_image[:, y, x] = f(new_gridZ)

                        except ValueError:
                            if slices[1] - slices[0] < 0:
                                slices_r = np.array(slices).copy()
                                slices_r = list(slices_r)
                                slices_r.reverse()
                                old_gridZ = np.array(slices_r)
                            else:
                                old_gridZ = np.array(slices)
                            for x in range(new_columns):
                                for y in range(new_rows):
                                    f = interp1d(old_gridZ, IM[:, y, x], kind=self.interpolation_alg,
                                                 fill_value="extrapolate")
                                    new_image[:, y, x] = f(new_gridZ)

                    # check if all dicom tags are valid
                    CT = dc.read_file(mypath_file + onlyfiles[0])
                    invalid_tags = []
                    key = list(CT.keys())
                    for ki in key:
                        try:
                            CT[ki].VR
                        except KeyError:
                            invalid_tags.append(ki)

                    for ki in invalid_tags:
                        wrong_roi_this.append(name + ' invalid tag' + str(ki))

                    # save interpolated images
                    for im in range(len(new_image)):
                        im_nr = int(
                            im * float(len(onlyfiles)) / len(new_image))  # choose an original dicom file to be modify
                        CT = dc.read_file(mypath_file + onlyfiles[im_nr])  # read file to be modified
                        for ki in invalid_tags:
                            del CT[ki]

                        # to save information about used interpolation
                        CT.SeriesDescription = self.interpolation_alg
                        # change UID so it is treated as new image
                        CT.FrameOfReferenceUID = CT.FrameOfReferenceUID[:-2] + self.UID
                        CT.SeriesInstanceUID = CT.SeriesInstanceUID[:-2] + self.UID
                        CT.SOPInstanceUID = CT.SOPInstanceUID[:-1] + self.UID + str(im)
                        CT.Columns = new_columns  # adapt columns
                        CT.Rows = new_rows  # adapt rows
                        CT.PixelSpacing[0] = str(self.resolution)  # adapt XY resolution
                        CT.PixelSpacing[1] = str(self.resolution)
                        if self.dim == "3D":
                            CT.sliceThick = str(self.resolution)  # adapt slice thickness
                            # adapt slice location tag if exists (in 3D)
                            try:
                                if position == 'FFS':
                                    CT.SliceLocation = str(-new_gridZ[im])  # minus comes from the standard
                                else:
                                    CT.SliceLocation = str(new_gridZ[im])
                            except AttributeError:
                                pass
                            # adapt patient position tag for new z position of the image
                            CT.ImagePositionPatient[2] = str(new_gridZ[im])
                        # for PET images, calculate new slope and intercept
                        if self.image_type == 'PET':
                            vmax = int(np.max(new_image[im]) + 1)
                            if bits[0] == 'u':
                                rslope = float(vmax) / (2 ** int(CT.BitsAllocated))  # 16 bits
                            else:
                                rslope = float(vmax) / (2 ** (int(CT.BitsAllocated) - 1))  # 16 bits
                            new_image[im] = new_image[im] / rslope
                            CT.RescaleSlope = round(rslope, 5)
                            CT.RescaleIntercept = 0
                            array = np.array(new_image[im], dtype=bits)
                            CT.LargestImagePixelValue = np.max(array)
                            CT[0x28, 0x107].VR = 'US'
                        else:
                            array = np.array(new_image[im], dtype=bits)
                        data = array.tostring()  # convert to string
                        CT.PixelData = data  # data pixel data to new image
                        if self.cropStructure["ct_path"] != "":
                            CT.ImagePositionPatient[0] = crop_corner_x
                            CT.ImagePositionPatient[1] = crop_corner_y
                            # CT.ImagePositionPatient[2] = zct
                        # try:
                        CT.save_as(mypath_save + 'image' + str(im + 1) + '.dcm')  # save image
                        # except KeyError:
                        #    print "images cannot be saved, irregular Dicom Tag found"
                        #    wrong_roi_this.append(name)

                    del new_image
                    del CT
                    del onlyfiles

                elif self.image_type == 'IVIM':
                    maps_list = ['DSlow2', 'DFast2', 'F2']

                    for ivim_map in maps_list:
                        try:
                            makedirs(mypath_save + ivim_map)
                        except OSError:
                            if not path.isdir(mypath_save):
                                raise

                        mypath_ivim = mypath_file + ivim_map + os.sep  # IVIM subtype
                        # read all files in
                        onlyfiles = [f for f in listdir(mypath_ivim) if isfile(join(mypath_ivim, f))]

                        for i in range(len(onlyfiles)):
                            onlyfiles[i] = (
                                float(dc.read_file(mypath_ivim + onlyfiles[i]).ImagePositionPatient[2]), onlyfiles[i])

                        # sorting the files according to theirs slice position
                        onlyfiles.sort()
                        for i in range(len(onlyfiles)):
                            onlyfiles[i] = onlyfiles[i][1]

                        CT = dc.read_file(mypath_ivim + onlyfiles[0])  # example image
                        position = CT.PatientPosition  # HFS or FFS
                        xCTspace = float(CT.PixelSpacing[1])  # XY resolution
                        yCTspace = float(CT.PixelSpacing[0])  # XY resolution
                        xct = float(CT.ImagePositionPatient[0])  # x position of top left corner
                        yct = float(CT.ImagePositionPatient[1])  # y position of top left corner
                        columns = CT.Columns  # number of columns
                        rows = CT.Rows  # number of rows
                        # new grid of X for interpolation
                        new_gridX = np.arange(xct, xct + xCTspace * columns, self.resolution)
                        # original grid of X
                        old_gridX = np.arange(xct, xct + xCTspace * columns, xCTspace)
                        # new grid of Y for interpolation
                        new_gridY = np.arange(yct, yct + yCTspace * rows, self.resolution)
                        # original grid of Y
                        old_gridY = np.arange(yct, yct + yCTspace * rows, yCTspace)

                        if len(old_gridX) > columns:  # due to rounding
                            old_gridX = old_gridX[:-1]
                            old_gridY = old_gridY[:-1]

                        new_rows = len(new_gridY)  # number of new rows
                        new_columns = len(new_gridX)  # number of new columns

                        IM = []  # list of images
                        slices = []  # list of slices
                        for k in onlyfiles:  # list of slices
                            CT = dc.read_file(mypath_ivim + k)
                            slices.append(round(float(CT.ImagePositionPatient[2]), self.round_factor))

                            # read image data
                            data = CT.PixelData
                            data16 = np.array(np.fromstring(data, dtype=np.int16))  # converting to decimal
                            # recalculating for rows x columns
                            a = np.reshape(data16, (rows, columns))
                            del data
                            del data16

                            # interpolate XY
                            b_new = np.zeros((len(old_gridY), len(new_gridX)))
                            for j in range(len(a)):
                                f = interp1d(old_gridX, a[j], kind=self.interpolation_alg, fill_value="extrapolate")
                                b_new[j] = f(new_gridX)

                            a_new = np.zeros((len(new_gridY), len(new_gridX)))
                            for j in range(len(b_new[0])):
                                f = interp1d(old_gridY, b_new[j], kind=self.interpolation_alg, fill_value="extrapolate")
                                a_new[:, j] = f(new_gridY)

                            del b_new
                            del a
                            IM.append(a_new)
                            del a_new

                        if self.dim == "2D":
                            # skip z interpolation if dim = 2D
                            new_image = np.array(IM)
                        else:
                            # define z interpolation grid
                            IM = np.array(IM)
                            sliceThick = round(abs(slices[0] - slices[1]), self.round_factor)
                            # check slice sorting,for the interpolation function one need increasing slice position
                            if slices[1] - slices[0] < 0:
                                new_gridZ = np.arange(slices[-1], slices[0] + sliceThick, self.resolution)
                                old_gridZ = np.arange(slices[-1], slices[0] + sliceThick, sliceThick)
                                Image = IM.copy()
                                for j in range(len(IM)):
                                    IM[j] = Image[-j - 1]
                                del Image
                            else:
                                new_gridZ = np.arange(slices[0], slices[-1] + sliceThick, self.resolution)
                                old_gridZ = np.arange(slices[0], slices[-1] + sliceThick, sliceThick)
                            self.logger.info('new grid Z ' + ", ".join(map(str, new_gridZ)))
                            self.logger.info('old grid Z ' + ", ".join(map(str, old_gridZ)))

                            # matrix with zeros for the new image
                            new_image = np.zeros((len(new_gridZ), new_rows, new_columns))
                            # interpolate in z direction
                            try:
                                for x in range(new_columns):
                                    for y in range(new_rows):
                                        f = interp1d(old_gridZ, IM[:, y, x], kind=self.type_of_int,
                                                     fill_value="extrapolate")
                                        new_image[:, y, x] = f(new_gridZ)

                            except ValueError:
                                if slices[1] - slices[0] < 0:
                                    slices_r = np.array(slices).copy()
                                    slices_r = list(slices_r)
                                    slices_r.reverse()
                                    old_gridZ = np.array(slices_r)
                                else:
                                    old_gridZ = np.array(slices)
                                for x in range(new_columns):
                                    for y in range(new_rows):
                                        f = interp1d(old_gridZ, IM[:, y, x], kind=self.interpolation_alg,
                                                     fill_value="extrapolate")
                                        new_image[:, y, x] = f(new_gridZ)
                        # save interpolated images
                        for im in range(len(new_image)):
                            # choose an original dicom file to be modify
                            im_nr = int(im * float(len(onlyfiles)) / len(new_image))
                            CT = dc.read_file(mypath_ivim + onlyfiles[im_nr])  # read file to be modified
                            # change UID so it is treated as new image
                            CT.FrameOfReferenceUID = CT.FrameOfReferenceUID[:-2] + self.UID
                            CT.SeriesInstanceUID = CT.SeriesInstanceUID[:-2] + self.UID
                            CT.SOPInstanceUID = CT.SOPInstanceUID[:-1] + self.UID + str(im)
                            CT.Columns = new_columns  # adapt columns
                            CT.Rows = new_rows  # adapt rows
                            CT.PixelSpacing[0] = str(self.resolution)  # adapt XY resolution
                            CT.PixelSpacing[1] = str(self.resolution)
                            if self.dim == "3D":
                                CT.sliceThick = str(self.resolution)  # adapt slice thickness
                                # adapt slice location tag if exists
                                try:
                                    if position == 'FFS':
                                        CT.SliceLocation = str(-new_gridZ[im])  # minus comes from the standard
                                    else:
                                        CT.SliceLocation = str(new_gridZ[im])
                                except AttributeError:
                                    pass
                                # adapt patient position tag for new z position of the image
                                CT.ImagePositionPatient[2] = str(new_gridZ[im])
                            array = np.array(new_image[im], dtype=np.int16)
                            data = array.tostring()  # convert to string
                            CT.PixelData = data  # data pixel data to new image

                            CT.save_as(mypath_save + ivim_map + os.sep + 'image' + str(im + 1) + '.dcm')  # save image
                        del new_image
                        del CT
                        del onlyfiles

                # ------------------------------------------------------------------------------------------------------
                # resize structure
                self.logger.info("Resize structure Set for texture")
                RS_UID = ['1.2.840.10008.5.1.4.1.1.481.3', 'RT Structure Set Storage']  # structure set

                rs = []
                for f in listdir(mypath_file):
                    try:
                        # read only dicoms of certain modality
                        if isfile(join(mypath_file, f)) and dc.read_file(mypath_file + f).SOPClassUID in RS_UID:
                            rs.append(f)
                    except InvalidDicomError:  # not a dicom file
                        pass

                resize_rs = True
                if len(rs) != 1:
                    resize_rs = False
                else:
                    rs_name = mypath_file + rs[0]

                if not resize_rs:  # if there is no RS or too many RS files change folder name
                    os.rename(mypath_save[:-1], mypath_save[:-1] + '_noRS')
                else:
                    # the resize structure is not necessary in 2D interpolation. go on with next patient-folder
                    if self.dim == "2D" or self.dim == "2D_singleSlice":
                        # copy structure file into new resize folder
                        copyfile(join(rs_name), join(mypath_save, rs[0]))
                        # structure will not be resized
                        return {'wrong_roi': wrong_roi_this, 'list_voi': list_voi_this, 'empty_roi': empty_roi_this}

                    rs = dc.read_file(rs_name)  # read rs

                    list_organs = []  # ROI (name, number)
                    list_organs_names = []  # ROI names
                    for j in range(len(rs.StructureSetROISequence)):
                        list_organs.append(
                            [rs.StructureSetROISequence[j].ROIName, rs.StructureSetROISequence[j].ROINumber])
                        list_organs_names.append(rs.StructureSetROISequence[j].ROIName)

                    change_struct = []  # structure to be resized and which are available in RS
                    for j in self.list_structure:
                        if j in list_organs_names:
                            change_struct.append(j)

                    structure_nr_to_save = []
                    for s in range(len(change_struct)):
                        self.logger.info('processing structure: ' + change_struct[s])
                        try:
                            # read contour points for given structure
                            # M - 3D matrix filled with 1 inside contour and 0 outside
                            # xmin - minimum value of x in the contour
                            # ymin - minimum value of y in the contour
                            # st_nr - number of the ROI of the defined name
                            M, xmin, ymin, st_nr = InterpolateROI().structures(rs_name, change_struct[s], slices, xct,
                                                                               yct, xCTspace, yCTspace, len(slices),
                                                                               self.round_factor)

                            # rounding new patient position to the defined precision
                            contour = []  # list of contour points
                            insertedZ = []  # list of contour slices already inserted for the given ROI
                            for gz in range(len(new_gridZ)):
                                new_gridZ[gz] = round(new_gridZ[gz], self.round_factor)
                            for n_s in range(len(M) - 1):  # n_s slice number
                                # if two consecutive slices not empty - interpolate
                                if (M[n_s] != []) and (M[n_s + 1] != []):
                                    if self.round_factor == 2:
                                        # create an interpolation grid between those slices
                                        zi = np.linspace(old_gridZ[n_s], old_gridZ[n_s + 1], int(sliceThick / 0.01) + 1)
                                        # round interpolation grid according to specified precision
                                        for gz in range(len(zi)):
                                            zi[gz] = round(zi[gz], self.round_factor)
                                        # interpolate, X list of x positions of the interpolated contour, Y list of y
                                        # positions of the interpolated contour , interpolation type  texture find
                                        # polygon encompassing the sturcture
                                        X, Y = InterpolateROI().interpolate(self.interpolation_alg, M[n_s], M[n_s + 1],
                                                                            np.linspace(0, 1,
                                                                                        int(sliceThick / 0.01) + 1),
                                                                            'texture')
                                    elif self.round_factor == 3:
                                        # create an interpolation grid between those slices
                                        zi = np.linspace(old_gridZ[n_s], old_gridZ[n_s + 1],
                                                         int(sliceThick / 0.001) + 1)

                                        # round interpolation grid according to specified precision
                                        for gz in range(len(zi)):
                                            zi[gz] = round(zi[gz], self.round_factor)
                                        # interpolate, X list of x positions of the interpolated contour, Y list of y
                                        # positions of the interpolated contour, interpolation type  texture find
                                        # polygon encompassing the sturcture
                                        X, Y = InterpolateROI().interpolate(self.interpolation_alg, M[n_s], M[n_s + 1],
                                                                            np.linspace(0, 1,
                                                                                        int(sliceThick / 0.001) + 1),
                                                                            'texture')
                                    # check which position in the interpolation grid corresponds to the new slice
                                    # position
                                    for i in range(len(zi)):
                                        # insertedZ gathers all slice positions which are already filled in case that
                                        # slice position is on the ovelap of two slices from orignal
                                        if zi[i] in new_gridZ and zi[i] not in insertedZ:
                                            insertedZ.append(zi[i])
                                            for j in range(len(X[i])):  # substructures in the slice
                                                l = np.zeros((3 * len(X[i][j])))
                                                # this needs to be new position for structure!
                                                if self.cropStructure["ct_path"] != "":
                                                    xct = crop_corner_x
                                                    yct = crop_corner_y
                                                # convert to the original coordinates in mm
                                                l[::3] = (X[i][j] + xmin) * xCTspace + xct
                                                # convert to the original coordinates in mm
                                                l[1::3] = (Y[i][j] + ymin) * yCTspace + yct
                                                # convert to the original coordinates in mm
                                                l[2::3] = round(zi[i], self.round_factor)
                                                l.round(self.round_factor)
                                                # convert to string
                                                li = [str(round(ci, self.round_factor)) for ci in l]
                                                contour.append(li)  # add to contour list
                                                del l
                                                del li

                            # search for ROI number I'm interested in
                            st_nr = 1000000
                            for j in range(len(list_organs)):
                                if list_organs[j][0] == change_struct[s]:
                                    for k in range(len(rs.ROIContourSequence)):
                                        if rs.ROIContourSequence[k].ReferencedROINumber == list_organs[j][1]:
                                            st_nr = k
                                            break
                            self.logger.info('corresponding ROI number: ' + str(st_nr))
                            structure_nr_to_save.append(rs.ROIContourSequence[st_nr].ReferencedROINumber)

                            rs.StructureSetROISequence[st_nr].ROIName = change_struct[s]

                            # modify the rs file, replace old contour sequence with the new one
                            for j in range(len(contour)):
                                try:
                                    rs.ROIContourSequence[st_nr].ContourSequence[j].ContourData = contour[j]
                                    nr = len(contour[j]) // 3  # number of points
                                    rs.ROIContourSequence[st_nr].ContourSequence[j].NumberOfContourPoints = str(nr)
                                except IndexError:  # if the new contour is a longer sequence
                                    a = dc.dataset.Dataset()
                                    a.add_new((0x3006, 0x42), 'CS', 'CLOSED_PLANAR')
                                    a.add_new((0x3006, 0x46), 'IS', str(len(contour[j]) // 3))
                                    a.add_new((0x3006, 0x48), 'IS', str(j))  # sequence element number
                                    a.add_new((0x3006, 0x50), 'DS', contour[j])
                                    rs.ROIContourSequence[st_nr].ContourSequence.append(a)
                            # delete the sequence elements if the original sequence was longer than interpolated
                            for j in range(len(contour), len(rs.ROIContourSequence[st_nr].ContourSequence)):
                                del rs.ROIContourSequence[st_nr].ContourSequence[-1]

                            self.logger.info('length of new contour: ' + str(len(contour)))
                            self.logger.info(
                                'length of new contour sequence: ' + str(
                                    len(rs.ROIContourSequence[st_nr].ContourSequence)))
                            self.logger.info('the numbers above should be the same')
                        except IndexError:
                            empty_roi_this.append(name + '    ' + change_struct[s])
                            pass

                    # delete structures which were not resized
                    # modify separately just to be sure that sorting is correct ROIContourSequence,
                    # RTROIObservationsSequence, StructureSetROISequence
                    # ROIContourSequence
                    nr_del = []
                    for i in range(len(rs.ROIContourSequence)):
                        if rs.ROIContourSequence[i].ReferencedROINumber not in structure_nr_to_save:
                            nr_del.append(i)
                    nr_del.reverse()
                    for i in nr_del:
                        del rs.ROIContourSequence[i]
                    # RTROIObservationsSequence
                    nr_del = []
                    for i in range(len(rs.RTROIObservationsSequence)):
                        if rs.RTROIObservationsSequence[i].ReferencedROINumber not in structure_nr_to_save:
                            nr_del.append(i)
                    nr_del.reverse()
                    for i in nr_del:
                        del rs.RTROIObservationsSequence[i]
                    # StructureSetROISequence
                    nr_del = []
                    for i in range(len(rs.StructureSetROISequence)):
                        if rs.StructureSetROISequence[i].ROINumber not in structure_nr_to_save:
                            nr_del.append(i)
                    nr_del.reverse()
                    for i in nr_del:
                        del rs.StructureSetROISequence[i]

                    rs.save_as(mypath_save + 'RS.00001.dcm')  # save modified RS file

            except OSError:  # no path with data for the patient X
                rmdir(mypath_save)
                pass
            except KeyError:  # problem with image
                wrong_roi_this.append(name)
                pass
            except IndexError:
                list_voi_this.append(name)
                pass

            return {'wrong_roi': wrong_roi_this, 'list_voi': list_voi_this, 'empty_roi': empty_roi_this}

        with tqdm_joblib(tqdm(desc="Resizing texture", total=len(self.list_dir))):
            out = Parallel(n_jobs=self.n_jobs)(delayed(parfor)(name) for name in self.list_dir)

        if len(out) > 1:
            wrong_roi = reduce(lambda e1, e2: e1+e2, [e['wrong_roi'] for e in out])
            list_voi = reduce(lambda e1, e2: e1+e2, [e['list_voi'] for e in out])
            empty_roi = reduce(lambda e1, e2: e1+e2, [e['empty_roi'] for e in out])
        else:
            wrong_roi = out[0]['wrong_roi']
            list_voi = out[0]['list_voi']
            empty_roi = out[0]['empty_roi']
        if len(wrong_roi) != 0:
            config = open(self.mypath_s + os.sep + 'key_error.txt', 'w')
            for i in wrong_roi:
                config.write(i + '\n')
            config.close()
        if len(list_voi) != 0:
            config = open(self.mypath_s + os.sep + 'voi_problem.txt', 'w')
            for i in list_voi:
                config.write(i + '\n')
            config.close()
        if len(empty_roi) != 0:
            config = open(self.mypath_s + os.sep + 'empty_roi.txt', 'w')
            for i in empty_roi:
                config.write(i + '\n')
            config.close()
