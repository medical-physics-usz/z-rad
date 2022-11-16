import logging
import os
from os.path import isfile, join

import numpy as np
import pydicom as dc
import nibabel as nib
from joblib import Parallel, delayed
from tqdm import tqdm

from read import ReadImageStructure
from texture import Texture
from utils import tqdm_joblib
from zrad.export import export_results, preset


class main_texture_pet(object):
    """main_texture_pet(object)
    Main class to handle PET images, reads images and structures, calls radiomics calculation and export class to export results
    Type: object
    Attributes:
    sb - Status bar in the frame
    file_type - dicom or nifti, influences reading in the files
    path_image - path to the patients subfolders
    path_save - path to save radiomics results
    structure - list of structures to be analysed
    labels - label number in the nifti file, list of numbers, each number corresponds to different contour
    pixNr number of analyzed bins, if not specified  = none
    binSize - bin size for the analysis, if not specified = none
    l_ImName - list of patients subfolders (here are data to be analysed)
    save_as - name of text files to save the radiomics results
    Dim - string variable of value 2D or 3D for dimensionality of calculation
    SUV - bool, correct PET intensities to SUV
    wv - bool, calculate wavelet
    local - switch to calculate local radiomics, set to False for now
    cropStructure - 
    exportList - list of matrices/features to be calculated and exported
    """

    def __init__(self, sb, file_type, path_image, path_save, structures, labels, pixNr, binSize, l_ImName, save_as, dim, SUV, wv, local,
                 cropStructure, exportList, n_jobs):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Start")
        self.n_jobs = n_jobs
        image_modality = ['PET']
        dicomProblem = []

        def parfor(ImName):
            self.logger.info("Patient " + ImName)
            to_return_3d = list()
            meanWV = False  # calculated modified WV transform

            for structure in structures:
                self.logger.info("Structure " + structure)
                structure = [structure]

                # read in CT data if needed
                try:
                    # calc the shift between CT and PET if crop True
                    self.logger.info("Read in PET")
                    mypath_image = path_image + ImName + os.sep
                    self.logger.debug("PET Path " + mypath_image)
                    PET_UID = ['1.2.840.10008.5.1.4.1.1.20', 'Positron Emission Tomography Image Storage',
                               '1.2.840.10008.5.1.4.1.1.128']  # PET
                    read = ReadImageStructure(file_type, PET_UID, mypath_image, structure, wv, dim, local)
                    dicomProblem.append([ImName, read.listDicomProblem])
                    
                    if file_type == 'dicom':
                        # parameters to recalculate intensities to SUV
                        # to extract dicom header value for SUV
                        sample_image = dc.read_file(mypath_image + read.onlyfiles[0])
                        inter = float(sample_image.RescaleIntercept)
                        if SUV and sample_image.Units == 'BQML':
                            try:
                                if sample_image.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose != '':
                                    dose = float(sample_image.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
                                else:
                                    dose = 0
                                HL = float(sample_image.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
                                h_start = 3600 * float(
                                    sample_image.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime[:2])
                                h_stop = 3600 * float(sample_image.AcquisitionTime[:2])
                                m_start = 60 * float(
                                    sample_image.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime[2:4])
                                m_stop = 60 * float(sample_image.AcquisitionTime[2:4])
                                s_start = float(
                                    sample_image.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime[4:])
                                s_stop = float(sample_image.AcquisitionTime[4:])
                                time = (h_stop + m_stop + s_stop - h_start - m_start - s_start)
                                activity = dose * np.exp(-time * np.log(2) / HL)
                                self.logger.info('activity ' + str(activity))
                                if sample_image.PatientWeight != '' and float(sample_image.PatientWeight) != 0.0:
                                    weight = float(sample_image.PatientWeight) * 1000
                                else:
                                    read.stop_calc = 'Patient weight attribute to calc SUV missing!'
                                    weight = np.nan
                                self.logger.info('weight' + str(weight))
                            except AttributeError:
                                read.stop_calc = 'attribute to calc SUV missing'
                                activity = np.nan
                        elif SUV and sample_image.Units == 'GML':
                            activity = 1.
                            weight = 1.
                        elif not SUV:  # no SUV correction
                            activity = 1.
                            weight = 1.
    
                        self.logger.info('units ' + sample_image.Units)
    
                        bitsRead = str(dc.read_file(mypath_image + read.onlyfiles[1]).BitsAllocated)
                        sign = int(dc.read_file(mypath_image + read.onlyfiles[1]).PixelRepresentation)
                        if sign == 1:
                            bitsRead = 'int' + bitsRead
                        elif sign == 0:
                            bitsRead = 'uint' + bitsRead
    
                        IM_matrix = []  # list containing the images matrix
                        for f in read.onlyfiles:
                            data = dc.read_file(mypath_image + f).PixelData
                            slope = float(dc.read_file(mypath_image + f).RescaleSlope)
                            data16 = np.array(np.fromstring(data, dtype=bitsRead))  # converting to decimal
                            data16 = (data16 * slope + inter) / (activity / weight)  # correcting for SUV
                            # recalculating for rows x columns
                            a = []
                            for j in range(read.rows):
                                a.append(data16[j * read.columns: (j + 1) * read.columns])
                            a = np.array(a)
                            IM_matrix.append(np.array(a))
                        IM_matrix = np.array(IM_matrix)
                        contour_matrix = ''  # only for nifti
    
                        if cropStructure["crop"]:
                            self.logger.info("Start: Cropping")
    
                            cropStructure["data"] = []
                            cropStructure["readCT"] = []
                            CT_matrix = []  # list containing the images matrix
    
                            self.logger.info("Read in provided CT for cropping")
                            crop_patient_path = cropStructure["ct_path"] + os.sep + ImName + os.sep
                            CT_UID = ['1.2.840.10008.5.1.4.1.1.2', '1.2.840.10008.5.1.4.1.1.2.1']
                            read_crop_CT = ReadImageStructure(CT_UID, crop_patient_path, structure, wv, dim, local)
    
                            dicomProblem.append(["crop_" + ImName, read_crop_CT.listDicomProblem])
                            # parameters to recalculate intensities HU
                            inter = float(dc.read_file(crop_patient_path + read_crop_CT.onlyfiles[1]).RescaleIntercept)
                            slope = float(dc.read_file(crop_patient_path + read_crop_CT.onlyfiles[1]).RescaleSlope)
    
                            bitsReadCT = dc.read_file(crop_patient_path + read_crop_CT.onlyfiles[1]).BitsAllocated
                            sign = int(dc.read_file(crop_patient_path + read_crop_CT.onlyfiles[1]).PixelRepresentation)
    
                            if sign == 1:
                                bitsReadCT = 'int' + bitsRead
                            elif sign == 0:
                                bitsReadCT = 'uint' + bitsRead
    
                            for f in read_crop_CT.onlyfiles:
                                data = dc.read_file(crop_patient_path + f).PixelData
                                data16 = np.array(np.fromstring(data, dtype=bitsReadCT))  # converitng to decimal
                                data16 = data16 * slope + inter
                                # recalculating for rows x columns
                                data_matrix_CT = []
                                for j in range(read_crop_CT.rows):
                                    data_matrix_CT.append(data16[j * read_crop_CT.columns:(j + 1) * read_crop_CT.columns])
                                data_matrix_CT = np.array(data_matrix_CT)
                                CT_matrix.append(np.array(data_matrix_CT))
                            CT_matrix = np.array(CT_matrix)
                            cropStructure["data"] = [CT_matrix]
                            cropStructure["readCT"] = read_crop_CT
                        if cropStructure["crop"]:
                            print("shape of matrices of PET and CT", IM_matrix.shape, CT_matrix.shape)
                            
                    elif file_type == 'nifti':
                        # nifti file type assumes that conversion to eg HU or SUV has already been performed
                        # the folder should contain two file, one contour mask with the name corresponding to the name given in GUI and the other file with the image
                        if read.stop_calc == '':
                            image_name = ''
                            contour_name = ''
                            for f in read.onlyfiles:
                                if isfile(join(mypath_image, f)) and structure[0] in f:  # nifti only handles one ROI
                                    contour_name = f
                                else: 
                                    image_name = f
                            img = nib.load(mypath_image + image_name)
                            contour = nib.load(mypath_image + contour_name)   
                            slope = img.header['scl_slope']
                            intercept = img.header['scl_inter']
                            if np.isnan(slope):
                                slope = 1.
                            if np.isnan(intercept):
                                intercept = 0
                            IM_matrix = img.get_fdata().transpose(2,1,0)  * slope + intercept
                            contour_matrix = contour.get_fdata().transpose(2,1,0) 
                            for lab in labels:
                                ind = np.where(contour_matrix == lab)
                                contour_matrix[ind] = 100
                            ind = np.where(contour_matrix != 100)
                            contour_matrix[ind] = 0
                            ind = np.where(contour_matrix == 100)
                            contour_matrix[ind] = 1
                            activity = 1.
                            weight = 1.
                        else:
                            IM_matrix = ''
                            contour_matrix = ''        

                except OSError:  # error if there is not directory
                    continue
                except IndexError:  # empty folder
                    continue
                except NameError:  # if none of 3 SUV conditions fulfilled and activity and weight not defined
                    continue
                # Texture(arguments).ret() -> function for texture calculation
                # arguments: image, stucture name, image corner x, image corner x, columns, pixelSpacing, HFS or FFS,
                # structure file, list of slice positions, patient number, path to save the textutre maps,
                # map name (eg. AIF1), pixel discretization, site
                # function returns: number of removed points, minimum values, maximum values, structre used for
                # calculations, mean,std, cov, skewness, kurtosis, enenrgy, entropy, contrast, corrrelation, homogenity,
                # coarseness, neighContrast, busyness, complexity, intensity varation, size variation, fractal dimension,
                # number of points used in the calculations, histogram (values bigger/smaller than median)
                if activity == 0:
                    read.stop_calc = 'activity 0'
                elif np.isnan(weight):
                    read.stop_calc = 'undefined weight'
                lista_results = Texture(file_type, [IM_matrix], contour_matrix, read.structure_f, read.columns, read.rows, read.xCTspace,
                                        read.slices, path_save, ImName, pixNr, binSize, image_modality, wv, local,
                                        cropStructure, read.stop_calc, meanWV, read.Xcontour, read.Xcontour_W, read.Ycontour,
                                        read.Ycontour_W, read.Xcontour_Rec, read.Ycontour_Rec).ret()
                #            elif dim == '2D': #not working
                #                lista_results = Texture2D(sb,IM_matrix, structure, x_ct,y_ct, columns, rows, xCTspace, patientPos, rs, slices, path_save, ImName, pixNr, prefix).ret()
                features_3d = [[ImName, lista_results[2], lista_results[:2], lista_results[3:-1], lista_results[-1]]]
                to_return_3d.append(features_3d)

            return to_return_3d

        with tqdm_joblib(tqdm(desc="Extracting intensity and texture features", total=len(l_ImName))):
            out = Parallel(n_jobs=self.n_jobs)(delayed(parfor)(ImName) for ImName in l_ImName)

        final_file, wave_names, par_names = preset(wv, path_save, save_as, image_modality)
        feature_vectors = [feature_vec for batch in out for feature_vec in batch]
        for feature_vec in feature_vectors:
            final_file = export_results(feature_vec, final_file, image_modality, wave_names, wv)
        final_file.close()
