import logging
import os

import numpy as np
import pydicom as dc
from joblib import Parallel, delayed

from export import Export
from read import ReadImageStructure
from structure import Structures
from texture import Texture


class main_texture_mr(object):
    """
    Main class to handle MR images, reads images and structures, normalize MR image according to two specified ROI,
    calls radiomics calculation and export class to export results
    Type: object
    Attributes:
    sb - Status bar in the frame
    path_image - path to the patients subfolders
    path_save - path to save radiomics results
    structure - list of structures to be analysed
    pixNr number of analyzed bins, if not specified  = none
    binSize - bin size for the analysis, if not specified = none
    l_ImName - list of patients subfolders (here are data to be analysed)
    save_as - name of text files to save the radiomics results
    Dim - string variable of value 2D or 3D for dimensionality of calculation
    struct_norm1 - ROI1 for the normalization coefficients for MR image based on two normal structures (for example
    muscle and white matter) and linear function
    struct_norm2 - ROI2 for the normalization
    wv - bool, calculate wavelet
    exportList - list of matrices/features to be calculated and exported
    """

    def __init__(self, sb, path_image, path_save, structures, pixNr, binSize, l_ImName, save_as, dim, struct_norm1,
                 struct_norm2, wv, local, cropStructure, exportList, n_jobs):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Start")
        self.n_jobs = n_jobs
        image_modality = ['MR']
        dicomProblem = []
        MR_UID = ['1.2.840.10008.5.1.4.1.1.4']  # MR

        def parfor(ImName):
            self.logger.info("Patient " + ImName)
            mypath_image = path_image + ImName + os.sep
            to_return_3d = list()

            for structure in structures:
                try:
                    read = ReadImageStructure(MR_UID, mypath_image, [structure], wv, dim, local)
                    dicomProblem.append([ImName, read.listDicomProblem])

                    # MR intensities normalization
                    if struct_norm1 != '':
                        norm_slope, norm_inter = self.normalization(struct_norm1, struct_norm2, read, mypath_image)
                    else:
                        norm_slope = 1  # to allow for calculation with no normalization
                        norm_inter = 0

                    bitsRead = str(dc.read_file(mypath_image + read.onlyfiles[1]).BitsAllocated)
                    sign = int(dc.read_file(mypath_image + read.onlyfiles[1]).PixelRepresentation)
                    if sign == 1:
                        bitsRead = 'int' + bitsRead
                    elif sign == 0:
                        bitsRead = 'uint' + bitsRead

                    IM_matrix = []  # list containing the images matrix
                    for f in read.onlyfiles:
                        data = dc.read_file(mypath_image + f).PixelData
                        data16 = np.array(np.fromstring(data, dtype=bitsRead))  # converting to decimal
                        data16 = data16 * norm_slope + norm_inter
                        # recalculating for rows x columns
                        a = []
                        for j in range(read.rows):
                            a.append(data16[j * read.columns:(j + 1) * read.columns])
                        a = np.array(a)
                        IM_matrix.append(np.array(a))
                    IM_matrix = np.array(IM_matrix)

                except OSError:  # error if there is not directory
                    import pdb
                    pdb.set_trace()
                    continue
                except IndexError:  # empty folder
                    import pdb
                    pdb.set_trace()
                    continue

                    # Texture(arguments).ret() -> function for texture calculation
                    # arguments: image, structure name, image corner x, image corner x, columns, pixelSpacing, HFS or FFS, structure file, list of slice positions, patient number, path to save the textutre maps, map name (eg. AIF1), pixel discretization, site
                    # function returns: number of removed points, minimum values, maximum values, structre used for calculations, mean,std, cov, skewness, kurtosis, enenrgy, entropy, contrast, corrrelation, homogenity, coarseness, neighContrast, busyness, complexity, intensity varation, size variation, fractal dimension, number of points used in the calculations, histogram (values bigger/smaller than median)
                stop_calc = ''  # in case something would be wrong with the image tags
                if dim == '3D':
                    lista_results = Texture([IM_matrix], read.structure_f, read.columns, read.rows, read.xCTspace,
                                            read.slices, path_save, ImName, pixNr, binSize, image_modality, wv, local,
                                            cropStructure, stop_calc, read.Xcontour, read.Xcontour_W, read.Ycontour,
                                            read.Ycontour_W, read.Xcontour_Rec, read.Ycontour_Rec).ret()

                #                elif dim == '2D': #not working
                #                    lista_results = Texture2D(sb,IM_matrix, structure, x_ct,y_ct, columns, rows, xCTspace, patientPos, rs, slices, path_save, ImName, pixNr, prefix).ret()

                features_3d = [[ImName, lista_results[2], lista_results[:2], lista_results[3:-1], lista_results[-1]]]
                to_return_3d.append(features_3d)
            return to_return_3d

        out = Parallel(n_jobs=self.n_jobs, verbose=20)(delayed(parfor)(ImName) for ImName in l_ImName)

        final_file, wave_names, par_names = Export().Preset(exportList, wv, local, path_save, save_as, image_modality,
                                                            path_image)

        feature_vectors = [feature_vec for batch in out for feature_vec in batch]
        for feature_vec in feature_vectors:
            final_file = Export().ExportResults(feature_vec, final_file, par_names, image_modality,
                                                wave_names, wv, local)
        final_file.close()

    def normalization(self, struct_norm1, struct_norm2, read, mypath_image):
        """Get the normalization coefficients for MR image based on two normal structures (for example muscle and white
        matter) and linear function.
        """
        # False for the wavelet and local argument, none for dim
        struct1 = Structures(read.rs, [struct_norm1], read.slices, read.x_ct, read.y_ct,
                             read.xCTspace, len(read.slices), False, None, False)
        norm1_Xcontour = struct1.Xcontour
        norm1_Ycontour = struct1.Ycontour

        struct2 = Structures(read.rs, [struct_norm2], read.slices, read.x_ct, read.y_ct, read.xCTspace,
                             len(read.slices), False, None,
                             False)  # False for the wavelet and local  argument, none for dim
        norm2_Xcontour = struct2.Xcontour
        norm2_Ycontour = struct2.Ycontour

        # to read only slices where there is a contour
        ind = []
        for f in range(len(norm1_Xcontour)):
            if norm1_Xcontour[f] != []:
                ind.append(f)
        for f in range(len(norm2_Xcontour)):
            if norm2_Xcontour[f] != []:
                ind.append(f)

        zmin = np.min(ind)
        zmax = np.max(ind)

        onlyfiles = read.onlyfiles[zmin:zmax + 1]
        slices = read.slices[zmin:zmax + 1]
        norm1_Xcontour = norm1_Xcontour[zmin:zmax + 1]
        norm1_Ycontour = norm1_Ycontour[zmin:zmax + 1]
        norm2_Xcontour = norm2_Xcontour[zmin:zmax + 1]
        norm2_Ycontour = norm2_Ycontour[zmin:zmax + 1]

        IM_matrix = []  # list containing the images matrix
        for i in onlyfiles:
            data = dc.read_file(mypath_image + i).PixelData
            data16 = np.array(np.fromstring(data, dtype=np.int16))  # converting to decimal
            # recalculating for rows x columns
            a = []
            for j in range(read.rows):
                a.append(data16[j * read.columns:(j + 1) * read.columns])
            a = np.array(a)
            IM_matrix.append(np.array(a))

        IM_matrix = np.array(IM_matrix)

        v1 = []  # values for structure 1
        v2 = []  # values for structure 2

        for i in range(len(norm1_Xcontour)):  # slices
            for j in range(len(norm1_Xcontour[i])):  # sub-structures in the slice
                for k in range(len(norm1_Xcontour[i][j])):
                    v1.append(IM_matrix[i][norm1_Ycontour[i][j][k]][norm1_Xcontour[i][j][k]])

        for i in range(len(norm2_Xcontour)):  # slices
            for j in range(len(norm2_Xcontour[i])):  # sub-structures in the slice
                for k in range(len(norm2_Xcontour[i][j])):
                    v2.append(IM_matrix[i][norm2_Ycontour[i][j][k]][norm2_Xcontour[i][j][k]])

        f1 = np.mean(v1)
        f2 = np.mean(v2)

        fa = (800 - 300) / (f1 - f2)  # find coefficients of a linear function
        fb = 800 - f1 * fa

        return fa, fb
