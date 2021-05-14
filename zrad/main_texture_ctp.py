import os
from os import makedirs
from os.path import isdir

import numpy as np
import pydicom as dc
import matplotlib.pyplot as plt
import scipy.stats as st

from export import Export
from read import ReadImageStructure
from texture import Texture


class main_texture_ctp(object):
    """
    Main class to handle CTP images, reads images and structures, remove outliers (could be improved) and calls radiomics calculation and export class to export results
    Type: object
    Attributes: 
    sb – Status bar in the frame 
    file_type - dicom or nifti, influences reading in the files
    path_image - path to the patients subfolders
    path_save - path to save radiomics results
    structure - list of structures to be analysed
    labels - label number in the nifti file, list of numbers, each number corresponds to different contour
    pixNr number of analyzed bins, if not specified  = none
    binSize – bin size for the analysis, if not specified = none
    l_ImName – list of patients subfolders (here are data to be analysed)
    save_as – name of text files to save the radiomics results
    Dim – string variable of value 2D or 3D for dimensionality of calculation
    outlier – bool, correct for outliers
    wv – bool, calculate wavelet  
    exportList – list of matrices/features to be calculated and exported
    """

    def __init__(self, sb, file_type, path_image, path_save, structure, labels, pixNr, binSize, l_ImName, save_as, dim, outlier_corr, wv,
                 local, cropStructure, exportList):
        final = []  # list with results
        image_modality = ['BV', 'MTT', 'BF']
        dicomProblem = []
        meanWV = False
        for ImName in l_ImName:
            print('patient', ImName)
            try:
                sb.SetStatusText('Load ' + ImName)

                mypath_image = path_image + ImName + os.sep
                UID = ['CTP']

                # none for dimension
                read = ReadImageStructure(file_type, UID, mypath_image, structure, wv, dim, local, image_modality)

                dicomProblem.append([ImName, read.listDicomProblem])

                l_IM_matrix = []  # list containing different perfusion maps
                for m_name in range(len(image_modality)):
                    IM_matrix = []  # list containing the images matrix
                    for f in read.onlyfiles[m_name]:
                        data = dc.read_file(mypath_image + f).PixelData
                        data16 = np.array(np.fromstring(data, dtype=np.float16))  # converting to decimal
                        # recalculating for rows x columns
                        a = []
                        for j in range(read.rows):  # (0, rows):
                            a.append(data16[j * read.columns:(j + 1) * read.columns])
                        a = np.array(a)
                        IM_matrix.append(np.array(a))
                    IM_matrix = np.array(IM_matrix)

                    l_IM_matrix.append(IM_matrix)
                contour_matrix = '' # only for nifti

                # pre-processing - remove points outside 3 sigma
                points_remove = []
                if outlier_corr == True:
                    self.Xcontour = read.Xcontour
                    self.Xcontour_W = read.Xcontour_W
                    self.Ycontour = read.Ycontour
                    self.Ycontour_W = read.Ycontour_W
                    for m in range(len(l_IM_matrix)):
                        # returns list of indices for outliers and the number of points left with gray value != nan
                        p_remove, norm_points = self.MatrixRemove(l_IM_matrix[m], image_modality[m], ImName, path_save)
                        points_remove.append(p_remove)
                    for m in range(len(l_IM_matrix)):
                        # place NaN in the place of outliers
                        l_IM_matrix[m] = self.remove_points(l_IM_matrix[m], points_remove)

                    del points_remove
                    del p_remove
                    del norm_points

            except OSError:  # error if there is not directory
                continue
            except IndexError:  # empty folder
                continue

                # for m_names in arange(0, len(prefix)):
                # Texture(arguments).ret() -> function for texture calculation
                # arguments: image, structure name, image corner x, image corner x, columns, pixelSpacing, HFS or FFS,
                # structure file, list of slice positions, patient number, path to save the texture maps,
                # map name (eg. AIF1), pixel discretization, site
                # function returns: number of removed points, minimum values, maximum values,
                # structure used for calculations, mean,std, cov, skewness, kurtosis, energy, entropy, contrast,
                # correlation, homogenity, coarseness, neighContrast, busyness, complexity, intensity variation,
                # size variation, fractal dimension, number of points used in the calculations,
                # histogram (values bigger/smaller than median)
            sb.SetStatusText('Calculate ' + ImName)
            stop_calc = ''  # in case something would be wrong with the image tags
            lista_results = Texture(sb, file_type, l_IM_matrix, contour_matrix, read.structure_f, read.columns, read.rows, read.xCTspace,
                                    read.slices, path_save, ImName, pixNr, binSize, image_modality, wv, cropStructure,
                                    stop_calc, meanWV, read.Xcontour, read.Xcontour_W, read.Ycontour, read.Ycontour_W).ret()

            # final list contains of the sublist for each patient, sublist contains of [patient number,
            # structure used for calculations, list of texture parameters, number of points used for calculations]
            final.append([ImName, lista_results[2], lista_results[:2], lista_results[3:-1], lista_results[-1]])

        # save the results
        Export(final, exportList, wv, path_save, save_as, image_modality, path_image)

    def MatrixRemove(self, imap, perf, ImName, path):
        """read the gray values in the ROI, remove the NaN and fir gaussian function"""
        v = []  # read gray values in ROI
        for i in range(len(self.Xcontour)):  # slices
            for j in range(len(self.Xcontour[i])):  # sub-structres in the slice
                for k in range(len(self.Xcontour[i][j])):
                    v.append(imap[i][self.Ycontour[i][j][k]][self.Xcontour[i][j][k]])

        # remove NaN
        ind = np.where(np.isnan(np.array(v)))[0]
        for j in np.arange(1, len(ind) + 1):
            v.pop(ind[-j])
        # fitting gaussian function and deleting ouliers.
        remove = self.histfit(v, 15, perf, ImName, path)
        p_remove = []
        for j in range(len(remove)):
            p_remove.append(
                np.where(np.array(imap) == remove[j]))  # which indices corresponds to the values that should be removed
        return p_remove, len(v)

    def histfit(self, x, N_bins, name, ImName, path):
        """
        x - data
        N_bins - number of bins in the histogram
         
        plot a histrogram and a guassian funcation with mean and SD calcualte from data
        """

        x = np.array(x, dtype=np.float64)
        plt.figure(figsize=(20, 20))
        plt.subplot(121)
        n, bins, patches = plt.hist(x, N_bins, normed=True, facecolor='green', alpha=0.75)

        bincenters = 0.5 * (bins[1:] + bins[:-1])

        y = st.norm.pdf(bincenters, loc=np.mean(x), scale=np.std(np.array(x)))

        plt.plot(bincenters, y, 'r--', linewidth=1, label='std: ' + str(round(np.std(x), 2)))
        plt.plot([np.mean(x) + 3 * np.std(x), np.mean(x) + 3 * np.std(x)], [0, 0.1], 'b--')
        plt.plot([np.mean(x) - 3 * np.std(x), np.mean(x) - 3 * np.std(x)], [0, 0.1], 'b--')
        # check which values are outside the range of 3 sigma
        ind1 = np.where(np.array(x) > (np.mean(x) + 3 * np.std(x)))[0]
        ind2 = np.where(np.array(x) < (np.mean(x) - 3 * np.std(x)))[0]

        v = []  # outliers
        for i in ind1:
            v.append(x[i])
        for i in ind2:
            v.append(x[i])
        x = list(x)
        leg = str(len(v) * 100. / len(x))
        for i in v:  # remove outlier from data vector
            x.remove(i)
        plt.legend()
        # plot a histogram after the removal
        plt.subplot(122)
        n, bins, patches = plt.hist(x, N_bins, normed=True, facecolor='green', alpha=0.75,
                                   label='removed points: ' + leg)
        plt.legend()
        # save the figure
        try:
            makedirs(path + 'gauss' + os.sep)
        except OSError:
            if not isdir(path + 'gauss' + os.sep):
                raise
        plt.savefig(path + 'gauss' + os.sep + 'hist_' + name + '_' + ImName + '.png')
        plt.close()
        return v  # retruns outliers values

    def remove_points(self, M, p):
        """insert NaN in the place of the outliers"""
        for i in p:
            if i != []:
                for j in i:
                    for k in range(len(j[0])):
                        M[j[0][k]][j[1][k]][j[2][k]] = np.nan
        return M
