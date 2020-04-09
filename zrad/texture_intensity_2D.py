# -*- coding: utf-8 -*-s

# import libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from os import makedirs
from os.path import isdir
import scipy.optimize as optimization


class Intensity(object):
    # feature initialisation (image biomarker standardisation initiative)

    def __init__(self):
        self.histogram = []
        # features
        self.min = []
        self.max = []
        self.mean = []  # 3.3.1
        self.std = []
        self.cov = []
        self.skewness = []
        self.kurtosis = []
        self.var = []
        self.median = []
        self.percentile10 = []
        self.percentile90 = []
        self.iqr = []
        self.Hrange = []
        self.mad = []
        self.rmad = []
        self.H_energy = []
        self.H_entropy = []
        self.rms = []
        self.H_uniformity = []

    def histogram_calculation(self, M, name, ImName, pixNr, path, w, structure):
        """ calculate and plot the histogram """
        M1 = []  # take all values except of nan
        for m in M:
            for i in range(len(m)):
                for j in range(len(m[i])):
                    if np.isnan(m[i][j]):
                        pass
                    else:
                        M1.append(m[i][j])

        matplotlib.rcParams.update({'font.size': 24})

        fig = plt.figure(300, figsize=(20, 20))
        try:
            fig.text(0.5, 0.95, ImName + ' ' + name)
            plt.hist(M1)
            try:
                makedirs(path + 'histogram\\')
            except OSError:
                if not isdir(path + 'histogram\\'):
                    raise
        except ValueError:
            pass

        fig.savefig(
            path + 'histogram\\' + name + '_' + ImName + '_' + structure + '_' + pixNr + '_' + str(w) + '.png')
        plt.close()
        self.histogram = M1

    def feature_calculation(self, interval):
        M1 = np.array(self.histogram)

        # standard deviation
        self.std = np.std(M1)

        # 3.3.1     mean intensity
        self.mean = np.mean(M1)

        # 3.3.2     intensity variance
        self.var = np.std(M1) ** 2

        # 3.3.3     intensity skewness
        miu = np.mean(M1)
        nom = 0
        denom = 0
        for i in M1:
            nom += (i - miu) ** 3
            denom += (i - miu) ** 2
        nom = nom / float(len(M1))
        denom = denom / float(len(M1))
        denom = np.sqrt(denom ** 3)
        self.skewness = nom / denom

        # 3.3.4     kurtosis
        miu = np.mean(M1)
        nom = 0
        denom = 0
        for i in M1:
            nom += (i - miu) ** 4
            denom += (i - miu) ** 2
        nom = nom / float(len(M1))
        denom = (denom / float(len(M1))) ** 2
        self.kurtosis = (nom / denom) - 3
        del miu

        # 3.3.5     median intensity
        self.median = np.median(M1)

        # 3.3.6     minimum intensity
        self.min = np.min(M1)

        # 3.3.7     10th intensity percentile
        p10 = np.percentile(M1, 10)
        self.percentile10 = p10
        # 3.3.8     90th intensity percentile
        p90 = np.percentile(M1, 90)
        self.percentile90 = p90

        # 3.3.9     maximum intensity
        self.max = np.max(M1)

        # 3.3.10 interquartile range
        self.iqr = np.percentile(M1, 75) - np.percentile(M1, 25)

        # 3.3.11    intensity  range
        self.Hrange = np.max(M1) - np.min(M1)

        # 3.3.12    mean absolute deviation
        self.mad = np.sum(abs((np.array(M1) - np.mean(M1)))) / float(len(M1))

        # 3.3.13    robust mean absolute deviation
        temp = list(M1)
        ind1 = np.where(np.array(temp) < p10)[0]
        for i in range(1, len(ind1) + 1):
            temp.pop(ind1[-i])
        ind2 = np.where(np.array(temp) > p90)[0]
        for i in range(1, len(ind2) + 1):
            temp.pop(ind2[-i])
        self.rmad = np.sum(abs((np.array(temp) - np.mean(temp)))) / float(len(temp))

        # 3.3.14    median absolute deviation

        # 3.3.15    intensity-based coefficient of variation
        miu = np.mean(M1)
        cov = 0
        for i in M1:
            cov += (i - miu) ** 2
        self.cov = np.sqrt(cov / float(len(M1))) / miu

        # 3.3.16    intensity-based quartile coefficient of dispersion

        # 3.3.17    intensity-based energy
        self.H_energy = np.sum(M1 ** 2)

        # 3.3.18    root mean square intensity
        self.rms = np.sqrt(np.sum(M1 ** 2) / len(M1))

        #   3.4.18   H entropy
        vmin = np.min(M1)
        dM1 = ((M1 - vmin) // interval) + 1

        s = set(dM1)
        sl = list(s)
        w = []
        for si in range(len(sl)):
            i = np.where(dM1 == sl[si])[0]
            w.append(len(i))
        p = 1.0 * np.array(w) / np.sum(w)

        self.H_entropy = -np.sum(p * np.log2(p))

        # 3.4.19    H_uniformity
        vmin = np.min(M1)
        dM1 = ((M1 - vmin) // interval) + 1

        g = list(set(dM1))
        p = []
        for gi in range(len(g)):
            ind = np.where(np.array(dM1) == g[gi])[0]
            p.append(len(ind) * 1.0 / len(dM1))
        self.H_uniformity = np.sum(np.array(p) ** 2)

    # noinspection PyTypeChecker
    def return_features(self, dictionary, m_wv):
        """ dictionary, m_wv (wavelet method) as input """
        # check if intensity features already in there
        dictionary["%shist_min" % m_wv] = self.min
        dictionary["%shist_max" % m_wv] = self.max
        dictionary["%shist_mean" % m_wv] = round(self.mean, 3)
        dictionary["%shist_sd" % m_wv] = round(self.std, 3)
        dictionary["%shist_coefficientOfVariation" % m_wv] = round(self.cov, 3)
        dictionary["%shist_skewness" % m_wv] = round(self.skewness, 3)
        dictionary["%shist_kurtosis" % m_wv] = round(self.kurtosis, 3)
        dictionary["%shist_variance" % m_wv] = round(self.var, 3)
        dictionary["%shist_median" % m_wv] = round(self.median, 3)
        dictionary["%shist_percentile10" % m_wv] = round(self.percentile10, 3)
        dictionary["%shist_percentile90" % m_wv] = round(self.percentile90, 3)
        dictionary["%shist_interquartileRange" % m_wv] = round(self.iqr, 3)  # interquartile range
        dictionary["%shist_range" % m_wv] = round(self.Hrange, 3)
        dictionary["%shist_meanAbsoluteDeviation" % m_wv] = round(self.mad, 3)  # mean absolute deviation
        dictionary["%shist_robustMeanAbsoluteDeviation" % m_wv] = round(self.rmad, 3)  # robust mean absolute deviation
        dictionary["%shist_energy" % m_wv] = round(self.H_energy, 3)
        dictionary["%shist_entropy" % m_wv] = round(self.H_entropy, 3)
        dictionary["%shist_rms" % m_wv] = round(self.rms, 3)  # root mean square
        dictionary["%shist_uniformity" % m_wv] = round(self.H_uniformity, 3)

        return dictionary


class GLCM(object):
    # feature initialisation (image biomarker standardisation initiative)
    def __init__(self, name, matrix, lista_t, n_bits):
        self.name = name
        self.matrix = matrix
        self.nrdirections = len(lista_t)
        self.directions = lista_t
        self.p_minus = []
        self.p_plus = []
        self.norm_matrix = []
        self.n_bits = n_bits
        # features:
        self.joint_max = []  # 3.6.1
        self.joint_average = []
        self.joint_variance = []
        self.joint_entropy = []
        self.diff_average = []  # 3.6.5
        self.diff_var = []
        self.diff_entropy = []
        self.sum_average = []
        self.sum_var = []
        self.sum_entropy = []  # 3.6.10
        self.angular_sec_moment = []
        self.contrast = []
        self.dissimilarity = []
        self.invers_diff = []
        self.norm_invers_diff = []  # 3.6.15
        self.invers_diff_moment = []
        self.norm_invers_diff_moment = []
        self.invers_var = []
        self.correlation = []
        self.autocorrelation = []  # 3.6.20
        self.cluster_tendency = []
        self.cluster_shade = []
        self.cluster_prominence = []
        self.info_corr1 = []
        self.info_corr2 = []  # 3.6.25
        # other features
        self.MCC = []  # maximal correlation coefficient

        # average
        self.joint_max_av = []  # 3.6.1
        self.joint_average_av = []
        self.joint_variance_av = []
        self.joint_entropy_av = []
        self.diff_average_av = []  # 3.6.5
        self.diff_var_av = []
        self.diff_entropy_av = []
        self.sum_average_av = []
        self.sum_var_av = []
        self.sum_entropy_av = []  # 3.6.10
        self.angular_sec_moment_av = []
        self.contrast_av = []
        self.dissimilarity_av = []
        self.invers_diff_av = []
        self.norm_invers_diff_av = []  # 3.6.15
        self.invers_diff_moment_av = []
        self.norm_invers_diff_moment_av = []
        self.invers_var_av = []
        self.correlation_av = []
        self.autocorrelation_av = []  # 3.6.20
        self.cluster_tendency_av = []
        self.cluster_shade_av = []
        self.cluster_prominence_av = []
        self.info_corr1_av = []
        self.info_corr2_av = []  # 3.6.25
        # other features
        self.MCC_av = []

    def matrix_calculation(self):
        """
         returns different (merged) GLCM matrices
         GLCM_mbs: GLCM of slices - directions are merged by slices (slices, x, y)
         GLCM_mbd: GLCM of directions - slices are merged by directions (directions, x, y)
         GLCM_mf: both slices and directions merged (x, y)
         GLCM: GLCM for each slice and direction - nothing merged (directions, slices, x, y)
         """
        # create for each slice one matrix for all directions
        matrix = self.matrix
        directions = self.directions
        glcm_mbs = np.zeros((len(matrix), self.n_bits, self.n_bits))  # directions merged
        glcm_mbd = np.zeros((len(directions), self.n_bits, self.n_bits))  # slices merged
        glcm_mf = np.zeros((self.n_bits, self.n_bits))  # fully merged
        glcm = np.zeros((len(directions), len(matrix), self.n_bits, self.n_bits))
        comatrix = np.zeros((len(directions), self.n_bits, self.n_bits))
        comatrix_merged = np.zeros((self.n_bits, self.n_bits))
        for i in range(len(matrix)):  # for every slice
            for y in range(len(matrix[i])):  # for every row in slice i
                for x in range(len(matrix[i][y])):  # for every column in row y of slice i
                    for d in range(len(directions)):  # 4 directions (0-3) in 2D
                        if 0 <= i + directions[d][2] < len(matrix) and 0 <= y + directions[d][1] < len(matrix[0]) \
                                and 0 <= x + directions[d][0] < len(matrix[0][0]):
                            # to check for index error
                            value1 = matrix[i][y][x]  # value in the structure matrix
                            value2 = matrix[i + directions[d][2]][y + directions[d][1]][
                                x + directions[d][0]]  # neighborhood value
                            if not np.isnan(value1) and not np.isnan(value2):
                                y_cm = int(value1)
                                x_cm = int(value2)
                                # add a count to the matrix element for positive and negative direction
                                if len(directions) == 4:  # 2D case
                                    glcm_mbs[i][y_cm][x_cm] += 1.  # directions merged
                                    glcm_mbs[i][x_cm][y_cm] += 1.
                                    glcm_mbd[d][y_cm][x_cm] += 1.  # slices merged
                                    glcm_mbd[d][x_cm][y_cm] += 1.
                                    glcm_mf[y_cm][x_cm] += 1.  # direction and slices merged
                                    glcm_mf[x_cm][y_cm] += 1.
                                    glcm[d][i][y_cm][x_cm] += 1.  # non-merged
                                    glcm[d][i][x_cm][y_cm] += 1.
                                else:  # 3D case
                                    comatrix[d][y_cm][x_cm] += 1.  # as volume, without merging
                                    comatrix[d][x_cm][y_cm] += 1.
                                    comatrix_merged[y_cm][x_cm] += 1.  # as volume, with merging (directions)
                                    comatrix_merged[x_cm][y_cm] += 1.
        if len(directions) == 4:
            return glcm, glcm_mbs, glcm_mbd, glcm_mf
        else:
            return comatrix, comatrix_merged

    def norm_marginal_calculation(self, glcm_matrix):
        """
        input: the glcm's matrix from one method
        returns:
        normalized (merged) GLCM matrices
        GLCM_mbs: GLCM of slices - directions are merged by slices (1, slices, x, y)
        GLCM_mbd: GLCM of directions - slices are merged by directions (directions, 1, x, y)
        GLCM_mf: both slices and directions merged (1, 1, x, y)
        GLCM: GLCM for each slice and direction - nothing merged (directions, slices, x, y)
        norm_matrix: normalized GLCM
        p_plus: diagonal probability. row - slice, column - probability
        p_minus: cross diagonal probability per slice
        """
        # normalize matrix
        for dd in range(self.nrdirections):
            if len(np.shape(glcm_matrix)) == 4:  # 4 d matrix (dir-slices-Ng-Ng)
                matrix_t = glcm_matrix[dd]  # norm only one directional matrix at once
            else:
                matrix_t = glcm_matrix  # len(ax0) is number of slices or directions
            if len(np.shape(matrix_t)) == 2:  # expand dim for fully merged matrix
                matrix_t = np.expand_dims(matrix_t, axis=0)
            norm_matrix = np.zeros(np.shape(matrix_t))
            for i in range(len(matrix_t)):
                counts = np.sum(matrix_t[i])  # sum of counts for each slice
                if counts != 0:
                    norm_matrix[i] = matrix_t[i] / counts
                else:
                    norm_matrix[i] = np.nan
            # marginal probabilities for each slice
            p_minus = np.zeros((len(matrix_t), self.n_bits))
            p_plus = np.zeros((len(matrix_t), 2 * self.n_bits + 3))
            for ax0 in range(len(norm_matrix)):  # for each slice
                for ax1 in range(len(norm_matrix[ax0])):
                    for ax2 in range(len(norm_matrix[ax0][ax1])):
                        p_minus[ax0][abs(ax1 - ax2)] += norm_matrix[ax0][ax1][ax2]  # diagonal probability
                        p_plus[ax0][abs(ax1 + ax2) + 2] += norm_matrix[ax0][ax1][ax2]  # cross diagonal probability
            self.p_minus.append(p_minus)
            self.p_plus.append(p_plus)
            self.norm_matrix.append(norm_matrix)
            # exit dd loop if we don't have to loop for 4th dim..
            if len(np.shape(glcm_matrix)) != 4:  # no directional matrix
                break

    def glcm_feature_calculation(self):
        """ Calculate features from 3.6
        Parameters
        -----------
        matrix to calculate features from, marginal (cross) diagonal probabilities
        norm_matrix has forms:
            d - slices - Ng - Ng
            1 - slices - Ng - Ng
            1 - directions - Ng - Ng
            1 - 1 - Ng - Ng

        returns
        -----------
        features (3.6.1 - 3.6.25)
        self.featurename is a vector with the feature value for each matrix slice
        self.featurename_average is a number with the averaged feature value
        """
        for dd in range(self.nrdirections):
            nrslices = len(self.norm_matrix[0])  # gives = 1 for glcm_mf, or slices, or directions
            matrix = self.norm_matrix[dd]  # only loops 4 times for glcm non merged

            for x in range(nrslices):  # matrix[x] is 2dimensional
                # 3.6.1 Joint Maximum
                self.joint_max.append(np.max(matrix[x]))

                # 3.6.2 Joint Average
                s = 0
                for i in range(len(matrix[x])):
                    s += (i + 1) * np.sum(matrix[x][i])  # i+1 gray values starting from 1 not from 0
                self.joint_average.append(s)

                # 3.6.3 Joint variance
                var = 0
                miu = 0
                for i in range(len(matrix[x])):
                    miu += (i + 1) * np.sum(matrix[x][i])  # i+1 gray values starting from 1 not from 0
                ind = np.where(matrix[x] != 0)  # non-zero entries only to speed up calculation
                for j in range(len(ind[0])):
                    var += (ind[0][j] + 1 - miu) ** 2 * matrix[x][ind[0][j]][
                        ind[1][j]]  # i+1 gray values starting from 1 not from 0
                if var == 0:
                    var = np.nan
                self.joint_variance.append(var)

                # 3.6.4 joint entropy / entropy
                entropy = 0
                ind = np.where(matrix[x] != 0)  # non-zero entries only to speed up calculation
                for j in range(len(ind[0])):
                    s4 = (matrix[x][ind[0][j]][ind[1][j]]) * np.log2(matrix[x][ind[0][j]][ind[1][j]])
                    if np.isnan(s4):
                        pass
                    else:
                        entropy += -s4
                if entropy == 0:  # if empty matrix
                    entropy = np.nan
                self.joint_entropy.append(entropy)
                del ind

                # 3.6.5     difference average
                # 3.6.6     difference variance
                a = 0
                v = 0
                for k in range(len(self.p_minus[dd][x])):
                    a += k * self.p_minus[dd][x][k]
                for k in range(len(self.p_minus[dd][x])):
                    v += (k - a) ** 2 * self.p_minus[dd][x][k]
                self.diff_average.append(a)
                self.diff_var.append(v)
                del a
                del v

                # 3.6.7     difference entropy
                e = 0
                for i in range(len(self.p_minus[dd][x])):
                    if self.p_minus[dd][x][i] != 0:
                        e += -self.p_minus[dd][x][i] * np.log2(self.p_minus[dd][x][i])
                self.diff_entropy.append(e)
                del e

                # 3.6.8     sum average
                # 3.6.9     sum variance
                a = 0
                v = 0
                for k in range(2, len(self.p_plus[dd][x])):
                    a += k * self.p_plus[dd][x][k]
                for k in range(2, len(self.p_plus[dd][x])):
                    v += (k - a) ** 2 * self.p_plus[dd][x][k]
                self.sum_average.append(a)
                self.sum_var.append(v)

                # 3.6.10 sum entropy
                e = 0
                for i in range(2, len(self.p_plus[dd][x])):
                    if self.p_plus[dd][x][i] != 0:
                        e += -self.p_plus[dd][x][i] * np.log2(self.p_plus[dd][x][i])
                self.sum_entropy.append(e)

                # 3.6.11 angular second moment // ENERGY
                energy = 0
                ind = np.where(matrix[x] != 0)  # non-zero entries only to speed up calculation
                for j in range(len(ind[0])):
                    energy += (matrix[x][ind[0][j]][ind[1][j]]) ** 2
                self.angular_sec_moment.append(energy)
                del ind

                # 3.6.12    contrast
                contrast = 0
                ind = np.where(matrix[x] != 0)  # non-zero entries only to speed up calculation
                for j in range(len(ind[0])):
                    contrast += ((ind[0][j] - ind[1][j]) ** 2) * matrix[x][ind[0][j]][ind[1][j]]
                if contrast == 0:
                    contrast = np.nan
                del ind
                self.contrast.append(contrast)

                # 3.6.13    dissimilarity
                ds = 0
                ind = np.where(np.array(matrix[x]) != 0)  # non-zero entries only to speed up calculation
                for i in range(len(ind[0])):
                    ds += abs(ind[0][i] - ind[1][i]) * matrix[x][ind[0][i]][ind[1][i]]
                self.dissimilarity.append(ds)
                del ind

                # 3.6.14     inverse difference (a measure of homogeneity)
                # 3.6.15     normalized
                homo = 0
                nhomo = 0
                ind = np.where(matrix[x] != 0)  # non-zero entries only to speed up calculation
                for j in range(len(ind[0])):
                    homo += matrix[x][ind[0][j]][ind[1][j]] / (1 + abs(ind[0][j] - ind[1][j]))
                    nhomo += matrix[x][ind[0][j]][ind[1][j]] / (1 + abs(ind[0][j] - ind[1][j]) / float(len(matrix[x])))
                if homo == 0:
                    homo = np.nan
                del ind
                self.invers_diff.append(homo)
                self.norm_invers_diff.append(nhomo)
                del homo
                del nhomo

                # 3.6.16    inverse different moment
                # 3.6.17    normalised inverse different moment     (previously named homogeneity)..........
                homo = 0
                nhomo = 0
                ind = np.where(matrix[x] != 0)  # non-zero entries only to speed up calculation
                for j in range(len(ind[0])):
                    homo += matrix[x][ind[0][j]][ind[1][j]] / (1 + (ind[0][j] - ind[1][j]) ** 2)
                    nhomo += matrix[x][ind[0][j]][ind[1][j]] / (
                                1 + ((ind[0][j] - ind[1][j]) / float(len(matrix[x]))) ** 2)
                if homo == 0:
                    homo = np.nan
                del ind
                self.invers_diff_moment.append(homo)
                self.norm_invers_diff_moment.append(nhomo)

                # 3.6.18    inverse variance
                f = 0
                for i in range(len(matrix[x])):
                    for j in range(i + 1, len(matrix[x][0])):
                        f += matrix[x][i][j] / (i - j) ** 2
                self.invers_var.append(2 * f)

                # 3.6.19    correlation
                mean = 0
                for i in range(len(matrix[x])):
                    mean += (i + 1) * np.sum(matrix[x][i])  # i+1 gray values starting from 1 not from 0
                std = 0
                for i in range(len(matrix[x])):
                    std += ((i + 1 - mean) ** 2) * np.sum(matrix[x][i])  # i+1 gray values starting from 1 not from 0
                std = np.sqrt(std)
                ind = np.where(np.array(matrix[x]) != 0)  # non-zero entries only to speed up calculation
                corr = 0
                for i in range(len(ind[0])):
                    corr += (ind[0][i] + 1) * (ind[1][i] + 1) * matrix[x][ind[0][i]][
                        ind[1][i]]  # i+1 gray values starting from 1 not from 0
                corr = (corr - mean ** 2) / std ** 2
                self.correlation.append(corr)
                del ind

                # 3.6.20    autocorrelation
                c = 0
                ind = np.where(np.array(matrix[x]) != 0)  # non-zero entries only to speed up calculation
                for i in range(len(ind[0])):
                    c += (ind[0][i] + 1) * (ind[1][i] + 1) * matrix[x][ind[0][i]][
                        ind[1][i]]  # i+1 gray values starting from 1 not from 0
                self.autocorrelation.append(c)
                del ind

                # 3.6.21        cluster tendency
                # 3.6.22        cluster shade
                # 3.6.23        cluster prominence
                mean = 0
                for i in range(len(matrix[x])):
                    mean += (i + 1) * np.sum(matrix[x][i])
                clust_t = 0
                clust_s = 0
                clust_p = 0
                ind = np.where(np.array(matrix[x]) != 0)  # non-zero entries only to speed up calculation
                for i in range(len(ind[0])):
                    clust_t += ((ind[0][i] + ind[1][i] + 2 - 2 * mean) ** 2) * matrix[x][ind[0][i]][
                        ind[1][i]]  # i+1 gray values starting from 1 not from 0
                    clust_s += ((ind[0][i] + ind[1][i] + 2 - 2 * mean) ** 3) * matrix[x][ind[0][i]][ind[1][i]]
                    clust_p += ((ind[0][i] + ind[1][i] + 2 - 2 * mean) ** 4) * matrix[x][ind[0][i]][ind[1][i]]
                self.cluster_tendency.append(clust_t)
                self.cluster_shade.append(clust_s)
                self.cluster_prominence.append(clust_p)
                del mean
                del ind

                # 3.6.24    information correlation 1
                # 3.6.25    information correlation 2
                hxy = self.joint_entropy[
                    x + dd * nrslices]  # all values are appended. for second direction we need to "jump" over all slices of first direction
                X = []
                for i in range(len(matrix[x])):
                    X.append(np.sum(matrix[x][i]))
                hxy1 = 0
                hxy2 = 0
                for i in range(len(matrix[x])):
                    for j in range(len(matrix[x][i])):
                        if X[i] * X[j] != 0:
                            hxy1 += -matrix[x][i][j] * np.log2(X[i] * X[j])
                            hxy2 += -X[i] * X[j] * np.log2(X[i] * X[j])
                hx = 0
                for i in range(len(X)):
                    if X[i] != 0:
                        hx += -X[i] * np.log2(X[i])
                try:
                    f12 = (hxy - hxy1) / hx
                except ZeroDivisionError:
                    f12 = np.nan
                if hxy > hxy2:
                    f13 = 0
                else:
                    f13 = np.sqrt(1 - np.exp(-2 * (hxy2 - hxy)))
                self.info_corr1.append(f12)
                self.info_corr2.append(f13)

                # MCC - maximal correlation coefficient
                try:
                    Q = np.zeros((len(matrix[x]), len(matrix[x][0])))
                    X = []
                    for i in range(len(matrix[x])):
                        X.append(np.sum(matrix[x][i]))

                    for i in range(len(matrix[x])):
                        for j in range(len(matrix[x][i])):
                            for k in range(len(X)):
                                if (X[i] * X[k]) != 0:
                                    Q[i][j] += matrix[x][i][k] * matrix[x][j][k] / (X[i] * X[k])

                    lmcc = np.linalg.eigvals(Q)

                    lmcc.sort()
                    try:
                        self.MCC.append(lmcc[-2] ** 0.5)
                    except IndexError:  # due to not sufficient number of bits in wavelet transform
                        self.MCC.append(np.nan)  # I replaced it because np.mean will give nan value if nan in it
                        # self.MCC.append('')
                except np.linalg.linalg.LinAlgError:
                    self.MCC.append(np.nan)

            if len(self.norm_matrix) != 4:  # no 4 dimensional matrix
                break

        # calculate average for all features
        self.joint_max_av = np.nanmean(self.joint_max)  # 3.6.1
        self.joint_average_av = np.nanmean(self.joint_average)
        self.joint_variance_av = np.nanmean(self.joint_variance)
        self.joint_entropy_av = np.nanmean(self.joint_entropy)
        self.diff_average_av = np.nanmean(self.diff_average)  # 3.6.5
        self.diff_var_av = np.nanmean(self.diff_var)
        self.diff_entropy_av = np.nanmean(self.diff_entropy)
        self.sum_average_av = np.nanmean(self.sum_average)
        self.sum_var_av = np.nanmean(self.sum_var)
        self.sum_entropy_av = np.nanmean(self.sum_entropy)  # 3.6.10
        self.angular_sec_moment_av = np.nanmean(self.angular_sec_moment)
        self.contrast_av = np.nanmean(self.contrast)
        self.dissimilarity_av = np.nanmean(self.dissimilarity)
        self.invers_diff_av = np.nanmean(self.invers_diff)
        self.norm_invers_diff_av = np.nanmean(self.norm_invers_diff)  # 3.6.15
        self.invers_diff_moment_av = np.nanmean(self.invers_diff_moment)
        self.norm_invers_diff_moment_av = np.nanmean(self.norm_invers_diff_moment)
        self.invers_var_av = np.nanmean(self.invers_var)
        self.correlation_av = np.nanmean(self.correlation)
        self.autocorrelation_av = np.nanmean(self.autocorrelation)  # 3.6.20
        self.cluster_tendency_av = np.nanmean(self.cluster_tendency)
        self.cluster_shade_av = np.nanmean(self.cluster_shade)
        self.cluster_prominence_av = np.nanmean(self.cluster_prominence)
        self.info_corr1_av = np.nanmean(self.info_corr1)
        self.info_corr2_av = np.nanmean(self.info_corr2)  # 3.6.25
        self.MCC_av = np.mean(self.MCC)  # leave it np.mean ... in case of index error np.nan is wanted...

    # noinspection PyTypeChecker
    def return_features(self, dictionary, m, m_wv):
        dictionary["{}GLCM_jointMax" .format(m_wv + m)] = round(self.joint_max_av, 3)  # 3.6.1
        dictionary["{}GLCM_jointAverage" .format(m_wv + m)] = round(self.joint_average_av, 3)
        dictionary["{}GLCM_jointVariance" .format(m_wv + m)] = round(self.joint_variance_av, 3)
        dictionary["{}GLCM_jointEntropy" .format(m_wv + m)] = round(self.joint_entropy_av, 3)
        dictionary["{}GLCM_differenceAverage" .format(m_wv + m)] = round(self.diff_average_av, 3)  # 3.6.5
        dictionary["{}GLCM_differenceVariance" .format(m_wv + m)] = round(self.diff_var_av, 3)
        dictionary["{}GLCM_differenceEntropy" .format(m_wv + m)] = round(self.diff_entropy_av, 3)
        dictionary["{}GLCM_sumAverage" .format(m_wv + m)] = round(self.sum_average_av, 3)
        dictionary["{}GLCM_sumVariance" .format(m_wv + m)] = round(self.sum_var_av, 3)
        dictionary["{}GLCM_sumEntropy" .format(m_wv + m)] = round(self.sum_entropy_av, 3)  # 3.6.10
        dictionary["{}GLCM_angular_sec_moment" .format(m_wv + m)] = round(self.angular_sec_moment_av, 3)
        dictionary["{}GLCM_contrast" .format(m_wv + m)] = round(self.contrast_av, 3)
        dictionary["{}GLCM_dissimilarity" .format(m_wv + m)] = round(self.dissimilarity_av, 3)
        dictionary["{}GLCM_inverseDifference" .format(m_wv + m)] = round(self.invers_diff_av, 3)
        dictionary["{}GLCM_inverseDifferenceNormalized" .format(m_wv + m)] = round(self.norm_invers_diff_av, 3)  # 3.6.15
        dictionary["{}GLCM_inverseDifferenceMoment" .format(m_wv + m)] = round(self.invers_diff_moment_av, 3)
        dictionary["{}GLCM_inverseDifferenceMomentNormalized" .format(m_wv + m)] = round(self.norm_invers_diff_moment_av, 3)
        dictionary["{}GLCM_inverseVariance" .format(m_wv + m)] = round(self.invers_var_av, 3)
        dictionary["{}GLCM_correlation" .format(m_wv + m)] = round(self.correlation_av, 3)
        dictionary["{}GLCM_autocorrelation" .format(m_wv + m)] = round(self.autocorrelation_av, 3)  # 3.6.20
        dictionary["{}GLCM_clusterTendency" .format(m_wv + m)] = round(self.cluster_tendency_av, 3)
        dictionary["{}GLCM_clusterShade" .format(m_wv + m)] = round(self.cluster_shade_av, 3)
        dictionary["{}GLCM_clusterProminence" .format(m_wv + m)] = round(self.cluster_prominence_av, 3)
        dictionary["{}GLCM_informationCorrelation1" .format(m_wv + m)] = round(self.info_corr1_av, 3)
        dictionary["{}GLCM_informationCorrelation2" .format(m_wv + m)] = round(self.info_corr2_av, 3)  # 3.6.25
        # other features
        dictionary["{}GLCM_maximalCorrelationCoefficient" .format(m_wv + m)] = round(self.MCC_av, 3)

        return dictionary


# noinspection PyPep8Naming
class GLRLM_GLSZM_GLDZM_NGLDM(object):
    # feature initialisation 3.7 (image biomarker standardisation initiative)
    def __init__(self, name, matrix, matrix_type, n_bits, lista_t):
        self.name = name
        self.matrix_type = matrix_type
        self.matrix = matrix
        self.glrlm_matrix = []
        self.matrix_norm = []
        self.n_bits = n_bits
        self.directionvector = lista_t

        # features from 3.7 / 3.8 / 3.9
        # GLRLM: x = run
        # GLSZM: x = zone
        # GLDZM: x = distance
        self.shortSmall_x_emphasis = []  # 3.7.1
        self.longLarge_x_emphasis = []
        self.low_grey_level_x_emphasis = []
        self.high_grey_level_x_emphasis = []
        self.shortSmall_x_low_grey_level_emphasis = []  # 3.7.5
        self.shortSmall_x_high_grey_level_emphasis = []
        self.longLarge_x_low_grey_level_emphasis = []
        self.longLarge_x_high_grey_level_emphasis = []
        self.grey_level_nonuniform = []
        self.normalised_grey_level_nonuniform = []  # 3.7.10
        self.x_lengthSize_nonuniform = []
        self.normalised_x_lengthSize_nonuniform = []
        self.x_percentage = []
        self.grey_level_var = []
        self.x_lengthSize_var = []  # 3.7.15
        self.xsize_entropy = []
        self.dependence_count_energy = []  # 3.11.16

        # average
        self.shortSmall_x_emphasis_av = []  # 3.7.1
        self.longLarge_x_emphasis_av = []
        self.low_grey_level_x_emphasis_av = []
        self.high_grey_level_x_emphasis_av = []
        self.shortSmall_x_low_grey_level_emphasis_av = []  # 3.7.5
        self.shortSmall_x_high_grey_level_emphasis_av = []
        self.longLarge_x_low_grey_level_emphasis_av = []
        self.longLarge_x_high_grey_level_emphasis_av = []
        self.grey_level_nonuniform_av = []
        self.normalised_grey_level_nonuniform_av = []  # 3.7.10
        self.x_lengthSize_nonuniform_av = []
        self.normalised_x_lengthSize_nonuniform_av = []
        self.x_percentage_av = []
        self.grey_level_var_av = []
        self.x_lengthSize_var_av = []  # 3.7.15
        self.xsize_entropy_av = []
        self.dependence_count_energy_av = []  # 3.11.16

    def matrix_calculation(self):
        if self.matrix_type == "GLRLM":
            # for each slice separately
            nr_slices = len(self.matrix)
            max_runlength = max(np.shape(self.matrix))  # check conditions for max run length??????????!!!!
            # a)  #direction and slices separate
            glrlm = np.zeros((len(self.directionvector), nr_slices, self.n_bits, max_runlength))
            # b) directions merged per slice (average then over slices)
            glrlm_mbs = np.zeros((nr_slices, self.n_bits, max_runlength))
            # c) slices merged per direction (then average over directions)
            glrlm_mbd = np.zeros((len(self.directionvector), self.n_bits, max_runlength))
            # d) merge all
            glrlm_mf = np.zeros((self.n_bits, max_runlength))
            # 2D calculation - for every slice
            for d in range(len(self.directionvector)):
                m = np.copy(self.matrix)
                for z in range(nr_slices):
                    for y in range(len(m[z])):  # for every row
                        for x in range(len(m[z][y])):  # for every column
                            grey_value1 = m[z][y][x]
                            if np.isnan(grey_value1):
                                continue
                            rl = 1  # run length = 1 if no adjacent voxel has the same value (index = 0)
                            if 0 <= z + self.directionvector[d][2] < len(m) and 0 <= y + self.directionvector[d][1] < \
                                    len(m[0]) and 0 <= x + self.directionvector[d][0] < len(m[0][0]):
                                # check for index error
                                grey_value2 = m[z + self.directionvector[d][2]][y + self.directionvector[d][1]][
                                    x + self.directionvector[d][0]]
                                while grey_value1 == grey_value2:  # find elements along d which have same value
                                    m[z + rl * self.directionvector[d][2]][y + rl * self.directionvector[d][1]][
                                        x + rl * self.directionvector[d][0]] = np.nan
                                    # set matrix element at grey value 2 to nan, so that we don't use it a 2nd time
                                    rl += 1  # go one step further
                                    if 0 <= z + rl * self.directionvector[d][2] < len(m) and 0 <= y + rl * \
                                            self.directionvector[d][1] < len(m[0]) and 0 <= x + rl * \
                                            self.directionvector[d][0] < len(m[0][0]):
                                        grey_value2 = \
                                        m[z + rl * self.directionvector[d][2]][y + rl * self.directionvector[d][1]][
                                            x + rl * self.directionvector[d][0]]
                                    else:  # next value won't be in index anymore. we don't change value2, but need to exit
                                        break
                                if rl > max_runlength:
                                    # add zero values to matrix
                                    max_runlength = rl

                            glrlm[d][z][int(grey_value1)][rl - 1] += 1.  # 0ind equals 1 in runlength
                            glrlm_mbs[z][int(grey_value1)][rl - 1] += 1.
                            glrlm_mbd[d][int(grey_value1)][rl - 1] += 1.
                            glrlm_mf[int(grey_value1)][rl - 1] += 1.

            # check if voxels are not accounted for twice:
            se = np.sum(glrlm_mf, axis=0)  # sums over columns
            elements = 1 * se[0] + 2 * se[1] + 3 * se[2] + 4 * se[3] + 5 * se[
                4]  # + 6 * se[5] + 7 * se[6] + 8 * se[7] + 9 * se[8]  # each column weighted according to its run length
            ee = np.sum(~np.isnan(self.matrix)) * 4

            return glrlm, glrlm_mbs, glrlm_mbd, glrlm_mf

        if self.matrix_type == "GLSZM" or self.matrix_type == "GLDZM":
            """ Distance Matrix """
            # distance to the ROI border, this distance is not exactly the same as in the IBSI, it just searching
            # for the closest nan voxel not a closest nan voxel from the original ROI
            dm = np.array(self.matrix).copy()
            (indz, indy, indx) = np.where(~np.isnan(self.matrix))
            for i in range(len(indz)):
                dist = []  # vector of distances for one voxel
                z = self.matrix[:, indy[i], indx[i]]
                nanz = np.where(np.isnan(z))[0]
                d = []
                if len(nanz) != 0:
                    d = list(abs(nanz - indz[i]))
                d.append(indz[i] + 1 - 0)
                d.append(len(self.matrix) - indz[i])
                dist.append(np.min(d))

                y = self.matrix[indz[i], :, indx[i]]
                nany = np.where(np.isnan(y))[0]
                d = []
                if len(nany) != 0:
                    d = list(abs(nany - indy[i]))
                d.append(indy[i] + 1 - 0)
                d.append(len(self.matrix[0]) - indy[i])
                dist.append(np.min(d))

                x = self.matrix[indz[i], indy[i], :]
                nanx = np.where(np.isnan(x))[0]
                d = []
                if len(nanx) != 0:
                    d = list(abs(nanx - indx[i]))
                d.append(indx[i] + 1 - 0)
                d.append(len(self.matrix[0][0]) - indx[i])
                dist.append(np.min(d))

                dm[indz[i], indy[i], indx[i]] = np.min(dist)
            dist_matrix = np.array(dm)

            '''gray-level size zone matrix'''
            '''gray-level distance zone matrix'''
            # adapted
            # Guillaume Thibault et al., ADVANCED STATISTICAL MATRICES FOR TEXTURE CHARACTERIZATION: APPLICATION TO
            # DNA CHROMATIN AND MICROTUBULE NETWORK CLASSIFICATION
            GLSZM = []  # 2. method: 2D matrices merged
            GLDZM = []
            m = np.array(self.matrix).copy()
            m.dtype = np.float
            Smax = 1  # maximal size
            Dmax = 1  # maximal distance
            for i in range(self.n_bits):
                GLSZM.append([0])
                GLDZM.append([0])
            GLSZM_perslice = np.zeros((len(m), self.n_bits, Smax))
            GLDZM_perslice = np.zeros((len(m), self.n_bits, Dmax))
            for k in range(len(self.matrix)):
                for i in range(len(self.matrix[k])):
                    for j in range(len(self.matrix[k][i])):
                        if np.isnan(m[k][i][j]):  # skip nan values
                            pass
                        else:
                            grey_value = int(m[k][i][j])
                            size = 1
                            m[k][i][j] = np.nan
                            points = self.neighbor(k, i, j, m, grey_value)  # searching for neighbors with same value
                            size += len(points)
                            zone = [[k, i, j]]  # contains coordinates of points in the zone
                            for ni in points:  # add new coord to zone
                                zone.append(ni)
                                m[ni[0]][ni[1]][ni[2]] = np.nan
                            while len(points) != 0:  # LOOK FOR NEIGHBORS OF NEIGHBORS
                                p = []
                                for n in range(len(points)):
                                    poin = self.neighbor(points[n][0], points[n][1], points[n][2], m, grey_value)
                                    for ni in poin:
                                        zone.append(ni)
                                        m[ni[0]][ni[1]][ni[2]] = np.nan
                                        p.append(ni)
                                        size += 1
                                points = p

                            if size > Smax:  # check if matrix (max zone size) needs to be enlarged
                                for s in range(len(GLSZM)):
                                    for si in range(size - Smax):
                                        GLSZM[s].append(0)
                                GLSZM_perslice = np.append(GLSZM_perslice,
                                                           np.zeros((len(self.matrix), len(GLSZM), size - Smax)),
                                                           axis=2)
                                Smax = size
                            #     GLSZM[v][size - 1] += 1
                            # else:
                            #     GLSZM[v][size - 1] += 1
                            GLSZM[grey_value][size - 1] += 1
                            GLSZM_perslice[k][grey_value][size - 1] += 1

                            # define minimum distance
                            distance = []
                            for zi in zone:
                                distance.append(dist_matrix[zi[0], zi[1], zi[2]])  # add coordinates to distance
                            min_distance = int(np.min(distance))  # find minimum distance
                            if min_distance > Dmax:
                                for s in range(len(GLDZM)):
                                    for si in range(min_distance - Dmax):
                                        GLDZM[s].append(0)
                                GLDZM_perslice = np.append(GLDZM_perslice, np.zeros(
                                    (len(self.matrix), len(GLDZM), min_distance - Dmax)), axis=2)
                                Dmax = min_distance
                            #     GLDZM[v][min_distance - 1] += 1
                            # else:
                            #     GLDZM[v][min_distance - 1] += 1
                            GLDZM[grey_value][min_distance - 1] += 1
                            GLDZM_perslice[k][grey_value][min_distance - 1] += 1

            for i in range(len(GLSZM)):
                GLSZM[i] = np.array(GLSZM[i])
            GLSZM = np.array(GLSZM)  # no normalization according to IBSI /float(np.sum(GLSZM))
            norm_GLSZM = np.sum(GLSZM)
            GLSZM.astype(np.float)

            GLDZM = np.array(GLDZM)  # no normalization according to IBSI /float(np.sum(GLSZM))
            norm_GLDZM = np.sum(GLDZM)
            GLDZM.astype(np.float)

            return GLSZM, GLDZM, GLSZM_perslice, GLDZM_perslice

        if self.matrix_type == "NGLDM":
            '''neighborhood gray-level dependence matrix'''
            matrix = np.copy(self.matrix)
            s = []
            for i in range(self.n_bits):
                s.append([0])
            maxSize = 0
            ngldm_nonmerged = np.zeros((len(self.matrix), self.n_bits, 1))
            for k in range(len(matrix)):
                for v in range(self.n_bits):
                    index = np.where(matrix[k] == v)  # search for a value level
                    for ind in range(len(index[0])):
                        temp = []
                        numerator = 0
                        # enable for 3D
                        # for z in [-1, 1]:
                        #     for y in [-1, 0, 1]:
                        #         for x in [-1, 0, 1]:
                        #             if 0 <= k + z < len(matrix) and 0 <= index[0][ind] + y < len(matrix[0]) and \
                        #                     0 <= index[1][ind] + x < len(matrix[0][0]):  # check for the index error
                        #                 temp.append(matrix[k + z][index[0][ind] + y][index[1][ind] + x])
                        #             else:
                        #                 numerator += 1
                        for y in [-1, 1]:
                            for x in [-1, 0, 1]:
                                if 0 <= index[0][ind] + y < len(matrix[0]) and 0 <= index[1][ind] + x < len(
                                        matrix[0][0]):
                                    temp.append(matrix[k][index[0][ind] + y][index[1][ind] + x])  # add neighb to temp
                                else:
                                    numerator += 1  # if not in range, one neighbor less
                        y = 0
                        for x in [-1, 1]:
                            if 0 <= index[1][ind] + x < len(matrix[0][0]):
                                temp.append(matrix[k][index[0][ind] + y][index[1][ind] + x])
                            else:
                                numerator += 1

                        ind_nan = np.where(np.isnan(np.array(temp)))[0]  # find nan neighbours
                        for n in range(1, len(ind_nan) + 1):  # remove nan neighbours
                            temp.pop(ind_nan[-n])
                        numerator += len(ind_nan)
                        if numerator != 8:  # 8 would mean no neighbours in 2D
                            # in 3D:   numerator != 26:  # if it has neigbourhood
                            size = len(np.where(np.array(temp) == v)[0])  # neighbours with the same value
                            if size > maxSize:
                                for gray in range(len(s)):
                                    for app in range(maxSize, size):
                                        s[gray].append(0)
                                ngldm_nonmerged = np.append(ngldm_nonmerged, np.zeros(
                                    (len(self.matrix), self.n_bits, size - maxSize)), axis=2)
                                maxSize = size
                            s[int(v)][size] += 1
                            ngldm_nonmerged[k][int(v)][size] += 1

            ngldm = np.array(s)
            ngldm_nonmerged = np.array(ngldm_nonmerged)
            ngldm.astype(np.float)
            ngldm_nonmerged.astype(np.float)
            return ngldm_nonmerged, ngldm

    def neighbor(self, z, y, x, matrix, v):
        """search for neighbours with the same gray level in 2D"""
        points = []
        for k in range(1):  # don't look for neighbors in z direction for 2D
            for i in range(-1, 2):
                for j in range(-1, 2):
                    try:
                        if matrix[z + k][y + i][x + j] == v and z + k >= 0 and y + i >= 0 and x + j >= 0:
                            points.append([z + k, y + i, x + j])
                    except IndexError:
                        pass
        return points

    def feature_calculation(self, matrix_dimension_type, matrix_method):
        """ Calculate features from 3.7 """
        # adapt form of matrix so that we have 4 dimensions for all cases
        matrix_t = matrix_method
        if matrix_dimension_type == "1)":
            matrix_t = matrix_method  # d, slice, Ng, run length//zone size//distance
            nv = np.sum(np.sum(~np.isnan(self.matrix), axis=1), axis=1)  # sum of voxels for each slice
        elif matrix_dimension_type == "2)":
            # add an add. dimension so that indices will match
            matrix_t = np.expand_dims(matrix_method, axis=0)  # 1, slice, Ng, run length// zone size// d
            nv = np.sum(np.sum(~np.isnan(self.matrix), axis=1), axis=1)
            if self.matrix_type == "GLRLM":
                nv = nv * len(self.directionvector)  # *4 because directions merged for GLRLM
        elif matrix_dimension_type == "3)":
            matrix_t = np.expand_dims(matrix_method, axis=1)  # d, 1, Ng, run length// zone size// d
            nv = [np.sum(~np.isnan(self.matrix))]  # sum of all voxels of whole matrix
        elif matrix_dimension_type == "4)":
            matrix_t = np.expand_dims(np.expand_dims(matrix_method, axis=0), axis=0)  # 1, 1, Ng, run length
            nv = [np.sum(~np.isnan(self.matrix))]
            if self.matrix_type == "GLRLM":
                nv = [nv[0] * len(self.directionvector)]  # *4 because directions merged for GLRLM

        nrdirection = len(matrix_t)
        nrslices = len(matrix_t[0])
        for dd in range(nrdirection):  # for all directions
            matrix = matrix_t[dd]

            for x in range(nrslices):
                sum_elements = float(np.sum(matrix[x]))  # sum over all elements of glrlrm / glszm / gldzm...

                if sum_elements == 0:
                    mylist = [self.shortSmall_x_emphasis, self.longLarge_x_emphasis, self.low_grey_level_x_emphasis,
                              self.high_grey_level_x_emphasis, self.shortSmall_x_low_grey_level_emphasis,
                              self.shortSmall_x_high_grey_level_emphasis, self.longLarge_x_low_grey_level_emphasis,
                              self.longLarge_x_high_grey_level_emphasis, self.grey_level_nonuniform,
                              self.normalised_grey_level_nonuniform, self.x_lengthSize_nonuniform,
                              self.normalised_x_lengthSize_nonuniform, self.x_percentage, self.grey_level_var,
                              self.x_lengthSize_var, self.xsize_entropy, self.dependence_count_energy]
                    # appends nan to list of variable used for example if the normalization factor equal 0
                    # -> zerodivisionerror
                    for i in mylist:
                        i.append(np.nan)
                    continue

                # 3.7.1     short runs emphasis // small zone emphasis
                sse = 0
                for i in range(len(matrix[x])):
                    for j in range(len(matrix[x][i])):
                        sse += matrix[x][i][j] / float(np.uint(j + 1) ** 2)  # place 0 in the list corresponds to size 1
                self.shortSmall_x_emphasis.append(sse / sum_elements)

                # 3.7.2     long runs emphasis // large zone emphasis
                lse = 0
                for i in range(len(matrix[x])):
                    for j in range(len(matrix[x][i])):
                        lse += matrix[x][i][j] * float(np.uint(j + 1) ** 2)  # place 0 in the list corresponds to size 1
                self.longLarge_x_emphasis.append(lse / sum_elements)

                # 3.7.3     low grey level run emphasis // lgl zone emphasis
                lgse = 0
                for i in range(len(matrix[x])):  # grey value
                    for j in range(len(matrix[x][i])):  # run length
                        lgse += matrix[x][i][j] / float(np.uint(i + 1) ** 2)  # otherwise level 0 is not included
                self.low_grey_level_x_emphasis.append(lgse / sum_elements)

                # 3.7.4     high grey level run emphasis    //  hgl zone emphasis
                hgse = 0
                for i in range(len(matrix[x])):
                    for j in range(len(matrix[x][i])):
                        hgse += matrix[x][i][j] * float(np.uint(i + 1) ** 2)  # otherwise level 0 is not included
                self.high_grey_level_x_emphasis.append(hgse / sum_elements)

                # 3.7.5     short run low grey level emphasis   // small zone lgl emphasis
                sslge = 0
                for i in range(len(matrix[x])):
                    for j in range(len(matrix[x][i])):
                        sslge += matrix[x][i][j] / float((np.uint(j + 1) ** 2 * (i + 1) ** 2))
                        # otherwise level 0 is not included
                self.shortSmall_x_low_grey_level_emphasis.append(sslge / sum_elements)

                # 3.7.6
                sshge = 0
                for i in range(len(matrix[x])):
                    for j in range(len(matrix[x][i])):
                        sshge += matrix[x][i][j] * float((i + 1) ** 2) / float(
                            np.uint(j + 1) ** 2)  # otherwise level 0 is not included
                self.shortSmall_x_high_grey_level_emphasis.append(sshge / sum_elements)

                # 3.7.7     long run low grey level emphasis
                lslge = 0
                for i in range(len(matrix[x])):
                    for j in range(len(matrix[x][i])):
                        lslge += matrix[x][i][j] * float(np.uint(j + 1) ** 2) / float(
                            (i + 1) ** 2)  # otherwise level 0 is not included
                self.longLarge_x_low_grey_level_emphasis.append(lslge / sum_elements)

                # 3.7.8     long run high grey level emphasis   // large zone hgl emphasis
                lshge = 0
                for i in range(len(matrix[x])):
                    for j in range(len(matrix[x][i])):
                        lshge += matrix[x][i][j] * float(np.uint(j + 1) ** 2 * (i + 1) ** 2)
                        # otherwise level 0 is not included
                self.longLarge_x_high_grey_level_emphasis.append(lshge / sum_elements)

                # 3.7.9     # grey level non-uniformity
                # 3.7.10    # " normalised
                var = 0
                matrix_tt = matrix[x] / np.sqrt(sum_elements)
                for m in range(len(matrix[x])):
                    s = 0
                    for n in range(len(matrix[x][m])):
                        s += matrix_tt[m][n]
                    var += s ** 2  # to avoid overflow error
                self.grey_level_nonuniform.append(var)
                self.normalised_grey_level_nonuniform.append(float(var) / sum_elements)

                # 3.7.11
                # 3.7.12   norm run length non-uniformity   // normalised zone size non-uniformity
                var = 0
                matrix_temp = matrix[x] / np.sqrt(sum_elements)
                for n in range(len(matrix_temp[0])):
                    s = 0
                    for m in range(len(matrix_temp)):
                        s += matrix_temp[m][n]
                    var += s ** 2  # to avoid overflow
                self.x_lengthSize_nonuniform.append(var)
                self.normalised_x_lengthSize_nonuniform.append(float(var) / sum_elements)

                # 3.7.13    run percentage      // zone percentage
                # nv = 0
                # for j in range(len(matrix[x][0])):  # loop over column
                #     nv += (j + 1) * np.sum(matrix[x][:, j])    # sum over column, multiply with run length / zone size
                # rpc = sum_elements / float(nv)
                # self.x_percentage.append(rpc)

                self.x_percentage.append(sum_elements / float(nv[x]))   # x chooses slice
                # original
                # nv = 0
                # for i in range(len(matrix[x])):
                #     for j in range(len(matrix[x][i])):
                #         nv += (j + 1) * matrix[x][i][j]
                # rpc = sum_elements / float(nv)
                # self.x_percentage.append(rpc)

                # 3.7.14        grey level variance
                pmatrix = matrix[x] / float(sum_elements)
                miu = 0
                for i in range(len(matrix[x])):
                    miu += (i + 1) * np.sum(pmatrix[i])
                glv = 0
                for i in range(len(matrix[x])):
                    glv += ((i + 1 - miu) ** 2) * np.sum(pmatrix[i])
                self.grey_level_var.append(glv)
                del pmatrix

                # 3.7.15        ?????????????? check if it is right lsv = run length variance    // zone size variance
                pmatrix = matrix[x] / float(sum_elements)
                miu = 0
                for i in range(len(matrix[x])):
                    for j in range(len(matrix[x][i])):
                        miu += (j + 1) * np.sum(pmatrix[i][j])
                lsv = 0
                for i in range(len(matrix[x])):
                    for j in range(len(matrix[x][i])):
                        lsv += ((j + 1 - miu) ** 2) * pmatrix[i][j]
                self.x_lengthSize_var.append(lsv)
                del pmatrix

                # 3.7.16    run entropy     // zone size entropy
                pmatrix = matrix[x] / float(sum_elements)
                ind = np.where(pmatrix != 0)
                e = 0
                for i in range(len(ind[0])):
                    e += -pmatrix[ind[0][i]][ind[1][i]] * np.log2(pmatrix[ind[0][i]][ind[1][i]])
                self.xsize_entropy.append(e)

                # 3.11.16   dependence count energy
                p_ngldm = matrix[x] / float(sum_elements)
                e = np.sum(p_ngldm ** 2)
                self.dependence_count_energy.append(e)

        # calculate average for all features
        self.shortSmall_x_emphasis_av = np.nanmean(self.shortSmall_x_emphasis)  # 3.7.1
        self.longLarge_x_emphasis_av = np.nanmean(self.longLarge_x_emphasis)
        self.low_grey_level_x_emphasis_av = np.nanmean(self.low_grey_level_x_emphasis)
        self.high_grey_level_x_emphasis_av = np.nanmean(self.high_grey_level_x_emphasis)
        self.shortSmall_x_low_grey_level_emphasis_av = np.nanmean(self.shortSmall_x_low_grey_level_emphasis)  # 3.7.5
        self.shortSmall_x_high_grey_level_emphasis_av = np.nanmean(self.shortSmall_x_high_grey_level_emphasis)
        self.longLarge_x_low_grey_level_emphasis_av = np.nanmean(self.longLarge_x_low_grey_level_emphasis)
        self.longLarge_x_high_grey_level_emphasis_av = np.nanmean(self.longLarge_x_high_grey_level_emphasis)
        self.grey_level_nonuniform_av = np.nanmean(self.grey_level_nonuniform)
        self.normalised_grey_level_nonuniform_av = np.nanmean(self.normalised_grey_level_nonuniform)  # 3.7.10         #nanss   ...................
        self.x_lengthSize_nonuniform_av = np.nanmean(self.x_lengthSize_nonuniform)
        self.normalised_x_lengthSize_nonuniform_av = np.nanmean(self.normalised_x_lengthSize_nonuniform)
        self.x_percentage_av = np.nanmean(self.x_percentage)
        self.grey_level_var_av = np.nanmean(self.grey_level_var)
        self.x_lengthSize_var_av = np.nanmean(self.x_lengthSize_var)  # 3.7.15
        self.xsize_entropy_av = np.nanmean(self.xsize_entropy)
        self.dependence_count_energy_av = np.nanmean(self.dependence_count_energy)  # 3.11.16

    # noinspection PyTypeChecker
    def return_features(self, dictionary, m, m_wv):
        """ m: method applied for feature calculation"""
        if self.matrix_type == "GLRLM":
            dictionary["{}GLRLM_sre" .format(m_wv + m)] = round(self.shortSmall_x_emphasis_av, 3)  # 3.7.1
            dictionary["{}GLRLM_lre" .format(m_wv + m)] = round(self.longLarge_x_emphasis_av, 3)
            dictionary["{}GLRLM_lgle" .format(m_wv + m)] = round(self.low_grey_level_x_emphasis_av, 3)
            dictionary["{}GLRLM_hgle" .format(m_wv + m)] = round(self.high_grey_level_x_emphasis_av, 3)
            dictionary["{}GLRLM_srlge" .format(m_wv + m)] = round(self.shortSmall_x_low_grey_level_emphasis_av, 3)  # 3.7.5
            dictionary["{}GLRLM_srhge" .format(m_wv + m)] = round(self.shortSmall_x_high_grey_level_emphasis_av, 3)
            dictionary["{}GLRLM_lrlge" .format(m_wv + m)] = round(self.longLarge_x_low_grey_level_emphasis_av, 3)
            dictionary["{}GLRLM_lrhge" .format(m_wv + m)] = round(self.longLarge_x_high_grey_level_emphasis_av, 3)
            dictionary["{}GLRLM_glnu" .format(m_wv + m)] = round(self.grey_level_nonuniform_av, 3)
            dictionary["{}GLRLM_glnuNorm" .format(m_wv + m)] = round(self.normalised_grey_level_nonuniform_av, 3)  # 3.7.10
            dictionary["{}GLRLM_rlnu" .format(m_wv + m)] = round(self.x_lengthSize_nonuniform_av, 3)
            dictionary["{}GLRLM_rlnuNorm" .format(m_wv + m)] = round(self.normalised_x_lengthSize_nonuniform_av, 3)
            dictionary["{}GLRLM_runPercentage" .format(m_wv + m)] = round(self.x_percentage_av, 3)
            dictionary["{}GLRLM_glVar" .format(m_wv + m)] = round(self.grey_level_var_av, 3)
            dictionary["{}GLRLM_rlVar" .format(m_wv + m)] = round(self.x_lengthSize_var_av, 3)  # 3.7.15
            dictionary["{}GLRLM_runEntropy" .format(m_wv + m)] = round(self.xsize_entropy_av, 3)

        if self.matrix_type == "GLSZM":
            # features from 3.8
            dictionary["{}GLSZM_sze" .format(m_wv + m)] = round(self.shortSmall_x_emphasis_av, 3)  # 3.7.1
            dictionary["{}GLSZM_lze" .format(m_wv + m)] = round(self.longLarge_x_emphasis_av, 3)
            dictionary["{}GLSZM_lgze" .format(m_wv + m)] = round(self.low_grey_level_x_emphasis_av, 3)
            dictionary["{}GLSZM_hgze" .format(m_wv + m)] = round(self.high_grey_level_x_emphasis_av, 3)
            dictionary["{}GLSZM_szlge" .format(m_wv + m)] = round(self.shortSmall_x_low_grey_level_emphasis_av, 3)  # 3.7.5
            dictionary["{}GLSZM_szhge" .format(m_wv + m)] = round(self.shortSmall_x_high_grey_level_emphasis_av, 3)
            dictionary["{}GLSZM_lzlge" .format(m_wv + m)] = round(self.longLarge_x_low_grey_level_emphasis_av, 3)
            dictionary["{}GLSZM_lzhge" .format(m_wv + m)] = round(self.longLarge_x_high_grey_level_emphasis_av, 3)
            dictionary["{}GLSZM_glnu" .format(m_wv + m)] = round(self.grey_level_nonuniform_av, 3)
            dictionary["{}GLSZM_glnuNorm" .format(m_wv + m)] = round(self.normalised_grey_level_nonuniform_av, 3)  # 3.7.10
            dictionary["{}GLSZM_zsnu" .format(m_wv + m)] = round(self.x_lengthSize_nonuniform_av, 3)
            dictionary["{}GLSZM_zsnuNorm" .format(m_wv + m)] = round(self.normalised_x_lengthSize_nonuniform_av, 3)
            dictionary["{}GLSZM_zonePercentage" .format(m_wv + m)] = round(self.x_percentage_av, 3)
            dictionary["{}GLSZM_glVar" .format(m_wv + m)] = round(self.grey_level_var_av, 3)
            dictionary["{}GLSZM_zsVar" .format(m_wv + m)] = round(self.x_lengthSize_var_av, 3)  # 3.7.15
            dictionary["{}GLSZM_zsEntropy" .format(m_wv + m)] = round(self.xsize_entropy_av, 3)

        if self.matrix_type == "GLDZM":  # 3.9
            dictionary["{}GLDZM_sde" .format(m_wv + m)] = round(self.shortSmall_x_emphasis_av, 3)  # 3.7.1
            dictionary["{}GLDZM_lde" .format(m_wv + m)] = round(self.longLarge_x_emphasis_av, 3)
            dictionary["{}GLDZM_lgze" .format(m_wv + m)] = round(self.low_grey_level_x_emphasis_av, 3)
            dictionary["{}GLDZM_hgze" .format(m_wv + m)] = round(self.high_grey_level_x_emphasis_av, 3)
            dictionary["{}GLDZM_sdlge" .format(m_wv + m)] = round(self.shortSmall_x_low_grey_level_emphasis_av, 3)  # 3.7.5
            dictionary["{}GLDZM_sdhge" .format(m_wv + m)] = round(self.shortSmall_x_high_grey_level_emphasis_av, 3)
            dictionary["{}GLDZM_ldlge" .format(m_wv + m)] = round(self.longLarge_x_low_grey_level_emphasis_av, 3)
            dictionary["{}GLDZM_ldhge" .format(m_wv + m)] = round(self.longLarge_x_high_grey_level_emphasis_av, 3)
            dictionary["{}GLDZM_glnu" .format(m_wv + m)] = round(self.grey_level_nonuniform_av, 3)
            dictionary["{}GLDZM_glnuNorm" .format(m_wv + m)] = round(self.normalised_grey_level_nonuniform_av, 3)  # 3.7.10
            dictionary["{}GLDZM_zdnu" .format(m_wv + m)] = round(self.x_lengthSize_nonuniform_av, 3)
            dictionary["{}GLDZM_zdnuNorm" .format(m_wv + m)] = round(self.normalised_x_lengthSize_nonuniform_av, 3)
            # this is redundant to GLSZM: GLDZM won't give right value for voxel amount
            # dictionary["{}GLDZM_zone_percentage" .format(m_wv + m)] = dictionary["{}GLSZM_zone_percentage"]]
            dictionary["{}GLDZM_zonePercentage" .format(m_wv + m)] = round(self.x_percentage_av, 3)
            dictionary["{}GLDZM_glVar" .format(m_wv + m)] = round(self.grey_level_var_av, 3)
            dictionary["{}GLDZM_zdVar" .format(m_wv + m)] = round(self.x_lengthSize_var_av, 3)  # 3.7.15
            dictionary["{}GLDZM_zdEntropy" .format(m_wv + m)] = round(self.xsize_entropy_av, 3)

        if self.matrix_type == "NGLDM":
            # features from 3.11
            dictionary["{}NGLDM_lde" .format(m_wv + m)] = round(self.shortSmall_x_emphasis_av, 3)  # 3.7.1
            dictionary["{}NGLDM_hde" .format(m_wv + m)] = round(self.longLarge_x_emphasis_av, 3)
            dictionary["{}NGLDM_lgce" .format(m_wv + m)] = round(self.low_grey_level_x_emphasis_av, 3)
            dictionary["{}NGLDM_hgce" .format(m_wv + m)] = round(self.high_grey_level_x_emphasis_av, 3)
            dictionary["{}NGLDM_ldlge" .format(m_wv + m)] = round(self.shortSmall_x_low_grey_level_emphasis_av, 3)  # 3.7.5
            dictionary["{}NGLDM_ldhge" .format(m_wv + m)] = round(self.shortSmall_x_high_grey_level_emphasis_av, 3)
            dictionary["{}NGLDM_hdlge" .format(m_wv + m)] = round(self.longLarge_x_low_grey_level_emphasis_av, 3)
            dictionary["{}NGLDM_hdhge" .format(m_wv + m)] = round(self.longLarge_x_high_grey_level_emphasis_av, 3)
            dictionary["{}NGLDM_glnu" .format(m_wv + m)] = round(self.grey_level_nonuniform_av, 3)
            dictionary["{}NGLDM_glnuNorm" .format(m_wv + m)] = round(self.normalised_grey_level_nonuniform_av, 3)  # 3.7.10
            dictionary["{}NGLDM_dcnu" .format(m_wv + m)] = round(self.x_lengthSize_nonuniform_av, 3)
            dictionary["{}NGLDM_dcnuNorm" .format(m_wv + m)] = round(self.normalised_x_lengthSize_nonuniform_av, 3)  # not listed for NGLDM!
            dictionary["{}NGLDM_dcPercentage" .format(m_wv + m)] = round(self.x_percentage_av, 3)  # 3.11.12
            dictionary["{}NGLDM_glVar" .format(m_wv + m)] = round(self.grey_level_var_av, 3)  # 3.11.13
            dictionary["{}NGLDM_dcVar" .format(m_wv + m)] = round(self.x_lengthSize_var_av, 3)  # 3.11.14
            dictionary["{}NGLDM_dcEntropy" .format(m_wv + m)] = round(self.xsize_entropy_av, 3)
            dictionary["{}NGLDM_dcEnergy" .format(m_wv + m)] = round(self.dependence_count_energy_av, 3)  # 3.11.16

        return dictionary


class NGTDM(object):
    # feature initialisation 3.10 (image biomarker standardisation initiative)
    def __init__(self, name, matrix, dimension, n_bits):
        """ name: NGTDM-2d (or NGTDM-3d), dimension: 2D (or 3D)"""
        self.name = name
        self.matrix = matrix  # original matrix
        self.ngtdm_matrix = []
        self.ngtdm_voxelcount = []
        self.dim = dimension
        self.n_bits = n_bits
        self.ngtdm_merged = []  # calculated per slice (only 2D neighbours, then merged)
        self.ngtdm_voxelcount_merged = []

        # features
        self.coarseness = []  # 3.10.1
        self.contrast = []
        self.busyness = []
        self.complexity = []
        self.strength = []  # 3.10.5

    def ngtdm_matrix_calculation(self):
        # Amadasun et al. Textural Features Corresponding to Textural Properties
        """neighborhood gray-tone difference matrix
        dimensions: 2d or 3d - calculate slices separately or not
        ngtdm_matrix: 2D: ax0 = slices, ax1 = average neighbour level per grey level.
                    3D: av. neigh. level per grey level
        ngtdm_norm: 2D: ax0 = slices, ax1 = voxel count per grey level.
                    3D: voxel count per grey level"""

        matrix = self.matrix
        nr_slices = len(self.matrix)
        s = np.zeros((nr_slices, self.n_bits))  # difference from average neighborhood gray value
        ni = np.zeros((nr_slices, self.n_bits))  # number of voxels of a given gray level
        if self.dim == "2D" or self.dim == "2D_singleSlice":
            max_neighbours = 8
        else:
            max_neighbours = 26
        for k in range(nr_slices):
            for v in range(self.n_bits):  # for each grey level
                index = np.where(matrix[k] == v)  # search for a value level
                for ind in range(len(index[0])):    # for each element of grey value v look at neighbors
                    temp = []
                    numerator = 0
                    # get values of all neighbours
                    if self.dim == "3D":
                        for z in [-1, 1]:  # neighbors in z direction only if 3d..
                            for y in [-1, 0, 1]:
                                for x in [-1, 0, 1]:
                                    if 0 <= k + z < len(matrix) and 0 <= index[0][ind] + y < len(matrix[0]) and \
                                            0 <= index[1][ind] + x < len(matrix[0][0]):  # check for the index error
                                        temp.append(matrix[k + z][index[0][ind] + y][index[1][ind] + x])
                                    else:
                                        numerator += 1
                    for y in [-1, 1]:
                        for x in [-1, 0, 1]:
                            if 0 <= index[0][ind] + y < len(matrix[0]) and 0 <= index[1][ind] + x < len(matrix[0][0]):
                                temp.append(matrix[k][index[0][ind] + y][index[1][ind] + x])    # add neighbor grey value to temp
                            else: # if neighbor out of index
                                numerator += 1
                    for x in [-1, 1]:
                        if 0 <= index[1][ind] + x < len(matrix[0][0]):
                            temp.append(matrix[k][index[0][ind]][index[1][ind] + x])
                        else:
                            numerator += 1
                    ind_nan = np.where(np.isnan(np.array(temp)))[0]     # no nan values in neighborhood for calculation
                    for n in range(1, len(ind_nan) + 1):
                        temp.pop(ind_nan[-n])
                    numerator += len(ind_nan)
                    if numerator != 26 and self.dim == "3D" or numerator != 8 and self.dim == "2D" or numerator != 8 and self.dim == "2D_singleSlice":
                        a = abs(v - (float(np.sum(temp)) / (max_neighbours - numerator)))
                        s[k][v] += a
                        ni[k][v] += 1
        if self.dim == "2D" or self.dim == "2D_singleSlice":
            self.ngtdm_merged = np.sum(s, axis=0)    # calculated per slice (only 2D neighbours, then merged)
            self.ngtdm_voxelcount_merged = np.sum(ni, axis=0)
        if self.dim == "3D":
            s = np.sum(s, axis=0)
            ni = np.sum(ni, axis=0)
        self.ngtdm_matrix = s  # average neighbor level difference per value
        self.ngtdm_voxelcount = ni  # voxel count per level

    def feature_calculation(self, method):
        """ calculates feature for 2d or 3d case. for 3d it returns one value, for 2d it returns one value per slice"""
        s = self.ngtdm_matrix
        ni = self.ngtdm_voxelcount
        if method == "merged-2d":
            s = self.ngtdm_merged
            ni = self.ngtdm_voxelcount_merged
            self.coarseness = []
            self.contrast = []
            self.busyness = []
            self.complexity = []
            self.strength = []

        if len(np.shape(s)) == 1:  # for the 2d-merged and 3d case
            s = np.expand_dims(s, axis=0)
            ni = np.expand_dims(ni, axis=0)

        for x in range(len(s)):  # len s = nr slices or =1 for 3d
            # 3.10.1 coarseness
            f = 0
            ind = np.where(np.array(ni[x]) != 0)[0]
            for i in ind:
                f += s[x][i] * ni[x][i] / np.sum(ni[x])
            if f == 0:
                f = np.nan
            else:
                f = 1. / (0.000000001 + f)
            self.coarseness.append(f)
            del f

            # 3.10.2 contrast
            try:
                ng = len(np.where(np.array(ni[x]) != 0)[0])
                ind = np.where(np.array(ni[x]) != 0)[0]
                s1 = 0
                for i in ind:
                    for j in ind:
                        s1 += float(ni[x][i]) / (np.sum(ni[x])) * float(ni[x][j]) / (np.sum(ni[x])) * (i - j) ** 2
                s2 = 0
                ind = np.where(np.array(s[x]) != 0)[0]
                for i in ind:
                    s2 += s[x][i]

                f = (1. / (ng * (ng - 1))) * s1 * (1. / (np.sum(ni[x]))) * s2
                if f == 0:
                    self.contrast.append(np.nan)
                else:
                    self.contrast.append(f)
                del f
            except ZeroDivisionError:
                self.contrast.append('')  # not sure if this works '' !!!!!!!!!!!!!!!!!!!!!!! (return '')

            # 3.10.3 busyness
            try:
                nom = 0
                denom = 0
                ind = np.where(np.array(ni[x]) != 0)[0]
                for i in ind:
                    nom += s[x][i] * ni[x][i] / np.sum(ni[x])
                    for j in ind:
                        denom += abs(float((i + 1) * ni[x][i]) / (np.sum(ni[x])) - float((j + 1) * ni[x][j]) / (
                            np.sum(ni[x])))  # to adapt i = [1:Ng]
                if nom / denom == 0:
                    self.busyness.append(np.nan)
                else:
                    self.busyness.append(nom / denom)  # don't divide by 2 to adapt for the oncoray
            except ZeroDivisionError:
                self.busyness.append('')

            # 3.10.4 complexity
            ind = np.where(np.array(ni[x]) != 0)[0]
            s1 = 0
            for i in ind:
                for j in ind:
                    s1 += (abs(i - j) / (float(ni[x][i]) + float(ni[x][j]))) * (
                            (s[x][i] * float(ni[x][i]) / (np.sum(ni[x]))) + (
                                s[x][j] * float(ni[x][j]) / (np.sum(ni[x]))))
            if s1 == 0:
                s1 = np.nan
            self.complexity.append(s1)

            # 3.10.5 strength
            ind = np.where(np.array(ni[x]) != 0)[0]
            s1 = 0
            for i in ind:
                for j in ind:
                    s1 += ((float(ni[x][i]) + float(ni[x][j])) / np.sum(ni[x])) * (i - j) ** 2
            s2 = np.sum(s[x])
            strength = s1 / s2
            if s2 == 0:
                strength = 0
            self.strength.append(strength)

        if self.dim == "2D" and method != "merged":  # for method 1: features have to be averaged over slices
            self.coarseness = round(np.nanmean(self.coarseness), 4)  # too high values ????????
            self.contrast = round(np.nanmean([nn for nn in self.contrast if nn != '']), 4)  # get rid of '' for mean calc
            self.busyness = round(np.nanmean([nn for nn in self.busyness if nn != '']), 4)  # '' values
            self.complexity = round(np.nanmean(self.complexity), 4)  # nan values
            self.strength = round(np.mean(self.strength), 4)  # 0 values

    def return_features(self, dictionary, m, m_wv):
        dictionary["{}NGTDM_coarseness" .format(m_wv + m)] = self.coarseness
        dictionary["{}NGTDM_contrast" .format(m_wv + m)] = self.contrast
        dictionary["{}NGTDM_busyness" .format(m_wv + m)] = self.busyness
        dictionary["{}NGTDM_complexity" .format(m_wv + m)] = self.complexity
        dictionary["{}NGTDM_strength" .format(m_wv + m)] = self.strength

        return dictionary


class CMS_MTV(object):

    def __init__(self, matrix, path, pixNr, ImName, matrix_v, xCTspace, zCTspace, structure, dimension):
        self.matrix_v = matrix_v
        self.matrix = matrix
        self.voxel_size = xCTspace
        self.slice_thickness = zCTspace
        self.path = path
        self.pixNr = pixNr
        self.ImName = ImName
        self.structure = structure
        self.dim = dimension

        # features
        self.cms = self.center_mass_shift()
        self.mtv = self.metabolic_tumor_volume()
        self.frac_dim = self.fractal()

    def center_mass_shift(self):  # shift it into shape features!!!!!!!!!!!!!!!! ?????????????????????
        # calculated on original gray values
        if self.dim == "2D" or self.dim == "2D_singleSlice":
            matrix = np.copy(self.matrix_v)
            # calculate center of mass shift for each slice. average results
            cms_2d = []
            for s in range(len(self.matrix)):
                ind = np.where(~np.isnan(matrix[s]))
                ind_r = list(ind)

                ind_r[0] = ind_r[0] * self.voxel_size
                ind_r[1] = ind_r[1] * self.voxel_size
                # ind_r[2] = ind_r[2] * self.voxel_size
                # geometrical center
                geo = np.array([np.sum(ind_r[0]), np.sum(ind_r[1])])
                geo = geo / float(len(ind[0]))
                # weighted according to grey value
                gl = np.array([np.sum(ind_r[0] * matrix[s][ind]), np.sum(ind_r[1] * matrix[s][ind])])
                gl = gl / np.sum(matrix[s][ind])
                cms = geo - gl
                cms_2d.append(np.sqrt(np.sum(cms ** 2)))
            return np.mean(cms_2d)
        else:
            matrix = np.copy(self.matrix_v)
            ind = np.where(~np.isnan(matrix))
            ind_r = list(ind)

            ind_r[0] = ind_r[0] * self.voxel_size
            ind_r[1] = ind_r[1] * self.voxel_size
            ind_r[2] = ind_r[2] * self.voxel_size
            # geometrical center
            geo = np.array([np.sum(ind_r[0]), np.sum(ind_r[1]), np.sum(ind_r[2])])
            geo = geo / float(len(ind[0]))
            # weighted according to grey value
            gl = np.array(
                [np.sum(ind_r[0] * matrix[ind]), np.sum(ind_r[1] * matrix[ind]), np.sum(ind_r[2] * matrix[ind])])
            gl = gl / np.sum(matrix[ind])
            cms = geo - gl
            cms = np.sqrt(np.sum(cms ** 2))
            return cms

    def metabolic_tumor_volume(self):
        # calculated on original gray values
        percent = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        mtv = []
        # find vmax
        vmax = np.nanmax(self.matrix_v)
        for p in percent:
            ind = np.where(self.matrix_v > p * vmax)
            if len(ind) != 3:
                mtv.append('')
            else:
                # calculate volume of voxels with a high activity
                if self.dim == "3D":
                    vol = len(ind[0]) * self.voxel_size ** 3
                else:  # in 2D case, there is no interpolation in z direction. voxel size not isotropic.
                    vol = len(ind[0]) * self.voxel_size ** 2 * self.slice_thickness
                mtv.append(vol)
        return mtv

    def fractal(self):  # check if this is corrclty implemented for 2 D cases!!!!!!!!
        # https://en.wikipedia.org/wiki/Box_counting
        """fractal dimension"""
        m = self.matrix
        path = self.path
        pixNr = self.pixNr
        ImName = self.ImName
        try:
            def func_lin(x, a, b):
                return x * a + b

            if self.dim != "3D":
                maxR = np.min([len(m[0]), len(m[0][0])])  # take min of x / y dimension for 2D cases
            else:
                maxR = np.min([len(m), len(m[0]), len(m[0][0])])  # take min of all directions

            frac = []
            for r in range(2, maxR + 1):  # because log(1) = 0
                N = 0
                if self.dim == "2D":  # in 2D several slices case, we want to iterate through each slice.
                    rz = 1
                    rzz = 0
                else:
                    rz = r
                    rzz = r
                for z in range(len(m), rz):  #
                    for y in range(len(m[0]), r):
                        for x in range(len(m[0][0]), r):
                            m = np.array(m)
                            matrix = m[z:z + rzz, y:y + r, x:x + r]  # doesn't produce indexerror
                            ind = len(np.where(np.isnan(matrix))[0])
                            if ind < (r ** 3) and self.dim == "3D" or ind < (r**2) and self.dim != "3D":
                                N += 1
                frac.append(np.log(N))
            x = np.log(range(2, maxR + 1))
            xdata = np.transpose(x)
            x0 = np.array([0, 0])  # initial guess
            result = optimization.curve_fit(func_lin, xdata, frac, x0)
            fit = func_lin(x, result[0][0], result[0][1])
            plt.figure(2000)
            ax = plt.subplot(111)
            plt.plot(x, frac, 'o')
            plt.plot(x, fit, label='dim = ' + str(-round(result[0][0], 2)))
            plt.xlabel('ln(r)')
            plt.ylabel('ln(N(r))')
            plt.legend()
            # print path+'fractals\\'+ImName+'.png'
            try:
                makedirs(path + 'fractals\\')
            except OSError:
                if not isdir(path + 'fractals\\'):
                    raise
            plt.savefig(path + 'fractals\\' + ImName + '_' + self.structure + '_' + pixNr + '.png')
            plt.close()
            return -result[0][0]
        except TypeError:
            return ''
            pass

    def return_features(self, dictionary, m_wv):
        dictionary["%sfrac_dim" % m_wv] = round(self.frac_dim, 3)
        dictionary["%scms" % m_wv] = self.cms
        dictionary["%smtv2" % m_wv] = self.mtv[0]
        dictionary["%smtv3" % m_wv] = self.mtv[1]
        dictionary["%smtv4" % m_wv] = self.mtv[2]
        dictionary["%smtv5" % m_wv] = self.mtv[3]
        dictionary["%smtv6" % m_wv] = self.mtv[4]
        dictionary["%smtv7" % m_wv] = self.mtv[5]

        return dictionary
