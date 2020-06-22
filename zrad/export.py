import sys
from os import makedirs
from os.path import isdir

from numpy import array, float64


class Export(object):
    def Preset(self, exportList, wv, local, path_save, save_as, perf_names, path_image):
        try:
            makedirs(path_save)
        except OSError:
            if not isdir(path_save):
                raise

        calcGLDZM = True
        calcNGLDM = True
        final_file = open(path_save + 'texture_' + save_as + '.txt', 'w')
        # names of the texture parameters
        par_names = ['Mean', 'SD', 'COV', 'skewness', 'kurtosis', 'var', 'median', 'percentile10', 'percentile90',
                     'iqr', 'Hrange', 'mad', 'rmad', 'H_energy', 'H_entropy', 'rms', 'H_uniformity',
                     'energy', 'entropy', 'contrast', 'correlation', 'homogenity', 'homogenity_n', 'idiff', 'idiff_n',
                     'variance', 'sum_average', 'sum_entropy', 'sum_variance', 'diff_entropy', 'diff_variance', 'IMC1',
                     'IMC2', 'MCC', 'joint_max', 'joint_average', 'diff_average', 'dissimilarity', 'inverse_variance',
                     'autocorrelation', 'clust_tendency', 'clust_shade', 'clust_prominence',
                     'M_energy', 'M_entropy', 'M_contrast', 'M_correlation', 'M_homogenity', 'M_homogenity_n',
                     'M_idiff', 'M_idiff_n', 'M_variance', 'M_sum_average', 'M_sum_entropy', 'M_sum_variance',
                     'M_diff_entropy', 'M_diff_variance', 'M_IMC1', 'M_IMC2', 'M_MCC', 'M_joint_max', 'M_joint_average',
                     'M_diff_average', 'M_dissimilarity', 'M_inverse_variance', 'M_autocorrelation', 'M_clust_tendency',
                     'M_clust_shade', 'M_clust_prominence',
                     'coarseness', 'neighContrast', 'busyness', 'complexity', 'strength',
                     'len_intensityVar', 'len_intensityVar_n', 'len_sizeVar', 'len_sizeVar_n', 'len_sse', 'len_lse',
                     'len_lgse', 'len_hgse', 'len_sslge', 'len_sshge', 'len_lslge', 'len_lshge', 'len_rpc',
                     'len_grey_lev_var', 'len_zone_size_var', 'len_size_entropy',
                     'M_len_intensityVar', 'M_len_intensityVar_n', 'M_len_sizeVar', 'M_len_sizeVar_n', 'M_len_sse',
                     'M_len_lse', 'M_len_lgse', 'M_len_hgse', 'M_len_sslge', 'M_len_sshge', 'M_len_lslge',
                     'M_len_lshge', 'M_len_rpc', 'M_len_grey_lev_var', 'M_len_zone_size_var', 'M_len_size_entropy',
                     'intensityVar', 'intensityVar_n', 'sizeVar', 'sizeVar_n', 'sse', 'lse', 'lgse', 'hgse', 'sslge',
                     'sshge', 'lslge', 'lshge', 'rpc', 'grey_lev_var', 'zone_size_var', 'size_entropy']
        if calcGLDZM:
            app = ['GLDZM_intensityVar', 'GLDZM_intensityVar_n', 'GLDZM_sizeVar', 'GLDZM_sizeVar_n', 'GLDZM_sse',
                   'GLDZM_lse', 'GLDZM_lgse', 'GLDZM_hgse', 'GLDZM_sslge', 'GLDZM_sshge', 'GLDZM_lslge', 'GLDZM_lshge',
                   'GLDZM_rpc', 'GLDZM_grey_lev_var', 'GLDZM_zone_size_var', 'GLDZM_size_entropy']
            par_names = par_names + app
        if calcNGLDM:
            app = ['NGLDM_intensityVar', 'NGLDM_intensityVar_n', 'NGLDM_sizeVar', 'NGLDM_sizeVar_n', 'NGLDM_sse',
                   'NGLDM_lse', 'NGLDM_lgse', 'NGLDM_hgse', 'NGLDM_sslge', 'NGLDM_sshge', 'NGLDM_lslge', 'NGLDM_lshge',
                   'NGLDM_grey_lev_var', 'NGLDM_zone_size_var', 'NGLDM_size_entropy', 'NGLDM_energy']
            par_names = par_names + app
        app = ['fractal_dim', 'center_mass_shift', 'MTV20%', 'MTV30%', 'MTV40%', 'MTV50%', 'MTV60%', 'MTV70%']
        par_names = par_names + app

        if wv:
            wave_names = ['', 'HHH', 'HHL', 'HLH', 'HLL', 'LHH', 'LHL', 'LLH', 'LLL']
        else:
            wave_names = ['']

        # #write the header

        final_file.write('patient')
        final_file.write('\t')
        final_file.write('organ')
        final_file.write('\t')
        final_file.write('vmin')
        final_file.write('\t')
        final_file.write('vmax')
        final_file.write('\t')
        final_file.write('voxels')
        final_file.write('\t')
        for i in range(len(perf_names)):
            for k in range(len(wave_names)):
                for j in range(len(par_names)):
                    newFeatName = wave_names[k] + "_" + par_names[j]
                    if len(perf_names) > 1:
                        newFeatName = perf_names[i] + "_" + newFeatName
                    newFeatName = newFeatName.lstrip("_")
                    final_file.write(newFeatName)
                    if i + j + k < len(perf_names) + len(wave_names) + len(par_names) - 3:
                        final_file.write("\t")

        final_file.write('\n')
        sys.stdout.flush()

        return final_file, wave_names, par_names

    def ExportResults(self, final, final_file, par_names, perf_names, wave_names, wv, local):
        # for now fixed
        calcGLDZM = True
        calcNGLDM = True

        # write the results
        # results are save as follow
        # one line corresponds to one patient
        # texture parameters are save in groups regarding perfusion maps and then the with the order from par_names list
        # rearrange the final list so wavelets have the order 'HHH', 'HHL', 'HLH', 'HLL', 'LHH', 'LHL', 'LLH', 'LLL'
        # instead of the 'LLL', 'HHH', 'HHL', 'HLH', 'HLL', 'LHH', 'LHL', 'LLH'
        if wv:
            try:
                temp = array(final[0][3], float64)
            except ValueError:
                temp = array(final[0][3])
            lll = temp[:, 1].copy()
            temp[:, 1:8] = temp[:, 2:9]  # 0 as in the new version each patient is saved separately
            temp[:, 8] = lll
            final[0][3] = temp

        if wv:
            for i in range(len(final)):
                final_file.write(str(final[i][0]))  # patient
                final_file.write('\t')
                final_file.write(str(final[i][1]))  # organ
                final_file.write('\t')
                final_file.write(str(final[i][2][0]))  # vmin list
                final_file.write('\t')
                final_file.write(str(final[i][2][1]))  # vmax list
                final_file.write('\t')
                final_file.write(str(final[i][4]))  # voxel list
                final_file.write('\t')
                for k in range(len(perf_names)):
                    for l in range(len(wave_names)):
                        for j in range(len(final[i][3])):
                            # results for a specific texture parameter
                            final_file.write(str(final[i][3][j][k * len(wave_names) + l]))
                            final_file.write('\t')
                final_file.write('\n')
        elif local:
            si = 0  # number of subvolume
            for i in range(len(final)):
                final_file.write(str(final[i][0]))
                final_file.write('\t')
                final_file.write(str(final[i][1][si]))
                final_file.write('\t')
                final_file.write(str(final[i][2][0]))
                final_file.write('\t')
                final_file.write(str(final[i][2][1]))
                final_file.write('\t')
                final_file.write(str(final[i][4]))
                final_file.write('\t')
                final_file.write(str(final[i][5]))  # voxels list
                final_file.write('\t')
                for k in range(len(perf_names)):
                    for j in range(len(par_names)):
                        final_file.write(str(final[i][3][j][si]))  # results for a specific texture parameter
                        final_file.write('\t')
                final_file.write('\n')
                si += 1
        elif not wv:
            for i in range(len(final)):
                final_file.write(str(final[i][0]))  # patient
                final_file.write('\t')
                final_file.write(str(final[i][1]))  # organ
                final_file.write('\t')
                final_file.write(str(final[i][2][0]))  # vmin list
                final_file.write('\t')
                final_file.write(str(final[i][2][1]))  # vmax list
                final_file.write('\t')
                final_file.write(str(final[i][4]))  # voxel list
                final_file.write('\t')
                for k in range(len(perf_names)):
                    for l in range(len(wave_names)):
                        for j in range(len(final[i][3])):
                            # results for a specific texture parameter
                            final_file.write(str(final[i][3][j][k * len(wave_names) + l]))
                            final_file.write('\t')
                final_file.write('\n')
        sys.stdout.flush()

        return final_file
