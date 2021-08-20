import sys
from os import makedirs
from os.path import isdir

from numpy import array, float64


class Export(object):
    def Preset(self, exportList, wv, local, path_save, save_as, perf_names):
        try:
            makedirs(path_save)
        except OSError:
            if not isdir(path_save):
                raise

        calcGLDZM = True
        calcNGLDM = True
        final_file = open(path_save + 'texture_' + save_as + '.txt', 'w')
        # names of the texture parameters
        par_names = ['hist_mean', 'hist_SD', 'hist_coeffOfVar', 'hist_skewness', 'hist_kurtosis', 'hist_variance', 'hist_median', 'hist_percentile10', 'hist_percentile90',
                     'hist_IQR', 'hist_range', 'hist_meanAbsDeviation', 'hist_robustMeanAbsDeviation', 'hist_energy', 'hist_entropy', 'hist_RMS', 'hist_uniformity',
                     'GLCM_energy', 'GLCM_entropy', 'GLCM_contrast', 'GLCM_correlation', 'GLCM_homogeneity', 'GLCM_homogeneity_norm', 'GLCM_inverese_diff', 'GLCM_inverese_diff_norm',
                     'GLCM_variance', 'GLCM_sum_average', 'GLCM_sum_entropy', 'GLCM_sum_variance', 'GLCM_diff_entropy', 'GLCM_diff_variance', 'GLCM_IMC1',
                     'GLCM_IMC2', 'GLCM_MCC', 'GLCM_joint_max', 'GLCM_joint_average', 'GLCM_diff_average', 'GLCM_dissimilarity', 'GLCM_inverse_variance',
                     'GLCM_autocorrelation', 'GLCM_clust_tendency', 'GLCM_clust_shade', 'GLCM_clust_prominence',
                     'mGLCM_energy', 'mGLCM_entropy', 'mGLCM_contrast', 'mGLCM_correlation', 'mGLCM_homogeneity', 'mGLCM_homogeneity_norm', 'mGLCM_inverese_diff', 'mGLCM_inverese_diff_norm',
                     'mGLCM_variance', 'mGLCM_sum_average', 'mGLCM_sum_entropy', 'mGLCM_sum_variance', 'mGLCM_diff_entropy', 'mGLCM_diff_variance', 'mGLCM_IMC1',
                     'mGLCM_IMC2', 'mGLCM_MCC', 'mGLCM_joint_max', 'mGLCM_joint_average', 'mGLCM_diff_average', 'mGLCM_dissimilarity', 'mGLCM_inverse_variance',
                     'mGLCM_autocorrelation', 'mGLCM_clust_tendency', 'mGLCM_clust_shade', 'mGLCM_clust_prominence',
                     'NGTDM_coarseness', 'NGTDM_contrast', 'NGTDM_busyness', 'NGTDM_complexity', 'NGTDM_strength',
                     'GLRLM_GLnonuniformity', 'GLRLM_GLnonuniformity_norm', 'GLRLM_RLnonuniformity', 'GLRLM_RLnonuniformity_norm', 'GLRLM_shortRunEmp', 'GLRLM_longRunEmp',
                     'GLRLM_lowGL_run_emp', 'GLRLM_highGL_run_emp', 'GLRLM_shortRun_lowGL_emp', 'GLRLM_shortRun_highGL_emp', 'GLRLM_longRun_lowGL_emp', 'GLRLM_longRun_highGL_emp', 'GLRLM_runPercentage',
                     'GLRLM_LRvar', 'GLRLM_RLvar', 'GLRLM_entropy',
                     'mGLRLM_GLnonuniformity', 'mGLRLM_GLnonuniformity_norm', 'mGLRLM_RLnonuniformity', 'mGLRLM_RLnonuniformity_norm', 'mGLRLM_shortRunEmp', 'mGLRLM_longRunEmp',
                     'mGLRLM_lowGL_run_emp', 'mGLRLM_highGL_run_emp', 'mGLRLM_shortRun_lowGL_emp', 'mGLRLM_shortRun_highGL_emp', 'mGLRLM_longRun_lowGL_emp', 'mGLRLM_longRun_highGL_emp', 'mGLRLM_runPercentage',
                     'mGLRLM_LRvar', 'mGLRLM_RLvar', 'mGLRLM_entropy',
                     'GLSZM_GLnonuniformity', 'GLSZM_GLnonuniformity_norm', 'GLSZM_ZSnonuniformity', 'GLSZM_ZSnonuniformity_norm', 'GLSZM_smallZoneEmp', 'GLSZM_largeZoneEmp',
                     'GLSZM_lowGL_zone_emp', 'GLSZM_highGL_zone_emp', 'GLSZM_smallZone_lowGL_emp', 'GLSZM_smallZone_highGL_emp', 'GLSZM_largeZone_lowGL_emp', 'GLSZM_largeZone_highGL_emp', 'GLSZM_zonePercentage',
                     'GLSZM_GLvar', 'GLSZM_ZSvar', 'GLSZM_entropy']
        if calcGLDZM:
            app = ['GLDZM_GLnonuniformity', 'GLDZM_GLnonuniformity_norm', 'GLDZM_ZSnonuniformity', 'GLDZM_ZSnonuniformity_norm', 'GLDZM_smallDistanceEmp', 'GLDZM_largeDistanceEmp',
                    'GLDZM_lowGL_zone_emp', 'GLDZM_highGL_zone_emp', 'GLDZM_smallDistance_lowGL_emp', 'GLDZM_smallDistance_highGL_emp', 'GLDZM_largeDistance_lowGL_emp', 'GLDZM_largeDistance_highGL_emp', 'GLDZM_zonePercentage',
                    'GLDZM_GLvar', 'GLDZM_ZSvar', 'GLDZM_entropy']
            par_names = par_names + app
        if calcNGLDM:
            app = ['NGLDM_GLnonuniformity', 'NGLDM_GLnonuniformity_norm', 'NGLDM_DCnonuniformity', 'NGLDM_DCnonuniformity_norm', 'NGLDM_lowDependenceEmp', 'NGLDM_highDependenceEmp',
                    'NGLDM_lowGL_count_emp', 'NGLDM_highGL_count_emp', 'NGLDM_lowDependence_lowGL_emp', 'NGLDM_lowDependence_highGL_emp', 'NGLDM_highDependence_lowGL_emp', 'NGLDM_highDependence_highGL_emp',
                    'NGLDM_GLvar', 'NGLDM_DCvar', 'NGLDM_DCenergy', 'NGLDM_DCentropy']
            par_names = par_names + app
        app = ['fractal_dim', 'center_mass_shift', 'MaxIntensityTumorVolume20%', 'MaxIntensityTumorVolume30%', 'MaxIntensityTumorVolume40%', 'MaxIntensityTumorVolume50%', 'MaxIntensityTumorVolume60%', 'MaxIntensityTumorVolume70%']
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
