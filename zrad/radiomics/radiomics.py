import sys

import numpy as np

from .radiomics_definitions import MorphologicalFeatures, LocalIntensityFeatures, IntensityBasedStatFeatures, GLCM, \
    GLRLM_GLSZM_GLDZM_NGLDM, NGTDM
from ..image import Image
from ..toolbox_logic import handle_uncaught_exception, validate_mask

sys.excepthook = handle_uncaught_exception


class Radiomics:

    def __init__(self,
                 aggr_dim='3D', aggr_method='AVER',
                 intensity_range=None, outlier_range=None,
                 number_of_bins=None, bin_size=None,
                 slice_weighting=False, slice_median=False):

        self.patient_morphological_mask = None
        self.patient_intensity_mask = None

        if slice_weighting and slice_median:
            raise ValueError('Only one slice median averaging is not supported with weighting strategy.')

        else:
            self.slice_weighting = slice_weighting
            self.slice_median = slice_median

        self.calc_intensity_mask = False
        if intensity_range is not None:
            self.calc_intensity_mask = True
            self.intensity_range = intensity_range
            self.discret_min_val = intensity_range[0]

        self.calc_outlier_mask = False
        if str(outlier_range).strip().replace('.', '').isdigit():
            self.calc_outlier_mask = True
            self.outlier_range = outlier_range

        self.calc_discr_bin_number = False
        self.calc_discr_bin_size = False

        if number_of_bins is not None:
            self.calc_discr_bin_number = True
            self.bin_number = number_of_bins

        if bin_size is not None:
            self.calc_discr_bin_size = True
            self.bin_size = bin_size

        if aggr_dim in ['2D', '2.5D', '3D']:
            self.aggr_dim = aggr_dim
        else:
            raise ValueError(f"Wrong aggregation dim {aggr_dim}. Available '2D', '2.5D', and '3D'.")

        if aggr_method in ['MERG', 'AVER', 'SLICE_MERG', 'DIR_MERG']:
            self.aggr_method = aggr_method
        else:
            raise ValueError(f"Wrong aggregation dim {aggr_method}. "
                             "Available 'MERG', 'AVER', 'SLICE_MERG', and 'DIR_MERG'.")


        self.morph_columns = ['morph_volume', 'morph_vol_approx', 'morph_area_mesh', 'morph_av', 'morph_comp_1',
            'morph_comp_2', 'morph_sph_dispr', 'morph_sphericity', 'morph_asphericity', 'morph_com', 'morph_diam',
            'morph_pca_maj_axis', 'morph_pca_min_axis', 'morph_pca_least_axis', 'morph_pca_elongation',
            'morph_pca_flatness', 'morph_vol_dens_aabb', 'morph_area_dens_aabb', 'morph_vol_dens_aee',
            'morph_area_dens_aee', 'morph_vol_dens_conv_hull', 'morph_area_dens_conv_hull', 'morph_integ_int']
        self.intensity_columns = ['loc_peak_loc', 'loc_peak_glob', 'stat_mean', 'stat_var', 'stat_skew', 'stat_kurt',
            'stat_median', 'stat_min', 'stat_p10', 'stat_p90', 'stat_max', 'stat_iqr', 'stat_range', 'stat_mad',
            'stat_rmad', 'stat_medad', 'stat_cov', 'stat_qcod', 'stat_energy', 'stat_rms',]
        self.discretised_intensity_columns = ['ih_mean', 'ih_var', 'ih_skew', 'ih_kurt', 'ih_median', 'ih_min',
            'ih_p10', 'ih_p90', 'ih_max', 'ih_mode', 'ih_iqr', 'ih_range', 'ih_mad', 'ih_rmad', 'ih_medad', 'ih_cov',
            'ih_qcod', 'ih_entropy', 'ih_uniformity', 'ih_max_grad', 'ih_max_grad_g', 'ih_min_grad', 'ih_min_grad_g']
        self.glcm_columns = ['cm_joint_max', 'cm_joint_avg', 'cm_joint_var', 'cm_joint_entr', 'cm_diff_avg',
            'cm_diff_var', 'cm_diff_entr', 'cm_sum_avg', 'cm_sum_var', 'cm_sum_entr', 'cm_energy', 'cm_contrast',
            'cm_dissimilarity', 'cm_inv_diff', 'cm_inv_diff_norm', 'cm_inv_diff_mom', 'cm_inv_diff_mom_norm',
            'cm_inv_var', 'cm_corr', 'cm_auto_corr', 'cm_clust_tend', 'cm_clust_shade', 'cm_clust_prom',
            'cm_info_corr1', 'cm_info_corr2']
        self.glrlm_columns = ['rlm_sre', 'rlm_lre', 'rlm_lgre', 'rlm_hgre', 'rlm_srlge', 'rlm_srhge', 'rlm_lrlge',
            'rlm_lrhge', 'rlm_glnu', 'rlm_glnu_norm', 'rlm_rlnu', 'rlm_rlnu_norm', 'rlm_r_perc', 'rlm_gl_var',
            'rlm_rl_var', 'rlm_rl_entr']
        self.glszm_columns = ['szm_sze', 'szm_lze', 'szm_lgze', 'szm_hgze', 'szm_szlge', 'szm_szhge', 'szm_lzlge',
            'szm_lzhge', 'szm_glnu', 'szm_glnu_norm', 'szm_zsnu', 'szm_zsnu_norm', 'szm_z_perc', 'szm_gl_var',
            'szm_zs_var', 'szm_zs_entr']
        self.gldzm_columns = ['dzm_sde', 'dzm_lde', 'dzm_lgze', 'dzm_hgze', 'dzm_sdlge', 'dzm_sdhge',
            'dzm_ldlge', 'dzm_ldhge', 'dzm_glnu', 'dzm_glnu_norm', 'dzm_zdnu', 'dzm_zdnu_norm', 'dzm_z_perc',
            'dzm_gl_var', 'dzm_zd_var', 'dzm_zd_entr']
        self.ngtdm_columns = ['ngt_coarseness', 'ngt_contrast', 'ngt_busyness', 'ngt_complexity', 'ngt_strength']
        self.ngldm_columns = ['ngl_lde', 'ngl_hde', 'ngl_lgce', 'ngl_hgce', 'ngl_ldlge', 'ngl_ldhge', 'ngl_hdlge',
            'ngl_hdhge', 'ngl_glnu', 'ngl_glnu_norm', 'ngl_dcnu', 'ngl_dcnu_norm', 'ngl_dc_perc', 'ngl_gl_var',
            'ngl_dc_var', 'ngl_dc_entr', 'ngl_dc_energy']

    def _rename_to_ibsi_standards(self, columns, aggr_dim, aggr_method):

        new_columns = []

        el_aggr_dim = '2_5D' if aggr_dim == '2.5D' else aggr_dim

        aggr_method_map = {'AVER': 'avg', 'DIR_MERG': 'avg', 'SLICE_MERG': 'comb', 'MERG': 'comb'}

        for el in columns:
            if el.startswith(('cm', 'rlm')):
                el_aggr_method = aggr_method_map.get(aggr_method, '')
                new_columns.append(f'{el}_{el_aggr_dim}_{el_aggr_method}')
            elif el.startswith(('szm', 'dzm', 'ngl')):
                new_columns.append(f'{el}_{el_aggr_dim}')
            else:
                new_columns.append(el)
        return new_columns

    def _morphology_and_intensity_extraction_preprocessing(self, mask):
        if self.slice_2d:
            mask_validated = mask
        else:
            mask_validated = validate_mask(mask, '3D')
        self.patient_morphological_mask = mask_validated.copy()
        self.patient_morphological_mask.array = self.patient_morphological_mask.array.astype(np.int8)
        self.patient_intensity_mask = mask_validated.copy()
        self.patient_intensity_mask.array = np.where(self.patient_intensity_mask.array > 0, self.patient_image.array, np.nan)
        self.apply_outlier_removal_and_intensity_truncation()

    def _ih_and_texture_extraction_preprocessing(self, mask):
        if self.aggr_dim != '3D':
            if self.slice_2d:
                mask_validated = mask
            else:
                mask_validated = validate_mask(mask, self.aggr_dim)
            self.patient_morphological_mask = mask_validated.copy()
            self.patient_morphological_mask.array = self.patient_morphological_mask.array.astype(np.int8)
            self.patient_intensity_mask = mask_validated.copy()
            self.patient_intensity_mask.array = np.where(self.patient_intensity_mask.array > 0, self.patient_image.array, np.nan)
            self.apply_outlier_removal_and_intensity_truncation()

    def _set_images_and_preprocess(self, image, mask, filtered_image, use_texture_preprocessing=False):
        """Set image properties and run common preprocessing steps."""
        self.slice_2d = True if 1 in image.array.shape else False
        self.orig_patient_image = image
        self.patient_image = filtered_image if filtered_image else image

        self._morphology_and_intensity_extraction_preprocessing(mask)
        if use_texture_preprocessing:
            self._ih_and_texture_extraction_preprocessing(mask)
            self.apply_discretisation()

    def extract_morphological_features(self, image, mask, filtered_image=None):
        """Extract morphological features."""
        if 1 in image.array.shape:
            self.morphological_features_ = {}
        else:
            self._set_images_and_preprocess(image, mask, filtered_image)
            self.calc_morphological_features()
            self.features_ = self.morphological_features_

    def extract_intensity_features(self, image, mask, filtered_image=None):
        """Extract intensity features."""
        self._set_images_and_preprocess(image, mask, filtered_image)
        self.calc_intensity_features()
        self.features_ = self.intensity_features_

    def extract_discr_intensity_features(self, image, mask, filtered_image=None):
        """Extract discretised intensity features."""
        self._set_images_and_preprocess(image, mask, filtered_image, use_texture_preprocessing=True)
        self.calc_discretised_intensity_features()
        self.features_ = self.intensity_features_

    def extract_glcm_features(self, image, mask, filtered_image=None):
        """Extract GLCM features."""
        self._set_images_and_preprocess(image, mask, filtered_image, use_texture_preprocessing=True)
        self.calc_glcm_features()
        self.features_ = self.glcm_features_

    def extract_glrlm_features(self, image, mask, filtered_image=None):
        """Extract GLRLM features."""
        self._set_images_and_preprocess(image, mask, filtered_image, use_texture_preprocessing=True)
        self.calc_glrlm_features()
        self.features_ = self.glrlm_features_


    def extract_glszm_gldzm_features(self, image, mask, filtered_image=None):
        """Extract GLSZM and GLDZM features."""
        self._set_images_and_preprocess(image, mask, filtered_image, use_texture_preprocessing=True)
        self.calc_glszm_gldzm_features()
        self.features_ = (self.glszm_features_ | self.gldzm_features_)

    def extract_ngtdm_features(self, image, mask, filtered_image=None):
        """Extract NGTDM features."""
        self._set_images_and_preprocess(image, mask, filtered_image, use_texture_preprocessing=True)
        self.calc_ngtdm_features()
        self.features_ = self.ngtdm_features_

    def extract_ngldm_features(self, image, mask, filtered_image=None):
        """Extract NGLDM features."""
        self._set_images_and_preprocess(image, mask, filtered_image, use_texture_preprocessing=True)
        self.calc_ngldm_features()
        self.features_ = self.ngldm_features_

    def extract_features(self, image, mask, filtered_image=None):
        """Extract All features."""
        self.extract_morphological_features(image, mask, filtered_image)
        self.extract_intensity_features(image, mask, filtered_image)
        self.extract_discr_intensity_features(image, mask, filtered_image)
        self.extract_glcm_features(image, mask, filtered_image)
        self.extract_glrlm_features(image, mask, filtered_image)
        self.extract_glszm_gldzm_features(image, mask, filtered_image)
        self.extract_ngtdm_features(image, mask, filtered_image)
        self.extract_ngldm_features(image, mask, filtered_image)
        self.features_ = (self.morphological_features_ | self.intensity_features_ | self.discr_intensity_features_
                          | self.glcm_features_ | self.glrlm_features_ | self.glszm_features_ | self.gldzm_features_ |
                          self.ngtdm_features_, self.ngldm_features_)

    def apply_discretisation(self):
        if self.calc_discr_bin_size:
            if self.calc_intensity_mask:
                self.patient_intensity_mask = self.apply_bin_size_discr(self.patient_intensity_mask,
                                                                        self.discret_min_val,
                                                                        self.bin_size)
            else:
                self.patient_intensity_mask = self.apply_bin_size_discr(self.patient_intensity_mask,
                                                                        np.nanmin(self.patient_intensity_mask.array),
                                                                        self.bin_size)
        if self.calc_discr_bin_number:
            self.patient_intensity_mask = self.apply_bin_number_discr(self.patient_intensity_mask, self.bin_number)

    def apply_outlier_removal_and_intensity_truncation(self):
        if self.calc_intensity_mask:
            intensity_range_mask = np.where((self.orig_patient_image.array <= self.intensity_range[1])
                                            & (self.orig_patient_image.array >= self.intensity_range[0]),
                                            1, 0)
            self.patient_intensity_mask = Image(array=np.where((intensity_range_mask > 0)
                                                               & (~np.isnan(self.patient_intensity_mask.array)),
                                                               self.patient_intensity_mask.array, np.nan),
                                                origin=self.patient_intensity_mask.origin,
                                                spacing=self.patient_intensity_mask.spacing,
                                                direction=self.patient_intensity_mask.direction,
                                                shape=self.patient_intensity_mask.shape)
        if self.calc_outlier_mask:
            flattened_image = np.where(self.patient_morphological_mask.array > 0,
                                       self.orig_patient_image.array, np.nan).ravel()
            valid_values = flattened_image[~np.isnan(flattened_image)]
            mean = np.mean(valid_values)
            std = np.std(valid_values)
            outlier_mask = np.where((self.orig_patient_image.array <= mean + self.outlier_range * std)
                                    & (self.orig_patient_image.array >= mean - self.outlier_range * std)
                                    & (~np.isnan(self.patient_intensity_mask.array)),
                                    1, 0)

            self.patient_intensity_mask = Image(array=np.where((outlier_mask > 0)
                                                               & (~np.isnan(self.patient_intensity_mask.array)),
                                                               self.patient_intensity_mask.array, np.nan),
                                                origin=self.patient_intensity_mask.origin,
                                                spacing=self.patient_intensity_mask.spacing,
                                                direction=self.patient_intensity_mask.direction,
                                                shape=self.patient_intensity_mask.shape)

    def apply_bin_size_discr(self, image, min_val, bin_size):

        return Image(array=np.floor((image.array - min_val) / bin_size) + 1,
                     origin=image.origin,
                     spacing=image.spacing,
                     direction=image.direction,
                     shape=image.shape)

    def apply_bin_number_discr(self, image, bin_number):

        return Image(array=np.where(image.array != np.nanmax(image.array),
                                    np.floor(bin_number * (image.array - np.nanmin(image.array))
                                             / (np.nanmax(image.array) - np.nanmin(image.array))) + 1, bin_number),
                     origin=image.origin,
                     spacing=image.spacing,
                     direction=image.direction,
                     shape=image.shape)

    def calc_intensity_features(self):

        local_intensity_features = LocalIntensityFeatures(self.patient_image.array,
                                                          self.patient_intensity_mask.array,
                                                          (self.patient_image.spacing[::-1]))

        local_intensity_features.calc_local_intensity_peak()
        local_intensity_features.calc_global_intensity_peak()

        intensity_features = IntensityBasedStatFeatures()
        intensity_features.calc_mean_intensity(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_variance(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_skewness(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_kurtosis(self.patient_intensity_mask.array)
        intensity_features.calc_median_intensity(self.patient_intensity_mask.array)
        intensity_features.calc_min_intensity(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_10th_percentile(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_90th_percentile(self.patient_intensity_mask.array)
        intensity_features.calc_max_intensity(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_iqr(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_range(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_mean_abs_deviation(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_robust_mean_abs_deviation(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_median_abs_deviation(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_variation_coef(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_quartile_coef_dispersion(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_energy(self.patient_intensity_mask.array)
        intensity_features.calc_root_mean_square_intensity(self.patient_intensity_mask.array)

        self.intensity_based_features = [local_intensity_features.local_intensity_peak,
                                         local_intensity_features.global_intensity_peak,
                                         intensity_features.mean_intensity,
                                         intensity_features.intensity_variance,
                                         intensity_features.intensity_skewness,
                                         intensity_features.intensity_kurtosis,
                                         intensity_features.median_intensity,
                                         intensity_features.min_intensity,
                                         intensity_features.intensity_10th_percentile,
                                         intensity_features.intensity_90th_percentile,
                                         intensity_features.max_intensity,
                                         intensity_features.intensity_iqr,
                                         intensity_features.intensity_range,
                                         intensity_features.intensity_based_mean_abs_deviation,
                                         intensity_features.intensity_based_robust_mean_abs_deviation,
                                         intensity_features.intensity_based_median_abs_deviation,
                                         intensity_features.intensity_based_variation_coef,
                                         intensity_features.intensity_based_quartile_coef_dispersion,
                                         intensity_features.intensity_based_energy,
                                         intensity_features.root_mean_square_intensity]

        self.intensity_features_ = dict(zip(self.intensity_columns, self.intensity_based_features))

    def calc_discretised_intensity_features(self):

        discr_intensity_features = IntensityBasedStatFeatures()
        discr_intensity_features.calc_mean_intensity(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_variance(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_skewness(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_kurtosis(self.patient_intensity_mask.array)
        discr_intensity_features.calc_median_intensity(self.patient_intensity_mask.array)
        discr_intensity_features.calc_min_intensity(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_10th_percentile(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_90th_percentile(self.patient_intensity_mask.array)
        discr_intensity_features.calc_max_intensity(self.patient_intensity_mask.array)
        discr_intensity_features.calc_discretised_intensity_mode(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_iqr(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_range(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_based_mean_abs_deviation(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_based_robust_mean_abs_deviation(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_based_median_abs_deviation(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_based_variation_coef(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_based_quartile_coef_dispersion(self.patient_intensity_mask.array)
        discr_intensity_features.calc_discretised_intensity_entropy(self.patient_intensity_mask.array)
        discr_intensity_features.calc_discretised_intensity_uniformity(self.patient_intensity_mask.array)
        discr_intensity_features.calc_max_hist_gradient(self.patient_intensity_mask.array)  # 3.4.20
        discr_intensity_features.calc_max_hist_gradient_intensity(self.patient_intensity_mask.array)  # 3.4.21
        discr_intensity_features.calc_min_hist_gradient(self.patient_intensity_mask.array)  # 3.4.22
        discr_intensity_features.calc_min_hist_gradient_intensity(self.patient_intensity_mask.array)  # 3.4.23

        self.discr_intensity_based_features = [discr_intensity_features.mean_intensity,
                                               discr_intensity_features.intensity_variance,
                                               discr_intensity_features.intensity_skewness,
                                               discr_intensity_features.intensity_kurtosis,
                                               discr_intensity_features.median_intensity,
                                               discr_intensity_features.min_intensity,
                                               discr_intensity_features.intensity_10th_percentile,
                                               discr_intensity_features.intensity_90th_percentile,
                                               discr_intensity_features.max_intensity,
                                               discr_intensity_features.intensity_hist_mode,
                                               discr_intensity_features.intensity_iqr,
                                               discr_intensity_features.intensity_range,
                                               discr_intensity_features.intensity_based_mean_abs_deviation,
                                               discr_intensity_features.intensity_based_robust_mean_abs_deviation,
                                               discr_intensity_features.intensity_based_median_abs_deviation,
                                               discr_intensity_features.intensity_based_variation_coef,
                                               discr_intensity_features.intensity_based_quartile_coef_dispersion,
                                               discr_intensity_features.discret_intensity_entropy,
                                               discr_intensity_features.discret_intensity_uniformity,
                                               discr_intensity_features.max_hist_gradient,
                                               discr_intensity_features.max_hist_gradient_intensity,
                                               discr_intensity_features.min_hist_gradient,
                                               discr_intensity_features.min_hist_gradient_intensity]

        self.discr_intensity_features_ = dict(zip(self.discretised_intensity_columns, self.discr_intensity_based_features))

    def calc_glcm_features(self):
        glcm = GLCM(image=self.patient_intensity_mask.array.T, slice_weight=self.slice_weighting,
                    slice_median=self.slice_median)
        if self.aggr_dim == '3D':
            glcm.calc_glc_3d_matrix()
            if self.aggr_method == 'AVER':
                glcm.calc_3d_averaged_glcm_features()
            elif self.aggr_method == 'MERG':
                glcm.calc_3d_merged_glcm_features()

        elif self.aggr_dim == '2.5D' or self.aggr_dim == '2D':
            glcm.calc_glc_2d_matrices()
            if self.aggr_method == 'DIR_MERG':
                glcm.calc_2_5d_direction_merged_glcm_features()
            elif self.aggr_method == 'MERG':
                glcm.calc_2_5d_merged_glcm_features()
            elif self.aggr_method == 'AVER':
                glcm.calc_2d_averaged_glcm_features()
            elif self.aggr_method == 'SLICE_MERG':
                glcm.calc_2d_slice_merged_glcm_features()

        self.glcm_features = [glcm.joint_max,
                              glcm.joint_average,
                              glcm.joint_var,
                              glcm.joint_entropy,
                              glcm.dif_average,
                              glcm.dif_var,
                              glcm.dif_entropy,
                              glcm.sum_average,
                              glcm.sum_var,
                              glcm.sum_entropy,
                              glcm.ang_second_moment,
                              glcm.contrast,
                              glcm.dissimilarity,
                              glcm.inv_diff,
                              glcm.norm_inv_diff,
                              glcm.inv_diff_moment,
                              glcm.norm_inv_diff_moment,
                              glcm.inv_variance,
                              glcm.cor,
                              glcm.autocor,
                              glcm.cluster_tendency,
                              glcm.cluster_shade,
                              glcm.cluster_prominence,
                              glcm.inf_cor_1,
                              glcm.inf_cor_2]

        self.glcm_features_ = dict(zip(self._rename_to_ibsi_standards(self.glcm_columns, self.aggr_dim, self.aggr_method), self.glcm_features))

    def calc_glrlm_features(self):
        glrlm = GLRLM_GLSZM_GLDZM_NGLDM(image=self.patient_intensity_mask.array.T, slice_weight=self.slice_weighting,
                                        slice_median=self.slice_median)
        if self.aggr_dim == '3D':

            glrlm.calc_glrl_3d_matrix()
            if self.aggr_method == 'AVER':
                glrlm.calc_3d_averaged_glrlm_features()
            elif self.aggr_method == 'MERG':
                glrlm.calc_3d_merged_glrlm_features()

        elif self.aggr_dim == '2.5D' or self.aggr_dim == '2D':
            glrlm.calc_glrl_2d_matrices()

            if self.aggr_method == 'DIR_MERG':
                glrlm.calc_2_5d_direction_merged_glrlm_features()
            elif self.aggr_method == 'MERG':
                glrlm.calc_2_5d_merged_glrlm_features()
            elif self.aggr_method == 'AVER':
                glrlm.calc_2d_averaged_glrlm_features()
            elif self.aggr_method == 'SLICE_MERG':
                glrlm.calc_2d_slice_merged_glrlm_features()

        self.glrlm_features = [glrlm.short_runs_emphasis,
                               glrlm.long_runs_emphasis,
                               glrlm.low_grey_level_run_emphasis,
                               glrlm.high_gr_lvl_emphasis,
                               glrlm.short_low_gr_lvl_emphasis,
                               glrlm.short_high_gr_lvl_emphasis,
                               glrlm.long_low_gr_lvl_emphasis,
                               glrlm.long_high_gr_lvl_emphasis,
                               glrlm.non_uniformity,
                               glrlm.norm_non_uniformity,
                               glrlm.length_non_uniformity,
                               glrlm.norm_length_non_uniformity,
                               glrlm.percentage,
                               glrlm.gr_lvl_var,
                               glrlm.length_var,
                               glrlm.entropy]

        self.glrlm_features_ = dict(zip(self._rename_to_ibsi_standards(self.glrlm_columns, self.aggr_dim, self.aggr_method), self.glrlm_features))

    def calc_glszm_gldzm_features(self):
        def extract_features(glszm_gldzm):
            return [
                glszm_gldzm.short_runs_emphasis,
                glszm_gldzm.long_runs_emphasis,
                glszm_gldzm.low_grey_level_run_emphasis,
                glszm_gldzm.high_gr_lvl_emphasis,
                glszm_gldzm.short_low_gr_lvl_emphasis,
                glszm_gldzm.short_high_gr_lvl_emphasis,
                glszm_gldzm.long_low_gr_lvl_emphasis,
                glszm_gldzm.long_high_gr_lvl_emphasis,
                glszm_gldzm.non_uniformity,
                glszm_gldzm.norm_non_uniformity,
                glszm_gldzm.length_non_uniformity,
                glszm_gldzm.norm_length_non_uniformity,
                glszm_gldzm.percentage,
                glszm_gldzm.gr_lvl_var,
                glszm_gldzm.length_var,
                glszm_gldzm.entropy,
            ]

        glszm_gldzm = GLRLM_GLSZM_GLDZM_NGLDM(
            image=self.patient_intensity_mask.array.T,
            slice_weight=self.slice_weighting,
            slice_median=self.slice_median,
        )

        if self.aggr_dim == '3D':
            glszm_gldzm.calc_glsz_gldz_3d_matrices(self.patient_morphological_mask.array.T)
            glszm_gldzm.calc_3d_glszm_features()
            self.glszm_features_ = dict(zip(self._rename_to_ibsi_standards(self.glszm_columns, self.aggr_dim, self.aggr_method), extract_features(glszm_gldzm)))

            glszm_gldzm.reset_fields()
            glszm_gldzm.calc_3d_gldzm_features()
            self.gldzm_features_ = dict(zip(self._rename_to_ibsi_standards(self.gldzm_columns, self.aggr_dim, self.aggr_method), extract_features(glszm_gldzm)))

        else:
            glszm_gldzm.calc_glsz_gldz_2d_matrices(self.patient_morphological_mask.array.T)
            if self.aggr_dim == '2.5D':
                glszm_gldzm.calc_2_5d_glszm_features()
                self.glszm_features_ = dict(zip(self._rename_to_ibsi_standards(self.glszm_columns, self.aggr_dim, self.aggr_method), extract_features(glszm_gldzm)))

                glszm_gldzm.reset_fields()
                glszm_gldzm.calc_2_5d_gldzm_features()
                self.gldzm_features_ = dict(zip(self._rename_to_ibsi_standards(self.gldzm_columns, self.aggr_dim, self.aggr_method), extract_features(glszm_gldzm)))
            else:
                glszm_gldzm.calc_2d_glszm_features()
                self.glszm_features_ = dict(zip(self._rename_to_ibsi_standards(self.glszm_columns, self.aggr_dim, self.aggr_method), extract_features(glszm_gldzm)))

                glszm_gldzm.reset_fields()
                glszm_gldzm.calc_2d_gldzm_features()
                self.gldzm_features_ = dict(zip(self._rename_to_ibsi_standards(self.gldzm_columns, self.aggr_dim, self.aggr_method), extract_features(glszm_gldzm)))

    def calc_ngtdm_features(self):
        ngtdm = NGTDM(image=self.patient_intensity_mask.array.T, slice_weight=self.slice_weighting,
                      slice_median=self.slice_median)
        if self.aggr_dim == '3D':
            ngtdm.calc_ngtd_3d_matrix()
            ngtdm.calc_3d_ngtdm_features()
        elif self.aggr_dim == '2.5D':
            ngtdm.calc_ngtd_2d_matrices()
            ngtdm.calc_2_5d_ngtdm_features()
        elif self.aggr_dim == '2D':
            ngtdm.calc_ngtd_2d_matrices()
            ngtdm.calc_2d_ngtdm_features()

        ngtdm_features = [ngtdm.coarseness,
                               ngtdm.contrast,
                               ngtdm.busyness,
                               ngtdm.complexity,
                               ngtdm.strength]

        self.ngtdm_features_ = dict(zip(self._rename_to_ibsi_standards(self.ngtdm_columns, self.aggr_dim, self.aggr_method), ngtdm_features))

    def calc_ngldm_features(self):
        ngldm = GLRLM_GLSZM_GLDZM_NGLDM(image=self.patient_intensity_mask.array.T, slice_weight=self.slice_weighting,
                                        slice_median=self.slice_median)
        if self.aggr_dim == '3D':
            ngldm.calc_ngld_3d_matrix()
            ngldm.calc_3d_ngldm_features()
        elif self.aggr_dim == '2.5D':
            ngldm.calc_ngld_2d_matrices()
            ngldm.calc_2_5d_ngldm_features()
        elif self.aggr_dim == '2D':
            ngldm.calc_ngld_2d_matrices()
            ngldm.calc_2d_ngldm_features()

        ngldm_features = [ngldm.short_runs_emphasis,
                               ngldm.long_runs_emphasis,
                               ngldm.low_grey_level_run_emphasis,
                               ngldm.high_gr_lvl_emphasis,
                               ngldm.short_low_gr_lvl_emphasis,
                               ngldm.short_high_gr_lvl_emphasis,
                               ngldm.long_low_gr_lvl_emphasis,
                               ngldm.long_high_gr_lvl_emphasis,
                               ngldm.non_uniformity,
                               ngldm.norm_non_uniformity,
                               ngldm.length_non_uniformity,
                               ngldm.norm_length_non_uniformity,
                               ngldm.percentage,
                               ngldm.gr_lvl_var,
                               ngldm.length_var,
                               ngldm.entropy,
                               ngldm.energy]

        self.ngldm_features_ = dict(zip(self._rename_to_ibsi_standards(self.ngldm_columns, self.aggr_dim, self.aggr_method), ngldm_features))

    def calc_morphological_features(self):
        morf_features = MorphologicalFeatures(self.patient_morphological_mask.array,
                                              (self.patient_morphological_mask.spacing[::-1]))
        morf_features.calc_mesh()
        morf_features.calc_vol_and_area_mesh()
        morf_features.calc_vol_count()
        morf_features.calc_surf_to_vol_ratio()
        morf_features.calc_compactness_1()
        morf_features.calc_compactness_2()
        morf_features.calc_spherical_disproportion()
        morf_features.calc_sphericity()
        morf_features.calc_asphericity()
        morf_features.calc_centre_of_shift(self.patient_intensity_mask.array)
        morf_features.calc_convex_hull()
        morf_features.calc_max_diameter()
        morf_features.calc_pca()
        morf_features.calc_major_minor_least_axes_len()
        morf_features.calc_elongation()
        morf_features.calc_flatness()
        morf_features.calc_vol_and_area_densities_aabb()
        morf_features.calc_vol_density_aee()
        morf_features.calc_area_density_aee()
        morf_features.calc_vol_density_ch()
        morf_features.calc_area_density_ch()
        morf_features.calc_integrated_intensity(self.patient_intensity_mask.array)

        morph_features = [morf_features.vol_mesh,
                               morf_features.vol_count,
                               morf_features.area_mesh,
                               morf_features.surf_to_vol_ratio,
                               morf_features.compactness_1,
                               morf_features.compactness_2,
                               morf_features.spherical_disproportion,
                               morf_features.sphericity,
                               morf_features.asphericity,
                               morf_features.centre_of_shift,
                               morf_features.max_diameter,
                               morf_features.major_axis_len,
                               morf_features.minor_axis_len,
                               morf_features.least_axis_len,
                               morf_features.elongation,
                               morf_features.flatness,
                               morf_features.vol_density_aabb,
                               morf_features.area_density_aabb,
                               morf_features.vol_density_aee,
                               morf_features.area_density_aee,
                               morf_features.vol_density_ch,
                               morf_features.area_density_ch,
                               morf_features.integrated_intensity]

        self.morphological_features_ = dict(zip(self.morph_columns, morph_features))
